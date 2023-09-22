import torch
from torch import nn
import torch.nn.functional as F
from opt import get_opts
import os
import glob
import imageio
import numpy as np
import cv2
import math
import random
from einops import rearrange

# data
from torch.utils.data import DataLoader
from datasets import dataset_dict
from datasets.ray_utils import axisangle_to_R, get_rays
from datasets.stylize_tools.utils import Stylizer

# models
from kornia.utils.grid import create_meshgrid3d
from models.networks import NGP
from models.implicit_mask import implicit_mask
from models.rendering import render

# optimizer, losses
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from losses import NeRFLoss

# metrics
from torchmetrics import PeakSignalNoiseRatio

# pytorch-lightning
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from utils import slim_ckpt

# render path
from tqdm import trange
from utils import load_ckpt

import warnings; warnings.filterwarnings("ignore")


def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min())
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img


class StylizeSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        print('start to stylize')
        self.save_hyperparameters(hparams)

        self.loss = NeRFLoss()
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)

        self.rgb_act = 'Sigmoid'
        # import ipdb; ipdb.set_trace()
        self.model = NGP(scale=self.hparams.scale, rgb_act=self.rgb_act, use_skybox=hparams.use_skybox, embed_a=hparams.embed_a, embed_a_len=hparams.embed_a_len)
        assert hparams.weight_path is not None
        load_ckpt(self.model, self.hparams.weight_path, prefixes_to_ignore=['embedding_a'])
      
        self.N_imgs = 0
        if hparams.dataset_name == 'kitti':
            self.N_imgs = 2 * hparams.train_frames
        else:
            if os.path.exists(os.path.join(hparams.root_dir, 'images')):
                img_dir_name = 'images'
            elif os.path.exists(os.path.join(hparams.root_dir, 'rgb')):
                img_dir_name = 'rgb'
            elif os.path.exists(os.path.join(hparams.root_dir, f'images_{int(1/hparams.downsample)}')):
                img_dir_name = f'images_{int(1/hparams.downsample)}'
            
            self.N_imgs = len(os.listdir(os.path.join(hparams.root_dir, img_dir_name)))
        if hparams.embed_a:
            self.embedding_a = torch.nn.Embedding(self.N_imgs, hparams.embed_a_len)   
            load_ckpt(self.embedding_a, self.hparams.weight_path, model_name='embedding_a', prefixes_to_ignore=['model', 'normal_net'])
        
        if hparams.embed_msk:
            self.msk_model = implicit_mask()
            load_ckpt(self.msk_model, self.hparams.weight_path, model_name='msk_model', prefixes_to_ignore=['embedding_a'])

        G = self.model.grid_size
        self.model.register_buffer('density_grid',
            torch.zeros(self.model.cascades, G**3))
        self.model.register_buffer('grid_coords',
            create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))
        
        self.frame_series = []

        self.stylizer = Stylizer(hparams.styl_img_path, hparams)

    def forward(self, batch, split):
        if split=='train':
            poses = self.poses[batch['img_idxs']]
            directions = self.directions[batch['pix_idxs']]
        else:
            poses = batch['pose']
            directions = self.directions

        if self.hparams.embed_a and split=='train':
            embedding_a = self.embedding_a(batch['img_idxs'])
        elif self.hparams.embed_a and split=='test':
            embedding_a = self.embedding_a(torch.tensor([0], device=directions.device))

        rays_o, rays_d = get_rays(directions, poses)
                
        kwargs = {'test_time': split!='train',
                  'random_bg': self.hparams.random_bg,
                  'use_skybox': self.hparams.use_skybox,
                  'render_rgb': hparams.render_rgb,
                  'render_depth': hparams.render_depth,
                  'render_normal': hparams.render_normal,
                  'render_semantic': hparams.render_semantic,
                  'img_wh': self.img_wh,
                  'stylize': True
                  }
        if self.hparams.dataset_name in ['colmap', 'nerfpp', 'tnt', 'kitti']:
            kwargs['exp_step_factor'] = 1/256
        if self.hparams.embed_a:
            embedding_a = self.embedding_a(torch.tensor([0]).cuda()).detach().expand_as(embedding_a)
            kwargs['embedding_a'] = embedding_a

        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float32):
            if split == 'train':
                return render(self.model, rays_o, rays_d, **kwargs)
            else:
                chunk_size = 8192
                all_ret = {}
                for i in range(0, rays_o.shape[0], chunk_size):
                    ret = render(self.model, rays_o[i:i+chunk_size], rays_d[i:i+chunk_size], **kwargs)
                    for k in ret:
                        if k not in all_ret:
                            all_ret[k] = []
                        all_ret[k].append(ret[k])
                for k in all_ret:
                    if k in ['total_samples']:
                        continue
                    all_ret[k] = torch.cat(all_ret[k], 0)
                # all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret and k not in ['total_samples']}
                all_ret['total_samples'] = torch.sum(torch.tensor(all_ret['total_samples']))
                return all_ret

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'downsample': self.hparams.downsample,
                  'use_sem': False,
                  'depth_mono': self.hparams.depth_mono,
                  'sem_conf_path': self.hparams.sem_conf_path,
                  'sem_ckpt_path': self.hparams.sem_ckpt_path}
        if self.hparams.dataset_name == 'kitti':
            kwargs['scene'] = self.hparams.kitti_scene
            kwargs['start'] = self.hparams.start
            kwargs['train_frames'] = self.hparams.train_frames
            center_pose = []
            for i in self.hparams.center_pose:
                center_pose.append(float(i))
            val_list = []
            for i in self.hparams.val_list:
                val_list.append(int(i))
            kwargs['center_pose'] = center_pose
            kwargs['val_list'] = val_list
            kwargs['sequence'] = self.hparams.sequence
            kwargs['scene_type'] = self.hparams.scene_type
        self.train_dataset = dataset(split=self.hparams.split, **kwargs)
        self.train_dataset.batch_size = self.hparams.batch_size
        self.train_dataset.ray_sampling_strategy = self.hparams.ray_sampling_strategy

        self.test_dataset = dataset(split='test', **kwargs)
        
        self.img_wh = self.test_dataset.img_wh

    def configure_optimizers(self):
        # define additional parameters
        self.register_buffer('directions', self.train_dataset.directions.to(self.device))
        self.register_buffer('poses', self.train_dataset.poses.to(self.device))
            
        net_params = []        
        
        for n, p in self.model.named_parameters():
            net_params += [p] 
                            
        opts = []
        self.net_opt = Adam([{'params': net_params}], lr=self.hparams.lr, eps=1e-15)
            
        opts += [self.net_opt]
        net_sch = CosineAnnealingLR(self.net_opt,
                                    self.hparams.num_epochs,
                                    self.hparams.lr/30)

        return opts, [net_sch]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=16,
                          persistent_workers=True,
                          batch_size=None,
                          pin_memory=True)
    
    def on_train_start(self):
        self.stylized_rgb = []
        for i in trange(len(self.train_dataset.poses), desc='Stylizing training views'):
            batch = {}
            batch['pose'] = self.train_dataset.poses[i].cuda()
            results = self(batch, split='test')
            semantic_labels = rearrange((results['semantic'].squeeze(-1).cuda()).to(torch.uint8), '(h w) -> h w', h=self.img_wh[1])
            rgb_gt = rearrange((self.train_dataset.rays[i].cuda()*255).to(torch.uint8), '(h w) c -> h w c', h=self.img_wh[1])
            rgb_stylized = self.stylizer.forward(rgb_gt, semantic_labels).reshape(-1, 3)
            self.stylized_rgb.append(rgb_stylized.cuda())
        
        self.stylized_rgb = torch.stack(self.stylized_rgb, dim=0)

    def training_step(self, batch, batch_nb, *args):
        tensorboard = self.logger.experiment

        if self.hparams.embed_msk:
            w, h = self.img_wh
            uv = torch.tensor(batch['uv']).cuda()
            img_idx = torch.tensor(batch['img_idxs']).cuda()
            uvi = torch.zeros((uv.shape[0], 3)).cuda()
            uvi[:, 0] = (uv[:, 0]-h/2) / h
            uvi[:, 1] = (uv[:, 1]-w/2) / w
            uvi[:, 2] = (img_idx - self.N_imgs/2) / self.N_imgs
            mask = self.msk_model(uvi)

        results = self(batch, split='train')
        batch['rgb'] = self.stylized_rgb[batch['img_idxs'], batch['pix_idxs']]

        loss_kwargs = {'dataset_name': self.hparams.dataset_name,
                       'stylize': True}
        if self.hparams.embed_msk:
            loss_kwargs['mask'] = mask
        loss_d = self.loss(results, batch, **loss_kwargs)

        loss = sum(lo.mean() for lo in loss_d.values())

        with torch.no_grad():
            self.train_psnr(results['rgb'], batch['rgb'])
        self.log('lr', self.net_opt.param_groups[0]['lr'])
        self.log('train/loss', loss)
        self.log('train/psnr', self.train_psnr, True)
        if self.global_step%1000 == 0:
            batch = self.test_dataset[0]
            for i in batch:
                if isinstance(batch[i], torch.Tensor):
                    batch[i] = batch[i].cuda()
            results = self(batch, split='test')
            w, h = self.img_wh
            rgb_gt = rearrange(self.stylized_rgb[0], '(h w) c -> c h w', h=h)
            rgb_pred = rearrange(results['rgb'], '(h w) c -> c h w', h=h)
            depth_pred = depth2img(rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
            depth_pred = rearrange(depth_pred, 'h w c -> c h w', h=h)
            tensorboard.add_image('img/render0_gt', rgb_gt.cpu().numpy(), self.global_step)
            tensorboard.add_image('img/render0', rgb_pred.cpu().numpy(), self.global_step)

        return loss

    def on_train_end(self):
        torch.cuda.empty_cache()

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


if __name__ == '__main__':
    hparams = get_opts()
    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')
    system = StylizeSystem(hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.dataset_name}/{hparams.exp_name}',
                              filename='stylized',
                              save_weights_only=True,
                              every_n_epochs=hparams.num_epochs,
                              save_on_train_epoch_end=True,
                              save_top_k=-1)
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

    logger = TensorBoardLogger(save_dir=f"logs/{hparams.dataset_name}",
                               name=hparams.exp_name,
                               default_hp_metric=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      check_val_every_n_epoch=0,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='gpu',
                      devices=1,
                      num_sanity_val_steps=-1 if hparams.val_only else 0,
                      precision=32)

    trainer.fit(system)

    ckpt_ = \
        slim_ckpt(f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/stylized.ckpt')
    torch.save(ckpt_, f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/stylized_slim.ckpt')