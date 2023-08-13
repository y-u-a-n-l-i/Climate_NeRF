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
from models.rendering import render, MAX_SAMPLES

# optimizer, losses
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from losses import NeRFLoss

# metrics
from torchmetrics import (
    PeakSignalNoiseRatio, 
    StructuralSimilarityIndexMeasure
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# pytorch-lightning
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.distributed import all_gather_ddp_if_available

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


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        print('start to stylize')
        self.save_hyperparameters(hparams)

        self.warmup_steps = 256
        self.update_interval = 16 # the interval to update density grid

        self.loss = NeRFLoss()
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1)
        if self.hparams.eval_lpips:
            self.val_lpips = LearnedPerceptualImagePatchSimilarity('vgg')
            for p in self.val_lpips.net.parameters():
                p.requires_grad = False

        self.rgb_act = 'None' if self.hparams.use_exposure else 'Sigmoid'
        # import ipdb; ipdb.set_trace()
        self.model = NGP(scale=self.hparams.scale, rgb_act=self.rgb_act, use_skybox=hparams.use_skybox, embed_a=hparams.embed_a, embed_a_len=hparams.embed_a_len)
        assert hparams.ckpt_path is not None
      
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
        
        if hparams.embed_msk:
            self.msk_model = implicit_mask()

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
                  'use_skybox': self.hparams.use_skybox if self.global_step>=self.warmup_steps else False,
                  'render_rgb': hparams.render_rgb,
                  'render_depth': hparams.render_depth,
                  'render_normal': hparams.render_normal,
                  'render_sem': hparams.render_semantic,
                  'img_wh': self.img_wh,
                  'stylize': True
                  }
        if self.hparams.dataset_name in ['colmap', 'nerfpp', 'tnt']:
            kwargs['exp_step_factor'] = 1/256
        if self.hparams.use_exposure:
            kwargs['exposure'] = batch['exposure']
        if self.hparams.embed_a:
            embedding_a = self.embedding_a(torch.tensor([0]).cuda()).detach()
            kwargs['embedding_a'] = embedding_a

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
                  'use_sem': self.hparams.render_semantic,
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
            
        load_ckpt(self.model, self.hparams.ckpt_path, prefixes_to_ignore=['embedding_a', 'normal_net'])
        net_params = []        
        if self.hparams.embed_msk:
            load_ckpt(self.msk_model, self.hparams.ckpt_path, model_name='msk_model', prefixes_to_ignore=['embedding_a', 'normal_net.params'])
        
        embeda_params = []
        if self.hparams.embed_a:
            load_ckpt(self.embedding_a, self.hparams.ckpt_path, model_name='embedding_a', prefixes_to_ignore=['model', 'normal_net'])
            for n, p in self.embedding_a.named_parameters():
                embeda_params += [p]
                            
        opts = []
        self.net_opt = Adam([{'params': net_params}, 
                        {'params': embeda_params, 'lr': 1e-6}], lr=self.hparams.lr, eps=1e-15)
            
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

    # def val_dataloader(self):
    #     return DataLoader(self.test_dataset,
    #                       num_workers=8,
    #                       batch_size=None,
    #                       pin_memory=True)

    def on_train_start(self):
        self.stylized_rgb = []
        for i in trange(len(self.train_dataset.poses)):
            batch = {}
            batch['pose'] = self.train_dataset.poses[i].cuda()
            results = self(batch, split='test')
            semantic_labels = rearrange((results['semantic'].squeeze(-1).cuda()).to(torch.uint8), '(h w) -> h w', h=self.img_wh[1])
            rgb_gt = rearrange((self.train_dataset.rays[i].cuda()*255).to(torch.uint8), '(h w) c -> h w c', h=self.img_wh[1])
            rgb_stylized = self.stylizer.forward(rgb_gt, semantic_labels).reshape(-1, 3)
            self.stylized_rgb.append(rgb_stylized.cuda())
        
        torch.stack(self.stylized_rgb, dim=0)

    def training_step(self, batch, batch_nb, *args):
        tensorboard = self.logger.experiment

        results = self(batch, split='train')

        batch['rgb'] = self.stylized_rgb[batch['img_idxs'], batch['pix_idxs']]

        loss_kwargs = {'dataset_name': self.hparams.dataset_name,
                       'up_sem': False,
                       'distill': True}
        loss_d = self.loss(results, batch, **loss_kwargs)

        loss = sum(lo.mean() for lo in loss_d.values())

        with torch.no_grad():
            self.train_psnr(results['rgb'], batch['rgb'])
        self.log('lr', self.net_opt.param_groups[0]['lr'])
        self.log('train/loss', loss)
        self.log('train/psnr', self.train_psnr, True)
        if self.global_step%10000 == 0 and self.global_step>0:
            print('[val in training]')
            batch = self.test_dataset[0]
            for i in batch:
                if isinstance(batch[i], torch.Tensor):
                    batch[i] = batch[i].cuda()
            results = self(batch, split='test')
            w, h = self.img_wh
            rgb_pred = rearrange(results['rgb'], '(h w) c -> c h w', h=h)
            depth_pred = depth2img(rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
            depth_pred = rearrange(depth_pred, 'h w c -> c h w', h=h)
            tensorboard.add_image('img/render0', rgb_pred.cpu().numpy(), self.global_step)

        return loss

    def on_train_end(self):
        torch.cuda.empty_cache()
        if not self.hparams.no_save_test:
            self.val_dir = f'results/{self.hparams.dataset_name}/{self.hparams.exp_name}'
            os.makedirs(self.val_dir, exist_ok=True)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


if __name__ == '__main__':
    hparams = get_opts()
    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')
    system = NeRFSystem(hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.dataset_name}/{hparams.exp_name}',
                              filename='{epoch:d}',
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
                      devices=hparams.num_gpus,
                      strategy=DDPPlugin(find_unused_parameters=False)
                               if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=-1 if hparams.val_only else 0,
                      precision=32)

    trainer.fit(system)

    if not hparams.val_only: # save slimmed ckpt for the last epoch
        ckpt_ = \
            slim_ckpt(f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}.ckpt')
        torch.save(ckpt_, f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}_slim.ckpt')

    if (not hparams.no_save_test) and \
       hparams.dataset_name=='nsvf' and \
       'Synthetic' in hparams.root_dir: # save video
        imgs = sorted(glob.glob(os.path.join(system.val_dir, '*.png')))
        imageio.mimsave(os.path.join(system.val_dir, 'rgb.mp4'),
                        [imageio.imread(img) for img in imgs[::2]],
                        fps=30, macro_block_size=1)
        imageio.mimsave(os.path.join(system.val_dir, 'depth.mp4'),
                        [imageio.imread(img) for img in imgs[1::2]],
                        fps=30, macro_block_size=1)
    
    if not hparams.val_only: # save slimmed ckpt for the last epoch
        ckpt_ = \
            slim_ckpt(f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}.ckpt')
        torch.save(ckpt_, f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}_slim.ckpt')

    # if not hparams.render_origin:
    #     from render_metaball import render_for_test  
    #     render_for_test(hparams, split='train')
    # else:
    #     from render import render_for_test
    #     render_for_test(hparams, split='train')
