import random
import torch
from torch import nn
from opt import get_opts
import os
import glob
import imageio
import numpy as np
import cv2
import random
import math
from einops import rearrange

# data
from torch.utils.data import DataLoader
import torch.nn.functional as F
from datasets import dataset_dict
from datasets.ray_utils import axisangle_to_R, get_rays
from datasets.snow import SnowSeed

# models
from kornia.utils.grid import create_meshgrid3d
from models.networks import NGP
from models.mb_networks import NGP_mb, vis_net
from models.rendering import render

# optimizer, losses
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# pytorch-lightning
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from utils import slim_ckpt, load_ckpt
from utility.cal_vertical import get_ground_plane, get_vertical_R

from tqdm import trange
from utils import load_ckpt
import trimesh

from torch import autograd

import warnings; warnings.filterwarnings("ignore")


def depth2img(depth):
    # depth = (depth-depth.min())/(depth.max()-depth.min())
    depth = depth/16
    depth = np.clip(depth, a_min=0., a_max=1.)
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img

def semantic2img(sem_label, classes):
    # depth = (depth-depth.min())/(depth.max()-depth.min())
    level = 1/(classes-1)
    sem_color = level * sem_label
    # depth = np.clip(depth, a_min=0., a_max=1.)
    sem_color = cv2.applyColorMap((sem_color*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return sem_color


class SnowSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
                        
        self.rgb_act = 'Sigmoid'
        self.normal_model = None
        self.model = NGP(scale=self.hparams.scale, rgb_act=self.rgb_act, use_skybox=hparams.use_skybox, embed_a=hparams.embed_a, embed_a_len=hparams.embed_a_len)
        load_ckpt(self.model, self.hparams.weight_path, prefixes_to_ignore=['embedding_a'])

        self.origin = NGP(scale=self.hparams.scale, rgb_act=self.rgb_act, use_skybox=hparams.use_skybox, embed_a=hparams.embed_a, embed_a_len=hparams.embed_a_len)
        load_ckpt(self.origin, self.hparams.weight_path_origin_scene, prefixes_to_ignore=['embedding_a'])

        ###
        img_dir_name = None
        if os.path.exists(os.path.join(hparams.root_dir, 'images')):
            img_dir_name = 'images'
        elif os.path.exists(os.path.join(hparams.root_dir, 'rgb')):
            img_dir_name = 'rgb'

        if hparams.dataset_name == 'kitti':
            self.N_imgs = 2 * hparams.train_frames
        elif hparams.dataset_name == 'mega':
            self.N_imgs = 1920 // 6
        else:
            self.N_imgs = len(os.listdir(os.path.join(hparams.root_dir, img_dir_name)))
        # self.N_imgs = self.N_imgs - math.ceil(self.N_imgs/8)
        if hparams.embed_a:
            self.embedding_a = torch.nn.Embedding(self.N_imgs, hparams.embed_a_len)   
            load_ckpt(self.embedding_a, self.hparams.weight_path, model_name='embedding_a', prefixes_to_ignore=['model'])
        ###
        
        self.samples_points = []
        self.samples_color = []
        self.cam_o = []
        self.cam_color = []
        self.center_o = []
        self.center_color = []

    def forward(self, batch, split, **kwargs):
        if split == 'train':
            poses = self.poses[batch['img_idxs']]
            directions = self.directions[batch['pix_idxs']]
            rays_o, rays_d = get_rays(directions, poses)
        else:
            poses = batch['pose']
            directions = self.directions
            rays_o, rays_d = get_rays(directions, poses)
        
        if self.hparams.embed_a and split=='train':
            embedding_a = self.embedding_a(batch['img_idxs'])
        elif self.hparams.embed_a and split=='test':
            embedding_a = self.embedding_a(torch.tensor([0], device=directions.device))
        kwargs_ = {'test_time': False,
                  'random_bg': self.hparams.random_bg,
                  'use_skybox': self.hparams.use_skybox,
                  'render_rgb': hparams.render_rgb,
                  'render_depth': hparams.render_depth,
                  'render_normal': hparams.render_normal,
                  'render_semantic': hparams.render_semantic,
                  'img_wh': self.img_wh,
                  'up_vector': self.up,
                  'ground_height': self.ground_height,
                  'R': self.R,
                  'R_inv': self.R_inv,
                  'scale': self.hparams.scale,
                  'cal_snow_occ': kwargs.get('cal_snow_occ', False),
                  'pred_shadow': kwargs.get('pred_shadow', False),
                  'mb_model': self.mb_model,
                  'sun_vis_net': self.sun_vis_net,
                  'snow_occ_net': self.snow_occ_net,
                  'sky_rays_o': batch['rays_o'],
                  'sky_rays_d': batch['rays_d'],
                  'origin_model': self.origin,
                  'make_snow': True,
                  "sky_label": self.hparams.sky_label
                  }
        if self.hparams.dataset_name in ['colmap', 'nerfpp', 'tnt']:
            kwargs_['exp_step_factor'] = 1/256
        if self.hparams.embed_a:
            embedding_a = self.embedding_a(torch.tensor([0]).cuda()).detach().expand_as(embedding_a)
            kwargs_['embedding_a'] = embedding_a

        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float32):
            res = render(self.model, rays_o, rays_d, **kwargs_)
        return res

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                'downsample': self.hparams.downsample,
                'render_train': self.hparams.render_train,
                'use_shadow': self.hparams.shadow_hint,
                'shadow_ckpt_path': self.hparams.shadow_ckpt_path}
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
        if self.hparams.dataset_name == 'mega':
            kwargs['mega_frame_start'] = self.hparams.mega_frame_start
            kwargs['mega_frame_end'] = self.hparams.mega_frame_end
        self.train_dataset = dataset(split=self.hparams.split, **kwargs)
        self.train_dataset.batch_size = self.hparams.batch_size
        self.train_dataset.ray_sampling_strategy = self.hparams.ray_sampling_strategy

        ground_plane = get_ground_plane(self.hparams, self.train_dataset, self.model.cuda(), embedding_a=(self.embedding_a.cuda())(torch.LongTensor([0]).cuda()).detach(), ground_label=self.hparams.ground_label)
        up = F.normalize(torch.FloatTensor([ground_plane[0], ground_plane[1], ground_plane[2]]).cuda(), dim=0)
        camera_up_mean = torch.FloatTensor(self.train_dataset.up).cuda()
        if torch.sum(up*camera_up_mean)<0:
            up *= -1
        ground_height = ground_plane[3]
        self.register_buffer('up', up)
        self.register_buffer('ground_height', torch.tensor(ground_height).cuda())
        print(f"processed up_vector: {self.up}")
        print(f"processes ground_height: {self.ground_height}")
        self.sky_height = torch.tensor(0.5).cuda() + self.ground_height

        R, R_inv = get_vertical_R(self.up)
        self.register_buffer('R', R)
        self.register_buffer('R_inv', R_inv)
                
        self.mb_model = NGP_mb(scale=self.hparams.scale, up=self.up, ground_height=self.ground_height,
                               R=self.R, R_inv=self.R_inv, interval=self.hparams.mb_size, rgb_act=self.rgb_act)

        self.sun_vis_net = vis_net(scale=self.hparams.scale)
        self.snow_occ_net = vis_net(scale=self.hparams.scale)

        self.img_wh = self.train_dataset.img_wh
        self.snow_seed = SnowSeed(self.up, height=self.sky_height, R=self.R, interval=self.hparams.mb_size/2, range=self.hparams.scale)
        self.snow_seed.batch_size = self.hparams.batch_size

    def configure_optimizers(self):
        # define additional parameters
        self.register_buffer('directions', self.train_dataset.directions.to(self.device))
        self.register_buffer('poses', self.train_dataset.poses.to(self.device))
            
        if self.hparams.cal_snow_occ or self.hparams.pred_shadow:
            load_ckpt(self.mb_model, self.hparams.ckpt_path, model_name='mb_model', prefixes_to_ignore=['embedding_a', 'normal_net.params'])

        net_params = []
        net_to_optimize = ['mb_model', 'sun_vis_net', 'snow_occ_net']
        for n, p in self.named_parameters():
            for selected_n in net_to_optimize:
                if selected_n in n:
                    net_params += [p]
        
        opts = []
        self.net_opt = Adam(net_params, lr=1e-2, eps=1e-15)
        opts = [self.net_opt] 

        return opts

    def train_dataloader(self):
        
        return [DataLoader(self.train_dataset,
                        num_workers=16,
                        persistent_workers=True,
                        batch_size=None,
                        pin_memory=True),
                DataLoader(self.snow_seed,
                        num_workers=16,
                        persistent_workers=True,
                        batch_size=None,
                        pin_memory=True)]

    def training_step(self, batch, *args):        
        loss_items = []
        fwd_kwargs = {'cal_snow_occ': True, 'pred_shadow': self.hparams.shadow_hint}
        batch_ = batch[0]
        batch_['rays_o'] = batch[1]['rays_o']
        batch_['rays_d'] = batch[1]['rays_d']
        results = self(batch_, split='train', **fwd_kwargs)
        loss_snow_alpha = torch.mean((results['alphas'] - results['alphas_mb'])**2)
        loss_items.append(loss_snow_alpha)
        loss_items.append(torch.mean((results['rgbs'] - results['rgbs_mb'])**2))
        loss_snow_occ = torch.mean(torch.abs(results['pred_t'] - results['model_t'])**2)
        loss_items.append(loss_snow_occ)
        if self.hparams.shadow_hint:
            loss_pred_shadow = F.binary_cross_entropy(results['pred_shadow'], batch_['shadow'], reduction='mean')
            loss_items.append(loss_pred_shadow)

        loss = sum(loss_items)
        self.log('lr', self.net_opt.param_groups[0]['lr'])
        self.log('train/loss', loss)
        self.log('train/loss_snow_alpha', loss_snow_alpha, True)
        self.log('train/loss_snow_occ', loss_snow_occ, True)
        if self.hparams.shadow_hint:
            self.log('train/loss_pred_shadow', loss_pred_shadow, True)
        return loss

    def on_train_end(self):
        delattr(self, "origin")
        torch.cuda.empty_cache()

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def on_save_checkpoint(self, checkpoint):
        keys = []
        for k in checkpoint["state_dict"].keys():
            keys.append(k)
        for n in keys:
            if "origin" in n:
                checkpoint["state_dict"].pop(n)

if __name__ == '__main__':
    random.seed(20220806)
    torch.manual_seed(20220806)
    torch.cuda.manual_seed_all(20220806)
    np.random.seed(20220806)
    hparams = get_opts()
    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')
    system = SnowSystem(hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.dataset_name}/{hparams.exp_name}',
                              filename='model_with_snow',
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
                      precision=32,
                      detect_anomaly=True)

    trainer.fit(system)

    if not hparams.val_only: # save slimmed ckpt for the last epoch
        ckpt_ = \
            slim_ckpt(f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/model_with_snow.ckpt')
        torch.save(ckpt_, f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/model_with_snow_slim.ckpt')