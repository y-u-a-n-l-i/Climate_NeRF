import os
import torch
try:
    # for backward compatibility
    import imageio.v2 as imageio
except ModuleNotFoundError:
    import imageio
import numpy as np
import cv2
from tqdm import trange
from models.networks import NGP
from models.rendering import render
from datasets import dataset_dict
from datasets.ray_utils import get_rays
from utils import load_ckpt
from opt import get_opts
from einops import rearrange

def render_for_test(hparams, split='test'):
    os.makedirs(os.path.join(hparams.root_dir, 'normal_up'), exist_ok=True)
    out_dir = os.path.join(hparams.root_dir, 'normal_up')
    rgb_act = 'None' if hparams.use_exposure else 'Sigmoid'
    model = NGP(scale=hparams.scale, rgb_act=rgb_act, use_skybox=hparams.use_skybox, embed_a=hparams.embed_a, embed_a_len=hparams.embed_a_len).cuda()
    if split=='train':
        ckpt_path = f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}_slim.ckpt'
    else:
        ckpt_path = hparams.ckpt_path        
    print(f'ckpt specified: {ckpt_path} !')
    load_ckpt(model, ckpt_path, prefixes_to_ignore=['embedding_a', 'msk_model'])
    if os.path.exists(os.path.join(hparams.root_dir, 'images')):
        img_dir_name = 'images'
    elif os.path.exists(os.path.join(hparams.root_dir, 'rgb')):
        img_dir_name = 'rgb'

    if hparams.dataset_name=='tnt':
        N_imgs = len(os.listdir(os.path.join(hparams.root_dir, img_dir_name)))
        embed_a_length = hparams.embed_a_len
        if hparams.embed_a:
            embedding_a = torch.nn.Embedding(N_imgs, embed_a_length).cuda() 
            load_ckpt(embedding_a, ckpt_path, model_name='embedding_a', \
                prefixes_to_ignore=["model", "msk_model"])
            embedding_a = embedding_a(torch.tensor([0]).cuda())        
        
    dataset = dataset_dict[hparams.dataset_name]
    kwargs = {'root_dir': hparams.root_dir,
            'downsample': 1.0,
            'render_normal_mask': True}
    dataset = dataset(split='test', **kwargs)
    w, h = dataset.img_wh
    render_traj_rays = dataset.render_normal_rays
    
    for img_idx in trange(len(render_traj_rays)):
        # if img_idx>0:
        #     break
        rays = render_traj_rays[img_idx][:, :6].cuda()
        render_kwargs = {'test_time': True,
                    'T_threshold': 1e-2,
                    'use_skybox': hparams.use_skybox,
                    'render_rgb': hparams.render_rgb,
                    'render_depth': hparams.render_depth,
                    'render_normal': hparams.render_normal,
                    'distill_normal': hparams.render_normal,
                    'img_wh': dataset.img_wh,
                    'normal_model': normal_model}
        if hparams.dataset_name in ['colmap', 'nerfpp']:
            render_kwargs['exp_step_factor'] = 1/256
        if hparams.embed_a:
            render_kwargs['embedding_a'] = embedding_a
        results = render(model, rays[:, :3], rays[:, 3:6], **render_kwargs)
        
        normal = rearrange(results['normal'].cpu().numpy(), '(h w) c -> h w c', h=h)
        up_mask = np.zeros_like(normal)
        # import ipdb; ipdb.set_trace()
        up_mask[..., :] = dataset.up
        # up_mask[..., :] = np.array([0.0039,-0.7098,-0.6941])
        valid_mask = np.linalg.norm(normal.reshape(-1, 3), axis=-1, keepdims=True)!=0
        theta_between_up = np.matmul(up_mask.reshape(-1, 3)[:, None, :], normal.reshape(-1, 3)[:, :, None]).squeeze(-1)\
                        /(np.linalg.norm(up_mask.reshape(-1, 3), axis=-1, keepdims=True)*np.linalg.norm(normal.reshape(-1, 3), axis=-1, keepdims=True)-1e-6)
        # near_up = np.logical_and(theta_between_up>0.7, theta_between_up<=1.)
        # import ipdb; ipdb.set_trace()
        near_up = np.logical_and(theta_between_up>0.5, valid_mask)
        # near_up = theta_between_up>0.3
        near_up = np.reshape(near_up, (h, w))
        weight = np.reshape(theta_between_up, (h, w))
        normal = (near_up*255*weight).astype(np.uint8)
        imageio.imsave(os.path.join(out_dir, f'msk_{"%05d" % img_idx}.pgm'), normal)
        torch.cuda.synchronize()

if __name__ == '__main__':
    hparams = get_opts()
    if hparams.normal_distillation_only:
        assert hparams.ckpt_path is not None, "No ckpt specified when distilling normals"
        hparams.num_epochs = 0
    render_for_test(hparams)