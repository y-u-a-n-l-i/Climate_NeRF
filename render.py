import os
import torch
import torch.nn.functional as F
import imageio
import numpy as np
import cv2
import math 
from PIL import Image
from tqdm import trange
from models.networks import NGP
from models.mb_networks import NGP_mb, vis_net
from models.rendering import render, MAX_SAMPLES
from datasets import dataset_dict
from datasets.ray_utils import get_rays
from utils import load_ckpt, guided_filter
from opt import get_opts
from einops import rearrange
from simulate import get_simulator

def depth2img(depth, scale=16):
    depth = depth/scale
    depth = np.clip(depth, a_min=0., a_max=1.)
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img

def semantic2img(sem_label, classes):
    level = 1/(classes-1)
    sem_color = level * sem_label
    sem_color = cv2.applyColorMap((sem_color*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return sem_color

def render_chunks(model, rays_o, rays_d, chunk_size, **kwargs):
    chunk_n = math.ceil(rays_o.shape[0]/chunk_size)
    d = kwargs.get('depth_smooth', None)
    results = {}
    for i in range(chunk_n):
        rays_o_chunk = rays_o[i*chunk_size: (i+1)*chunk_size]
        rays_d_chunk = rays_d[i*chunk_size: (i+1)*chunk_size]
        if d is not None:
            kwargs['depth_smooth'] = d[i*chunk_size: (i+1)*chunk_size]
        ret = render(model, rays_o_chunk, rays_d_chunk, **kwargs)
        for k in ret:
            if k not in results:
                results[k] = []
            results[k].append(ret[k])
    for k in results:
        if k in ['total_samples']:
            continue
        results[k] = torch.cat(results[k], 0)
    return results

def render_for_test(hparams, split='test'):
    os.makedirs(os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}'), exist_ok=True)
    rgb_act = 'Sigmoid'
    if hparams.use_skybox:
        print('render skybox!')
    model = NGP(scale=hparams.scale, rgb_act=rgb_act, use_skybox=hparams.use_skybox, embed_a=hparams.embed_a, embed_a_len=hparams.embed_a_len).cuda()
    if split=='train':
        ckpt_path = f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}_slim.ckpt'
    else:
        ckpt_path = hparams.weight_path 

    load_ckpt(model, ckpt_path, prefixes_to_ignore=['embedding_a', 'msk_model'])
    print('Loaded checkpoint: {}'.format(ckpt_path))
    
    if os.path.exists(os.path.join(hparams.root_dir, 'images')):
        img_dir_name = 'images'
    elif os.path.exists(os.path.join(hparams.root_dir, 'rgb')):
        img_dir_name = 'rgb'
    
    if hparams.dataset_name == 'kitti':
        N_imgs = 2 * hparams.train_frames
    elif hparams.dataset_name == 'mega':
        N_imgs = 1920 // 6
    else:
        N_imgs = len(os.listdir(os.path.join(hparams.root_dir, img_dir_name)))
    
    embed_a_length = hparams.embed_a_len
    if hparams.embed_a:
        embedding_a = torch.nn.Embedding(N_imgs, embed_a_length).cuda() 
        load_ckpt(embedding_a, ckpt_path, model_name='embedding_a', \
            prefixes_to_ignore=["model", "msk_model"])
        embedding_a = embedding_a(torch.tensor([0]).cuda())        
        
    dataset = dataset_dict[hparams.dataset_name]
    kwargs = {'root_dir': hparams.root_dir,
            'downsample': hparams.downsample,
            'render_train': hparams.render_train,
            'render_traj': hparams.render_traj,
            'anti_aliasing_factor': hparams.anti_aliasing_factor}
    if hparams.dataset_name == 'kitti':
            kwargs['scene'] = hparams.kitti_scene
            kwargs['start'] = hparams.start
            kwargs['train_frames'] = hparams.train_frames
            center_pose = []
            for i in hparams.center_pose:
                center_pose.append(float(i))
            val_list = []
            for i in hparams.val_list:
                val_list.append(int(i))
            kwargs['center_pose'] = center_pose
            kwargs['val_list'] = val_list
    if hparams.dataset_name == 'mega':
            kwargs['mega_frame_start'] = hparams.mega_frame_start
            kwargs['mega_frame_end'] = hparams.mega_frame_end

    dataset = dataset(split='test', **kwargs)
    w, h = dataset.img_wh
    if hparams.render_traj:
        render_traj_rays = dataset.render_traj_rays
    else:
        # render_traj_rays = dataset.rays
        render_traj_rays = {}
        print("generating rays' origins and directions!")
        for img_idx in trange(len(dataset.poses)):
            rays_o, rays_d = get_rays(dataset.directions.cuda(), dataset[img_idx]['pose'].cuda())
            render_traj_rays[img_idx] = torch.cat([rays_o, rays_d], 1).cpu()

    frames_dir = f'results/{hparams.dataset_name}/{hparams.exp_name}/frames'
    os.makedirs(frames_dir, exist_ok=True)
    if hparams.simulate:
        simulate_kwargs = {
            'depth_bound': hparams.depth_bound,
            'sigma': hparams.sigma,
            'rgb_smog': hparams.rgb_smog, 
            'depth_path': hparams.depth_path,
            # water params
            'rgb_water': hparams.rgb_water,
            'water_height': hparams.water_height,
            'plane_path': hparams.plane_path,
            'refraction_idx': hparams.refraction_idx,
            'pano_path': hparams.pano_path,
            'v_forward': hparams.v_forward,
            'v_down': hparams.v_down,
            'v_right': hparams.v_right,
            'theta': hparams.gl_theta,
            'sharpness': hparams.gl_sharpness, 
            'wave_len': hparams.wave_len,
            'wave_ampl': hparams.wave_ampl,
            'refract_decay': hparams.refract_decay
        }
        simulator = get_simulator(
            effect=hparams.simulate,
            device='cuda',
            **simulate_kwargs
        )
        if hparams.simulate == 'snow':
            dict_ = torch.load(ckpt_path)
            up = dict_['up'].cuda()
            ground_height = dict_['ground_height'].item()
            R = dict_['R'].cuda()
            R_inv = dict_['R_inv'].cuda()
            mb_model = NGP_mb(scale=hparams.scale, up=up, ground_height=ground_height,
                               R=R, R_inv=R_inv, interval=hparams.mb_size, rgb_act=rgb_act).cuda()
            # for _ in trange(50, desc='updating occ grids'):
            #     with torch.cuda.amp.autocast(enabled=True, dtype=torch.float32):
            #         model.update_density_grid(0.01*MAX_SAMPLES/3**0.5, warmup=True, aux_model=mb_model)
            load_ckpt(mb_model, ckpt_path, model_name='mb_model')
            snow_occ_net = vis_net(scale=hparams.scale).cuda()
            load_ckpt(snow_occ_net, ckpt_path, model_name='snow_occ_net')
            if hparams.shadow_hint:
                sun_vis_net = vis_net(scale=hparams.scale).cuda()
                load_ckpt(sun_vis_net, ckpt_path, model_name='sun_vis_net')
    
    depth_load = None
    if hparams.depth_path and hparams.simulate == 'water':
        print('Load depth:', hparams.depth_path)
        depth_load = torch.FloatTensor(np.load(hparams.depth_path))

    frame_series = []
    depth_raw_series = []
    depth_series = []
    points_series = []
    normal_series = []
    semantic_series = []

    for img_idx in trange(len(render_traj_rays)):
        rays = render_traj_rays[img_idx][:, :6].cuda()
        render_kwargs = {
            'img_idx': img_idx,
            'test_time': True,
            'T_threshold': 1e-2,
            'use_skybox': hparams.use_skybox,
            'render_rgb': hparams.render_rgb,
            'render_depth': hparams.render_depth,
            'render_normal': hparams.render_normal,
            'render_semantic': hparams.render_semantic,
            'img_wh': dataset.img_wh,
            'anti_aliasing_factor': hparams.anti_aliasing_factor,
            'snow': hparams.simulate == 'snow'
        }
        if hparams.dataset_name in ['colmap', 'nerfpp', 'tnt', 'kitti']:
            render_kwargs['exp_step_factor'] = 1/256
        if hparams.embed_a:
            render_kwargs['embedding_a'] = embedding_a
        if hparams.simulate:
            render_kwargs['simulator'] = simulator
            render_kwargs['simulate_effect'] = hparams.simulate
        if depth_load is not None:
            d = depth_load[img_idx]
            d = guided_filter(d, d, hparams.gf_r, hparams.gf_eps)
            if hparams.anti_aliasing_factor > 1:
                a = hparams.anti_aliasing_factor
                size = (int(a*h), int(a*w))
                d = F.interpolate(d[None, None], size=size)[0, 0]
            d = d.flatten().cuda()
            render_kwargs['depth_smooth'] = d
        if hparams.simulate == 'snow':
            render_kwargs['mb_model'] = mb_model
            render_kwargs['snow_occ_net'] = snow_occ_net
            render_kwargs['cal_snow_occ'] = True
            if hparams.shadow_hint:
                render_kwargs['sun_vis_net'] = sun_vis_net
                render_kwargs['pred_shadow'] = True

        rays_o = rays[:, :3]
        rays_d = rays[:, 3:6]
        results = {}
        chunk_size = hparams.chunk_size
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float32):
            if chunk_size > 0:
                results = render_chunks(model, rays_o, rays_d, chunk_size, **render_kwargs)
            else:
                results = render(model, rays_o, rays_d, **render_kwargs)

        if hparams.render_rgb:
            rgb_frame = None
            if hparams.anti_aliasing_factor > 1.0:
                h_new = int(h*hparams.anti_aliasing_factor)
                rgb_frame = rearrange(results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h_new)
                rgb_frame = Image.fromarray((rgb_frame*255).astype(np.uint8)).convert('RGB')
                rgb_frame = np.array(rgb_frame.resize((w, h), Image.Resampling.BICUBIC))
            else:
                rgb_frame = rearrange(results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
                rgb_frame = (rgb_frame*255).astype(np.uint8)
            frame_series.append(rgb_frame)
            cv2.imwrite(os.path.join(frames_dir, '{:0>3d}-rgb.png'.format(img_idx)), cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
        if hparams.render_semantic:
            sem_frame = semantic2img(rearrange(results['semantic'].squeeze(-1).cpu().numpy(), '(h w) -> h w', h=h), 7)
            semantic_series.append(sem_frame)
        if hparams.render_depth:
            depth_raw = rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h)
            depth_raw_series.append(depth_raw)
            depth = depth2img(depth_raw, scale=2*hparams.scale)
            depth_series.append(depth)
            cv2.imwrite(os.path.join(frames_dir, '{:0>3d}-depth.png'.format(img_idx)), cv2.cvtColor(depth, cv2.COLOR_RGB2BGR))
        
        if hparams.render_points:
            points = rearrange(results['points'].cpu().numpy(), '(h w) c -> h w c', h=h)
            points_series.append(points)

        if hparams.render_normal:
            normal = rearrange(results['normal_pred'].cpu().numpy(), '(h w) c -> h w c', h=h)+1e-6            
            normal_series.append((255*(normal+1)/2).astype(np.uint8))
                        
        torch.cuda.synchronize()
    
    print(f"saving to results/{hparams.dataset_name}/{hparams.exp_name}")
    if hparams.render_rgb:
        imageio.mimsave(os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}', 'render_traj.mp4' if not hparams.render_train else "circle_path.mp4"),
                        frame_series,
                        fps=30, macro_block_size=1)

    if hparams.render_semantic:
        imageio.mimsave(os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}', 'render_semantic.mp4' if not hparams.render_train else "circle_path_semantic.mp4"),
                        semantic_series,
                        fps=30, macro_block_size=1)
    if hparams.render_depth:
        imageio.mimsave(os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}', 'render_traj_depth.mp4' if not hparams.render_train else "circle_path_depth.mp4"),
                        depth_series,
                        fps=30, macro_block_size=1)
    
    if hparams.render_depth_raw:
        depth_raw_all = np.stack(depth_raw_series) #(n_frames, h ,w)
        path = f'results/{hparams.dataset_name}/{hparams.exp_name}/depth_raw.npy'
        np.save(path, depth_raw_all)

    if hparams.render_points:
        points_all = np.stack(points_series)
        path = f'results/{hparams.dataset_name}/{hparams.exp_name}/points.npy'
        np.save(path, points_all)

    if hparams.render_normal:
        imageio.mimsave(os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}', 'render_traj_normal.mp4' if not hparams.render_train else "circle_path_normal.mp4"),
                        normal_series,
                        fps=30, macro_block_size=1)

if __name__ == '__main__':
    hparams = get_opts()
    render_for_test(hparams)