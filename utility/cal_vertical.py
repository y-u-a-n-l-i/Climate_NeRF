import torch
from opt import get_opts
import os
import torch_scatter

# data
import torch.nn.functional as F
from datasets.ray_utils import get_rays

# models
from models.rendering import render
from render import render_chunks

from tqdm import trange

import warnings; warnings.filterwarnings("ignore")
import open3d as o3d

def construct_vox_points_closest(xyz_val, vox_res, partition_xyz=None, space_min=None, space_max=None):
    # xyz, N, 3
    xyz = xyz_val if partition_xyz is None else partition_xyz
    if space_min is None:
        xyz_min, xyz_max = torch.min(xyz, dim=-2)[0], torch.max(xyz, dim=-2)[0]
        space_edge = (xyz_max - xyz_min) * 1.05
        xyz_mid = (xyz_max + xyz_min) / 2
        space_min = xyz_mid - space_edge / 2
    else:
        space_edge = space_max - space_min
        mask = (xyz_val - space_min[None,...])>0
        mask *= (space_max[None,...] - xyz_val)>0
        mask = torch.prod(mask, dim=-1) > 0
        xyz_val = xyz_val[mask, :]
    construct_vox_sz = space_edge / vox_res
    xyz_shift = xyz_val - space_min[None, ...]
    sparse_grid_idx, inv_idx = torch.unique(torch.floor(xyz_shift / construct_vox_sz[None, ...]).to(torch.int32), dim=0, return_inverse=True)
    xyz_centroid = torch_scatter.scatter_mean(xyz_val, inv_idx, dim=0)
    # xyz_centroid = construct_vox_sz[None, ...]*(torch.floor((scatter_mean(xyz_val, inv_idx, dim=0)-space_min[None, ...]) / construct_vox_sz) + 0.5) + space_min[None, ...]
    xyz_centroid_prop = xyz_centroid[inv_idx,:]
    xyz_residual = torch.norm(xyz_val - xyz_centroid_prop, dim=-1)

    _, min_idx = torch_scatter.scatter_min(xyz_residual, inv_idx, dim=0)
    return xyz_centroid, sparse_grid_idx, min_idx

@torch.no_grad()
def get_ground_plane(hparams, dataset, model, embedding_a, ground_label=0):
    print("calculating ground plane")
    train_rays = {}
    print("generating rays' origins and directions")
    for img_idx in trange(len(dataset.poses)):
        rays_o, rays_d = get_rays(dataset.directions.cuda(), dataset.poses[img_idx].cuda())
        train_rays[img_idx] = torch.cat([rays_o, rays_d], 1).cpu()

    ground_points = []
    for img_idx in trange(len(train_rays)):
        rays = train_rays[img_idx][:, :6].cuda()
        render_kwargs = {
            'img_idx': img_idx,
            'test_time': True,
            'T_threshold': 1e-2,
            'use_skybox': hparams.use_skybox,
            'render_depth': True,
            'render_semantic': True,
            'img_wh': dataset.img_wh,
            'anti_aliasing_factor': hparams.anti_aliasing_factor
        }
        if hparams.dataset_name in ['colmap', 'nerfpp', 'tnt', 'kitti']:
            render_kwargs['exp_step_factor'] = 1/256
        if hparams.embed_a:
            render_kwargs['embedding_a'] = embedding_a

        rays_o = rays[:, :3]
        rays_d = rays[:, 3:6]
        results = {}
        chunk_size = hparams.chunk_size
        # with torch.cuda.amp.autocast(enabled=True, dtype=torch.float32):
        if chunk_size > 0:
            results = render_chunks(model, rays_o, rays_d, chunk_size, **render_kwargs)
        else:
            results = render(model, rays_o, rays_d, **render_kwargs)
        ground_points.append(results['depth_xyz'][(results['semantic']==ground_label).squeeze(-1)])
    
    ground_points = torch.cat(ground_points, dim=0)
    ground_points, _, _ = construct_vox_points_closest(ground_points.cuda() if len(ground_points) < 99999999 else ground_points[::(len(ground_points)//99999999+1),...].cuda(), 512, 
                                                       space_min=torch.FloatTensor([-1.0, -1.0, -1.0]).cuda(), 
                                                       space_max=torch.FloatTensor([1.0, 1.0, 1.0]).cuda())
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ground_points.cpu().numpy())
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.005,
                                            ransac_n=1000,
                                            num_iterations=1000)
    print("done")
    return plane_model

@torch.no_grad()
def get_vertical_R(up):
    y_axis = torch.FloatTensor([0, -1, 0]).cuda()
        
    identical = torch.FloatTensor([[1, 0, 0], 
                                    [0, 1, 0], 
                                    [0, 0, 1]]).cuda()
    cross = torch.linalg.cross(y_axis, up)
    skewed_cross = torch.FloatTensor([[0, -cross[2], cross[1]],
                                        [cross[2], 0, -cross[0]],
                                        [-cross[1], cross[0], 0]]).cuda()
    cos_theta = torch.sum(y_axis* up)
    R = identical + skewed_cross + 1/(1+cos_theta)*skewed_cross @ skewed_cross
    R_inv = torch.linalg.inv(R)

    return R, R_inv