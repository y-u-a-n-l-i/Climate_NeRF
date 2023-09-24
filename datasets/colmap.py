import torch
import numpy as np
import os
import glob
from tqdm import tqdm

from .ray_utils import *
from .color_utils import read_image, Shadow_predictor
from .colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary
from mmseg.apis import inference_model, init_model
import mmcv
from mmseg.utils import get_classes

from .base import BaseDataset

def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)

def center_poses(poses, pts3d):
    """
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
        pts3d: (N, 3) reconstructed point cloud

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pts3d_centered: (N, 3) centered point cloud
    """

    pose_avg = average_poses(poses, pts3d) # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg # convert to homogeneous coordinate for faster computation
                                 # by simply adding 0, 0, 0, 1 as the last row
    pose_avg_inv = np.linalg.inv(pose_avg_homo)
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1) # (N_images, 4, 4) homogeneous coordinate

    poses_centered = pose_avg_inv @ poses_homo # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3] # (N_images, 3, 4)

    pts3d_centered = pts3d @ pose_avg_inv[:, :3].T + pose_avg_inv[:, 3:].T

    return poses_centered, pts3d_centered


class ColmapDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        self.read_intrinsics(**kwargs)

        if kwargs.get('read_meta', True):
            self.read_meta(split, **kwargs)

    def read_intrinsics(self, **kwargs):
        # Step 1: read and scale intrinsics (same for all images)
        camdata = read_cameras_binary(os.path.join(self.root_dir, 'sparse/0/cameras.bin'))
        h = int(camdata[1].height*self.downsample)
        w = int(camdata[1].width*self.downsample)
        self.img_wh = (w, h)

        if camdata[1].model == 'SIMPLE_RADIAL':
            fx = fy = camdata[1].params[0]*self.downsample
            cx = camdata[1].params[1]*self.downsample
            cy = camdata[1].params[2]*self.downsample
        elif camdata[1].model in ['PINHOLE', 'OPENCV']:
            fx = camdata[1].params[0]*self.downsample
            fy = camdata[1].params[1]*self.downsample
            cx = camdata[1].params[2]*self.downsample
            cy = camdata[1].params[3]*self.downsample
        else:
            raise ValueError(f"Please parse the intrinsics for camera model {camdata[1].model}!")
        self.K = torch.FloatTensor([[fx, 0, cx],
                                    [0, fy, cy],
                                    [0,  0,  1]])
        self.directions = get_ray_directions(h, w, self.K, anti_aliasing_factor=kwargs.get('anti_aliasing_factor', 1.0))

    def read_meta(self, split, **kwargs):
        # Step 2: correct poses
        # read extrinsics (of successfully reconstructed images)
        imdata = read_images_binary(os.path.join(self.root_dir, 'sparse/0/images.bin'))
        img_names = [imdata[k].name for k in imdata]
        
        perm = np.argsort(img_names)
        if '360' in self.root_dir and self.downsample<1: # mipnerf360 data
            folder = f'images_{int(1/self.downsample)}'
        else:
            folder = 'images'

        sem_model = None
        if kwargs.get('use_sem', False):
            config_file = kwargs.get('sem_conf_path', 'pretrained/mmseg/segformer_mit-b5_8x1_1024x1024_160k_cityscapes.py')
            checkpoint_file = kwargs.get('sem_ckpt_path', 'pretrained/mmseg/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth')
            palette = 'cityscapes'
            print(f"Load segmentation model from {checkpoint_file}")
            sem_model = init_model(config_file, checkpoint=checkpoint_file, device='cuda')
            sem_model.CLASSES = get_classes(palette)

        shadow_predictor = None
        if kwargs.get('use_shadow', False):
            shadow_predictor = Shadow_predictor(kwargs.get('shadow_ckpt_path', 'pretrained/mtmt/iter_10000.pth'))

        # read successfully reconstructed images and ignore others
        img_paths = [os.path.join(self.root_dir, folder, name)
                     for name in sorted(img_names)]
        w2c_mats = []
        bottom = np.array([[0, 0, 0, 1.]])
        for k in imdata:
            im = imdata[k]
            R = im.qvec2rotmat(); t = im.tvec.reshape(3, 1)
            w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
        w2c_mats = np.stack(w2c_mats, 0)
        poses = np.linalg.inv(w2c_mats)[perm, :3] # (N_images, 3, 4) cam2world matrices

        pts3d = read_points3d_binary(os.path.join(self.root_dir, 'sparse/0/points3D.bin'))
        pts3d = np.array([pts3d[k].xyz for k in pts3d]) # (N, 3)

        # self.poses, self.pts3d = center_poses(poses, pts3d)
        self.poses, self.pts3d = poses, pts3d
        self.up = torch.FloatTensor(-normalize(self.poses[:, :3, 1].mean(0)))
        print(f"scene up {self.up}")
        scale = np.linalg.norm(self.poses[..., 3], axis=-1).max()
        print(f"scene scale {scale}")
        self.poses[..., 3] /= scale
        self.pts3d /= scale
        
        self.rays = []
        if kwargs.get('use_sem', False):
            self.labels = []
        if kwargs.get('use_shadow', False):
            self.shadows = []
        if split == 'test_traj': # use precomputed test poses
            self.poses = create_spheric_poses(1.2, self.poses[:, 1, 3].mean(), )
            self.poses = torch.FloatTensor(self.poses)
            return

        if 'HDR-NeRF' in self.root_dir: # HDR-NeRF data
            if 'syndata' in self.root_dir: # synthetic
                # first 17 are test, last 18 are train
                self.unit_exposure_rgb = 0.73
                if split=='train':
                    img_paths = sorted(glob.glob(os.path.join(self.root_dir,
                                                            f'train/*[024].png')))
                    self.poses = np.repeat(self.poses[-18:], 3, 0)
                elif split=='test':
                    img_paths = sorted(glob.glob(os.path.join(self.root_dir,
                                                            f'test/*[13].png')))
                    self.poses = np.repeat(self.poses[:17], 2, 0)
                else:
                    raise ValueError(f"split {split} is invalid for HDR-NeRF!")
            else: # real
                self.unit_exposure_rgb = 0.5
                # even numbers are train, odd numbers are test
                if split=='train':
                    img_paths = sorted(glob.glob(os.path.join(self.root_dir,
                                                    f'input_images/*0.jpg')))[::2]
                    img_paths+= sorted(glob.glob(os.path.join(self.root_dir,
                                                    f'input_images/*2.jpg')))[::2]
                    img_paths+= sorted(glob.glob(os.path.join(self.root_dir,
                                                    f'input_images/*4.jpg')))[::2]
                    self.poses = np.tile(self.poses[::2], (3, 1, 1))
                elif split=='test':
                    img_paths = sorted(glob.glob(os.path.join(self.root_dir,
                                                    f'input_images/*1.jpg')))[1::2]
                    img_paths+= sorted(glob.glob(os.path.join(self.root_dir,
                                                    f'input_images/*3.jpg')))[1::2]
                    self.poses = np.tile(self.poses[1::2], (2, 1, 1))
                else:
                    raise ValueError(f"split {split} is invalid for HDR-NeRF!")
        else:
            # use every 8th image as test set
            if split=='train':
                img_paths = [x for i, x in enumerate(img_paths) if i%8!=0]
                self.poses = np.array([x for i, x in enumerate(self.poses) if i%8!=0])
            elif split=='test':
                render_c2w_f64 = torch.FloatTensor(self.poses)
                
                img_paths = [x for i, x in enumerate(img_paths) if i%8==0]
                self.poses = np.array([x for i, x in enumerate(self.poses) if i%8==0])
                if kwargs.get('render_traj', False):
                    # render_c2w_f64 = create_spheric_poses(1.2, self.poses[:, 1, 3].mean())
                    # render_c2w_f64 = torch.FloatTensor(render_c2w_f64)
                    render_c2w_f64 = generate_interpolated_path(self.poses, 120)[400:800]
                # if kwargs.get('render_train', False):
                #     print('render_train!')
                #     all_render_poses = []
                #     for i, pose in enumerate(self.poses):
                #         if len(all_render_poses) >= 600:
                #             break
                #         all_render_poses.append(torch.FloatTensor(pose))
                #         if i>0 and i<len(self.poses)-1:
                #             pose_new = (pose*3+self.poses[i+1])/4
                #             all_render_poses.append(torch.FloatTensor(pose_new))
                #             pose_new = (pose+self.poses[i+1])/2
                #             all_render_poses.append(torch.FloatTensor(pose_new))
                #             pose_new = (pose+self.poses[i+1]*3)/4
                #             all_render_poses.append(torch.FloatTensor(pose_new))
                #     render_c2w_f64 = torch.stack(all_render_poses)

        print(f'Loading {len(img_paths)} {split} images ...')
        for img_path in tqdm(img_paths):
            buf = [] # buffer for ray attributes: rgb, etc
            
            img = read_image(img_path, self.img_wh)
            buf += [torch.FloatTensor(img)]

            if 'HDR-NeRF' in self.root_dir: # get exposure
                folder = self.root_dir.split('/')
                scene = folder[-1] if folder[-1] != '' else folder[-2]
                if scene in ['bathroom', 'bear', 'chair', 'desk']:
                    e_dict = {e: 1/8*4**e for e in range(5)}
                elif scene in ['diningroom', 'dog']:
                    e_dict = {e: 1/16*4**e for e in range(5)}
                elif scene in ['sofa']:
                    e_dict = {0:0.25, 1:1, 2:2, 3:4, 4:16}
                elif scene in ['sponza']:
                    e_dict = {0:0.5, 1:2, 2:4, 3:8, 4:32}
                elif scene in ['box']:
                    e_dict = {0:2/3, 1:1/3, 2:1/6, 3:0.1, 4:0.05}
                elif scene in ['computer']:
                    e_dict = {0:1/3, 1:1/8, 2:1/15, 3:1/30, 4:1/60}
                elif scene in ['flower']:
                    e_dict = {0:1/3, 1:1/6, 2:0.1, 3:0.05, 4:1/45}
                elif scene in ['luckycat']:
                    e_dict = {0:2, 1:1, 2:0.5, 3:0.25, 4:0.125}
                e = int(img_path.split('.')[0][-1])
                buf += [e_dict[e]*torch.ones_like(img[:, :1])]

            self.rays += [torch.cat(buf, 1)]

            if kwargs.get('use_sem', False):
                result = inference_model(sem_model, img_path)
                label = result.pred_sem_seg.data.reshape(-1).cpu()
                label[torch.logical_or(label==0, label==1)] = 0 # road
                label[torch.logical_and(label<=7, label>=2)] = 1
                label[label==8] = 2
                label[label==9] = 3
                label[label==10] = 4
                label[torch.logical_or(label==11, label==12)] = 5
                label[label>=13] = 6
                self.labels += [label]
            
            if kwargs.get('use_shadow', False):
                shadows += [shadow_predictor.predict(img_path)]            

        self.rays = torch.stack(self.rays) # (N_images, hw, ?)
        self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)
        
        if kwargs.get('use_sem', False):            
            self.labels = torch.LongTensor(torch.stack(self.labels))
        
        if kwargs.get('use_shadow', False):
            self.shadows = torch.FloatTensor(torch.stack(self.shadows)).unsqueeze(-1)
        
        if split=='test':
            self.render_traj_rays = self.get_path_rays(render_c2w_f64)
            
    def get_path_rays(self, c2w_list):
        rays = {}
        print(f'Loading {len(c2w_list)} camera path ...')
        for idx, pose in enumerate(tqdm(c2w_list)):
            render_c2w = np.array(c2w_list[idx][:3])
            rays_o, rays_d = \
                get_rays(self.directions, torch.FloatTensor(render_c2w))
            rays[idx] = torch.cat([rays_o, rays_d], 1).cpu() # (h*w, 6)

        return rays
