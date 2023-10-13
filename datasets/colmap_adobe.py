import torch
import numpy as np
import os
import glob
from tqdm import tqdm

from .ray_utils import *
from .color_utils import read_image, read_semantic
from .colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary

from .base import BaseDataset

TEST_FRAME_STEP = 2

def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)

def center_poses(poses, pts3d, box_scale):
    """
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
        pts3d: (N, 3) reconstructed point cloud

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pts3d_centered: (N, 3) centered point cloud
    """
    pts_max = np.max(pts3d, axis=0)
    pts_min = np.min(pts3d, axis=0)
    len_max = np.max(pts_max - pts_min)
    box_center = (pts_max + pts_min) / 2
    normalize_scale = (len_max/2)/box_scale * 1.02
    poses[:, :, -1] -= box_center[np.newaxis]
    poses[:, :, -1] /= normalize_scale
    # poses[:, :, -1] += np.array([0, 0, -5])[np.newaxis] #
    pts3d -= box_center[np.newaxis]
    pts3d /= normalize_scale
    # pts3d += np.array([0, 0, -5])[np.newaxis] #
    return poses, pts3d


class AdobeDataset(BaseDataset):
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
            if kwargs.get('use_sem', False):
                semantics = f'semantic_{int(1/self.downsample)}'
        else:
            folder = 'images'
            if kwargs.get('use_sem', False):
                semantics = 'semantic'
        # read successfully reconstructed images and ignore others
        img_paths = [os.path.join(self.root_dir, folder, name)
                     for name in sorted(img_names)]
        if kwargs.get('use_sem', False):
            sem_paths = []
            for name in sorted(img_names):
                sem_file_name = os.path.splitext(name)[0]+'.pgm'             
                sem_paths.append(os.path.join(self.root_dir, semantics, sem_file_name))
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
        self.up = torch.FloatTensor(-normalize(poses[:, :3, 1].mean(0)))
        box_scale = kwargs.get('scale', 16)
        self.poses, self.pts3d = center_poses(poses, pts3d, box_scale)
        
        # print('center:', np.mean(self.pts3d, axis=0))
        # forward_dir = np.mean(self.poses[:, :, 2], axis=0)
        # forward_dir = forward_dir / np.linalg.norm(forward_dir)
        # shift = -forward_dir * kwargs.get('scale', 16)
        # self.poses[:, :, 3] += shift
        # self.pts3d += shift

        # import vedo 
        # pts = vedo.Points(self.pts3d)
        # cam = vedo.Points(self.poses[:, :, -1], c='r')
        # vedo.show([pts, cam], axes=1)

        self.rays = []
        if kwargs.get('use_sem', False):
            self.labels = []
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
            if split=='test':
                n = self.poses.shape[0]
                anchor_n = 10
                stride = n // anchor_n
                anchor_ids = [i for i in range(0, n, stride)]
                pose_anchors = self.poses[anchor_ids]
                render_c2w = generate_interpolated_path(pose_anchors, 30)[:300]
                self.render_c2w = torch.FloatTensor(render_c2w)

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

        self.rays = torch.stack(self.rays) # (N_images, hw, ?)
        self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)
        
        if kwargs.get('use_sem', False):
            classes = kwargs.get('classes', 7)
            for sem_path in sem_paths:
                label = read_semantic(sem_path=sem_path, sem_wh=self.img_wh, classes=classes)
                self.labels += [label]
            
            self.labels = torch.LongTensor(np.stack(self.labels))
        
        if split=='test':
            self.render_path_rays = self.get_path_rays(self.render_c2w)
            
    def get_path_rays(self, c2w_list):
        rays = {}
        print(f'Loading {len(c2w_list)} camera path ...')
        for idx, pose in enumerate(tqdm(c2w_list)):
            render_c2w = np.array(c2w_list[idx][:3])
            rays_o, rays_d = \
                get_rays(self.directions, torch.FloatTensor(render_c2w))
            rays[idx] = torch.cat([rays_o, rays_d], 1).cpu() # (h*w, 6)

        return rays


def test():
    root_dir = '/hdd/datasets/adobe/statue'
    kwargs = {'scale': 4}
    dataset = AdobeDataset(root_dir, **kwargs)


if __name__ == '__main__':
    test()