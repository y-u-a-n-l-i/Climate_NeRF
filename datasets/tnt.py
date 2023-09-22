import torch
import glob
import numpy as np
import os
from PIL import Image
from einops import rearrange
from tqdm import tqdm

from .ray_utils import *
from .color_utils import read_image, Shadow_predictor
from mmseg.apis import inference_model, init_model
import mmcv
from mmseg.utils import get_classes

from .base import BaseDataset

def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)

class tntDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, cam_scale_factor=0.95, render_train=False, **kwargs):
        super().__init__(root_dir, split, downsample)

        def sort_key(x):
            if len(x) > 2 and x[-10] == "_":
                return x[-9:]
            return x
        
        img_dir_name = None
        depth_dir_name = None
        if os.path.exists(os.path.join(root_dir, 'images')):
            img_dir_name = 'images'
        elif os.path.exists(os.path.join(root_dir, 'rgb')):
            img_dir_name = 'rgb'
        
        img_files = sorted(os.listdir(os.path.join(root_dir, img_dir_name)), key=sort_key)
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

        if os.path.exists(os.path.join(root_dir, 'depth')):
            depth_dir_name = 'depth'
        
        if split == 'train': prefix = '0_'
        elif split == 'val': prefix = '1_'
        elif 'Synthetic' in self.root_dir: prefix = '2_'
        elif split == 'test': prefix = '1_' # test set for real scenes
        
        imgs = sorted(glob.glob(os.path.join(self.root_dir, img_dir_name, prefix+'*.png')), key=sort_key)
        
        depths = []
        if kwargs.get('depth_mono', False):            
            depths = sorted(glob.glob(os.path.join(self.root_dir, depth_dir_name, prefix+'*.npy')), key=sort_key)
        poses = sorted(glob.glob(os.path.join(self.root_dir, 'pose', prefix+'*.txt')), key=sort_key)
        
        for img_name in img_files:
            img_file_path = os.path.join(root_dir, img_dir_name, img_name)
            img = Image.open(img_file_path)
            w, h = img.width, img.height
            break
        
        w, h = int(w*downsample), int(h*downsample)
        self.K = np.loadtxt(os.path.join(root_dir, 'intrinsics.txt'), dtype=np.float32)
        if self.K.shape[0]>4:
            self.K = self.K.reshape((4, 4))
        self.K = self.K[:3, :3]
        self.K *= downsample
        self.K = torch.FloatTensor(self.K)
        
        self.img_wh = (w, h)
        self.directions = get_ray_directions(h, w, self.K, anti_aliasing_factor=kwargs.get('anti_aliasing_factor', 1.0))
        
########################################################## get g.t. poses:
        
        self.has_render_traj = False
        if split == "test" and not render_train:
            self.has_render_traj = os.path.exists(os.path.join(root_dir, 'camera_path'))
        all_c2w = []
        for pose_fname in poses:
            pose_path = pose_fname
            #  intrin_path = path.join(root, intrin_dir_name, pose_fname)
            #  (right, down, forward)
            cam_mtx = np.loadtxt(pose_path).reshape(-1, 4)
            if len(cam_mtx) == 3:
                bottom = np.array([[0.0, 0.0, 0.0, 1.0]])
                cam_mtx = np.concatenate([cam_mtx, bottom], axis=0)
            all_c2w.append(torch.from_numpy(cam_mtx))  # C2W (4, 4) OpenCV

        c2w_f64 = torch.stack(all_c2w)
        # center = c2w_f64[:, :3, 3].mean(axis=0)
        # # radius = np.linalg.norm((c2w_f64[:, :3, 3]-center), axis=0).mean(axis=0)
        # up = -normalize(c2w_f64[:, :3, 1].sum(0))
        self.up = -normalize(c2w_f64[:,:3,1].mean(0)).float()
        print(f'up vector: {self.up}')
########################################################### scale the scene
        norm_pose_files = sorted(
            os.listdir(os.path.join(root_dir, 'pose')), key=sort_key
        )
        norm_poses = np.stack(
            [
                np.loadtxt(os.path.join(root_dir, 'pose', x)).reshape(-1, 4)
                for x in norm_pose_files
            ],
            axis=0,
        )
        scale = np.linalg.norm(norm_poses[..., 3], axis=-1).max()
        print(f"scene scale {scale}")
###########################################################
        if self.has_render_traj or render_train:
            print("render camera path" if not render_train else "render train interpolation")
            all_render_c2w = []
            pose_names = [
                x
                for x in os.listdir(os.path.join(root_dir, "camera_path/pose" if not render_train else "pose"))
                if x.endswith(".txt")
            ]
            pose_names = sorted(pose_names, key=lambda x: int(x[-9:-4]))
            for x in pose_names:
                cam_mtx = np.loadtxt(os.path.join(root_dir, "camera_path/pose" if not render_train else "pose", x)).reshape(
                    -1, 4
                )
                if len(cam_mtx) == 3:
                    bottom = np.array([[0.0, 0.0, 0.0, 1.0]])
                    cam_mtx = np.concatenate([cam_mtx, bottom], axis=0)
                all_render_c2w.append(torch.from_numpy(cam_mtx))  # C2W (4, 4) OpenCV
            render_c2w_f64 = torch.stack(all_render_c2w)
############ here we generate the test trajectories
############ we store the poses in render_c2w_f64
            ###### do interpolation ######
            if render_train:
                all_render_c2w_new = []
                for i, pose in enumerate(all_render_c2w):
                    if len(all_render_c2w_new) >= 600:
                        break
                    all_render_c2w_new.append(pose)
                    if i>0 and i<len(all_render_c2w)-1:
                        pose_new = (pose*3+all_render_c2w[i+1])/4
                        all_render_c2w_new.append(pose_new)
                        pose_new = (pose+all_render_c2w[i+1])/2
                        all_render_c2w_new.append(pose_new)
                        pose_new = (pose+all_render_c2w[i+1]*3)/4
                        all_render_c2w_new.append(pose_new)

                render_c2w_f64 = torch.stack(all_render_c2w_new)
            # render_c2w_f64 = generate_interpolated_path(render_c2w_f64.numpy(), 2)
            self.c2w = render_c2w_f64

        if kwargs.get('render_normal_mask', False):
            print("render up normal mask for train data!")
            all_render_c2w = []
            pose_names = [
                x
                for x in os.listdir(os.path.join(root_dir, "pose"))
                if x.endswith(".txt") and x.startswith("0_")
            ]
            pose_names = sorted(pose_names, key=lambda x: int(x[-9:-4]))
            for x in pose_names:
                cam_mtx = np.loadtxt(os.path.join(root_dir, "pose", x)).reshape(
                    -1, 4
                )
                if len(cam_mtx) == 3:
                    bottom = np.array([[0.0, 0.0, 0.0, 1.0]])
                    cam_mtx = np.concatenate([cam_mtx, bottom], axis=0)
                all_render_c2w.append(torch.from_numpy(cam_mtx))  # C2W (4, 4) OpenCV
            render_normal_c2w_f64 = torch.stack(all_render_c2w)
############################################################ normalize by camera
        c2w_f64[..., 3] /= scale
        
        if self.has_render_traj or render_train:
            render_c2w_f64[..., 3] /= scale
        
        if kwargs.get('render_normal_mask', False):
            render_normal_c2w_f64 /=scale
                                    
        if kwargs.get('render_normal_mask', False):
            render_normal_c2w_f64 = np.array(render_normal_c2w_f64)
            # render_normal_c2w_f64 = pose_avg_inv @ render_normal_c2w_f64
            render_normal_c2w_f64 = render_normal_c2w_f64[:, :3]            
########################################################### gen rays
        classes = kwargs.get('classes', 7)
        self.imgs = imgs
        if split.startswith('train'):
            self.read_meta('train', imgs, c2w_f64, sem_model=sem_model, shadow_predictor=shadow_predictor)
            if len(depths)>0:
                self.depths_2d = self.read_depth(depths)
        else: # val, test
            self.read_meta(split, imgs, c2w_f64, sem_model=sem_model, shadow_predictor=shadow_predictor)
            if len(depths)>0:
                self.depths_2d = self.read_depth(depths)
            
            if self.has_render_traj or render_train:
                self.render_traj_rays = self.get_path_rays(render_c2w_f64)
            if kwargs.get('render_normal_mask', False):
                self.render_normal_rays = self.get_path_rays(render_normal_c2w_f64)

    def read_meta(self, split, imgs, c2w_list, sem_model=None, shadow_predictor=None):
        # rays = {} # {frame_idx: ray tensor}
        rays = []
        labels = []
        shadows = []

        self.poses = []
        print(f'Loading {len(imgs)} {split} images...')
        for idx, img_path in enumerate(tqdm(imgs)):
            c2w = np.array(c2w_list[idx][:3])
            self.poses += [c2w]

            img = read_image(img_path=img_path, img_wh=self.img_wh)
            if img.shape[-1] == 4:
                img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
            rays += [img]
            
            if sem_model is not None:
                result = inference_model(sem_model, img_path)
                label = result.pred_sem_seg.data.reshape(-1).cpu()
                label[torch.logical_or(label==0, label==1)] = 0 # road
                label[torch.logical_and(label<=7, label>=2)] = 1
                label[label==8] = 2
                label[label==9] = 3
                label[label==10] = 4
                label[torch.logical_or(label==11, label==12)] = 5
                label[label>=13] = 6
                labels += [label]
            
            if shadow_predictor is not None:
                shadows += [shadow_predictor.predict(img_path)]
        
        self.poses = torch.FloatTensor(np.stack(self.poses))
        self.rays = torch.FloatTensor(np.stack(rays))
        
        if sem_model is not None:
            self.labels = torch.LongTensor(torch.stack(labels))
        if shadow_predictor is not None:
            self.shadows = torch.FloatTensor(torch.stack(shadows)).unsqueeze(-1)
    
    def read_depth(self, depths):
        depths_ = []

        for depth in depths:
            depths_ += [rearrange(np.load(depth), 'h w -> (h w)')]
        return torch.FloatTensor(np.stack(depths_))
    
    def get_path_rays(self, c2w_list):
        rays = {}
        print(f'Loading {len(c2w_list)} camera path ...')
        for idx, pose in enumerate(tqdm(c2w_list)):
            render_c2w = np.array(c2w_list[idx][:3])

            rays_o, rays_d = \
                get_rays(self.directions, torch.FloatTensor(render_c2w))

            rays[idx] = torch.cat([rays_o, rays_d], 1).cpu() # (h*w, 6)

        return rays

# if __name__ == '__main__':
#     import pickle 
#     save_path = 'output/dataset_cameras/family.pkl'

#     kwargs = {
#         'root_dir': '../datasets/TanksAndTempleBG/Family',
#         'render_traj': True,
#     }
#     dataset = tntDataset(
#         split='test',
#         **kwargs
#     )
    
#     cam_info = {
#         'img_wh': dataset.img_wh,
#         'K': np.array(dataset.K),
#         'c2w': np.array(dataset.c2w)
#     }

#     # with open(save_path, 'wb') as file:
#     #     pickle.dump(cam_info, file, protocol=pickle.HIGHEST_PROTOCOL)

#     with open(save_path, 'rb') as file:
#         cam = pickle.load(file)
    
#     print('Image W*H:', cam['img_wh'])
#     print('Camera K:', cam['K'])
#     print('Camera poses:', cam['c2w'].shape)
