import numpy as np
import os
import imageio
from multiprocessing import Pool
import cv2
import copy
import torch
from .ray_utils import *
from .color_utils import *
from .base import BaseDataset
from tqdm import tqdm

def readVariable(fid, name, M, N):
    # rewind
    fid.seek(0, 0)
    # search for variable identifier
    line = 1
    success = 0
    while line:
        line = fid.readline()
        if line.startswith(name):
            success = 1
            break
    # return if variable identifier not found
    if success == 0:
        return None
    # fill matrix
    line = line.replace('%s:' % name, '')
    line = line.split()
    assert (len(line) == M * N)
    line = [float(x) for x in line]
    mat = np.array(line).reshape(M, N)
    return mat

def loadCalibrationCameraToPose(filename):
    # open file
    fid = open(filename, 'r')
    # read variables
    Tr = {}
    cameras = ['image_00', 'image_01', 'image_02', 'image_03']
    lastrow = np.array([0, 0, 0, 1]).reshape(1, 4)
    for camera in cameras:
        Tr[camera] = np.concatenate((readVariable(fid, camera, 3, 4), lastrow))
    # close file
    fid.close()
    return Tr

class KittiDataset(BaseDataset):
    def __init__(self, root_dir, split, downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)
        # path and initialization
        self.root_dir = root_dir
        self.split = split
        self.sequence = '2013_05_28_drive_0000_sync'
        self.start = kwargs['start']
        self.pseudo_root = os.path.join(root_dir, 'pspnet', self.sequence)
        self.cam2world_root = os.path.join(root_dir, 'data_poses', self.sequence, 'cam0_to_world.txt')
        self.visible_id = os.path.join(root_dir, 'visible_id', self.sequence)
        self.scene = kwargs['scene']
        self.img_root = os.path.join(root_dir, self.sequence)
        # load image_ids
        train_ids = np.arange(self.start, self.start + kwargs['train_frames'])
        test_ids = np.arange(self.start, self.start + kwargs['train_frames'])
        test_ids = np.array(kwargs['val_list'])
        if split == 'train':
            self.image_ids = train_ids
        elif split == 'val' or split == 'test':
            self.image_ids = test_ids

        self.shadow_predictor = None
        if kwargs.get('use_shadow', False):
            self.shadow_predictor = Shadow_predictor(kwargs.get('shadow_ckpt_path', 'pretrained/mtmt/iter_10000.pth'))

        # load intrinsics
        calib_dir = os.path.join(root_dir, 'calibration')
        self.intrinsic_file = os.path.join(calib_dir, 'perspective.txt')
        self.load_intrinsic(self.intrinsic_file)
        self.H = int(self.height * downsample)
        self.W = int(self.width  * downsample)
        self.img_wh = (self.W, self.H)
        self.K_00[:2] = self.K_00[:2] * downsample
        self.K_01[:2] = self.K_01[:2] * downsample
        self.intrinsic_00 = self.K_00[:, :-1]
        self.intrinsic_01 = self.K_01[:, :-1]
        self.directions = get_ray_directions(self.H, self.W, self.intrinsic_00, anti_aliasing_factor=kwargs.get('anti_aliasing_factor', 1.0))
        
        # load cam2world poses
        self.cam2world_dict_00 = {}
        self.cam2world_dict_01 = {}
        self.pose_file = os.path.join(root_dir, 'data_poses', self.sequence, 'poses.txt')
        poses = np.loadtxt(self.pose_file)
        frames = poses[:, 0]
        poses = np.reshape(poses[:, 1:], [-1, 3, 4])
        fileCameraToPose = os.path.join(calib_dir, 'calib_cam_to_pose.txt')
        self.camToPose = loadCalibrationCameraToPose(fileCameraToPose)['image_01']
        for line in open(self.cam2world_root, 'r').readlines():
            value = list(map(float, line.strip().split(" ")))
            self.cam2world_dict_00[value[0]] = np.array(value[1:]).reshape(4, 4)
        for frame, pose in zip(frames, poses):
            pose = np.concatenate((pose, np.array([0., 0., 0.,1.]).reshape(1, 4)))
            self.cam2world_dict_01[frame] = np.matmul(np.matmul(pose, self.camToPose), np.linalg.inv(self.R_rect))
        self.translation = np.array(kwargs['center_pose'])
        # load images
        self.visible_id = os.path.join(root_dir, 'visible_id', self.sequence)
        self.images_list_00 = {}
        self.images_list_01 = {}
        for idx in self.image_ids:
            frame_name = '%010d' % idx
            if os.path.exists(os.path.join(self.visible_id,frame_name + '.txt')) == False:
                continue
            image_file_00 = os.path.join(self.img_root, 'image_00/data_rect/%s.png' % frame_name)
            image_file_01 = os.path.join(self.img_root, 'image_01/data_rect/%s.png' % frame_name)
            if not os.path.isfile(image_file_00):
                raise RuntimeError('%s does not exist!' % image_file_00)
            self.images_list_00[idx] = image_file_00
            self.images_list_01[idx] = image_file_01

        # load metas
        self.build_metas(self.cam2world_dict_00, self.cam2world_dict_01, self.images_list_00, self.images_list_01)

    def load_intrinsic(self, intrinsic_file):
        with open(intrinsic_file) as f:
            intrinsics = f.read().splitlines()
        for line in intrinsics:
            line = line.split(' ')
            if line[0] == 'P_rect_00:':
                K = [float(x) for x in line[1:]]
                K = np.reshape(K, [3, 4])
                self.K_00 = K
            elif line[0] == 'P_rect_01:':
                K = [float(x) for x in line[1:]]
                K = np.reshape(K, [3, 4])
                intrinsic_loaded = True
                self.K_01 = K
            elif line[0] == 'R_rect_01:':
                R_rect = np.eye(4)
                R_rect[:3, :3] = np.array([float(x) for x in line[1:]]).reshape(3, 3)
            elif line[0] == "S_rect_01:":
                width = int(float(line[1]))
                height = int(float(line[2]))
        assert (intrinsic_loaded == True)
        assert (width > 0 and height > 0)
        self.width, self.height = width, height
        self.R_rect = R_rect

    def build_metas(self, cam2world_dict_00, cam2world_dict_01, images_list_00, images_list_01):
        self.imgs = []
        rays = []
        labels = []
        depths = []
        poses = []
        shadows = []
        for idx, frameId in tqdm(enumerate(self.image_ids)):
            pose = cam2world_dict_00[frameId]
            pose[:3, 3] = pose[:3, 3] - self.translation
            poses.append(pose)         
            image_path = images_list_00[frameId]
            self.imgs.append(image_path)
            image = (np.array(imageio.imread(image_path)) / 255.).astype(np.float32)
            image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
            # rays = build_rays(self.intrinsic_00, pose, image.shape[0], image.shape[1])
            rays_rgb = image.reshape(-1, 3)
            rays.append(rays_rgb)
            
            pseudo_label = cv2.imread(os.path.join(self.pseudo_root, self.scene,self.sequence[-9:-5]+'_{:010}.png'.format(frameId)), cv2.IMREAD_GRAYSCALE)
            pseudo_label = cv2.resize(pseudo_label, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            mask = np.logical_or(pseudo_label==7, pseudo_label==8)
            pseudo_label[mask] = 0 # road
            mask = np.logical_and(pseudo_label>=11, pseudo_label<=20)
            mask = np.logical_or(pseudo_label==35, mask)
            pseudo_label[mask] = 1 # building
            pseudo_label[pseudo_label==21] = 2 # vegetation
            pseudo_label[pseudo_label==22] = 3 # terrain
            pseudo_label[pseudo_label==23] = 4 # sky
            mask = np.logical_or(pseudo_label==24, pseudo_label==25)
            pseudo_label[mask] = 5 # person
            mask = np.logical_and(pseudo_label>=26, pseudo_label<=33)
            pseudo_label[mask] = 6 # vehicle
            pseudo_label[pseudo_label>6] = 256
            pseudo_label = pseudo_label.reshape(-1)
            labels.append(pseudo_label.astype(np.int32))
            
            depth = np.loadtxt("{}/sgm/{}/depth_{:010}_0.txt".format(self.root_dir, self.sequence, frameId)).astype(np.int32)
            depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            depth = depth.reshape(-1)
            
            depths.append(depth)

            if self.shadow_predictor is not None:
                shadows += [self.shadow_predictor.predict(image_path)]
            # input_tuples.append((rays, rays_rgb, frameId, intersection, pseudo_label, self.intrinsic_00, 0, depth))
        
        print('load meta_00 done')
    
        # if cfg.use_stereo == True:
        for idx, frameId in tqdm(enumerate(self.image_ids)):
            pose = cam2world_dict_01[frameId]
            pose[:3, 3] = pose[:3, 3] - self.translation
            poses.append(pose)
            image_path = images_list_01[frameId]
            self.imgs.append(image_path)
            image = (np.array(imageio.imread(image_path)) / 255.).astype(np.float32)
            image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
            # rays = build_rays(self.intrinsic_00, pose, image.shape[0], image.shape[1])
            rays_rgb = image.reshape(-1, 3)
            rays.append(rays_rgb)
            pseudo_label = 256 * np.ones_like(pseudo_label).astype(np.int32)
            labels.append(pseudo_label)
            depth = -1 * np.ones_like(pseudo_label)
            depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            depth = depth.reshape(-1)
            depths.append(depth)
            if self.shadow_predictor is not None:
                shadows += [self.shadow_predictor.predict(image_path)]
            # input_tuples.append((rays, rays_rgb, frameId, intersection, pseudo_label, self.intrinsic_01, 1, depth))
        print('load meta_01 done')
        # self.metas = input_tuples
        self.poses = torch.FloatTensor(np.stack(poses))
        self.up = -torch.nn.functional.normalize(self.poses[:,:3,1].mean(0), dim=0)
        print(f'up vector: {self.up}')
        scale = torch.norm(self.poses[..., 3], dim=-1).max()
        print(f"scene scale {scale}")
        self.poses[..., 3] /= 25
        self.poses = self.poses[:, :3,:]
        self.rays = torch.FloatTensor(np.stack(rays))
        self.labels = torch.LongTensor(np.stack(labels))
        self.depths_2d = torch.FloatTensor(np.stack(depths))
        render_c2w_f64 = generate_interpolated_path(self.poses.numpy(), 120)[:400]
        self.c2w = render_c2w_f64
        self.render_traj_rays = self.get_path_rays(render_c2w_f64)
        if self.shadow_predictor is not None:
            self.shadows = torch.FloatTensor(torch.stack(shadows)).unsqueeze(-1)
        
    def get_path_rays(self, c2w_list):
        rays = {}
        print(f'Loading {len(c2w_list)} camera path ...')
        for idx, pose in enumerate(c2w_list):
            render_c2w = np.array(c2w_list[idx][:3])

            rays_o, rays_d = \
                get_rays(self.directions, torch.FloatTensor(render_c2w))
            
            rays[idx] = torch.cat([rays_o, rays_d], 1).cpu() # (h*w, 6)

        return rays

if __name__ == '__main__':
    import pickle 
    save_path = 'output/dataset_cameras/kitti-1908.pkl'

    kwargs = {
        'root_dir': '../datasets/KITTI-360',
        'scene': 'seq0_1908-1971',
        'start': 1908,
        'train_frames': 64,
        'center_pose': [1040.42271023, 3738.8884705, 115.89219779],
        'val_list': [1915, 1925, 1935, 1945, 1955, 1965],
        'render_traj': True,
    }
    dataset = KittiDataset(
        split='test',
        **kwargs
    )
    
    cam_info = {
        'img_wh': dataset.img_wh,
        'K': np.array(dataset.K_00),
        'c2w': np.array(dataset.c2w)
    }

    with open(save_path, 'wb') as file:
        pickle.dump(cam_info, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(save_path, 'rb') as file:
        cam = pickle.load(file)
    
    print('Image W*H:', cam['img_wh'])
    print('Camera K:', cam['K'])
    print('Camera poses:', cam['c2w'].shape)