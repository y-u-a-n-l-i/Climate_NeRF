from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn.functional as F
from kornia import create_meshgrid

class SnowSeed(Dataset):
    """
    Define length and sampling method
    """
    def __init__(self, up, height, R, interval = 0.01, range = 2):
        self.up = F.normalize(up.cpu(), dim=0).float()
        self.height = height
        
        principle = self.up * height.cpu()
        resolution = int(2*range / interval)
        grid = create_meshgrid(resolution, resolution, True)[0] * range
        self.coord_2d = grid.reshape(-1, 2)
        u, v = grid.unbind(-1)
        zeros = torch.zeros_like(u)
        rays_o = torch.stack([u, zeros, v], -1).reshape(-1, 3, 1)
        # y_axis = torch.FloatTensor([0, -1, 0])
        
        # identical = torch.FloatTensor([[1, 0, 0], 
        #                                [0, 1, 0], 
        #                                [0, 0, 1]])
        # cross = torch.linalg.cross(y_axis, self.up)
        # skewed_cross = torch.FloatTensor([[0, -cross[2], cross[1]],
        #                                   [cross[2], 0, -cross[0]],
        #                                   [-cross[1], cross[0], 0]])
        # cos_theta = torch.sum(y_axis* self.up)
        # self.R = identical + skewed_cross + 1/(1+cos_theta)*skewed_cross @ skewed_cross
        # self.R_inv = torch.linalg.inv(self.R)
        self.R = R.cpu()
        self.rays_o = torch.matmul(self.R, rays_o).reshape(-1, 3) + principle
        self.rays_d = -self.up

    def __len__(self):
        # if self.split.startswith('train'):
        #     return 1000
        # return len(self.rays_o)
        return 1000

    def __getitem__(self, idx):
        # if self.split.startswith('train'):
        #     # training pose is retrieved in train.py
        #     if self.ray_sampling_strategy == 'all_images': # randomly select images
        #         img_idxs = np.random.choice(len(self.poses), self.batch_size)
        #     elif self.ray_sampling_strategy == 'same_image': # randomly select ONE image
        #         img_idxs = np.random.choice(len(self.poses), 1)[0]
        #     # randomly select pixels
        #     pix_idxs = np.random.choice(self.img_wh[0]*self.img_wh[1], self.batch_size)
        #     rays = self.rays[img_idxs, pix_idxs]
        #     sample = {'img_idxs': img_idxs, 'pix_idxs': pix_idxs,
        #             'rgb': rays[:, :3]}
        #     if hasattr(self, 'labels'):
        #         labels = self.labels[img_idxs, pix_idxs]
        #         sample['label'] = labels
        #     if self.rays.shape[-1] == 4: # HDR-NeRF data
        #         sample['exposure'] = rays[:, 3:]
        sample = {}
        rays_idx = np.random.choice(len(self.rays_o), self.batch_size)
        rays_o = self.rays_o[rays_idx]
        rays_d = self.rays_d.unsqueeze(0).repeat(rays_o.shape[0], 1)
        coord_2d = self.coord_2d[rays_idx]
        sample = {'rays_o': rays_o,
                  'rays_d': rays_d,
                  'coord_2d': coord_2d}

        return sample