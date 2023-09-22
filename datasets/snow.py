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
        u, v = grid.unbind(-1)
        zeros = torch.zeros_like(u)
        rays_o = torch.stack([u, zeros, v], -1).reshape(-1, 3, 1)
        self.R = R.cpu()
        self.rays_o = torch.matmul(self.R, rays_o).reshape(-1, 3) + principle
        self.rays_d = -self.up

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        sample = {}
        rays_idx = np.random.choice(len(self.rays_o), self.batch_size)
        rays_o = self.rays_o[rays_idx]
        rays_d = self.rays_d.unsqueeze(0).repeat(rays_o.shape[0], 1)
        sample = {'rays_o': rays_o,
                  'rays_d': rays_d}

        return sample