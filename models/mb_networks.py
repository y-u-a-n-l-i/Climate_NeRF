from venv import create
import torch
from torch import nn
import torch.nn.functional as F
import tinycudann as tcnn
import numpy as np
from .ref_util import *
import vren

def wrap_light(L, dir, wrap=8):
    '''
    Input:
    L: normalized light direction
    dir: direction related to the metaball center
    
    Output:
    diffuse_scale: a grey scale for diffuse color
    '''
    dot = torch.sum(L[None, :] * dir, dim=-1)
    diffuse_scale = (dot+wrap)/(1.+wrap)
    return diffuse_scale

def kernel_function(density_o, R, r):
    '''
    calculate densities for points inside metaballs
    
    Inputs:
    density_o: densities in the center (N_samples,)
    R: radius of metaballs (1)
    r: radius of samples inside metaballs (N_samples,)
    
    Output:
    density_r: densities of samples (N_samples, ) 
    '''
    r = torch.clamp(r, max=R)
    # density_r = (R/0.002)**(1/3) * (-4/9*(r/R)**6+17/9*(r/R)**4-22/9*(r/R)**2+1)*density_o
    density_r = 315/(64*torch.pi*1.5**7)*(1.5**2-(r/R)**2)**3*density_o
    density_r = torch.clamp(density_r, min=0)
    return density_r

def dkernel_function(density_o, R, r):
    '''
    calculate derivatives for densities inside metaballs
    
    Inputs:
    density_o: densities in the center (N_samples,)
    R: radius of metaballs (1)
    r: radius of samples inside metaballs (N_samples,)
    
    Output:
    ddensity_dr: derivatives of densities of samples (N_samples, ) 
    '''
    r = torch.clamp(r, max=R)
    # ddensity_dr = (R/0.002)**(1/3) * (-24/9*(r/R)**5+68/9*(r/R)**3-44/9*(r/R))*density_o
    ddensity_dr = -6*315/(64*torch.pi*1.5**7)*(1.5**2-(r/R)**2)**2*(r/R**2)*density_o
    ddensity_dr = torch.clamp(ddensity_dr, max=-1e-4)
    return ddensity_dr

def contract_to_unisphere(
    x: torch.Tensor,
    #  ord: Union[float, int] = float("inf"),
    eps: float = 1e-9,
):
    mag = torch.linalg.norm(x, ord=2, dim=-1, keepdim=True)
    mask = mag.squeeze(-1) > 1
    
    x[mask] = (2 - 1 / (mag[mask]+eps)) * (x[mask] / (mag[mask]+eps)) # [-2, 2]
    return (x+2)/4

class NGP_mb(nn.Module):
    def __init__(self, scale, up, ground_height, R, R_inv, interval, b=1.5, rgb_act='Sigmoid'):
        super().__init__()

        self.rgb_act = rgb_act

        # scene bounding box
        self.scale = scale
        self.register_buffer('center', torch.zeros(1, 3))
        self.register_buffer('xyz_min', -torch.ones(1, 3)*scale)
        self.register_buffer('xyz_max', torch.ones(1, 3)*scale)
        self.register_buffer('half_size', (self.xyz_max-self.xyz_min)/2)

        self.up = up
        self.ground_height = ground_height
        self.interval = interval
        self.R = R
        self.R_inv = R_inv
        self.mb_cascade = 5
        self.b = b

        # constants
        L_mb = 8; F_mb = 2; log2_T_mb = 19; N_min_mb = 32
        b_mb = np.exp(np.log(2048*scale/N_min_mb)/(L_mb-1))
        print(f'GridEncoding for metaball: Nmin={N_min_mb} b={b_mb:.5f} F={F_mb} L={L_mb}')

        self.mb_encoder = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "Grid",
	                "type": "Hash",
                    "n_levels": L_mb,
                    "n_features_per_level": F_mb,
                    "log2_hashmap_size": log2_T_mb,
                    "base_resolution": N_min_mb,
                    "per_level_scale": b_mb,
                    "interpolation": "Linear"
                })
        
        self.mb_net = nn.Sequential(
            nn.Linear(self.mb_encoder.n_output_dims, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.mb_act = nn.Sigmoid()
        # self.iso_color = nn.Parameter(torch.ones(3).cuda())
        self.grey_encoder = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "Grid",
	                "type": "Hash",
                    "n_levels": L_mb,
                    "n_features_per_level": F_mb,
                    "log2_hashmap_size": log2_T_mb,
                    "base_resolution": N_min_mb,
                    "per_level_scale": b_mb,
                    "interpolation": "Linear"
                })
        self.rgb_net = nn.Sequential(
            nn.Linear(self.grey_encoder.n_output_dims, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.rgb_act = nn.Sigmoid()

    def alpha(self, x):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            return_feat: whether to return intermediate feature

        Outputs:
            sigmas: (N)
        """
        x = (x-self.xyz_min)/(self.xyz_max-self.xyz_min)
        # x = contract_to_unisphere(x)
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float32):
            h = self.mb_encoder(x)
        h = self.mb_net(h)
        alphas = self.mb_act(h[:, 0])
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float32):
            h = self.grey_encoder(x)
        h = self.rgb_net(h)
        rgbs = self.rgb_act(h[:, 0:])
        
        return alphas, rgbs
    
    def forward(self, x, **kwargs):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            return_feat: whether to return intermediate feature

        Outputs:
            sigmas: (N)
        """
        x = torch.matmul(self.R_inv, x.reshape(-1, 3, 1)).reshape(-1, 3)
        alphas, rgbs = self.alpha(x)
        
        return alphas, rgbs.squeeze(-1)

    def forward_test(self, x, density_only=False, geometry_model=None, **kwargs):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            d: (N, 3) directions
        Outputs:
            sigmas: (N)
            rgbs: (N, 3)
        """
        with torch.no_grad():
            # convert x in colmap coordinate into snow falling coordinate
            x_sf = torch.matmul(self.R_inv, x.reshape(-1, 3, 1)).reshape(-1, 3) # in snow falling direction
            # x_sf = x
            N_samples = x_sf.shape[0]
            
            x_sf_vertices = []
            x_sf_dis = []
            x_sf_radius = []
            x_sf_grad = []
            
            for i in range(self.mb_cascade):
                radius = self.interval / self.b**i
                x_sf_coord = torch.floor(x_sf/radius) * radius
                offsets = radius * torch.FloatTensor([[0, 0, 0],
                                                    [0, 0, 1],
                                                    [0, 1, 0],
                                                    [1, 0, 0],
                                                    [0, 1, 1],
                                                    [1, 0, 1],
                                                    [1, 1, 0],
                                                    [1, 1, 1]]).cuda()
                x_sf_vertices_ = (x_sf_coord[:, None, :] + offsets[None, :, :]) # (N_samples, 8, 3)
                x_sf_dis_ = torch.norm(x_sf_vertices_-x_sf[:, None, :], dim=-1) # (N_samples, 8)
                x_sf_grad_ = (x_sf[:, None, :] - x_sf_vertices_) / (x_sf_dis_[..., None]) # (N_samples, 8, 3)
                x_sf_vertices.append(x_sf_vertices_.reshape(-1, 3))
                x_sf_dis.append(x_sf_dis_.reshape(-1))
                x_sf_radius.append(radius * torch.ones((x_sf_vertices_.shape[0] * x_sf_vertices_.shape[1])).cuda())
                x_sf_grad.append(x_sf_grad_.reshape(-1, 3))
            
            x_sf_vertices = torch.cat(x_sf_vertices, dim=0) # (N_samples * mb_cascade * 8, 3)
            x_sf_dis = torch.cat(x_sf_dis, dim=0) # (N_samples * mb_cascade * 8)
            x_sf_radius = torch.cat(x_sf_radius, dim=0) # (N_samples * mb_cascade * 8)
            x_sf_grad = torch.cat(x_sf_grad, dim=0) # (N_samples * mb_cascade * 8, 3)
            
            alpha, rgbs = self.alpha(x_sf_vertices) # (N_samples * mb_cascade * 8)

            x_sf_colmap = torch.matmul(self.R, x_sf_vertices.reshape(-1, 3, 1)).reshape(-1, 3)
            if geometry_model is not None:
                valid_mask = vren.test_occ(x_sf_colmap, geometry_model.density_bitfield, geometry_model.cascades,
                                           geometry_model.scale, geometry_model.grid_size)[0]
                alpha[valid_mask<1] *= 0

            if kwargs.get('cal_snow_occ', False):
                weighted_sigmoid = lambda x, weight, bias : 1./(1+torch.exp(-weight*(x-bias)))
                snow_occ = weighted_sigmoid(kwargs['snow_occ_net'](x_sf_colmap), 5, 0.6)
                alpha *= snow_occ

            rgbs = (rgbs+4)/(1+4) # value can be tuned
            if kwargs.get('pred_shadow', False):
                shadow = 0.7*(kwargs['sun_vis_net'](x_sf_colmap)>0.5).float() + 0.3 # value can be tuned
                rgbs *= shadow[:, None]
            
            density_c = alpha * kwargs.get('center_density', 2e3) # (N_samples * mb_cascade * 8)
            density_sample = kernel_function(density_c, x_sf_radius, x_sf_dis) # (N_samples * mb_cascade * 8)
            ddensity_sample_dxsf = dkernel_function(density_c, x_sf_radius, x_sf_dis)[:, None] * x_sf_grad
            
            densities = torch.chunk(density_sample, self.mb_cascade, dim=0)
            sigmas = torch.stack(densities, dim=-1) # (N_samples * 8, mb_cascade)
                    
            rgbs = torch.chunk(rgbs, self.mb_cascade, dim=0)
            rgbs = torch.stack(rgbs, dim=1).reshape(N_samples, 8*self.mb_cascade) * sigmas.view(N_samples, 8*self.mb_cascade) # (N_samples, 8*mb_cascade, 1)
            sigmas = torch.sum(sigmas.view(N_samples, self.mb_cascade*8), dim=-1)
            rgbs = (torch.sum(rgbs, dim=-1, keepdim=True) / (sigmas.view(N_samples, 1)+1e-6)).expand(-1, 3)
            
            weighted_sigmoid = lambda x, weight, bias : 1./(1+torch.exp(-weight*(x-bias)))
            thres_ratio = kwargs.get('mb_thres', 1/8)
            thres = weighted_sigmoid(sigmas, 50, kwargs.get('center_density', 2e3) * thres_ratio)
            sigmas = sigmas * thres
            if density_only:
                return sigmas 
            
            ddensities_dxsf = torch.chunk(ddensity_sample_dxsf, self.mb_cascade, dim=0)
            normals = torch.stack(ddensities_dxsf, dim=1) # (N_samples*8, mb_cascade, 3)
            normals = torch.sum(normals.reshape(N_samples, 8*self.mb_cascade, 3), dim=1) # (N_samples, 3)
            normals = -F.normalize(normals, dim=-1)
        rgbs = rgbs * wrap_light(torch.FloatTensor([0, -1., 0]).cuda(), normals)[:, None]
        
        return sigmas, rgbs, normals

class vis_net(nn.Module):
    def __init__(self, scale):
        super().__init__()
        
        self.register_buffer('xyz_min', -torch.ones(1, 3)*scale)
        self.register_buffer('xyz_max', torch.ones(1, 3)*scale)

        L = 8; F = 2; log2_T = 19; N_min = 32
        b = np.exp(np.log(2048*scale/N_min)/(L-1))
        print(f'GridEncoding for vis: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')

        self.vis_encoder = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "Grid",
	                "type": "Hash",
                    "n_levels": L,
                    "n_features_per_level": F,
                    "log2_hashmap_size": log2_T,
                    "base_resolution": N_min,
                    "per_level_scale": b,
                    "interpolation": "Linear"
                })
        
        self.vis_net = nn.Sequential(
            nn.Linear(self.vis_encoder.n_output_dims, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = (x-self.xyz_min)/(self.xyz_max-self.xyz_min)
        # x = contract_to_unisphere(x)
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float32):
            x_enc = self.vis_encoder(x)
        vis = self.vis_net(x_enc)
        return vis[:, 0]