from venv import create
import torch
from torch import nn
import torch.nn.functional as F
import tinycudann as tcnn
import vren
from .custom_functions import TruncExp, TruncTanh
import numpy as np
from einops import rearrange
from .rendering_old import NEAR_DISTANCE
from .ref_util import *

class Normal(nn.Module):
    def __init__(self, width=128, depth=5):
        super().__init__()
        
        self.normal_net = tcnn.NetworkWithInputEncoding(
                    n_input_dims=3, n_output_dims=3,
                    encoding_config={
                        "otype": "Frequency",
                        "n_frequencies": 6
                    },
                    network_config={
                        "otype": "CutlassMLP",
                        "activation": "ReLU",
                        "output_activation": "None",
                        "n_neurons": width,
                        "n_hidden_layers": depth,
                    }
                )
        # self.normal_net = nn.Sequential(
        #     nn.Linear(3, 64),
        #     nn.ReLU(True),
        #     nn.Linear(64, 64),
        #     nn.ReLU(True),
        #     nn.Linear(64, 64),
        #     nn.ReLU(True),
        #     nn.Linear(64, 64),
        #     nn.ReLU(True),
        #     nn.Linear(64, 3),            
        # )
        # self.activation = nn.Sigmoid()
        # self.activation = TruncExp.apply
    def forward(self, x, **kwargs):
        out = self.normal_net(x)
        # out = self.activation(out)
        return out
    
class NGP(nn.Module):
    def __init__(self, scale, rgb_act='Sigmoid', use_skybox=False, embed_a=False, embed_a_len=12, classes=7):
        super().__init__()

        self.rgb_act = rgb_act

        # scene bounding box
        self.scale = scale
        self.use_skybox = use_skybox
        self.embed_a = embed_a
        self.register_buffer('center', torch.zeros(1, 3))
        self.register_buffer('xyz_min', -torch.ones(1, 3)*scale)
        self.register_buffer('xyz_max', torch.ones(1, 3)*scale)
        self.register_buffer('half_size', (self.xyz_max-self.xyz_min)/2)

        # each density grid covers [-2^(k-1), 2^(k-1)]^3 for k in [0, C-1]
        self.cascades = max(1+int(np.ceil(np.log2(2*scale))), 1)
        self.grid_size = 128
        self.register_buffer('density_bitfield',
            torch.zeros(self.cascades*self.grid_size**3//8, dtype=torch.uint8))

        # constants
        L = 16; F = 2; log2_T = 19; N_min = 16
        b = np.exp(np.log(2048*scale/N_min)/(L-1))
        print(f'GridEncoding for spital: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')

        self.xyz_encoder = \
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
        
        self.xyz_net = nn.Sequential(
            nn.Linear(self.xyz_encoder.n_output_dims, 128),
            nn.Softplus(),
            nn.Linear(128, 1)
        )

        # constants
        L_ = 32; F_ = 2; log2_T_ = 21; N_min_ = 16
        b_ = np.exp(np.log(2048*scale/N_min_)/(L_-1))
        print(f'GridEncoding for RGB: Nmin={N_min_} b={b_:.5f} F={F_} T=2^{log2_T_} L={L_}')

        self.rgb_encoder = \
            tcnn.Encoding(3, {
                "otype": "HashGrid",
                "n_levels": L_,
                "n_features_per_level": F_,
                "log2_hashmap_size": log2_T_,
                "base_resolution": N_min_,
                "per_level_scale": b_,
                "interpolation": "Linear"
            })
        
        # self.xyz_net = nn.Sequential(
        #     nn.Linear(self.xyz_encoder.n_output_dims, 64),
        #     nn.ReLU(True),
        #     nn.Linear(64, 16)
        # )

        self.dir_encoder = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": 4,
                },
            )

        # self.normal_net = \
        #     tcnn.Network(
        #             n_input_dims=16, n_output_dims=3,
        #             network_config={
        #                 "otype": "CutlassMLP",
        #                 "activation": "ReLU",
        #                 "output_activation": self.rgb_act,
        #                 "n_neurons": 32,
        #                 "n_hidden_layers": 2,
        #             }
        #         )
        
        rgb_input_dim = self.rgb_encoder.n_output_dims + self.dir_encoder.n_output_dims + 3
        print(f'rgb_input_dim: {rgb_input_dim}')
        self.rgb_net = \
            tcnn.Network(
                n_input_dims=rgb_input_dim+embed_a_len if embed_a else rgb_input_dim,
                n_output_dims=3,
                network_config={
                    "otype": "CutlassMLP",
                    "activation": "ReLU",
                    "output_activation": rgb_act,
                    "n_neurons": 128,
                    "n_hidden_layers": 1,
                }
            )
                
        self.norm_pred_header = tcnn.Network(
                    n_input_dims=self.rgb_encoder.n_output_dims, n_output_dims=3,
                    network_config={
                        "otype": "CutlassMLP",
                        "activation": "ReLU",
                        "output_activation": "None",
                        "n_neurons": 32,
                        "n_hidden_layers": 1,
                    }
                )
        # self.norm_pred_act = nn.Tanh()

        self.semantic_header = tcnn.Network(
                    n_input_dims=self.rgb_encoder.n_output_dims, n_output_dims=classes,
                    network_config={
                        "otype": "CutlassMLP",
                        "activation": "ReLU",
                        "output_activation": "None",
                        "n_neurons": 32,
                        "n_hidden_layers": 1,
                    }
                )
        self.semantic_act = nn.Softmax(dim=-1)
            
        if use_skybox:
            print("Use skybox!")
            self.skybox_dir_encoder = \
                tcnn.Encoding(
                    n_input_dims=3,
                    encoding_config={
                        "otype": "SphericalHarmonics",
                        "degree": 3,
                    },
                )

            self.skybox_rgb_net = \
                tcnn.Network(
                    n_input_dims=9,
                    n_output_dims=3,
                    network_config={
                        "otype": "CutlassMLP",
                        "activation": "ReLU",
                        "output_activation": rgb_act,
                        "n_neurons": 32,
                        "n_hidden_layers": 1,
                    }
                )
            
        self.sigma_act = nn.Softplus(beta=100)
        if self.rgb_act == 'None': # rgb_net output is log-radiance
            for i in range(3): # independent tonemappers for r,g,b
                tonemapper_net = \
                    tcnn.Network(
                        n_input_dims=1, n_output_dims=1,
                        network_config={
                            "otype": "CutlassMLP",
                            "activation": "ReLU",
                            "output_activation": "Sigmoid",
                            "n_neurons": 64,
                            "n_hidden_layers": 1,
                        }
                    )
                setattr(self, f'tonemapper_net_{i}', tonemapper_net)

    def density(self, x, return_feat=False):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            return_feat: whether to return intermediate feature

        Outputs:
            sigmas: (N)
        """
        x = (x-self.xyz_min)/(self.xyz_max-self.xyz_min)
        # x = (x-self.center+1e-6)/torch.sqrt(torch.abs(x-self.center+1e-6)*self.scale)
        # x = (x+1)/2
        h = self.xyz_encoder(x)
        h = self.xyz_net(h)
        # sigmas = TruncExp.apply(h[:, 0])
        sigmas = self.sigma_act(h[:, 0])
        if return_feat: 
            feat_rgb = self.rgb_encoder(x)
            return sigmas, feat_rgb
        return sigmas

    def log_radiance_to_rgb(self, log_radiances, **kwargs):
        """
        Convert log-radiance to rgb as the setting in HDR-NeRF.
        Called only when self.rgb_act == 'None' (with exposure)

        Inputs:
            log_radiances: (N, 3)

        Outputs:
            rgbs: (N, 3)
        """
        if 'exposure' in kwargs:
            log_exposure = torch.log(kwargs['exposure'])
        else: # unit exposure by default
            log_exposure = 0

        out = []
        for i in range(3):
            inp = log_radiances[:, i:i+1]+log_exposure
            out += [getattr(self, f'tonemapper_net_{i}')(inp)]
        rgbs = torch.cat(out, 1)
        return rgbs

    @torch.enable_grad()
    def grad(self, x):
        x = x.requires_grad_(True)
        sigmas, feat_rgb = self.density(x, return_feat=True)
        grads = torch.autograd.grad(
                outputs=sigmas,
                inputs=x,
                grad_outputs=torch.ones_like(sigmas, requires_grad=False).cuda(),
                retain_graph=True
                )[0]
        return sigmas, feat_rgb, grads
    
    def forward(self, x, d, embed_a, **kwargs):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            d: (N, 3) directions

        Outputs:
            sigmas: (N)
            rgbs: (N, 3)
        """
        sigmas, feat_rgb, grads = self.grad(x)
        cnt = torch.sum(torch.isinf(grads))
        grads = grads.detach()
        if torch.any(torch.isnan(grads)):
            print('grads contains nan')
        if torch.any(torch.isinf(grads)):
            print('grads contains inf')
        normals_raw = -F.normalize(grads, p=2, dim=-1, eps=1e-6)
        if torch.any(torch.isnan(normals_raw)):
            print('normals_raw contains nan')
        if torch.any(torch.isinf(normals_raw)):
            print('normals_raw contains inf')
        
        # up_sem = self.up_label_header(feat_rgb)
        normals_pred = self.norm_pred_header(feat_rgb)
        normals_pred = -F.normalize(normals_pred, p=2, dim=-1, eps=1e-6)
        if torch.any(torch.isnan(normals_pred)):
            print('normals_pred contains nan')
        if torch.any(torch.isinf(normals_pred)):
            print('normals_pred contains inf')
        
        semantic = self.semantic_header(feat_rgb)
        semantic = self.semantic_act(semantic)
        # d = d/torch.norm(d, dim=1, keepdim=True)
        d = F.normalize(d, p=2, dim=-1, eps=1e-6)
        d = self.dir_encoder((d+1)/2)

        if self.embed_a:
            rgbs = self.rgb_net(torch.cat([x, d, feat_rgb, embed_a], 1))
        else:
            rgbs = self.rgb_net(torch.cat([x, d, feat_rgb], 1))
            

        if self.rgb_act == 'None': # rgbs is log-radiance
            if kwargs.get('output_radiance', False): # output HDR map
                rgbs = TruncExp.apply(rgbs)
            else: # convert to LDR using tonemapper networks
                rgbs = self.log_radiance_to_rgb(rgbs, **kwargs)
            
        return sigmas, rgbs, normals_raw, normals_pred, semantic, cnt
    
    @torch.no_grad()
    def forward_test(self, x, d, embed_a, **kwargs):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            d: (N, 3) directions

        Outputs:
            sigmas: (N)
            rgbs: (N, 3)
        """
        sigmas, feat_rgb = self.density(x, return_feat=True)
        
        normals_pred = self.norm_pred_header(feat_rgb)
        normals_pred = -F.normalize(normals_pred, p=2, dim=-1, eps=1e-6)
        normals_raw = torch.zeros_like(normals_pred).cuda()
        cnt = 0
        
        semantic = self.semantic_header(feat_rgb)
        semantic = self.semantic_act(semantic)
        # d = d/torch.norm(d, dim=1, keepdim=True)
        d = F.normalize(d, p=2, dim=-1, eps=1e-6)
        d = self.dir_encoder((d+1)/2)

        if self.embed_a:
            rgbs = self.rgb_net(torch.cat([x, d, feat_rgb, embed_a], 1))
        else:
            rgbs = self.rgb_net(torch.cat([x, d, feat_rgb], 1))
            

        if self.rgb_act == 'None': # rgbs is log-radiance
            if kwargs.get('output_radiance', False): # output HDR map
                rgbs = TruncExp.apply(rgbs)
            else: # convert to LDR using tonemapper networks
                rgbs = self.log_radiance_to_rgb(rgbs, **kwargs)
            
        return sigmas, rgbs, normals_raw, normals_pred, semantic, cnt
    
    def forward_skybox(self, d):
        if not self.use_skybox:
            return None
        d = d/torch.norm(d, dim=1, keepdim=True)
        d = self.skybox_dir_encoder((d+1)/2)
        rgbs = self.skybox_rgb_net(d)
        
        return rgbs     