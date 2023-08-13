import torch
import numpy as np
from torch import nn
import tinycudann as tcnn

class implicit_mask(nn.Module):
    def __init__(self, latent=32, W=128):
        super().__init__()
        
        # constants
        L = 8; F = 2; log2_T = 16; N_min = 16
        b = np.exp(np.log(2048/N_min)/(L-1))
        print(f'GridEncoding for mask: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')

        self.mask_encoder = \
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
        
        self.mask_net = nn.Sequential(
            nn.Linear(self.mask_encoder.n_output_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, uvi):
        uvi_enc = self.mask_encoder(uvi)
        mask = self.mask_net(uvi_enc)
        return mask
# class implicit_mask(nn.Module):
#     def __init__(self, latent=32, W=128):
#         super().__init__()
        
#         self.uv_encoder = tcnn.Encoding(2, {
#                                 "otype": "Frequency", 
#                                 "n_frequencies": 8   
#                             })
        
#         self.mask_mapping = nn.Sequential(
#                             nn.Linear(latent + self.uv_encoder.n_output_dims, W), nn.ReLU(True),
#                             nn.Linear(W, W), nn.ReLU(True),
#                             nn.Linear(W, W), nn.ReLU(True),
#                             nn.Linear(W, W), nn.ReLU(True),
#                             nn.Linear(W, 1), nn.Sigmoid())

#     def forward(self, uv, latent):
#         uv_enc = self.uv_encoder(uv)
#         mask = self.mask_mapping(torch.cat([uv_enc, latent], dim=-1))
#         return mask