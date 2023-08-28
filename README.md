# ClimateNeRF: Extreme Weather Synthesis in Neural Radiance Field

This is the official repo for PyTorch implementation of paper "ClimateNeRF: Extreme Weather Synthesis in Neural Radiance Field", ICCV 2023.

### [Paper](https://arxiv.org/abs/2211.13226) | [Project Page](https://climatenerf.github.io/)

## Prerequisites
This project is tested on:
- Ubuntu 18.04.6
- NVIDIA RTX 3090
- CUDA 11.3
- Python package manager `conda`

## Setup
- Create and activate environment by `conda create -n climatenerf python=3.8` and `conda activate climatenerf`.
- Install torch and torchvision by `pip install torch==1.11.0 torchvision==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113`
- Install `torch-scatter` by `pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu113.html`
- Install [PyTorch extension](https://github.com/NVlabs/tiny-cuda-nn#pytorch-extension) from `tinycudann`.
- Install mmsegmentation 
- Install remaining dependencies by `pip install -r requirements.txt`
- Install cuda extension with `pip install models/csrc`, `pip` >= 22.1 is needed.

## Acknowledgement
The code was built on [ngp_pl](https://github.com/kwea123/ngp_pl). Thanks [kwea123](https://github.com/kwea123) for the great project!