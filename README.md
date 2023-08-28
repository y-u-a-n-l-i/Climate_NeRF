# ClimateNeRF: Extreme Weather Synthesis in Neural Radiance Field

This is the official repo for PyTorch implementation of paper "ClimateNeRF: Extreme Weather Synthesis in Neural Radiance Field", ICCV 2023.

### [Paper](https://arxiv.org/abs/2211.13226) | [Project Page](https://climatenerf.github.io/)
https://github.com/y-u-a-n-l-i/Climate_NeRF/assets/68422992/59efd3e2-1dd2-4ce6-a07b-2d53a6a6c89e

## Prerequisites
This project is tested on:
- Ubuntu 18.04.6:
    - CUDA 11.3
- NVIDIA RTX 3090
- Python package manager `conda`

## Setup

### Environment
- Create and activate environment by `conda create -n climatenerf python=3.8` and `conda activate climatenerf`.
- Install torch and torchvision by `pip install torch==1.11.0 torchvision==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113`
- Install `torch-scatter` by `pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu113.html`
- Install [PyTorch extension](https://github.com/NVlabs/tiny-cuda-nn#pytorch-extension) from `tinycudann`.
- Install [mmsegmentation](https://mmsegmentation.readthedocs.io/en/latest/get_started.html) and download config and checkpoint files according to their instruction.
    - segmentation model `segformer_mit-b5_8xb1-160k_cityscapes-1024x1024` is recommended.
- Install remaining dependencies by `pip install -r requirements.txt`
- Install cuda extension with `pip install models/csrc`, `pip` >= 22.1 is needed.

<details>
    <summary> potential bugs </summary>

    - Bug: when installing `tinycudann`
    
    ```
    ...
    {PATH_TO}/tiny-cuda-nn/dependencies/json/json.hpp:3954:14: fatal error: filesystem: No such file or directory
     #include <filesystem>
              ^~~~~~~~~~~~
    ```

</details>

### Dataset



## Acknowledgement
The code was built on [ngp_pl](https://github.com/kwea123/ngp_pl). Thanks [kwea123](https://github.com/kwea123) for the great project!
