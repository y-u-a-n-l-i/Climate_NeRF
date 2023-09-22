# ClimateNeRF: Extreme Weather Synthesis in Neural Radiance Field

This is the official repo for PyTorch implementation of paper "ClimateNeRF: Extreme Weather Synthesis in Neural Radiance Field", ICCV 2023.

### [Paper](https://arxiv.org/abs/2211.13226) | [Project Page](https://climatenerf.github.io/)
https://github.com/y-u-a-n-l-i/Climate_NeRF/assets/68422992/59efd3e2-1dd2-4ce6-a07b-2d53a6a6c89e

## 🌦️ Prerequisites
This project is tested on:
- Ubuntu 18.04.6:
    - CUDA 11.3
- NVIDIA RTX 3090
- Python package manager `conda`

## 🌦️ Setup

### Environment
- Create and activate environment by `conda create -n climatenerf python=3.8` and `conda activate climatenerf`.
- Install torch and torchvision by `pip install torch==1.11.0 torchvision==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113`
- Install `torch-scatter` by `pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu113.html`
- Install [PyTorch extension](https://github.com/NVlabs/tiny-cuda-nn#pytorch-extension) from `tinycudann`.
- Install [mmsegmentation](https://mmsegmentation.readthedocs.io/en/latest/get_started.html) and download config and checkpoint files according to their instruction.
    - segmentation model `segformer_mit-b5_8xb1-160k_cityscapes-1024x1024` is recommended.
- Install remaining dependencies by `pip install -r requirements.txt`
- Install cuda extension with `pip install models/csrc`, `pip` >= 22.1 is needed. Recompile CUDA extension after any modifications.

<details>
<summary>potential bugs</summary>

1. Bug: when installing `tinycudann`
```
...
{PATH_TO}/tiny-cuda-nn/dependencies/json/json.hpp:3954:14: fatal error: filesystem: No such file or directory
    #include <filesystem>
            ^~~~~~~~~~~~
```
Solution in https://github.com/NVlabs/tiny-cuda-nn/issues/352 is recommended. If CUDA 11.3 is used, gcc-9 will be recommended.
</details>

### Dataset

1. TanksAndTemple dataset:

We use the download link from [ARF](https://github.com/Kai-46/ARF-svox2/blob/master/download_data.sh). Download and extract by:
```
pip install gdown
gdown 10Tj-0uh_zIIXf0FZ6vT7_te90VsDnfCU
unzip TanksAndTempleBG.zip && mv TanksAndTempleBG tnt
```

2. Kitti-360 dataset:

We use the [download link](https://drive.google.com/file/d/1oJF8e5m4yPrRArn6EPmqXguIl-au2FnT/view?pli=1) from [Panoptic NeRF](https://github.com/fuxiao0719/PanopticNeRF/tree/panopticnerf#data-preparation). We use the same data folder sructure as the one in [Panoptic NeRF](https://github.com/fuxiao0719/PanopticNeRF/tree/panopticnerf#data-preparation).

3. Colmap dataset:

We mainly test our project on garden scene in [mipnerf360 dataset](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip).

<<<<<<< HEAD
## Train

### Scene reconstruction with semantic predictions.
=======
### Model Parameters
- You can find the model checkpoints [here](https://uofi.box.com/s/hwcq1f69oo2he6w4pbwwtg3rdrs1pzui).
- Dowload plane parameters [here](https://uofi.box.com/s/pawqf4qmwpxcic09fk9sybc285r3yrrc), which are used in flood simulation. Please put the scene-specific `plane.npy` in the folder of dataset (e.g. `TanksAndTempleBG/Playground/plane.npy`)

## 🌦️Usage
The configuration of each scene could be adjusted in the config files under `configs/`, and we provide partial training/rendering/simulation scripts under `scripts/`.

In the following we use TanksAndTemple Playground scene as example, please edit the paths, experiment names accordingly. You can also run all the following together with `bash scripts/tanks/playground.sh`, and the output images and videos are under `results/`.

### Train
```
DATA_ROOT=/hdd/datasets/TanksAndTempleBG/Playground
SEM_CONF=/hdd/mmsegmentation/ckpts/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py
SEM_CKPT=/hdd/mmsegmentation/ckpts/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth

python train.py --config configs/Playground.txt --exp_name playground \
    --root_dir $DATA_ROOT --sem_conf_path $SEM_CONF --sem_ckpt_path $SEM_CKPT
```
### Novel View Synthesis
```
python render.py --config configs/Playground.txt --exp_name playground \
    --root_dir $DATA_ROOT \
    --weight_path $CKPT \
    --render_depth --render_depth_raw --render_normal --render_semantic
```

### 🌫️ Smog Simulation
```
python render.py --config configs/Playground.txt --exp_name playground-smog \
    --root_dir $DATA_ROOT \
    --weight_path $CKPT \
    --simulate smog --chunk_size -1 
```

### 🌊 Flood Simulation
```
python render.py --config configs/Playground.txt --exp_name playground-flood \
    --root_dir $DATA_ROOT \
    --weight_path $CKPT \
    --simulate water \
    --plane_path $DATA_ROOT/plane.npy \
    --anti_aliasing_factor 2 --chunk_size 600000
```

### ❄️ Snow Simulation



>>>>>>> c80579085c88d40ca728aa86f906bcc949055962
## Acknowledgement
The code was built on [ngp_pl](https://github.com/kwea123/ngp_pl). Thanks [kwea123](https://github.com/kwea123) for the great project!
