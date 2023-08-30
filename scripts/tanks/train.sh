DATA_ROOT=/hdd/datasets/TanksAndTempleBG/Train
SEM_CONF=/hdd/mmsegmentation/ckpts/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py
SEM_CKPT=/hdd/mmsegmentation/ckpts/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth

# train
python train.py --config configs/Train.txt --exp_name train-no \
    --root_dir $DATA_ROOT --sem_conf_path $SEM_CONF --sem_ckpt_path $SEM_CKPT

# Testing view
python render.py --config configs/Train.txt --exp_name train-no \
    --root_dir $DATA_ROOT --sem_conf_path $SEM_CONF --sem_ckpt_path $SEM_CKPT \
    --weight_path ckpts/tnt/train-no/epoch=79_slim.ckpt \
    --render_depth_raw

# Smog
python render.py --config configs/Train.txt --exp_name train-smog \
    --root_dir $DATA_ROOT --chunk_size -1 \
    --weight_path ckpts/tnt/train-no/epoch=79_slim.ckpt \
    --depth_path results/tnt/train-no/depth_raw.npy \
    --simulate smog --depth_bound 0.9 --sigma 0.5 --rgb 0.925 0.906 0.758 

# Flood
# wave param: plane_len=0.2, ampl_const=5e5
python render.py --config configs/Train.txt --exp_name train-flood \
    --root_dir $DATA_ROOT \
    --weight_path ckpts/tnt/train-no/epoch=79_slim.ckpt \
    --depth_path results/tnt/train-no/depth_raw.npy \
    --simulate water --water_height 0.0 --rgb 0.488 0.406 0.32 --refraction_idx 1.35 --gf_r 5 --gf_eps 0.1 \
    --plane_path $DATA_ROOT/plane.npy \
    --gl_theta 0.008 --gl_sharpness 500 --wave_len 0.2 --wave_ampl 500000 \
    --anti_aliasing_factor 2 --chunk_size 600000