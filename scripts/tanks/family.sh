DATA_ROOT=/hdd/datasets/TanksAndTempleBG/Family
SEM_CONF=/hdd/mmsegmentation/ckpts/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py
SEM_CKPT=/hdd/mmsegmentation/ckpts/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth

# train
python train.py --config configs/Family.txt --exp_name family \
    --root_dir $DATA_ROOT --sem_conf_path $SEM_CONF --sem_ckpt_path $SEM_CKPT

# Testing view
python render.py --config configs/Family.txt --exp_name family \
    --root_dir $DATA_ROOT --sem_conf_path $SEM_CONF --sem_ckpt_path $SEM_CKPT \
    --weight_path ckpts/tnt/family/epoch=79_slim.ckpt \
    --render_depth_raw

# Smog
python render.py --config configs/Family.txt --exp_name family-smog \
    --root_dir $DATA_ROOT --chunk_size -1 \
    --weight_path ckpts/tnt/family/epoch=79_slim.ckpt \
    --depth_path results/tnt/family/depth_raw.npy \
    --simulate smog --depth_bound 0.9 --sigma 0.5 --rgb 0.925 0.906 0.758 

# Flood
# wave param: plane_len=0.2, ampl_const=5e5
python render.py --config configs/Family.txt --exp_name family-flood \
    --root_dir $DATA_ROOT \
    --weight_path ckpts/tnt/family/epoch=79_slim.ckpt \
    --depth_path results/tnt/family/depth_raw.npy \
    --simulate water --water_height -0.02 --rgb 0.488 0.406 0.32 --refraction_idx 1.35 --gf_r 5 --gf_eps 0.1 \
    --plane_path $DATA_ROOT/plane.npy \
    --gl_theta 0.008 --gl_sharpness 500 --wave_len 1.0 --wave_ampl 2000000 \
    --anti_aliasing_factor 2 --chunk_size 600000