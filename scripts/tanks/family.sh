DATA_ROOT=/hdd/datasets/TanksAndTempleBG/Family
SEM_CONF=/hdd/mmsegmentation/ckpts/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py
SEM_CKPT=/hdd/mmsegmentation/ckpts/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth
CKPT=ckpts/tnt/family/epoch=79_slim.ckpt
# train
python train.py --config configs/Family.txt --exp_name family \
    --root_dir $DATA_ROOT --sem_conf_path $SEM_CONF --sem_ckpt_path $SEM_CKPT

# Testing view
python render.py --config configs/Family.txt --exp_name family \
    --root_dir $DATA_ROOT \
    --weight_path $CKPT \
    --render_depth --render_depth_raw --render_normal --render_semantic

# Smog
python render.py --config configs/Family.txt --exp_name family-smog \
    --root_dir $DATA_ROOT \
    --weight_path $CKPT \
    --simulate smog --chunk_size -1 

# Flood
# wave param: plane_len=0.2, ampl_const=5e5
python render.py --config configs/Family.txt --exp_name family-flood \
    --root_dir $DATA_ROOT \
    --weight_path $CKPT \
    --simulate water \
    --plane_path $DATA_ROOT/plane.npy \
    --anti_aliasing_factor 2  --chunk_size 600000