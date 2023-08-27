DATA_ROOT=/hdd/datasets/TanksAndTempleBG/Playground
SEM_CONF=/hdd/mmsegmentation/ckpts/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py
SEM_CKPT=/hdd/mmsegmentation/ckpts/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth

# train
# python train.py --config configs/Playground.txt --exp_name playground-no \
#     --root_dir $DATA_ROOT --sem_conf_path $SEM_CONF --sem_ckpt_path $SEM_CKPT

# Testing view
python render.py --config configs/Playground.txt --exp_name playground-no \
    --root_dir $DATA_ROOT --sem_conf_path $SEM_CONF --sem_ckpt_path $SEM_CKPT \
    --weight_path ckpts/tnt/playground-no/epoch=79_slim.ckpt 

# Smog
# python render.py --config configs/Playground.txt --exp_name playground-smog-sparse \
#     --root_dir ../datasets/TanksAndTempleBG/Playground --dataset_name tnt \
#     --batch_size 1024 --scale 8.0 --num_epochs 60 --render_path --render_rgb \
#     --embed_a --embed_a_len 8 --use_skybox \
#     --ckpt_load ckpts/tnt/playground-no/epoch=149_slim.ckpt \
#     --depth_path results/tnt/playground-no/depth_raw.npy \
#     --simulate smog --depth_bound 0.9 --sigma 0.5 --rgb 0.925 0.906 0.758 

# Render panorama
# python render_panorama.py --config configs/Playground.txt --exp_name playground-no \
#     --root_dir ../datasets/TanksAndTempleBG/Playground --dataset_name tnt \
#     --pano_hw 512 1024 --pano_radius 0.2 \
#     --batch_size 1024 --use_skybox \
#     --ckpt_load ckpts/tnt/playground-no/epoch=149_slim.ckpt 

# Flood
# wave param: plane_len=0.2, ampl_const=5e5
# python render.py --config configs/Playground.txt --exp_name playground-flood-refraction \
#     --root_dir ../datasets/TanksAndTempleBG/Playground --dataset_name tnt  \
#     --batch_size 1024 --use_skybox --render_path --render_rgb --render_mask \
#     --ckpt_load ckpts/tnt/playground-no/epoch=149_slim.ckpt \
#     --simulate water --water_height 0.0 --rgb 0.488 0.406 0.32 --refraction_idx 1.35 --gf_r 5 --gf_eps 0.1 \
#     --plane_path ../datasets/TanksAndTempleBG/Playground/plane.npy \
#     --pano_path results/tnt/playground-no/panorama/rgb/0.png --v_forward 1 0 0 --v_down 0 1 0 --v_right 0 0 1 \
#     --gl_theta 0.008 --gl_sharpness 500 --wave_len 0.2 --wave_ampl 500000 \
#     --anti_aliasing_factor 1 --chunk_size 600000

# gan+ngp: smog
# python train.py --config configs/Playground.txt --exp_name playground-smog-gan-ngp \
#     --root_dir ../datasets/TanksAndTempleBG/Playground --dataset_name tnt \
#     --batch_size 1024 --scale 8.0 --num_epochs 50 --render_path --render_rgb \
#     --use_skybox --climate smog \
#     --ckpt_load ckpts/tnt/playground-smog-gan-ngp/last.ckpt #--val_only