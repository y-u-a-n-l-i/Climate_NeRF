import argparse
import configargparse

def get_opts():
    # parser = argparse.ArgumentParser()
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='config file path')
    # common args for all datasets
    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='nerf',
                        choices=['nerf', 'nsvf', 'colmap', 'nerfpp', 'rtmv', 'tnt', 'kitti', 'mega'],
                        help='which dataset to train/test')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'trainval'],
                        help='use which split to train')
    parser.add_argument('--downsample', type=float, default=1.0,
                        help='downsample factor (<=1.0) for the images')
    parser.add_argument('--anti_aliasing_factor', type=float, default=1.0,
                        help='Render larger images then downsample')

    # model parameters
    parser.add_argument('--scale', type=float, default=0.5,
                        help='scene scale (whole scene must lie in [-scale, scale]^3')
    parser.add_argument('--use_skybox', action='store_true', default=False,
                        help='whether to use skybox')
    parser.add_argument('--use_exposure', action='store_true', default=False,
                        help='whether to train in HDR-NeRF setting')
    parser.add_argument('--embed_a', action='store_true', default=False,
                        help='whether to use appearance embeddings')
    parser.add_argument('--embed_a_len', type=int, default=4,
                        help='the length of the appearance embeddings')
    parser.add_argument('--embed_msk', action='store_true', default=False,
                        help='whether to use sigma embeddings')
    parser.add_argument('--num_classes', type=int, default=7,
                        help='total number of semantic classes')
    parser.add_argument('--sem_conf_path', type=str, default=None,
                        help='path to config file of semantic segmentation model')
    parser.add_argument('--sem_ckpt_path', type=str, default=None,
                        help='path to checkpoint file of semantic segmentation model')
    parser.add_argument('--styl_img_path', type=str, default=None,
                        help='path to style image')

    # for kitti 360 dataset
    parser.add_argument('--kitti_scene', type=str, default='seq0_1538-1601',
                        choices=['seq0_1538-1601', 'seq0_1728-1791', 'seq0_1908-1971', 'seq0_3353-3416'],
                        help='scene in kitti dataset')
    parser.add_argument('--start', type=int, default=1538,
                        help='start index of frames to train.')
    parser.add_argument('--train_frames', type=int, default=64,
                        help='total number of frames to train.')
    parser.add_argument('--center_pose', action="append", default=[],
                        help='center of poses in kitti-360 dataset')
    parser.add_argument('--val_list', action="append", default=[],
                        help='frames for val dataset')

    # for mega-nerf dataset
    parser.add_argument('--mega_frame_start', type=int, default=753)
    parser.add_argument('--mega_frame_end',   type=int, default=846)

    # training options
    parser.add_argument('--batch_size', type=int, default=8192,
                        help='number of rays in a batch')
    parser.add_argument('--ray_sampling_strategy', type=str, default='all_images',
                        choices=['all_images', 'same_image'],
                        help='''
                        all_images: uniformly from all pixels of ALL images
                        same_image: uniformly from all pixels of a SAME image
                        ''')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument('--depth_mono', action='store_true', default=False,
                        help='use 2D predicted depth')
    # experimental training options
    parser.add_argument('--random_bg', action='store_true', default=False,
                        help='''whether to train with random bg color (real dataset only)
                        to avoid objects with black color to be predicted as transparent
                        ''')

    # validation options
    parser.add_argument('--eval_lpips', action='store_true', default=False,
                        help='evaluate lpips metric (consumes more VRAM)')
    parser.add_argument('--val_only', action='store_true', default=False,
                        help='run only validation (need to provide ckpt_path)')
    parser.add_argument('--no_save_test', action='store_true', default=False,
                        help='whether to save test image and video')
    parser.add_argument('--render_traj', action='store_true', default=False,
                        help='render video on a trajectory')
    parser.add_argument('--render_train', action='store_true', default=False,
                        help='interpolate among training views to get camera trajectory')

    # misc
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint to load (including optimizers, etc)')
    parser.add_argument('--weight_path', type=str, default=None,
                        help='pretrained checkpoint to load (excluding optimizers, etc)')
    
    # render
    parser.add_argument('--render_rgb', action='store_true', default=False,
                        help='render rgb series')
    parser.add_argument('--render_depth', action='store_true', default=False,
                        help='render depth series')
    parser.add_argument('--render_depth_raw', action='store_true', default=False,
                        help='save depth_raw.npy')
    parser.add_argument('--render_normal', action='store_true', default=False,
                        help='render normal series')
    parser.add_argument('--render_semantic', action='store_true', default=False,
                        help='render semantic segmentation series')
    parser.add_argument('--normal_composite', action='store_true', default=False,
                        help='render normal+rgb composition series')
    parser.add_argument('--render_points', action='store_true', default=False,
                        help='render depth points')
    parser.add_argument('--chunk_size', type=int, default=131072, 
                        help='Divide image into chunks for rendering')
    
    # Simulation
    parser.add_argument('--simulate', default=None, 
                        help='Simulation effect: [smog, water, snow]')
    parser.add_argument('--depth_path', help='path to depth_raw.npy')
    # Smog
    parser.add_argument('--depth_bound', type=float, default=0.9)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--rgb', nargs='+', type=float, default=[1.0, 1.0, 1.0])
    # water
    parser.add_argument('--water_height', type=float, default=0.0)
    parser.add_argument('--plane_path', help='path to plane parameters')
    parser.add_argument('--refraction_idx', type=float, default=1.33)
    parser.add_argument('--gf_r', type=int, default=5, help='Guided filter, filter radius')
    parser.add_argument('--gf_eps', type=float, default=0.1, help='Guided filter, regularization term')
    parser.add_argument('--wave_len', type=float, default=1.0, help='Length of wave plane')
    parser.add_argument('--wave_ampl', type=int, default=2e6, help='Amplitude constant of wave')
    # panorama
    parser.add_argument('--pano_hw', type=int, nargs='+', default=[512, 1024], help='h, w of panorama image')
    parser.add_argument('--pano_radius', type=float, default=0.1, help='Set ray origin on the sphere at (0, 0, 0) with radius')
    parser.add_argument('--v_forward', type=float, nargs='+', default=[1, 0, 0])
    parser.add_argument('--v_down',    type=float, nargs='+', default=[0, 1, 0])
    parser.add_argument('--v_right',   type=float, nargs='+', default=[0, 0, 1])
    parser.add_argument('--pano_path', help='path to panorama image')
    # glossy 
    parser.add_argument('--gl_theta', type=float, default=0.01)
    parser.add_argument('--gl_sharpness', type=float, default=1000)
    parser.add_argument('--refract_decay', type=float, default=-1)
    
    return parser.parse_args()
