import gzip 
import os 
import numpy as np
from PIL import Image 
import tensorflow as tf
import torch 
from tfrecord.torch.dataset import TFRecordDataset


def test():
    # image_hash: Unique hash of the image data.
    # cam_idx: Numeric ID of the camera that took the given image.
    # equivalent_exposure: A value that is equivalent to a measure of camera exposure.
    # height: Image height in pixels.
    # width: Image width in pixels.
    # image: PNG-encoded RGB image data of shape [H, W, 3].
    # ray_origins: Camera ray origins in 3D of shape [H, W, 3]. Has pixel-wise correspondence to “image”.
    # ray_dirs: Normalized camera ray directions in 3D of shape [H, W, 3]. Has pixel-wise correspondence to “image”.
    # intrinsics: Camera intrinsic focal lengths (f_u, f_v) of shape [2].

    root_dir = '../datasets/block_nerf'
    files = [os.path.join(root_dir, name) for name in os.listdir(root_dir) if 'tfrecord' in name]
    # files = tf.io.gfile.glob('waymo_block_nerf_mission_bay_validation.tfrecord*')
    dataset = tf.data.TFRecordDataset(filenames=files, compression_type='GZIP')

    iterator = dataset.as_numpy_iterator()
    i = 0
    
    for entry in iterator:
        example = tf.train.Example()
        example.MergeFromString(entry)

        # Decode RGB image.
        img_data = example.features.feature['image'].bytes_list.value[0]
        rgb = tf.image.decode_png(img_data).numpy()
        image = Image.fromarray(rgb)
        path = os.path.join(root_dir, 'images', '{:0>5d}.png'.format(i))
        image.save(path)
        i += 1
        print(i)
        # print('=============')
        # cam_idx = example.features.feature['cam_idx'].int64_list.value[0]
        # print('cam_idx:', cam_idx)
        # height = example.features.feature['height'].int64_list.value[0]
        # width = example.features.feature['width'].int64_list.value[0]
        # print('h, w:', height, width)
        # print('Image shape', rgb.shape)
        # rays_o  = example.features.feature['ray_origins'].float_list.value
        # rays_o = np.array(rays_o).reshape(height, width, -1)
        # # print(rays_o.shape)
        # # print(rays_o[:3,:3,:])
        # # rays_d  = example.features.feature['ray_dirs'].float_list.value
        # # rays_d = np.array(rays_d).reshape(height, width, -1)
        # # print(rays_d.shape)
        # # print(rays_d[:3,:3,:])



if __name__ == '__main__':
    test()