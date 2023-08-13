from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union, Type

from bs4 import BeautifulStoneSoup
from datasets.mega_nerf.filesystem_dataset import FilesystemDataset
import torch
from torch.utils.data import DataLoader
from datasets.mega_nerf.image_metadata import ImageMetadata
from datasets.base import BaseDataset
from tqdm import tqdm 
from datasets.ray_utils import *
from einops import rearrange
import vedo

class MegaDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=0.25, train_every=6, **kwargs):
        super().__init__(root_dir, split, downsample)
        scale_factor = int(1 / downsample)
        start = kwargs.get('mega_frame_start', 0)
        end = kwargs.get('mega_frame_end', 10)
        train_items, val_items = get_image_metadata_partial(root_dir, start=start, end=end, scale_factor=scale_factor)
        items = train_items

        item = items[0]
        h, w = item.H, item.W
        self.img_wh = (w, h)
        
        fx, fy, sx, sy = item.intrinsics
        self.K = torch.tensor([
            [fx,  0, sx],
            [ 0, fy, sy],
            [ 0,  0,  1]
        ])
        self.directions = get_ray_directions(h, w, self.K, anti_aliasing_factor=kwargs.get('anti_aliasing_factor', 1.0))
        
        rot = torch.FloatTensor([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, -1]
        ])

        poses = []
        rays  = []
        for item in tqdm(items):
            img = item.load_image() / 255.0
            img = rearrange(img, 'h w c -> (h w) c')
            rays.append(img)
            
            c2w = item.c2w # [mega-nerf] x:right, y:up, z:backwards
            c2w = torch.cat([-c2w[:,1:2], c2w[:,0:1], c2w[:,2:]], -1)
            c2w = torch.cat([rot@c2w[:3, :3]@torch.inverse(rot), rot@c2w[:3,3:]], -1) 
            poses.append(c2w)
        
        poses = torch.stack(poses)
        cam_position = poses[:,:,-1]
        mean_position = torch.mean(cam_position, dim=0)
        poses[:,:,-1] -= mean_position[None]

        self.poses = poses #(N, 3, 4)
        self.rays  = torch.stack(rays)  #(N, H*W, 3)

        render_poses = generate_interpolated_path(poses.numpy(), n_interp=4)
        self.render_traj_rays = self.get_path_rays(render_poses)

    def get_path_rays(self, c2w_list):
        rays = {}
        print(f'Loading {len(c2w_list)} camera path ...')
        for idx, pose in enumerate(tqdm(c2w_list)):
            render_c2w = np.array(c2w_list[idx][:3])

            rays_o, rays_d = \
                get_rays(self.directions, torch.FloatTensor(render_c2w))

            rays[idx] = torch.cat([rays_o, rays_d], 1).cpu() # (h*w, 6)

        return rays

def test_dataset():
    root_dir = '../datasets/mega/building-pixsfm'
    kwargs = {
        'mega_frame_start': 753,
        'mega_frame_end': 846
    }
    dataset = MegaDataset(root_dir, split='train', **kwargs)

    print('poses:', dataset.poses.size())
    print('rays:', dataset.rays.size())
    print('render_traj_rays:', dataset.render_traj_rays.keys())
    poses = dataset.poses 
    cam_position = poses[:,:,-1]
    print(torch.mean(cam_position, dim=0))
    points = vedo.Points(cam_position)
    vedo.show(points, axes=1)
    # 753-846


def load_filesystem_dataset(
        dataset_path, 
        near:float=1,
        far:float=1e5,
        ray_altitude_range: List[float]=None,
        center_pixels: bool=True,
        device: torch.device=torch.device('cuda'),
        chunk_paths: List[Path]=None, 
        num_chunks: int=200,
        scale_factor: int=1, 
        disk_flush_size: int=10000000,
        split='train',
    ):
    train_items, val_items = get_image_metadata(dataset_path, scale_factor=scale_factor)
    items = None
    if split == 'train':
        items = train_items 
    else: 
        items = val_items 
    
    coordinate_info = torch.load(Path(dataset_path) / 'coordinates.pt', map_location='cpu')
    origin_drb = coordinate_info['origin_drb']
    pose_scale_factor = coordinate_info['pose_scale_factor']
    near = near / pose_scale_factor
    far  = far / pose_scale_factor

    dataset = FilesystemDataset(
        items, near, far, ray_altitude_range, center_pixels,
        device, [Path(x) for x in sorted(chunk_paths)], num_chunks, scale_factor, disk_flush_size
    )
    
    return dataset

def get_image_metadata(dataset_path, scale_factor:float=1, train_every=20) -> Tuple[List[ImageMetadata], List[ImageMetadata]]:
    dataset_path = Path(dataset_path)
    print('Get image metadta from:', dataset_path)

    train_path_candidates = sorted(list((dataset_path / 'train' / 'metadata').iterdir()))
    train_paths = [train_path_candidates[i] for i in
                range(0, len(train_path_candidates), train_every)]
    
    val_paths = sorted(list((dataset_path / 'val' / 'metadata').iterdir()))
    # train_paths += val_paths
    train_paths.sort(key=lambda x: x.name)
    val_paths.sort(key=lambda x: x.name)

    image_indices = {}
    for i, path in enumerate(train_paths+val_paths):
        image_indices[path.name] = i

    train_items = [
        get_metadata_item(x, image_indices[x.name], scale_factor, True) 
        for x in train_paths]
    val_items = [
        get_metadata_item(x, image_indices[x.name], scale_factor, True) 
        for x in val_paths]

    return train_items, val_items

def get_image_metadata_partial(dataset_path, start:int, end:int, scale_factor:float=1) -> Tuple[List[ImageMetadata], List[ImageMetadata]]:
    dataset_path = Path(dataset_path)
    print('Get image metadta from:', dataset_path)

    train_path_candidates = sorted(list((dataset_path / 'train' / 'metadata').iterdir()))
    start_id, end_id = 0, len(train_path_candidates)
    for i, path in enumerate(train_path_candidates):
        start_str = '{:0>6d}'.format(start)
        end_str   = '{:0>6d}'.format(end)
        if start_str in path.name:
            start_id = i
        if end_str in path.name:
            end_id = i+1
    train_paths = train_path_candidates[start_id:end_id]
    
    val_paths = sorted(list((dataset_path / 'val' / 'metadata').iterdir()))
    # train_paths += val_paths
    train_paths.sort(key=lambda x: x.name)
    val_paths.sort(key=lambda x: x.name)

    image_indices = {}
    for i, path in enumerate(train_paths+val_paths):
        image_indices[path.name] = i

    train_items = [
        get_metadata_item(x, image_indices[x.name], scale_factor, True) 
        for x in train_paths]
    val_items = [
        get_metadata_item(x, image_indices[x.name], scale_factor, True) 
        for x in val_paths]

    return train_items, val_items

def get_metadata_item(metadata_path: Path, image_index: int, scale_factor: int, is_val: bool) -> ImageMetadata:
    image_path = None
    for extension in ['.jpg', '.JPG', '.png', '.PNG']:
        candidate = metadata_path.parent.parent / 'rgbs' / '{}{}'.format(metadata_path.stem, extension)
        if candidate.exists():
            image_path = candidate
            break

    assert image_path.exists()

    metadata = torch.load(metadata_path, map_location='cpu')
    intrinsics = metadata['intrinsics'] / scale_factor
    # assert metadata['W'] % scale_factor == 0
    # assert metadata['H'] % scale_factor == 0
    # dataset_mask = metadata_path.parent.parent.parent / 'masks' / metadata_path.name
    # if cluster_mask_path is not None:
    #     if image_index == 0:
    #         main_print('Using cluster mask path: {}'.format(self.hparams.cluster_mask_path))
    #     mask_path = Path(self.hparams.cluster_mask_path) / metadata_path.name
    # elif dataset_mask.exists():
    #     if image_index == 0:
    #         main_print('Using dataset mask path: {}'.format(dataset_mask.parent))
    #     mask_path = dataset_mask
    # else:
    #     mask_path = None
    return ImageMetadata(image_path, metadata['c2w'], metadata['W'] // scale_factor, metadata['H'] // scale_factor,
                         intrinsics, image_index, None , is_val)


def test():
    dataset_path = '../datasets/mega/building-pixsfm'
    train_items, val_items = get_image_metadata(dataset_path, scale_factor=8)
    print(len(train_items))
    print(len(val_items))

    camera_positions = torch.cat([x.c2w[:3, 3].unsqueeze(0) for x in train_items + val_items])
    min_position = camera_positions.min(dim=0)[0]
    max_position = camera_positions.max(dim=0)[0]

    print('position min:', min_position)
    print('position max:', max_position)

    item = train_items[0]
    img = item.load_image()
    print(img.size())

    # dataset = get_dataset(dataset_path, scale_factor=8, chunk_paths=['results/test'])
    # print('Loaded dataset')
    # data_loader = DataLoader(dataset, batch_size=1024, 
    #                     shuffle=True, num_workers=0, pin_memory=True)
    # for idx, item in enumerate(data_loader):
    #     print(idx)
    #     print(item)
    #     break

if __name__ == '__main__':
    # test()
    test_dataset()