import os
import cv2,torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import plyfile
import skimage.measure
from models.networks import NGP
from opt import get_opts
from utils import load_ckpt

def convert_samples_to_ply(
    pytorch_3d_tensor,
    ply_filename_out,
    bbox,
    level=0.5,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply
    :param pytorch_3d_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to
    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    numpy_3d_tensor = pytorch_3d_tensor.numpy()
    voxel_size = list((bbox[1]-bbox[0]) / np.array(pytorch_3d_tensor.shape))

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_tensor, level=level, spacing=voxel_size
    )
    faces = faces[...,::-1] # inverse face orientation

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = bbox[0,0] + verts[:, 0]
    mesh_points[:, 1] = bbox[0,1] + verts[:, 1]
    mesh_points[:, 2] = bbox[0,2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    print("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)
    
hparams = get_opts()

os.makedirs(os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}'), exist_ok=True)
rgb_act = 'Sigmoid'
model = NGP(scale=hparams.scale, rgb_act=rgb_act, use_skybox=hparams.use_skybox, embed_a=hparams.embed_a, embed_a_len=hparams.embed_a_len).cuda()

ckpt_path = hparams.ckpt_path

print(f'ckpt specified: {ckpt_path} !')
load_ckpt(model, ckpt_path, prefixes_to_ignore=['embedding_a', 'msk_model'])

x_min, x_max = -1, 1
y_min, y_max = -0.3, 0.15
z_min, z_max = -1, 1

chunk_size = 128*128*32

xyz_min = torch.FloatTensor([[x_min, y_min, z_min]])
xyz_max = torch.FloatTensor([[x_max, y_max, z_max]])

dense_xyz = torch.stack(torch.meshgrid(
        torch.linspace(x_min, x_max, 512),
        torch.linspace(y_min, y_max, 128),
        torch.linspace(z_min, z_max, 512),
    ), -1).cuda()

samples = dense_xyz.reshape(-1, 3)
density = []
with torch.no_grad():
    for i in range(0, samples.shape[0], chunk_size):
        samples_ = samples[i:i+chunk_size]
        # with torch.cuda.amp.autocast(enabled=True, dtype=torch.float32):
        tmp = model.density(samples_)
        density.append(tmp.cpu())

density = torch.stack(density, dim=0)

density = density.reshape((dense_xyz.shape[0], dense_xyz.shape[1], dense_xyz.shape[2]))

bbox = torch.cat([xyz_min, xyz_max], dim=0)
# import ipdb; ipdb.set_trace()
convert_samples_to_ply(density.cpu(), os.path.join(f'results/{hparams.dataset_name}/{hparams.exp_name}', 'test.ply'), bbox=bbox.cpu(), level=10)