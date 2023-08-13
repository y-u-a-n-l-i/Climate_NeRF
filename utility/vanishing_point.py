import numpy as np
import os 
import cv2
from lu_vp_detect import VPDetection
import argparse 
from einops import rearrange
from tqdm import tqdm
from datasets import dataset_dict

def estimate_vertical_vec(vpd, image, pose, align_dim:int=1, align_neg=False):
    '''
    Estimate vertical vector with vanishing point and camera
    '''
    vps = vpd.find_vps(image) #(3, 3)
    vec_cam = np.cross(vps[0], vps[1])
    R_c2w = pose[:, :3].T
    vec_world = vec_cam @ R_c2w
    align_factor = -1 if align_neg else 1
    vec_world *= np.sign(vec_world[align_dim]) * align_factor
    return vec_world

def vectors_ransac(
    vectors, 
    n_iter:int,
    n_sample:int,
    threshold:float
):
    best_error = 1e8
    best_vector = None 
    n_vec = len(vectors)

    for _ in range(n_iter):
        idx_rand = np.random.permutation(n_vec)
        vectors_sample = vectors[idx_rand[:n_sample]]
        vectors_not_sample = vectors[idx_rand[n_sample:]]

        vectors_sample_sum = np.sum(vectors_sample, axis=0)
        vector_candidate = vectors_sample_sum / np.linalg.norm(vectors_sample_sum)
    
        error = 1 - (vectors_not_sample @ vector_candidate)**2
        # print('mean error:', np.mean(error))
        low_error = error < threshold
        if np.sum(low_error) > 0:
            vectors_inlier = np.concatenate([vectors_sample, vectors_not_sample[low_error]])
        else:
            vectors_inlier = vectors_sample 
        
        mean_error = np.mean(1 - (vectors_inlier @ vector_candidate)**2)
        if mean_error < best_error:
            best_error = mean_error 
            best_vector = vector_candidate

    print('Best error:', best_error)
    return best_vector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', default='tnt')
    parser.add_argument('-root_dir', help='Path to data folder')
    parser.add_argument('-length_thresh', type=float, default=50, help='Minimum length of the line in pixels')
    parser.add_argument('-repeat', type=int, default=10, help='Times of estimation for each image')
    parser.add_argument('-align_dim', type=int, default=1, help='Dimension of vector that is set in same direction')
    parser.add_argument('-align_neg', default=False, action='store_true', help='align_dim should be negative')
    parser.add_argument('-margin', type=float, default=0.01, help='Distance margin of camera closest to plane')
    parser.add_argument('-output', default='plane.npy')
    # RANSAC params
    parser.add_argument('-n_iter', type=int, default=100)
    parser.add_argument('-sample_rate', type=float, default=0.2)
    parser.add_argument('-ransac_thresh', type=float, default=0.005)
    parser.add_argument('-downsample', type=float, default=1.0)
    args = parser.parse_args()

    dataset_class = dataset_dict[args.dataset]
    dataset = dataset_class(root_dir=args.root_dir, downsample=args.downsample)

    K = dataset.K.numpy() #(3, 3)
    poses = dataset.poses.numpy() #(n, 3, 4)
    w, h = dataset.img_wh
    images = (dataset.rays * 255).numpy().astype(np.uint8)
    images = rearrange(images, 'n (h w) c -> n h w c', h=h) #(n, h, w, 3)
    n_images = images.shape[0]

    focal_length = K[0, 0]
    principal_point = (K[0, 2], K[1, 2])
    vpd = VPDetection(args.length_thresh, principal_point, focal_length)
    
    vec_list = []
    for i in tqdm(range(n_images)):
        for _ in range(args.repeat):
            vec = estimate_vertical_vec(vpd, images[i], poses[i], args.align_dim, args.align_neg)
            vec_list.append(vec)
    vec_list = np.stack(vec_list)
    normal = vectors_ransac(
        vec_list, 
        n_iter=args.n_iter,
        n_sample=int(len(vec_list) * args.sample_rate),
        threshold=args.ransac_thresh
    )

    cam_position = poses[:,:,-1] #(n, 3)
    # all cameras should be above the plane
    center2origin = np.min(cam_position @ normal) - args.margin
    center = center2origin * normal 

    plane_param = np.stack([normal, center])
    path = os.path.join(args.root_dir, args.output)
    np.save(path, plane_param)

def test():
    length_thresh = 50 # Minimum length of the line in pixels
    principal_point = (503.75,273.75) # Specify a list or tuple of two coordinates
                                # First value is the x or column coordinate
                                # Second value is the y or row coordinate
    focal_length = 1013.2 # Specify focal length in pixels
    seed = None # Or specify whatever ID you want (integer)

    vpd = VPDetection(length_thresh, principal_point, focal_length, seed)

    img = 'results/tnt/playground/frames/000-rgb.png' # Provide a path to the image
    # or you can read in the image yourself
    img = cv2.imread(img)

    # Run detection
    vps = vpd.find_vps(img)
    # unit vector from camera center to vanishing points
    # [v0, v1, v2] in camera coordinates

    # Display vanishing points
    print('vps')
    print(vps)

    # You can also access the VPs directly as a property
    # Only works when you run the algorithm first
    # vps = vpd.vps

    # You can also access the image coordinates for the vanishing points
    # Only works when you run the algorithm first
    vps_2D = vpd.vps_2D
    print('vps_2D')
    print(vps_2D)

if __name__ == '__main__':
    main()