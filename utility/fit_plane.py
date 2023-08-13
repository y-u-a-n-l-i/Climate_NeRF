import numpy as np 
from matplotlib import pyplot as plt

class Plane():
    def __init__(self, params=[1, 1, 1, 1]):
        # Define a 3D plane: 
        # ax + by + cz + d = 0
        # normal * (p - center) = 0
        a, b, c, d = params
        normal = np.array([a, b, c])
        norm = np.linalg.norm(normal)
        normal = normal / norm
        center = (-d/norm) * normal

        self.normal = normal 
        self.center = center
    
    def normal(self):
        return self.normal 
    
    def center(self):
        return self.center

    def move_by_distance(self, d):
        new_center = self.center + self.normal * d 
        self.center = new_center
    
    def square_error(self, points):
        '''
        Input
            points: (n, 3)
        Return 
            error: (n)
        '''
        diff = (points - self.center) @ self.normal
        error = diff ** 2
        return error
    
    def mean_square_error(self, points):
        return np.mean(self.square_error(points))
    
    def absolute_error(self, points):
        diff = (points - self.center) @ self.normal
        error = np.abs(diff)
        return error 
    
    def mean_absolute_error(self, points):
        return np.mean(self.absolute_error(points))


def plane_lse(points):
    '''
    Solve plane parameters that fit 3D points with least squre error
    Input
        points: (n, 3)
    Return 
        normal (3)
        constant
    '''
    n_point = len(points)
    ones = np.ones((n_point, 1))
    xy_ones = np.concatenate([points[:,:2], ones], axis=1)
    z = points[:, -1]

    sol = np.linalg.lstsq(xy_ones, -z)[0]
    a, b, d = sol
    # plane function: ax + by + z + d = 0
    plane = Plane([a, b, 1, d])
    return plane

def random_sample(points, n_sample):
    index = np.random.permutation(len(points))[:n_sample]
    points_sample = points[index]
    return points_sample

def plane_ransac(
    points, 
    n_iter:int,
    n_sample:int,
    threshold:float
):
    '''
    Get a plane fit the points best with RANSAC
    '''
    best_error = 1e8
    best_plane = None 
    n_points = len(points)

    for _ in range(n_iter):
        idx_rand = np.random.permutation(n_points)
        points_sample = points[idx_rand[:n_sample]]
        points_not_sample = points[idx_rand[n_sample:]]

        plane_sample = plane_lse(points_sample)
        error = plane_sample.absolute_error(points_not_sample)
        # print('mean error:', np.mean(error))
        low_error = error < threshold
        if np.sum(low_error) > 0:
            points_inlier = np.concatenate([points_sample, points_not_sample[low_error]])
        else:
            points_inlier = points_sample 
        
        mean_error = plane_sample.mean_absolute_error(points_inlier)
        if mean_error < best_error:
            best_error = mean_error 
            best_plane = plane_sample
    return best_plane

def test():
    pass 

if __name__ == '__main__':
    test()