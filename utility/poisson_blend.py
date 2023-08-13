try:
    # for backward compatibility
    import imageio.v2 as imageio
except ModuleNotFoundError:
    import imageio 
from matplotlib import pyplot as plt
import numpy as np
import argparse 
import scipy 

def poisson_blending(
    img_src,
    mask_tblr,
    img_tar, 
    pos_xy
):
    x_min, x_max, y_min, y_max = mask_tblr
    h_var, w_var = x_max - x_min, y_max - y_min 
    n_var = h_var * w_var 
    n_eqn = (h_var-1)*w_var + h_var*(w_var-1) + (h_var + w_var) * 2
    pos_x, pos_y = pos_xy
    var_idx = np.arange(h_var * w_var).reshape(h_var, w_var)

    img_blended = []
    for c in range(3):
        A = scipy.sparse.lil_matrix((n_eqn, n_var), dtype='double')
        b = np.zeros((n_eqn))
        img_src_c = img_src[:,:,c]
        img_tar_c = img_tar[:,:,c]
        i_eqn = 0

        # inside filling region
        for x in range(h_var - 1):
            for y in range(w_var):
                A[i_eqn, var_idx[x+1, y]] = 1
                A[i_eqn, var_idx[x, y]]   = -1
                b[i_eqn] = img_src_c[x_min+x+1, y_min+y] - img_src_c[x_min+x, y_min+y]
                i_eqn += 1

        for x in range(h_var):
            for y in range(w_var - 1):
                A[i_eqn, var_idx[x, y+1]] = 1
                A[i_eqn, var_idx[x, y]]   = -1
                b[i_eqn] = img_src_c[x_min+x, y_min+y+1] - img_src_c[x_min+x, y_min+y]
                i_eqn += 1
        # on the boundary
        for x in range(h_var): #left 
            A[i_eqn, var_idx[x, 0]] = 1
            diff = img_src_c[x_min+x, y_min] - img_src_c[x_min+x, y_min-1]
            b[i_eqn] = img_tar_c[pos_x+x, pos_y-1] + diff 
            i_eqn += 1
        for x in range(h_var): #right
            A[i_eqn, var_idx[x, -1]] = 1
            diff = img_src_c[x_min+x, y_max-1] - img_src_c[x_min+x, y_max]
            b[i_eqn] = img_tar_c[pos_x+x, pos_y+w_var] + diff 
            i_eqn += 1
        for y in range(w_var): #top
            A[i_eqn, var_idx[0, y]] = 1
            diff = img_src_c[x_min, y_min+y] - img_src_c[x_min-1, y_min+y]
            b[i_eqn] = img_tar_c[pos_x-1, pos_y+y] + diff
            i_eqn += 1
        for y in range(w_var): #bottom
            A[i_eqn, var_idx[-1, y]] = 1
            diff = img_src_c[x_max-1, y_min+y] - img_src_c[x_max, y_min+y]
            b[i_eqn] = img_tar_c[pos_x+h_var, pos_y+y] + diff 
            i_eqn += 1

        var = scipy.sparse.linalg.lsqr(A, b)[0]
        var = var.reshape(h_var, w_var)
        img_tar_c[pos_x:pos_x+h_var, pos_y:pos_y+w_var] = var
        img_blended.append(img_tar_c)
    img_blended = np.stack(img_blended, axis=-1)
    return img_blended

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-pano')
    parser.add_argument('-mask')
    parser.add_argument('-out')
    args = parser.parse_args()




if __name__ == '__main__':
    main()