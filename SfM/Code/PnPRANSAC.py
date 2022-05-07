from re import T
import numpy as np
from LinearPnP import PnP


def reproject_error(X_i, x_i, K_matrix, R, C):
    I = np.identity(3)
    IC = np.column_stack((I, -C))
    P = np.dot(K_matrix, np.dot(R, IC))
    p1_T, p2_T, p3_T = P
    
    Xi_hmg = np.hstack((X_i, 1))
    u, v = x_i
    
    u_proj = np.dot(p1_T, Xi_hmg) / np.dot(p3_T, Xi_hmg)
    v_proj = np.dot(p2_T, Xi_hmg) / np.dot(p3_T, Xi_hmg)
    
    e = (u-u_proj)**2 + (v-v_proj)**2
    
    return e


def apply_PnP_RANSAC(X_3d, x_, K_matrix, n_iterations=500, error_thresh=5):
    R_best = None
    t_best = None
    best_indices = list()
    
    for n in range(n_iterations):
        idxs = np.random.choice(len(X_3d), size=6)
        X_s = X_3d[idxs]
        x_s = x_[idxs]
        R, C = PnP(X_s, x_s, K_matrix)
        
        good_indices = list()
        if R is not None:
            for i in range(len(X_3d)):
                X_i = X_3d[i]
                x_i = x_[i]
                pnp_error = reproject_error(X_i, x_i, K_matrix, R, C)
                if pnp_error < error_thresh:
                    good_indices.append(i)
                    
        if len(best_indices) < len(good_indices):
            best_indices = good_indices
            R_best = R
            t_best = C
            
    return R_best, t_best