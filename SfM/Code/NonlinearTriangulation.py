import numpy as np
import scipy.optimize as opt


def nl_reprojection_error(X_tilde, point1, point2, P1, P2):
    p1_1T = P1[0, :]
    p2_1T = P1[1, :]
    p3_1T = P1[2, :]

    p1_2T = P2[0, :]
    p2_2T = P2[1, :]
    p3_2T = P2[2, :]
    
    u1, v1 = point1[0], point1[1]
    reprojected_u1 = np.dot(p1_1T, X_tilde) / np.dot(p3_1T, X_tilde)
    reprojected_v1 = np.dot(p2_1T, X_tilde) / np.dot(p3_1T, X_tilde)
    u1_error = u1 - reprojected_u1
    v1_error = v1 - reprojected_v1
    point1_error = u1_error**2 + v1_error**2
    
    u2, v2 = point2[0], point2[1]
    reprojected_u2 = np.dot(p1_2T, X_tilde) / np.dot(p3_2T, X_tilde)
    reprojected_v2 = np.dot(p2_2T, X_tilde) / np.dot(p3_2T, X_tilde)
    u2_error = u2 - reprojected_u2
    v2_error = v2 - reprojected_v2
    point2_error = u2_error**2 + v2_error**2
    
    total_error = point1_error + point2_error
    
    return total_error

def non_linear_triangulation(K_mtrx, point_set1, point_set2, X_world, R_init, C_init, R_sel, C_sel):
    
    I = np.identity(3)
    IC1 = np.column_stack((I, -C_init))
    P1 = np.dot(K_mtrx, np.dot(R_init, IC1))
    
    IC2 = np.column_stack((I, -C_sel))
    P2 = np.dot(K_mtrx, np.dot(R_sel, IC2))
    
    X_world_opt = list()
    for i in range(len(X_world)):
        # print(i)
        pt1 = point_set1[i]
        pt2 = point_set2[i]
        params = opt.least_squares(fun=nl_reprojection_error, x0=X_world[i], method='trf', args=[pt1, pt2, P1, P2])
        Xi_opt = params.x
        X_world_opt.append(Xi_opt)
    
    return np.array(X_world_opt)