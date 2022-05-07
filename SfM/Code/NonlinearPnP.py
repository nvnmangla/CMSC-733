import numpy as np
import scipy.optimize as opt
from Utils.TransformUtils import *


def non_linear_PnP(X_3d, point_set, K_matrix, R0, C0):
    pts = list()
    for i in range(len(point_set)):
        pts.append([point_set[i][0], point_set[i][1]])
    pts = np.array(pts)
    q = get_quaternion(R0)
    X0 = [q[0], q[1], q[2], q[3], C0[0], C0[1], C0[2]] 
    # print(X0)
    params = opt.least_squares(fun = nl_reprojection_error, x0=X0, method="trf", args=[X_3d, pts, K_matrix])
    X1 = params.x
    Q = X1[:4]
    C = X1[4:]
    R = get_rotation(Q)
    
    return R, C


def nl_reprojection_error(X0, X_3d, point_set, K_matrix):
    # print("X0 vector: ", X0)
    Q, C = X0[:4], X0[4:]
    # print("Q vals: ", Q)
    R = get_rotation(Q)
    
    I = np.identity(3)
    IC = np.column_stack((I, -C))
    P = np.dot(K_matrix, np.dot(R, IC))
    p1_T, p2_T, p3_T = P
    
    error_list = list()
    for i in range(len(point_set)):        
        X3d_i = np.hstack((X_3d[i], 1))
        u, v = point_set[i][0], point_set[i][1]
        
        u_proj = np.dot(p1_T, X3d_i) / np.dot(p3_T, X3d_i)
        v_proj = np.dot(p2_T, X3d_i) / np.dot(p3_T, X3d_i)

        e = (v-v_proj)**2 + (u - u_proj)**2
        error_list.append(e)
        
    mean_error = np.mean(error_list)
    
    return mean_error