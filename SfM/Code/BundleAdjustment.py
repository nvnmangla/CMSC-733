import numpy as np
import cv2
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
import time
from BuildVisibilityMatrix import *
from Utils.TransformUtils import *


def get_image_points(X_index, visiblity_matrix, img_pt_set):
    visible_features = img_pt_set[X_index]
    h, w = visiblity_matrix.shape
    image_points = []
    for i in range(h):
        for j in range(w):
            if visiblity_matrix[i,j] == 1:
                pt = visible_features[i, j]
                image_points.append(pt)
    return np.array(image_points).reshape(-1, 2)


def get_cam_point_indices(visiblity_matrix):
    camera_indices = []
    point_indices = []
    h, w = visiblity_matrix.shape
    for i in range(h):
        for j in range(w):
            if visiblity_matrix[i,j] == 1:
                camera_indices.append(j)
                point_indices.append(i)

    return np.array(camera_indices).reshape(-1), np.array(point_indices).reshape(-1)


def bundle_adjustment_sparsity(X_3D_bool, filtered_feature_flag, n_cameras):
    number_of_cam = n_cameras + 1
    X_index, visiblity_matrix = get_visibility_matrix(X_3D_bool.reshape(-1), filtered_feature_flag, n_cameras)
    n_observations = np.sum(visiblity_matrix)
    n_points = len(X_index[0])

    m = n_observations * 2
    n = number_of_cam * 6 + n_points * 3
    A = lil_matrix((m, n), dtype=int)
    # print(m, n)
    i = np.arange(n_observations)
    camera_indices, point_indices = get_cam_point_indices(visiblity_matrix)
    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1
    for s in range(3):
        A[2*i, (n_cameras)*6 + point_indices*3 + s] = 1
        A[2*i + 1, (n_cameras)*6 + point_indices*3 + s] = 1

    return A


def project_point(R, C, pt_3D, K_matrix):
    P2 = np.dot(K_matrix, np.dot(R, np.hstack((np.identity(3), -C.reshape(3,1)))))
    x3D_4 = np.hstack((pt_3D, 1))
    x_proj = np.dot(P2, x3D_4.T)
    x_proj /= x_proj[-1]
    return x_proj

    
def get_projection(points_3d, camera_params, K_matrix):
    x_proj = list()
    for i in range(len(camera_params)):
        R = get_rotation(camera_params[i, :3], 'e')
        C = camera_params[i, 3:].reshape(3,1)
        pt3D = points_3d[i]
        pt_proj = project_point(R, C, pt3D, K_matrix)[:2]
        x_proj.append(pt_proj)    
    return np.array(x_proj)

############################ Adapted from https://github.com/sakshikakde/SFM ########################################

def fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d, K):
    cam_num = n_cameras + 1
    camera_params = x0[:cam_num * 6].reshape((cam_num, 6))
    points_3d = x0[cam_num*6:].reshape((n_points, 3))
    points_proj = get_projection(points_3d[point_indices], camera_params[camera_indices], K)
    error_vec = (points_proj - points_2d).ravel()
    return error_vec


def bundle_adjust(X_all, X_found, img_pt_set, filtered_feature_flag, R_set_, C_set_, K_matrix, n_cameras):
    
    X_index, visiblity_matrix = get_visibility_matrix(X_found, filtered_feature_flag, n_cameras)
    points_3d = X_all[X_index]
    points_2d = get_image_points(X_index, visiblity_matrix, img_pt_set)

    RC_list = []
    for i in range(n_cameras+1):
        C, R = C_set_[i], R_set_[i]
        Q = get_euler_vec(R)
        RC = [Q[0], Q[1], Q[2], C[0], C[1], C[2]]
        RC_list.append(RC)
    RC_list = np.array(RC_list).reshape(-1, 6)

    x0 = np.hstack((RC_list.ravel(), points_3d.ravel()))
    n_points = points_3d.shape[0]

    camera_indices, point_indices = get_cam_point_indices(visiblity_matrix)
    
    A = bundle_adjustment_sparsity(X_found, filtered_feature_flag, n_cameras)
    t0 = time.time()
    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-10, method='trf',
                        args=(n_cameras, n_points, camera_indices, point_indices, points_2d, K_matrix))
    t1 = time.time()
    print('time to run BA :', t1-t0, 's \nA matrix shape: ' ,  A.shape, '\n############')
    
    x1 = res.x
    number_of_cam = n_cameras + 1
    optimized_camera_params = x1[:number_of_cam * 6].reshape((number_of_cam, 6))
    optimized_points_3d = x1[number_of_cam * 6:].reshape((n_points, 3))

    optimized_X_all = np.zeros_like(X_all)
    optimized_X_all[X_index] = optimized_points_3d

    optimized_C_set, optimized_R_set = [], []
    for i in range(len(optimized_camera_params)):
        R = get_rotation(optimized_camera_params[i, :3], 'e')
        C = optimized_camera_params[i, 3:].reshape(3,1)
        optimized_C_set.append(C)
        optimized_R_set.append(R)
    
    return optimized_R_set, optimized_C_set, optimized_X_all