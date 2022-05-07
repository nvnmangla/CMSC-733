import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

import Utils.ImageUtils as imutils
import Utils.MiscUtils as miscutils
from GetInliersRANSAC import *
from EssentialMatrixFromFundamentalMatrix import *
from ExtractCameraPose import *
from LinearTriangulation import *
from DisambiguateCameraPose import *
from NonlinearTriangulation import *
from PnPRANSAC import *
from LinearPnP import *
from NonlinearPnP import *
from BundleAdjustment import *


def main():
    parser = argparse.ArgumentParser()

    ################### Change input path here ############################################
    parser.add_argument('--InputPath', default="Data/", help='path to the matches and image files')
    # parser.add_argument('--SavePath', default="OutputFiles/", help='Save files here')
    parser.add_argument('--BA', default= True, type = lambda x: bool(int(x)), help='Do bundle adjustment or not')
    
    args = parser.parse_args()
    data_path = args.InputPath
    BA = bool(int(args.BA))
     
    
    K = np.array([[568.996140852, 0, 643.21055941], [0, 568.988362396, 477.982801038], [0, 0, 1]]).reshape(3, 3)

    image_set = imutils.get_ImageSet(data_path)
    img1, img2 = image_set[0], image_set[1]

    all_matches, match_bool_matrix, color_data  = imutils.organize_all_matches(data_path)

    filtered_feature_flag = np.zeros_like(match_bool_matrix)
    f_matrix = np.empty(shape=(6, 6), dtype=object)

    for i in range(0, 6):
        for j in range(i + 1, 6):
            idx = np.where(match_bool_matrix[:, i] & match_bool_matrix[:, j])
            pts1 = all_matches[idx, i].reshape(-1)
            pts2 = all_matches[idx, j].reshape(-1)
            print("Number of matches between images", i+1, "and", j+1, ":", len(pts1))
            idx = np.array(idx).reshape(-1)
            if len(idx) > 8:
                f, best_idx = get_inliers(pts1, pts2, idx, n_iterations=500, error_thresh=0.005)          
                f_matrix[i, j] = f
                filtered_feature_flag[best_idx, j] = 1
                filtered_feature_flag[best_idx, i] = 1

    m, n = 0, 1
    F = f_matrix[m, n]
    E = get_essential_matrix(F, K)

    R_set, C_set = extract_camera_pose(E)
    C_std = np.zeros((3, 1))
    R_std = np.identity(3)

    idx = np.where(filtered_feature_flag[:,m] & filtered_feature_flag[:,n])
    pt_set1 = all_matches[idx, m].reshape(-1)
    pt_set2 = all_matches[idx, n].reshape(-1)

    X_all_poses = list()
    for i in range(len(C_set)):
        X = linear_triangulation(K, C_std, R_std, C_set[i], R_set[i], pt_set1, pt_set2)
        X = X / X[:, 3].reshape(-1,1)
        X_all_poses.append(X)

    R_correct, C_correct, X_correct = disambiguate_cam_pose(R_set, C_set, X_all_poses)
    X_correct = X_correct / X_correct[:, 3].reshape(-1, 1)
    X_nl = non_linear_triangulation(K, pt_set1, pt_set2, X_correct, R_std, C_std, R_correct, C_correct)
    X_nl = X_nl / X_nl[:, 3].reshape(-1, 1)
    
    print("Mean error after Linear Triangulation:", miscutils.mean_reprojection_error(X_correct, pt_set1, pt_set2, R_std, C_std, R_correct, C_correct, K),\
          "and after Non-linear Triangulation:", miscutils.mean_reprojection_error(X_nl, pt_set1, pt_set2, R_std, C_std, R_correct, C_correct, K))
    

    # fig, axs = plt.subplots(nrows=1, ncols=2)
    # for i in range(len(X_all_poses)):
    #     curr_pose = X_all_poses[i]
    #     axs[0].scatter(curr_pose[:, 0], curr_pose[:, 2], s=5)
    # axs[1].scatter(X_correct[:, 0], X_correct[:, 2], s=5, label="Linear")
    # axs[1].scatter(X_nl[:, 0], X_nl[:, 2], s=5, label="Non-linear")
    # axs[1].legend(loc="upper right")
    # axs[0].set_xlim(-100, 100)
    # axs[0].set_ylim(-100, 100)
    # axs[1].set_xlim(-10, 20)
    # axs[1].set_ylim(0, 20)
    # plt.show()


    ##################################### Registering Cameras 1 and 2 ###################################################
    X_3D = np.zeros((all_matches.shape[0], 3))
    X_3D_bool = np.zeros((all_matches.shape[0], 1), dtype = int)

    X_3D[idx] = X_correct[:, :3]
    X_3D_bool[idx] = 1

    X_3D_bool[np.where(X_3D[:,2] < 0)] = 0

    C_Set = list()
    R_Set = list()

    C0 = np.zeros(3)
    R0 = np.eye(3)
    C_Set.append(C0)
    R_Set.append(R0)

    C_Set.append(C_correct)
    R_Set.append(R_correct)

    ###################################### Registering the other cameras ###################################################
    for i in range(2, 6):
        print("Registering")
        camera_psn_idx = np.where(X_3D_bool[:, 0] & filtered_feature_flag[:, i])[0]
        if len(camera_psn_idx) < 8:
            continue
        
        pt_set_i = all_matches[camera_psn_idx, i].reshape(-1)
        X_set_i = X_3D[camera_psn_idx, :].reshape(-1, 3)
        R_init, C_init = apply_PnP_RANSAC(X_set_i, pt_set_i, K, n_iterations = 1000, error_thresh = 5)
        # print(R_init, C_init)
        error_linear_PnP = reproj_error_PnP(X_set_i, pt_set_i, K, R_init, C_init)
        
        Ri, Ci = non_linear_PnP(X_set_i, pt_set_i, K, R_init, C_init)
        # print(Ri, Ci)
        error_non_linear_PnP = reproj_error_PnP(X_set_i, pt_set_i, K, Ri, Ci)
        print("")
        
        C_Set.append(Ci)
        R_Set.append(Ri)
        
        for j in range(0, i):
            print("Images:", i, j)
            idx_X_pts = np.where(filtered_feature_flag[:, j] & filtered_feature_flag[:, i])
            if (len(idx_X_pts[0]) < 8):
                continue
            
            pt_set1 = all_matches[idx_X_pts, j].reshape(-1)
            pt_set2 = all_matches[idx_X_pts, i].reshape(-1)
            
            X = linear_triangulation(K, C_Set[j], R_Set[j], Ci, Ri, pt_set1, pt_set2)
            X = X/X[:,3].reshape(-1,1)
            X = non_linear_triangulation(K, pt_set1, pt_set2, X, R_Set[j], C_Set[j], Ri, Ci)
            X = X/X[:,3].reshape(-1,1)
            
            X_3D[idx_X_pts] = X[:, :3]
            X_3D_bool[idx_X_pts] = 1
        
        if BA:
            R_Set, C_Set, X_3D = bundle_adjust(X_3D, X_3D_bool, all_matches, filtered_feature_flag, R_Set, C_Set, K, n_cameras=i)
            for k in range(0, i+1):
                idx_X_pts = np.where(X_3D_bool[:,0] & filtered_feature_flag[:, k])
                x = np.hstack((all_matches[idx_X_pts, k].reshape((-1, 1))))
                X = X_3D[idx_X_pts]
        
    X_3D_bool[X_3D[:,2] < 0] = 0

    feature_idx = np.where(X_3D_bool[:, 0])
    X = X_3D[feature_idx]
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]

    fig = plt.figure(figsize = (10, 10))
    plt.xlim(-250, 250)
    plt.ylim(0, 500)
    plt.scatter(x, z, marker='.', linewidths=0.5, color = 'lawngreen')
    for i in range(len(R_Set)):
        R1 = get_euler_vec(R_Set[i])
        R1 = np.rad2deg(R1)
        plt.plot(C_Set[i][0],C_Set[i][2], marker=(3, 0, int(R1[1])), markersize=15, linestyle='None')

    plt.savefig("bundles.png")
    plt.show()