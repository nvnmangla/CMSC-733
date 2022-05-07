import numpy as np
import cv2
import os
import argparse
import matplotlib.pyplot as plt


# keyPointPath = "Data/"
# file_name = "matching1.txt"
# # for file_name in os.listdir(keyPointPath):
# # if "matching" in file_name:
# primary_image_num = int(file_name.split("matching")[-1].split(".txt")[0])
# with open(os.path.join(keyPointPath, file_name)) as raw_data:
#     lines = raw_data.readlines()[1:]
#     all_coords = list()
#     for line in lines:
#         elems = line.split()
#         num_matches = int(elems[0])
#         rgb_vals = (int(elems[1]), int(elems[2]), int(elems[3]))
#         uv_coords_primary = (float(elems[4]), float(elems[5]))
#         # coords_list = list()
#         a = 1
#         while num_matches > 1:
#             img_num = int(elems[5+a])
#             uv_coords = (float(elems[6+a]), float(elems[7+a]))

#             all_coords.append(np.array([primary_image_num, img_num, uv_coords_primary, uv_coords], dtype=object))
#             a += 3
#             num_matches -= 1
    
#         # all_coords.append(np.array(coords_list, dtype=object))
#     all_coords = np.array(all_coords, dtype=object)

# print(len(all_coords[np.where(all_coords[:, 1] == 2)]))


# keyPointPath = "Data/"
# file_name = "matching1.txt"
# im1_matches = get_KeyPointMaches(keyPointPath, file_name)
# print(im1_matches)

# image_path = "Data/"
# image_list = list()
# for file_name in sorted(os.listdir(image_path)):
#     if ".jpg" in file_name:
#         # print(file_name)
#         image = cv2.imread(os.path.join(image_path, file_name))
#         image_list.append(image)

# cv2.imshow("concat", horizontal_concat)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# n_iterations = 500
# error_thresh = 0.002
# n_best_indices = 0
# best_indices = list()
# for n in range(n_iterations):
#     point_choices = np.random.choice(len(kpts1), size=8)
#     img1_choices = np.array([kpts1[choice] for choice in point_choices])
#     img2_choices = np.array([kpts2[choice] for choice in point_choices])
#     F_matrix = get_FundamentalMatrix(img1_choices, img2_choices)
    
#     good_indices = list()
#     for j in range(len(kpts1)):
#         F_error = get_F_error(kpts1[j], kpts2[j], F_matrix)
#         if F_error < error_thresh:
#             good_indices.append(j)
    
#     if n_best_indices < len(good_indices):
#         n_best_indices = len(good_indices)
#         best_indices = good_indices
         
# best_kpts1 = [kpts1[idx] for idx in best_indices]
# best_kpts2 = [kpts2[idx] for idx in best_indices]

# for i in range(len(best_kpts1)):
#     horizontal_concat = cv2.line(horizontal_concat, best_kpts1[i], plot_kpts2[i], (0, 255, 0), 1)
# cv2.imshow("concat", horizontal_concat)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# E_matrix = np.dot(K.T, np.dot(F, K))
# U, S, V_T = np.linalg.svd(E_matrix)
# S = np.diag(S)
# S[2, 2] = 0
# E_matrix = np.dot(np.dot(U, S), V_T)

# U, S, V_T = np.linalg.svd(E)
# W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

# translation_matrices = list()
# rotation_matrices = list()
# C1 = U[:, 2]
# R1 = np.dot(U, np.dot(W, V_T))
# translation_matrices.append(C1)
# rotation_matrices.append(R1)
# C2 = -U[:, 2]
# R2 = np.dot(U, np.dot(W, V_T))
# translation_matrices.append(C2)
# rotation_matrices.append(R2)
# C3 = U[:, 2]
# R3 = np.dot(U, np.dot(W.T, V_T))
# translation_matrices.append(C3)
# rotation_matrices.append(R3)
# C4 = -U[:, 2]
# R4 = np.dot(U, np.dot(W.T, V_T))
# translation_matrices.append(C4)
# rotation_matrices.append(R4)

# for i in range(len(rotation_matrices)):
#     if np.linalg.det(rotation_matrices[i]) < 0:
#         rotation_matrices[i] = -rotation_matrices[i]
#         translation_matrices[i] = -translation_matrices[i]

# IC_std = np.column_stack((I, C_std))
# P1 = np.dot(K, np.dot(R_std, IC_std))

# IC = np.column_stack((I, -C[0]))
# P2 = np.dot(K, np.dot(R[0], IC))

# p1_T = P1[0, :]
# p2_T = P1[1, :]
# p3_T = P1[2, :]

# pp1_T = P2[0, :]
# pp2_T = P2[1, :]
# pp3_T = P2[2, :]

# all_X = list()
# for i in range(len(best_kpts1)):
#     x = best_kpts1[i][0]
#     y = best_kpts1[i][1]
    
#     x_prime = best_kpts2[i][0]
#     y_prime = best_kpts2[i][1]
    
#     A = list()
#     A.append(y*p3_T - p2_T)
#     A.append(p1_T - x*p3_T)
#     A.append(y_prime*pp3_T - pp2_T)
#     A.append(pp1_T - x_prime*pp3_T)
#     A = np.array(A)
    
#     _, _, V_T = np.linalg.svd(A)
#     V = V_T.T
#     x = V[:, -1]
#     all_X.append(x)
    
def get_KeyPointMatches(matches_file_path, file_name):
    primary_image_num = int(file_name.split("matching")[-1].split(".txt")[0])
    with open(os.path.join(matches_file_path, file_name)) as raw_data:
        lines = raw_data.readlines()[1:]
        all_coords = list()
        for line in lines:
            elems = line.split()
            num_matches = int(elems[0])
            rgb_vals = (int(elems[1]), int(elems[2]), int(elems[3]))
            uv_coords_primary = (float(elems[4]), float(elems[5]))
            a = 1
            while num_matches > 1:
                img_num = int(elems[5+a])
                uv_coords = (float(elems[6+a]), float(elems[7+a]))
                all_coords.append(np.array([primary_image_num, img_num, uv_coords_primary, uv_coords], dtype=object))
                a += 3
                num_matches -= 1
        all_coords = np.array(all_coords, dtype=object)
    return all_coords


def get_ImageSet(image_path):
    image_list = list()
    for file_name in sorted(os.listdir(image_path)):
        if ".jpg" in file_name:
            image = cv2.imread(os.path.join(image_path, file_name))
            image_list.append(image)
    return image_list


def get_plot_points(keypoints2, image1_size):
    height, width = image1_size[0], image1_size[1]
    new_keypoints2 = list()
    for point in keypoints2:
        new_width2 = point[0] + width
        new_height2 = point[1]
        new_keypoints2.append((int(new_width2), (int(new_height2))))
    return new_keypoints2


def get_normalization_transform(points):
    """returns the normalization transform pertaining to a set of points

    Args:
        points (array): list of points

    Returns:
        array: Normalization tranform T
    """
    points_x = points[:, 0]
    points_y = points[:, 1]
    
    centroid_x = np.mean(points_x)
    centroid_y = np.mean(points_y)
    
    mean_distances_x = points_x - centroid_x
    mean_distances_y = points_y - centroid_y
    mean_distance = np.mean(mean_distances_x**2 + mean_distances_y**2)
    scale = np.sqrt(2/mean_distance)
    '''
    T = [scale   0   -scale*c(1)
         0     scale -scale*c(2)
         0       0      1      ];
    '''
    T = np.eye(3)
    T[0][0] = scale
    T[1][1] = scale
    T[0][2] = -scale*centroid_x
    T[1][2] = -scale*centroid_y
    
    return T


def get_FundamentalMatrix(pts_img1, pts_img2):
    """Computing the Fundamental MAtrix

    Args:
        pts_img1 (list): Points belonging to image 1
        pts_img2 (list): Points belonging to image 1

    Returns:
        array: Fundamental Matrix
    """
    pts_img1 = np.array(pts_img1)
    pts_img2 = np.array(pts_img2)
    
    # print(pts_img1[:, 1])
    T1 = get_normalization_transform(pts_img1)
    T2 = get_normalization_transform(pts_img2)
    
    pts_img1 = np.column_stack((pts_img1, np.array([1 for i in range(len(pts_img1))])))
    pts_img2 = np.column_stack((pts_img2, np.array([1 for i in range(len(pts_img2))])))
    
    normpts_img1 = np.dot(T1, pts_img1.T).T
    normpts_img2 = np.dot(T2, pts_img2.T).T
    
    A_matrix = list()
    for i in range(len(normpts_img1)):
        u1, v1 = normpts_img1[i][0], normpts_img1[i][1]
        u2, v2 = normpts_img2[i][0], normpts_img2[i][1]
        row = [u2*u1, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1, 1]
        A_matrix.append(row)
    
    A_matrix = np.array(A_matrix)
    _, _, V_T = np.linalg.svd(A_matrix)
    F_matrix = V_T.T[:, -1].reshape((3, 3)).copy()
    
    U, S, V_T = np.linalg.svd(F_matrix)
    S = np.diag(S)
    S[2, 2] = 0
    F_matrix = np.dot(np.dot(U, S), V_T)
    F_matrix = np.dot(np.dot(T2.T, F_matrix), T1)
    F_matrix = F_matrix / F_matrix[2, 2]
    
    return F_matrix


def get_F_error(pt1, pt2, F_matrix):
    x1 = np.array([pt1[0], pt1[1], 1])
    x2 = np.array([pt2[0], pt2[1], 1])
    error_F = np.dot(x2.T, np.dot(F_matrix, x1))
    return abs(error_F)
    

def get_inliers(point_set1, point_set2, num_iterations=500, error_thresh=0.005):
    n_best_indices = 0
    best_indices = list()
    best_F_matrix = None    
    for n in range(num_iterations):
        point_choices = np.random.choice(len(point_set1), size=8)
        img1_choices = np.array([point_set1[choice] for choice in point_choices])
        img2_choices = np.array([point_set2[choice] for choice in point_choices])
        F_matrix = get_FundamentalMatrix(img1_choices, img2_choices)
        
        good_indices = list()
        for j in range(len(point_set1)):
            F_error = get_F_error(point_set1[j], point_set2[j], F_matrix)
            if F_error < error_thresh:
                good_indices.append(j)
        
        if n_best_indices < len(good_indices):
            n_best_indices = len(good_indices)
            best_indices = good_indices
            best_F_matrix = F_matrix
            
    best_points1 = [point_set1[idx] for idx in best_indices]
    best_points2 = [point_set2[idx] for idx in best_indices]
    
    return best_F_matrix, best_indices, best_points1, best_points2


def plot_matches(image1, image2, point_set1, point_set2, best_indices):
    horizontal_concat = np.concatenate((image1, image2), axis=1)
    for i in range(len(point_set1)):
        if i in best_indices:
            horizontal_concat = cv2.line(horizontal_concat, point_set1[i], point_set2[i], (0, 255, 0), 1)
        else:
            horizontal_concat = cv2.line(horizontal_concat, point_set1[i], point_set2[i], (0, 0, 255), 1)
    cv2.imshow("concat", horizontal_concat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

def get_EssentialMatrix(F_matrix, K_matrix):
    E_matrix = np.dot(K_matrix.T, np.dot(F_matrix, K_matrix))
    U, S, V_T = np.linalg.svd(E_matrix)
    S = np.diag(S)
    S[2, 2] = 0
    E_matrix = np.dot(U, np.dot(S, V_T))
    return E_matrix


################################################
##############Extracting Camera Pose############
################################################
def extract_camera_poses(E_matrix):
    U, S, V_T = np.linalg.svd(E_matrix)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    translation_matrices = list()
    rotation_matrices = list()
    C1 = U[:, 2]
    R1 = np.dot(U, np.dot(W, V_T))
    translation_matrices.append(C1)
    rotation_matrices.append(R1)
    C2 = -U[:, 2]
    R2 = np.dot(U, np.dot(W, V_T))
    translation_matrices.append(C2)
    rotation_matrices.append(R2)
    C3 = U[:, 2]
    R3 = np.dot(U, np.dot(W.T, V_T))
    translation_matrices.append(C3)
    rotation_matrices.append(R3)
    C4 = -U[:, 2]
    R4 = np.dot(U, np.dot(W.T, V_T))
    translation_matrices.append(C4)
    rotation_matrices.append(R4)
    
    for i in range(len(rotation_matrices)):
        print(np.linalg.det(rotation_matrices[i]))
        if np.linalg.det(rotation_matrices[i]) < 0:
            rotation_matrices[i] = -rotation_matrices[i]
            translation_matrices[i] = -translation_matrices[i]
    
    return translation_matrices, rotation_matrices


def linear_triangulation(K, C1, R1, C2, R2, point_set1, point_set2):
    I = np.identity(3)
    IC1 = np.column_stack((I, C1))
    P1 = np.dot(K, np.dot(R1, IC1))
    
    IC2 = np.column_stack((I, C2))
    P2 = np.dot(K, np.dot(R2, IC2))
    
    p1_T = P1[0, :]
    p2_T = P1[1, :]
    p3_T = P1[2, :]

    pp1_T = P2[0, :]
    pp2_T = P2[1, :]
    pp3_T = P2[2, :]

    X_list = list()
    for i in range(len(point_set1)):
        x = point_set1[i][0]
        y = point_set1[i][1]
        
        x_prime = point_set2[i][0]
        y_prime = point_set2[i][1]
        
        A = list()
        A.append(y*p3_T - p2_T)
        A.append(p1_T - x*p3_T)
        A.append(y_prime*pp3_T - pp2_T)
        A.append(pp1_T - x_prime*pp3_T)
        A = np.array(A)
        
        _, _, V_T = np.linalg.svd(A)
        V = V_T.T
        X = V[:, -1] / V[-1, -1]
        X_list.append(X)

    return np.array(X_list)


data_path = "Data/"
file_name = "matching1.txt"
K = np.array([[568.996140852, 0, 643.21055941], [0, 568.988362396, 477.982801038], [0, 0, 1]])

image_set = get_ImageSet(data_path)

img1, img2 = image_set[0], image_set[1]
horizontal_concat = np.concatenate((img1, img2), axis=1)

all_im1_matches = get_KeyPointMatches(data_path, file_name)
kpts1 = [(int(pnt[0]),int(pnt[1])) for pnt in all_im1_matches[np.where(all_im1_matches[:, 1] == 2)][:, 2]]
kpts2 = [(int(pnt[0]),int(pnt[1])) for pnt in all_im1_matches[np.where(all_im1_matches[:, 1] == 2)][:, 3]]    
F_mtrx = get_FundamentalMatrix(kpts1, kpts2)
print(F_mtrx)
# F, best_indices, best_kpts1, best_kpts2 = get_inliers(kpts1, kpts2, num_iterations=2000, error_thresh=0.005)
# # print(F)
# # plot_kpts2 = get_plot_points(kpts2, (img1.shape[0], img1.shape[1]))
# # plot_matches(img1, img2, kpts1, plot_kpts2, best_indices)

# E = get_EssentialMatrix(F, K)

# C_set, R_set = extract_camera_poses(E)
# # print(C, R)

# C_std = np.zeros((3, 1))
# R_std = np.identity(3)

# X_all_poses = list()
# for i in range(len(C_set)):
#     pt_set1 = best_kpts1
#     pt_set2 = best_kpts2
#     X = linear_triangulation(K, C_std, R_std, C_set[i], R_set[i], pt_set1, pt_set2)
#     X_all_poses.append(X)

# # print(X_all_poses)

# fig =  plt.figure(figsize = (15, 15))
# colors = ['red', 'lawngreen', 'blue', 'yellow']
# for i in range(len(X_all_poses)):
#     X_curr = X_all_poses[i]
#     x_curr, z_curr = X_curr[:, 0], X_curr[:, 2]
#     # print(x_curr, z_curr)
#     plt.scatter(x_curr, z_curr, marker='.',linewidths = 0.5, color=colors[i])
# plt.show()