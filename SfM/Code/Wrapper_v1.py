import numpy as np
import cv2
import matplotlib.pyplot as plt

import Utils.ImageUtils as imutils
from GetInliersRANSAC import *
from EssentialMatrixFromFundamentalMatrix import *
from ExtractCameraPose import *
from LinearTriangulation import *
from DisambiguateCameraPose import *
from NonlinearTriangulation import *

data_path = "Data/"
file_name = "matching1.txt"
K = np.array([[568.996140852, 0, 643.21055941], [0, 568.988362396, 477.982801038], [0, 0, 1]]).reshape(3, 3)

image_set = imutils.get_ImageSet(data_path)
img1, img2 = image_set[0], image_set[1]

all_im1_matches = imutils.get_KeyPointMatches(data_path, file_name)
matches_img1 = [(pnt[0], pnt[1]) for pnt in all_im1_matches[np.where(all_im1_matches[:, 1] == 2)][:, 2]]
matches_img2 = [(pnt[0], pnt[1]) for pnt in all_im1_matches[np.where(all_im1_matches[:, 1] == 2)][:, 3]]

F, _, best_matches_img1, best_matches_img2 = get_inliers(matches_img1, matches_img2, n_iterations=500, error_thresh=0.005)
print(F)    
E = get_essential_matrix(F, K)

R_set, C_set = extract_camera_pose(E)
C_std = np.zeros((3, 1))
R_std = np.identity(3)

pt_set1 = best_matches_img1
pt_set2 = best_matches_img2
X_all_poses = list()
for i in range(len(C_set)):
    
    X = linear_triangulation(K, C_std, R_std, C_set[i], R_set[i], pt_set1, pt_set2)
    X = X / X[:, 3].reshape(-1,1)
    X_all_poses.append(X)

R_correct, C_correct, X_correct = disambiguate_cam_pose(R_set, C_set, X_all_poses)
X_correct = X_correct / X_correct[:, 3].reshape(-1, 1)
X_nl = non_linear_triangulation(X_correct, pt_set1, pt_set2, K, R_std, C_std, R_correct, C_correct)
X_nl = X_nl / X_nl[:, 3].reshape(-1, 1)

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


