import numpy as np
from EstimateFundamentalMatrix import *


def get_F_error(pt1, pt2, F_matrix):
    x1 = np.array([pt1[0], pt1[1], 1])
    x2 = np.array([pt2[0], pt2[1], 1])
    error_F = np.dot(x2.T, np.dot(F_matrix, x1))
    return abs(error_F)


def get_inliers(point_set1, point_set2, n_iterations=500, error_thresh=0.005):
    best_indices = list()
    best_F_matrix = None
    for n in range(n_iterations):
        point_choices = np.random.choice(len(point_set1), size=8)
        img1_choices = np.array([point_set1[choice] for choice in point_choices])
        img2_choices = np.array([point_set2[choice] for choice in point_choices])
        F_matrix = estimate_fundamental_matrix(img1_choices, img2_choices)
        
        good_indices = list()
        for j in range(len(point_set1)):
            F_error = get_F_error(point_set1[j], point_set2[j], F_matrix)
            if F_error < error_thresh:
                good_indices.append(j)
        
        if len(best_indices) < len(good_indices):
            best_indices = good_indices
            best_F_matrix = F_matrix
            
    best_points1 = np.array([point_set1[idx] for idx in best_indices])
    best_points2 = np.array([point_set2[idx] for idx in best_indices])
    
    return best_F_matrix, best_indices, best_points1, best_points2


def get_inliers(point_set1, point_set2, index_bin_mtrx, n_iterations=500, error_thresh=0.005):
    best_indices = list()
    best_F_matrix = None
    for n in range(n_iterations):
        point_choices = np.random.choice(len(point_set1), size=8)
        img1_choices = np.array([point_set1[choice] for choice in point_choices])
        img2_choices = np.array([point_set2[choice] for choice in point_choices])
        F_matrix = estimate_fundamental_matrix(img1_choices, img2_choices)
        
        good_indices = list()
        for j in range(len(point_set1)):
            F_error = get_F_error(point_set1[j], point_set2[j], F_matrix)
            if F_error < error_thresh:
                good_indices.append(index_bin_mtrx[j])
        
        if len(best_indices) < len(good_indices):
            best_indices = good_indices
            best_F_matrix = F_matrix
            
    # best_points1 = np.array([point_set1[idx] for idx in best_indices])
    # best_points2 = np.array([point_set2[idx] for idx in best_indices])
    
    return best_F_matrix, best_indices