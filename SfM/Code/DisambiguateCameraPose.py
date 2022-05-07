import numpy as np



def disambiguate_cam_pose(R_list, C_list, X_collection):
    num_good_points = 0
    best_idx = None
    for i in range(len(R_list)):
        R, C = R_list[i], C_list[i]
        r3 = R[2, :]
        X_set = X_collection[i]
        n_good_vals = 0
        for X_i in X_set:
            X_i = X_i / X_i[3]
            X_i = X_i[0:3]
            if r3.dot(X_i-C) > 0 and X_i[2] > 0:
                n_good_vals += 1
        
        if num_good_points < n_good_vals:
            num_good_points = n_good_vals
            best_idx = i
    
    R_final, C_final, X_final = R_list[best_idx], C_list[best_idx], X_collection[best_idx]
    return R_final, C_final, X_final


# X_all = np.array([[1, 3, 4, 9], [4, 3, 2, 7], [5, 7, 8, 9]])
# X_all = X_all/X_all[:,3].reshape(-1,1)
# print(X_all)