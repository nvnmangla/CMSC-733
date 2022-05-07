import numpy as np



def get_visibility_matrix(X_3D_bool, filtered_feature_flag, n_cameras):

    bin_temp = np.zeros((filtered_feature_flag.shape[0]), dtype = int)
    for n in range(n_cameras + 1):
        bin_temp = bin_temp | filtered_feature_flag[:,n]

    X_idx = np.where((X_3D_bool.reshape(-1)) & (bin_temp))
    
    visiblity_matrix = X_3D_bool[X_idx].reshape(-1,1)
    for n in range(n_cameras + 1):
        visiblity_matrix = np.hstack((visiblity_matrix, filtered_feature_flag[X_idx, n].reshape(-1,1)))

    o, c = visiblity_matrix.shape
    return X_idx, visiblity_matrix[:, 1:c]