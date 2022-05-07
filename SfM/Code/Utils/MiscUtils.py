import numpy as np



def reprojection_error(X_tilde, point1, point2, R1, C1, R2, C2, K_mtrx):
    I = np.identity(3)
    IC1 = np.column_stack((I, -C1))
    P1 = np.dot(K_mtrx, np.dot(R1, IC1))
    
    IC2 = np.column_stack((I, -C2))
    P2 = np.dot(K_mtrx, np.dot(R2, IC2))
    
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



def mean_reprojection_error(x3D, pts1, pts2, R1, C1, R2, C2, K_mtrx):    
    error_list = []
    for pt1, pt2, X in zip(pts1, pts2, x3D):
        total_error = reprojection_error(X, pt1, pt2, R1, C1, R2, C2, K_mtrx)
        error_list.append(total_error)
    return np.mean(error_list)