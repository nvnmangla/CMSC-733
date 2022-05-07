import numpy as np



def PnP(X_, x_, K_matrix):
    X_hmg = np.column_stack((X_, np.ones(len(X_))))
    x_hmg = list()
    for i in range(len(x_)):
        x_hmg.append([x_[i][0], x_[i][1], 1])
    x_hmg = np.array(x_hmg)
    x_norm = np.dot(np.linalg.inv(K_matrix), x_hmg.T).T
    z = np.zeros(4)
    for i in range(len(X_hmg)):
        Xi = X_hmg[i]
        u, v, w = x_norm[i]
        
        u_cross = np.array([[0, -1, v], [1, 0, -u], [-v, u, 0]])
        X_1 = np.hstack((Xi, z, z))
        X_2 = np.hstack((z, Xi, z))
        X_3 = np.hstack((z, z, Xi))
        X_tilde = np.row_stack((X_1, X_2, X_3))
        
        a = np.dot(u_cross, X_tilde)
        if i == 0:
            A = a
        else:
            A = np.vstack((A, a))
    
    U, S, V_T = np.linalg.svd(A)
    P = V_T[-1].reshape((3, 4))
    
    R = P[:, :3]
    Ur, S, Vr_T = np.linalg.svd(R)
    R = np.dot(Ur, Vr_T)
    
    C = P[:, 3]
    t = -np.dot(np.linalg.inv(R), C)
    
    if np.linalg.det(R) < 0:
        R = -R
        t = -t
    
    return R, t      


def reproj_error_PnP(X_3d, point_set, K_matrix, R, C):
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

    return np.mean(error_list)


# test_mtrx = np.array([[-1.46687266e+01, -6.56997882e+00, 4.42454214e+01], 
#              [-1.97812035e+01, -8.93804210e+00, 6.01332371e+01], 
#              [-1.97812035e+01, -8.93804210e+00, 6.01332371e+01],
#              [ 3.30410663e+00, -9.11116001e-02, 1.17288151e+01],
#              [-6.26036487e+00, -8.30772578e+00, 3.32543896e+01],
#              [ 1.50402121e+00, -7.12996320e+00, 1.65950149e+01],
#              [ 1.28860887e+00, -7.59330303e+00, 1.58884235e+01]])

# print(np.column_stack((test_mtrx, np.ones(len(test_mtrx)))))
# z = np.zeros(4)
# X_hmg1 = np.hstack((test_mtrx[0], 1, z, z))
# X_hmg2 = np.hstack((z, test_mtrx[0], 1, z))
# X_hmg3 = np.hstack((z, z, test_mtrx[0], 1))
# print(np.row_stack((X_hmg1, X_hmg2, X_hmg3)))