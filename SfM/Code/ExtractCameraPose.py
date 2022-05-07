import numpy as np



def extract_camera_pose(E_matrix):
    U, S, V_T = np.linalg.svd(E_matrix)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    R_list = list()
    C_list = list()
    
    C1, R1 = U[:, 2], np.dot(U, np.dot(W, V_T))
    C2, R2 = -U[:, 2], np.dot(U, np.dot(W, V_T))
    C3, R3 = U[:, 2], np.dot(U, np.dot(W.T, V_T))
    C4, R4 = -U[:, 2], np.dot(U, np.dot(W.T, V_T))
    
    C_list.append(C1)
    C_list.append(C2)
    C_list.append(C3)
    C_list.append(C4)
    
    R_list.append(R1)
    R_list.append(R2)
    R_list.append(R3)
    R_list.append(R4)
    
    for i in range(len(R_list)):
        # print(np.linalg.det(rotation_matices[i]))
        if (np.linalg.det(R_list[i]) < 0):
            R_list[i] = -R_list[i]
            C_list[i] = -C_list[i]

    return R_list, C_list
