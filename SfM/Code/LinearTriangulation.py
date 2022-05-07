import numpy as np



def linear_triangulation(K, C1, R1, C2, R2, point_set1, point_set2):
    I = np.identity(3)
    IC1 = np.column_stack((I, -C1))
    P1 = np.dot(K, np.dot(R1, IC1))
    
    IC2 = np.column_stack((I, -C2))
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
        X = V[:, -1]
        X_list.append(X)

    return np.array(X_list)