import numpy as np 
from scipy.spatial.transform import Rotation 


def get_quaternion(R):
    Q = Rotation.from_matrix(R)
    return Q.as_quat()


def get_euler_vec(R):
    euler = Rotation.from_matrix(R)
    return euler.as_rotvec()


def get_rotation(Q, type_='q'):
    if type_ == 'q':
        R = Rotation.from_quat(Q)
        return R.as_matrix()
    elif type_ == 'e':
        R = Rotation.from_rotvec(Q)
        return R.as_matrix()