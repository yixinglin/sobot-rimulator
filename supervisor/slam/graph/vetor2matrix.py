import numpy as np
from math import cos, sin, atan2

def v2t(v):
    """
    computes the homogeneous transform matrix T corresponding to the pose vector v
    :param v: the pose vector v = [x, y, theta]
    :return: a 3 x 3 homogeneous transform
    """
    c = cos(v[2, 0])
    s = sin(v[2, 0])
    T = np.array([[c, -s, v[0, 0]], [s, c, v[1, 0]], [0, 0, 1]])
    return T

def t2v(T):
    """
    computes the pose vector v from a homogeneous transform T
    :param T: a 3 x 3 homogeneous transform
    :return: the pose vector v = [x, y, theta]
    """
    x = T[0, 2]
    y = T[1, 2]
    theta = atan2(T[1, 0], T[0, 0])
    v = np.array([x, y, theta])
    return v.reshape(3,1)




