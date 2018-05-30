import numpy as np
import scipy.spatial

def distance_pp(p1, p2):
    ''' Calculates the euclidian distance between two points or sets of points
    >>> distance_pp(np.array([1, 0]), np.array([0, 1]))
    1.4142135623730951
    >>> distance_pp(np.array([[1, 0], [0, 0]]), np.array([0, 1]))
    array([ 1.41421356,  1.        ])
    >>> distance_pp(np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, -3]]))
    array([ 1.,  3.])
    '''
    return scipy.spatial.minkowski_distance(p1, p2)

def vector_equal(v1, v2):
    return v1.shape == v2.shape and np.allclose(v1, v2, rtol=1e-12, atol=1e-12, equal_nan=False)
