import numpy as np
import numpy.testing

def assert_float_equal(f1, f2):
    assert isinstance(f1, float) and isinstance(f2, float)
    np.testing.assert_allclose(f1, f2, rtol=1e-12, atol=1e-12, equal_nan=False)

def assert_vector_equal(v1, v2):
    np.testing.assert_allclose(v1, v2, rtol=1e-12, atol=1e-12, equal_nan=False)
    assert np.array(v1).shape == np.array(v2).shape

def assert_direction_equivalent(v1, v2):
    assert np.array(v1).shape == (3,) and np.array(v2).shape == (3,)
    assert np.allclose(v1, v2, atol=1e-12, equal_nan=False) or \
           np.allclose(v1, -v2, atol=1e-12, equal_nan=False)
