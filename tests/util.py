import numpy as np
import numpy.testing

def assert_vector_equal(v1, v2):
    np.testing.assert_allclose(v1, v2, rtol=1e-12, atol=1e-12, equal_nan=False)
