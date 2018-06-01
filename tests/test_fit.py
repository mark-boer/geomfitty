from geomfitty import fit
from .test_util import assert_vector_equal

import numpy as np
import pytest

# test centroid fit
class TestCentroid:
    def test_simple_call(self):
        data = np.array([[0,0], [1,1], [2,2]])
        c = fit.centroid_fit(data)
        assert_vector_equal(c, np.array([1,1]))

    def test_with_weigths(self):
        data = np.array([[0,0], [3,3]])
        c = fit.centroid_fit(data, weights=np.array([1,2]))
        assert_vector_equal(c, np.array([2,2]))

    def test_with_single_point(self):
        data = np.array([1,1])
        c = fit.centroid_fit(data)
        assert c.shape == (2,)
        assert_vector_equal(c, np.array([1,1]))

    def test_with_invalid_weights(self):
        data = np.array([[0,0], [1,1]])
        with pytest.raises(ValueError):
            fit.centroid_fit(data, weights=np.array([1,2,3]))

class TestLine:
    def test_fit_line(self):
        data = np.array([[1,1,0],[0,0,0]])
        line = fit.line_fit(data)
        assert_vector_equal(line.anchor_point, np.array([0.5, 0.5,0]))
        assert_vector_equal(line.direction, -np.array([1,1,0]) / np.sqrt(2))

class TestSphere:
    def test_fast_sphere_fit(self):
        data = np.array([[1,0,0],[0,1,0],[0,0,1],[-1,0,0]])
        sphere = fit.fast_sphere_fit(data)
        assert_vector_equal(np.array([0,0,0]), sphere.center)
        assert sphere.radius == 1.0

    def test_sphere_fit(self):
        data = np.array([[1,0,0],[0,1,0],[0,0,1],[-1,0,0]])
        sphere = fit.sphere_fit(data)
        assert_vector_equal(np.array([0,0,0]), sphere.center)
        assert sphere.radius == 1.0

    def test_sphere_fit_with_weights(self):
        data = np.array([[1,0,0],[0,1,0],[0,0,1],[-1,0,0]])
        sphere = fit.sphere_fit(data, weights=[1,1,1,1])
        assert_vector_equal(np.array([0,0,0]), sphere.center)
        assert sphere.radius == 1.0

