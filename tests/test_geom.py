from geomfitty import geom
from .util import assert_vector_equal

import pytest
import numpy as np

def test_asserts():
    assert_vector_equal([1,0,0], [1,0,0])
    assert_vector_equal(2, 2)

    with pytest.raises(AssertionError):
        assert_vector_equal([0,0], 0)

    with pytest.raises(AssertionError):
        assert_vector_equal(2, 2.5)


class AbstractTestGeom:
    def gen_random_shape(self):
        raise NotImplementedError

    def test_distance_to_a_single_point(self):
        shape = self.gen_random_shape()
        distance = shape.distance_to_point(np.random.uniform(size=(3,)))
        assert isinstance(distance, float)
        assert distance > 0

    def test_distance_to_a_multiple_points(self):
        shape = self.gen_random_shape()
        distance = shape.distance_to_point(np.random.uniform(size=(2,3)))
        assert distance.shape == (2,)
        assert np.all(distance > 0)

    @pytest.mark.skip
    def test_equality(self):
        shape = self.gen_random_shape()
        assert shape == shape

    @pytest.mark.skip
    def test_unequality(self):
        shape1 = self.gen_random_shape()
        shape2 = self.gen_random_shape()
        assert shape1 != shape2

class TestLine(AbstractTestGeom):
    def gen_random_shape(self):
        return geom.Line(np.random.uniform(size=(3,)), np.random.uniform(size=(3,)))

    def test_line_contains_anchor_point_and_direction(self):
        line = geom.Line([0,0,0], [1,0,0])
        assert_vector_equal(line.anchor_point, [0,0,0])
        assert_vector_equal(line.direction, [1,0,0])

    def test_line_direction_is_normalized(self):
        line = geom.Line([0,0,0], [2,0,0])
        assert_vector_equal(line.direction, [1,0,0])

    def test_distance_to_line_is_calculated(self):
        line = geom.Line([0,0,0], [1,0,0])
        assert line.distance_to_point([1,1,0]) == 1
        assert line.distance_to_point([1,1,1]) == np.sqrt(2)
        assert line.distance_to_point([1,3,4]) == 5

        line = geom.Line([1,0,0], [1,1,0])
        assert line.distance_to_point([3,2,4]) == 4
        assert line.distance_to_point([0,1,0]) == np.sqrt(2)

    def test_distance_to_line_can_take_multiple_points(self):
        line = geom.Line([0,0,0], [1,0,0])
        assert_vector_equal(line.distance_to_point([[1,1,0],[1,3,4]]), [1,5])

class TestPlane(AbstractTestGeom):
    def gen_random_shape(self):
        return geom.Plane(np.random.uniform(size=(3,)), np.random.uniform(size=(3,)))

class TestSphere(AbstractTestGeom):
    def gen_random_shape(self):
        return geom.Sphere(np.random.uniform(size=(3,)), np.random.uniform())

class TestCylinder(AbstractTestGeom):
    def gen_random_shape(self):
        return geom.Cylinder(np.random.uniform(size=(3,)), np.random.uniform(size=(3,)), np.random.uniform())

class TestCircle3D(AbstractTestGeom):
    def gen_random_shape(self):
        return geom.Circle3D(np.random.uniform(size=(3,)), np.random.uniform(size=(3,)), np.random.uniform())

    def test_distance_to_a_point(self):
        circle = geom.Circle3D([0,0,0], [1,0,0], 1)
        assert circle.distance_to_point([1,1,0]) == 1
        assert circle.distance_to_point([0,0,2]) == 1

class TestTorus(AbstractTestGeom):
    def gen_random_shape(self):
        return geom.Torus(np.random.uniform(size=(3,)), np.random.uniform(size=(3,)), np.random.uniform(), np.random.uniform())

    def test_distance_to_a_point(self):
        torus = geom.Torus([0,0,0], [1,0,0], 1, 0.5)
        assert torus.distance_to_point([1,1,0]) == 0.5
        assert torus.distance_to_point([0,0,2]) == 0.5
