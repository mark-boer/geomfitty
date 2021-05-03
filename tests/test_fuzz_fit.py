import numpy as np
import numpy.random
import pytest
from scipy import optimize  # type: ignore

from geomfitty import fit3d, geom3d

from .test_util import (
    assert_direction_equivalent,
    assert_float_equal,
    assert_vector_equal,
)


def brute_force_fit(Shape, args, points, weights=None, x0=None):
    csum = np.cumsum((0,) + args)

    def error(x, points, weights):
        geom = Shape(*(x[i:j] for i, j in zip(csum[:-1], csum[1:])))

        weights = 1.0 if weights is None else weights
        distances = geom.distance_to_point(points)
        return np.sum(weights * distances ** 2)

    x0 = np.random.uniform(low=-1, high=1, size=sum(args)) if x0 is None else x0
    results = optimize.minimize(error, x0=x0, args=(points, weights))
    return Shape(*(results.x[i:j] for i, j in zip(csum[:-1], csum[1:]))), results


def test_fuzz_line():
    # Create random data points along the z axis
    points = np.random.uniform(low=-1, high=1, size=(100, 3))
    points[:, 2] *= 10

    line1 = fit3d.line_fit(points)
    line2, results = brute_force_fit(
        geom3d.Line, (3, 3), points, x0=np.array([0, 0, 0, 0, 0, 1], dtype=np.float64)
    )

    assert_direction_equivalent(line1.direction, line2.direction)
    assert_float_equal(
        np.sum(line1.distance_to_point(points) ** 2),
        np.sum(line2.distance_to_point(points) ** 2),
    )


def test_fuzz_line_with_weights():
    # Create random data points along the z axis
    points = np.random.uniform(low=-1, high=1, size=(100, 3))
    points[:, 2] *= 10

    weights = np.random.uniform(size=(100,))

    line1 = fit3d.line_fit(points, weights=weights)
    line2, results = brute_force_fit(
        geom3d.Line,
        (3, 3),
        points,
        weights=weights,
        x0=np.array([0, 0, 0, 0, 0, 1], dtype=np.float64),
    )

    assert_direction_equivalent(line1.direction, line2.direction)
    assert_float_equal(
        np.sum(line1.distance_to_point(points) ** 2),
        np.sum(line2.distance_to_point(points) ** 2),
    )


def test_fuzz_plane():
    # Create random data points along the z axis
    points = np.random.uniform(low=-1, high=1, size=(100, 3))
    points[:, :2] *= 10

    plane1 = fit3d.plane_fit(points)
    plane2, results = brute_force_fit(
        geom3d.Plane, (3, 3), points, x0=np.array([0, 0, 0, 0, 0, 1], dtype=np.float64)
    )

    assert_direction_equivalent(plane1.normal, plane2.normal)
    assert_float_equal(
        np.sum(plane1.distance_to_point(points) ** 2),
        np.sum(plane2.distance_to_point(points) ** 2),
    )


def test_fuzz_plane_with_weights():
    # Create random data points along the z axis
    points = np.random.uniform(low=-1, high=1, size=(100, 3))
    points[:, :2] *= 10

    weights = np.random.uniform(size=(100,))

    plane1 = fit3d.plane_fit(points, weights=weights)
    plane2, results = brute_force_fit(
        geom3d.Plane,
        (3, 3),
        points,
        weights=weights,
        x0=np.array([0, 0, 0, 0, 0, 1], dtype=np.float64),
    )

    assert_direction_equivalent(plane1.normal, plane2.normal)
    assert_float_equal(
        np.sum(plane1.distance_to_point(points) ** 2),
        np.sum(plane2.distance_to_point(points) ** 2),
    )


@pytest.mark.parametrize("initial_guess", [geom3d.Sphere([0, 0, 0], 1), None])
def test_fuzz_sphere(initial_guess):
    points = np.random.uniform(low=-1, high=1, size=(3, 100))
    points /= np.linalg.norm(points, axis=0) * np.random.uniform(
        low=0.9, high=1.1, size=(100,)
    )
    points = points.T

    sphere1 = fit3d.sphere_fit(points, initial_guess=initial_guess)
    sphere2, results = brute_force_fit(
        geom3d.Sphere, (3, 1), points, x0=np.array([0, 0, 0, 1], dtype=np.float64)
    )

    assert_vector_equal(sphere1.center, sphere2.center)
    assert_float_equal(sphere1.radius, sphere2.radius)


@pytest.mark.parametrize("initial_guess", [geom3d.Sphere([0, 0, 0], 1), None])
def test_fuzz_sphere_with_weights(initial_guess):
    points = np.random.uniform(low=-1, high=1, size=(3, 100))
    points /= np.linalg.norm(points, axis=0) * np.random.uniform(
        low=0.9, high=1.1, size=(100,)
    )
    points = points.T

    weights = np.random.uniform(size=(100,))

    sphere1 = fit3d.sphere_fit(points, weights=weights, initial_guess=initial_guess)
    sphere2, results = brute_force_fit(
        geom3d.Sphere,
        (3, 1),
        points,
        weights=weights,
        x0=np.array([0, 0, 0, 1], dtype=np.float64),
    )

    assert_vector_equal(sphere1.center, sphere2.center)
    assert_float_equal(sphere1.radius, sphere2.radius)


@pytest.fixture(scope="function", params=[None, 1])
def weights(request):
    if request.param is None:
        return None
    return np.random.uniform(size=(100,))


# @pytest.mark.parametrize("initial_guess",  [None])
# TODO add intial_guess
def test_fuzz_cylinder(weights):
    initial_guess = geom3d.Cylinder([0, 0, 0], [0, 0, 1], 1)

    points = np.random.uniform(low=-2, high=2, size=(3, 100))
    points[:2] /= np.linalg.norm(points[:2], axis=0) * np.random.uniform(
        low=0.9, high=1.1, size=(100,)
    )
    points = points.T

    cylinder1 = fit3d.cylinder_fit(points, weights=weights, initial_guess=initial_guess)
    cylinder2, _ = brute_force_fit(
        geom3d.Cylinder,
        (3, 3, 1),
        points,
        weights=weights,
        x0=np.array([0, 0, 0, 0, 0, 1, 1], dtype=np.float64),
    )

    weights = 1.0 if weights is None else weights

    assert_float_equal(
        np.sum(weights * cylinder1.distance_to_point(points) ** 2),
        np.sum(weights * cylinder2.distance_to_point(points) ** 2),
    )

    print(np.sum(weights * cylinder1.distance_to_point(points) ** 2))
    print(np.sum(weights * cylinder1.distance_to_point(points) ** 2))

    assert_direction_equivalent(cylinder1.direction, cylinder2.direction)
    assert_float_equal(cylinder1.radius, cylinder2.radius)


def test_fuzz_circle():
    weights = None
    initial_guess = geom3d.Circle3D([0, 0, 0], [0, 0, 1], 1)

    points = np.random.uniform(low=-1, high=1, size=(3, 100))
    points[2] /= 10
    points[:2] /= np.linalg.norm(points[:2], axis=0) * np.random.uniform(
        low=0.9, high=1.1, size=(100,)
    )
    points = points.T

    circle1 = fit3d.circle3D_fit(points, weights=weights, initial_guess=initial_guess)
    circle2, results = brute_force_fit(
        geom3d.Circle3D,
        (3, 3, 1),
        points,
        weights=weights,
        x0=np.array([0, 0, 0, 0, 0, 1, 1], dtype=np.float64),
    )

    # assert np.sum(circle1.distance_to_point(points) ** 2) <= np.sum(circle2.distance_to_point(points) ** 2)

    assert_vector_equal(circle1.center, circle2.center)
    assert_direction_equivalent(circle1.direction, circle2.direction)
    assert_float_equal(circle1.radius, circle2.radius)
