from . import geom3d
from ._util import distance_point_point

import numpy as np
import numpy.linalg as la
from scipy import optimize


def centroid_fit(points, weights=None):
    """ Calculates the weighted average of a set of points
    This minimizes the sum of the squared distances between the points
    and the centroid.

    TODO add doctest
    """
    if points.ndim == 1:
        return points
    return np.average(points, axis=0, weights=weights)


def line_fit(points, weights=None) -> geom3d.Line:
    centroid = centroid_fit(points, weights)
    weights = 1.0 if weights is None else weights
    centered_points = points - centroid
    u, s, v = np.linalg.svd(
        np.matmul(weights * centered_points.transpose(), centered_points)
    )
    return geom3d.Line(anchor_point=centroid, direction=v[0])


def plane_fit(points, weights=None) -> geom3d.Plane:
    centroid = centroid_fit(points, weights)
    weights = 1.0 if weights is None else weights
    centered_points = points - centroid
    u, s, v = np.linalg.svd(
        np.matmul(weights * centered_points.transpose(), centered_points)
    )
    return geom3d.Plane(anchor_point=centroid, normal=v[2])


# TODO add weights
def fast_sphere_fit(points) -> geom3d.Sphere:
    A = np.append(points * 2, np.ones((points.shape[0], 1)), axis=1)
    f = np.sum(points ** 2, axis=1)
    C, _, _, _ = np.linalg.lstsq(A, f, rcond=None)
    center = C[0:3]
    radius = np.average(distance_point_point(points, center))
    return geom3d.Sphere(center=center, radius=radius)


def sphere_fit(
    points, weights=None, initial_guess: geom3d.Sphere = None
) -> geom3d.Sphere:
    initial_guess = initial_guess or fast_sphere_fit(points)

    def sphere_fit_residuals(center, points, weights):
        distances = distance_point_point(center, points)
        radius = np.average(distances, weights=weights)
        if weights is None:
            return distances - radius
        return (distances - radius) * np.sqrt(weights)

    results = optimize.least_squares(
        sphere_fit_residuals, x0=initial_guess.center, args=(points, weights)
    )
    if not results.success:
        return RuntimeError(results.message)

    radius = np.average(distance_point_point(points, results.x), weights=weights)
    return geom3d.Sphere(center=results.x, radius=radius)


# TODO be able to run wihtout initial_guess
def cylinder_fit(points, weights=None, initial_guess: geom3d.Cylinder = None):
    assert weights is None or len(points) == len(weights)

    def cylinder_fit_residuals(anchor_direction, points, weights):
        line = geom3d.Line(anchor_direction[:3], anchor_direction[3:])
        distances = line.distance_to_point(points)
        radius = np.average(distances, weights=weights)
        if weights is None:
            return distances - radius
        return (distances - radius) * np.sqrt(weights)

    x0 = np.concatenate([initial_guess.anchor_point, initial_guess.direction])
    results = optimize.least_squares(
        cylinder_fit_residuals, x0=x0, args=(points, weights)
    )
    if not results.success:
        return RuntimeError(results.message)

    line = geom3d.Line(results.x[:3], results.x[3:])
    distances = line.distance_to_point(points)
    radius = np.average(distances, weights=weights)
    return geom3d.Cylinder(results.x[:3], results.x[3:], radius)


# TODO be able to run wihtout initial_guess
def circle3D_fit(points, weights=None, initial_guess: geom3d.Circle3D = None):
    def circle_fit_residuals(circle_params, points, sqrt_w):
        circle = geom3d.Circle3D(
            circle_params[:3], circle_params[3:], la.norm(circle_params[3:])
        )
        distances = circle.distance_to_point(points)
        return distances * sqrt_w

    x0 = np.concatenate(
        [initial_guess.center, initial_guess.direction * initial_guess.radius]
    )
    results = optimize.least_squares(
        circle_fit_residuals,
        jac="2-point",
        method="lm",
        x0=x0,
        args=(points, 1.0 if weights is None else np.sqrt(weights)),
    )

    results = optimize.minimize(
        lambda *args: np.sum(circle_fit_residuals(*args) ** 2),
        x0=results.x,
        args=(points, 1.0 if weights is None else np.sqrt(weights)),
    )

    if not results.success:
        return RuntimeError(results.message)

    return geom3d.Circle3D(results.x[:3], results.x[3:], la.norm(results.x[3:]))


# TODO initial_guess
def torus_fit(points, weights=None, initial_guess: geom3d.Torus = None):
    def torus_fit_residuals(circle_params, points, weights):
        circle = geom3d.Circle3D(
            circle_params[:3], circle_params[3:], la.norm(circle_params[3:])
        )
        distances = circle.distance_to_point(points)
        radius = np.average(distances, weights=weights)
        weights = np.sqrt(weights) if weights is not None else 1.0
        return (distances - radius) * weights

    results = optimize.least_squares(
        torus_fit_residuals,
        x0=np.array([0, 0, 0, 0, 0, 1], dtype=np.float64),
        args=(points, weights),
    )

    if not results.success:
        return RuntimeError(results.message)

    circle3D = geom3d.Circle3D(results.x[:3], results.x[3:], la.norm(results.x[3:]))
    distances = circle3D.distance_to_point(points)
    minor_radius = np.average(distances, weights=weights)
    return geom3d.Torus(
        results.x[:3], results.x[3:], la.norm(results.x[3:]), minor_radius
    )
