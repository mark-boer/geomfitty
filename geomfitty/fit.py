from . import geom
from ._util import distance_point_point

import numpy as np
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


def line_fit(points, weights=None) -> geom.Line:
    centroid = centroid_fit(points, weights)
    weights = weights or np.ones(points.shape[0])
    centered_points = points - centroid
    u, s, v = np.linalg.svd(weights * centered_points.transpose() @ centered_points)
    return geom.Line(anchor_point=centroid, direction=v[0])


def plane_fit(points, weights=None) -> geom.Plane:
    centroid = centroid_fit(points, weights)
    weights = weights or np.ones(points.shape[0])
    centered_points = points - centroid
    u, s, v = np.linalg.svd(weights * centered_points.transpose() @ centered_points)
    return geom.Plane(anchor_point=centroid, normal=v[2])


# TODO add weights
def fast_sphere_fit(points) -> geom.Sphere:
    A = np.append(points * 2, np.ones((points.shape[0], 1)), axis=1)
    f = np.sum(points ** 2, axis=1)
    C, _, _, _ = np.linalg.lstsq(A, f)
    center = C[0:3]
    radius = np.average(distance_point_point(points, center))
    return geom.Sphere(center=center, radius=radius)


def sphere_fit(points, weights=None, initial_guess=None) -> geom.Sphere:
    initial_guess = initial_guess or fast_sphere_fit(points)

    def sphere_fit_residuals(center, points, weights):
        distances = distance_point_point(center, points)
        radius = np.average(distances, weights=weights)
        return (distances - radius) * (weights or 1)

    results = optimize.least_squares(
        sphere_fit_residuals, x0=initial_guess.center, args=(points, weights)
    )
    if not results.success:
        return RuntimeError(results.message)

    radius = np.average(distance_point_point(points, results.x), weights=weights)
    return geom.Sphere(center=results.x, radius=radius)


def cylinder_fit(points, weights=None, initial_guess=None):
    def cylinder_fit_residuals(anchor_direction, points, weights):
        line = geom.Line(anchor_direction[:3], anchor_direction[3:])
        distances = line.distance_to_point(points)
        radius = np.average(distances, weights=weights)
        return (distances - radius) * (weights or 1)

    # TODO try multiple guesses?
    results = optimize.least_squares(
        cylinder_fit_residuals, x0=np.array([0, 0, 0, 1, 0, 0]), args=(points, weights)
    )
    if not results.success:
        return RuntimeError(results.message)

    line = geom.Line(results.x[:3], results.x[3:])
    distances = line.distance_to_point(points)
    radius = np.average(distances, weights=weights)
    return geom.Cylinder(results.x[:3], results.x[3:], radius)
