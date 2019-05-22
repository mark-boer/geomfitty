import numpy as np

from ._descriptor import Position, Direction, PositiveNumber
from ._util import distance_point_point, distance_plane_point, distance_line_point
from abc import ABC, abstractmethod


class GeometricShape(ABC):
    @abstractmethod
    def distance_to_point(self, point):
        """ Calculates the smallest distance from a point to the shape
        """

    # @abstractmethod
    # def project_point(self, point):
    # pass


class Line(GeometricShape):
    anchor_point = Position()
    direction = Direction()

    def __init__(self, anchor_point, direction):
        self.anchor_point = anchor_point
        self.direction = direction

    def distance_to_point(self, point):
        return distance_line_point(self.anchor_point, self.direction, point)


class Plane(GeometricShape):
    anchor_point = Position()
    normal = Direction()

    def __init__(self, anchor_point, normal):
        self.anchor_point = anchor_point
        self.normal = normal

    def distance_to_point(self, point):
        return distance_plane_point(self.anchor_point, self.normal, point)


class Sphere(GeometricShape):
    center = Position()
    radius = PositiveNumber()

    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def distance_to_point(self, point):
        return np.abs(distance_point_point(point, self.center) - self.radius)


class Cylinder(Line):
    radius = PositiveNumber()

    def __init__(self, anchor_point, direction, radius):
        super().__init__(anchor_point, direction)
        self.radius = radius

    def distance_to_point(self, point):
        return np.abs(super().distance_to_point(point) - self.radius)


class Circle3D(GeometricShape):
    center = Position()
    direction = Direction()
    radius = PositiveNumber()

    def __init__(self, center, direction, radius):
        self.center = center
        self.direction = direction
        self.radius = radius

    def distance_to_point(self, point):
        delta_p = point - self.center
        x1 = np.expand_dims(np.dot(delta_p, self.direction), axis=1) @ np.atleast_2d(
            self.direction
        )
        x2 = delta_p - x1
        return np.sqrt(
            np.linalg.norm(x1, axis=-1) ** 2
            + (np.linalg.norm(x2, axis=-1) - self.radius) ** 2
        )


class Torus(Circle3D):
    minor_radius = PositiveNumber()

    def __init__(self, center, direction, major_radius, minor_radius):
        super().__init__(center, direction, major_radius)
        self.minor_radius = minor_radius

    @property
    def major_radius(self):
        return self.radius

    def distance_to_point(self, point):
        return np.abs(super().distance_to_point(point) - self.minor_radius)


class Cone(GeometricShape):
    anchor_point = Position()
    direction = Direction()
    orth_distance = PositiveNumber()
    phi = PositiveNumber()

    def __init__(self, anchor_point, direction, orth_distance, phi):
        self.anchor_point = anchor_point
        self.direction = direction
        self.orth_distance = orth_distance
        self.phi = phi

    # TODO this probably requires distance_plane point to return negative numbers aswell, depending on the side
    def distance_to_point(self, point):
        return np.abs(
            distance_line_point(self.anchor_point, self.direction, point)
            * np.cos(self.phi)
            + distance_plane_point(self.anchor_point, self.direction, point)
            * np.sin(self.phi)
            - self.orth_distance
        )
