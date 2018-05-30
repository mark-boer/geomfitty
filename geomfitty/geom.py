import numpy as np

from ._descriptor import Position, Direction, PositiveNumber
from ._util import distance_pp
from abc import ABC, abstractmethod

class GeometricShape(ABC):
    @abstractmethod
    def distance_to_point(self, point):
        ''' Calculates the smallest distance from a point to the shape
        '''

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
        delta_p = point - self.anchor_point
        return distance_pp(delta_p, np.expand_dims(np.dot(delta_p, self.direction), axis=1) @ np.atleast_2d(self.direction))

class Plane(GeometricShape):
    anchor_point = Position()
    normal = Direction()

    def __init__(self, anchor_point, normal):
        self.anchor_point = anchor_point
        self.normal = normal

    def distance_to_point(self, point):
        return np.abs(np.dot(point - self.anchor_point, self.normal))

class Sphere(GeometricShape):
    center = Position()
    radius = PositiveNumber()

    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def distance_to_point(self, point):
        return np.abs(distance_pp(point, self.center) - self.radius)

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
        pass

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
