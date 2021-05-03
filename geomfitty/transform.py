from abc import ABC, abstractmethod

import numpy as np


class Transform(ABC):
    @abstractmethod
    def transform_coords(self, coords):
        """ """

    # @abstractmethod
    # def transform_direction(self, directions):
    #     """
    #     """

    def transform(self, shape):
        """ """


class Translation(Transform):
    def transform_coords(self, coords):
        print("translate")


class Rotation(Transform):
    def transform_coords(self, coords):
        print("rotate")


class CoordTransform(Translation, Rotation):
    def transform_coords(self, coords):
        print("coord transform")
        Translation.transform_coords(self, coords)
        Rotation.transform_coords(self, coords)
        # super(Rotation, self).transform(coords)
