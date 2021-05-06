import functools
import numbers
from typing import List, Union

import numpy as np
import open3d as o3d  # type: ignore

from . import geom3d


def vec2vec_rotation(unit_vec_1, unit_vec_2):
    angle = np.arccos(np.dot(unit_vec_1, unit_vec_2))
    if angle < 1e-8:
        return np.identity(3, dtype=np.float64)

    if angle > (np.pi - 1e-8):
        # WARNING this only works because all geometries are rotationaly invariant
        # minus identity is not a proper rotation matrix
        return -np.identity(3, dtype=np.float64)

    rot_vec = np.cross(unit_vec_1, unit_vec_2)
    rot_vec /= np.linalg.norm(rot_vec)

    return o3d.geometry.get_rotation_matrix_from_axis_angle(angle * rot_vec)


@functools.singledispatch
def to_open3d_geom(geom):
    return geom


@to_open3d_geom.register  # type: ignore[no-redef]
def _(points: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


@to_open3d_geom.register  # type: ignore[no-redef]
def _(geom: geom3d.Line, length: numbers.Number = 1):
    points = (
        geom.anchor_point
        + np.stack([geom.direction, -geom.direction], axis=0) * length / 2
    )

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector([[0, 1]]),
    )
    return line_set


@to_open3d_geom.register  # type: ignore[no-redef]
def _(geom: geom3d.Sphere):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=geom.radius)
    mesh.translate(geom.center)

    return o3d.geometry.LineSet.create_from_triangle_mesh(mesh)


@to_open3d_geom.register  # type: ignore[no-redef]
def _(geom: geom3d.Plane, length: numbers.Number = 1):
    points = np.array([[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0]]) * length / 2

    mesh = o3d.geometry.TetraMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.tetras = o3d.utility.Vector4iVector(np.array([[0, 1, 2, 3]]))

    rotation = vec2vec_rotation([0, 0, 1], geom.normal)
    mesh.rotate(rotation)
    mesh.translate(geom.anchor_point)

    return o3d.geometry.LineSet.create_from_tetra_mesh(mesh)


@to_open3d_geom.register  # type: ignore[no-redef]
def _(geom: geom3d.Cylinder, length: numbers.Number = 1):
    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=geom.radius, height=length)

    mesh.remove_vertices_by_index([0, 1])

    rotation = vec2vec_rotation([0, 0, 1], geom.direction)
    mesh.rotate(rotation)
    mesh.translate(geom.anchor_point)

    return o3d.geometry.LineSet.create_from_triangle_mesh(mesh)


@to_open3d_geom.register  # type: ignore[no-redef]
def _(geom: geom3d.Circle3D):
    mesh = o3d.geometry.TriangleMesh.create_torus(
        torus_radius=geom.radius, tube_radius=1e-6
    )
    rotation = vec2vec_rotation([0, 0, 1], geom.direction)
    mesh.rotate(rotation)
    mesh.translate(geom.center)

    return o3d.geometry.LineSet.create_from_triangle_mesh(mesh)


@to_open3d_geom.register  # type: ignore[no-redef]
def _(geom: geom3d.Torus):
    mesh = o3d.geometry.TriangleMesh.create_torus(
        torus_radius=geom.major_radius, tube_radius=geom.minor_radius
    )
    rotation = vec2vec_rotation([0, 0, 1], geom.direction)
    mesh.rotate(rotation)
    mesh.translate(geom.center)

    return o3d.geometry.LineSet.create_from_triangle_mesh(mesh)


def plot(
    geometries_or_points: List[Union[geom3d.GeometricShape, np.ndarray]],
    display_coordinate_frame: bool = False,
):
    geometries = [to_open3d_geom(g) for g in geometries_or_points]
    if display_coordinate_frame:
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame())
    o3d.visualization.draw_geometries(geometries)
