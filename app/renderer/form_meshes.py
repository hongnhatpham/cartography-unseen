"""Shared meshes and matching collision pieces for the additive forms.

Templates live in a unit box and are built once. Tubes retain separate collision
pieces, so the empty space in a cage or between arch ribs stays empty.
"""
from __future__ import annotations

from functools import lru_cache
from math import pi

import numpy as np


def _tube(points: np.ndarray, radius: float, sides: int = 6) -> tuple[np.ndarray, np.ndarray]:
    rings = []
    angles = np.arange(sides) * 2 * pi / sides
    for index, point in enumerate(points):
        tangent = points[min(index + 1, len(points) - 1)] - points[max(0, index - 1)]
        tangent /= np.linalg.norm(tangent)
        axis = np.array([0., 1., 0.]) if abs(tangent[1]) < .9 else np.array([1., 0., 0.])
        u = np.cross(tangent, axis)
        u /= np.linalg.norm(u)
        v = np.cross(tangent, u)
        normals = np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v
        rings.append(np.c_[point + radius * normals, normals])
    triangles = []
    bounds = []
    for index in range(len(rings) - 1):
        vertices = np.concatenate((rings[index][:, :3], rings[index + 1][:, :3]))
        bounds.append(np.r_[vertices.min(axis=0), vertices.max(axis=0)])
        for side in range(sides):
            other = (side + 1) % sides
            triangles.extend((rings[index][side], rings[index + 1][side], rings[index + 1][other],
                              rings[index][side], rings[index + 1][other], rings[index][other]))
    return np.asarray(triangles, dtype="f4"), np.asarray(bounds)


def _rounded() -> np.ndarray:
    def vertex(u: float, v: float) -> np.ndarray:
        raw = np.array([np.cos(v) * np.cos(u), np.sin(v), np.cos(v) * np.sin(u)])
        position = np.sign(raw) * np.abs(raw) ** .45
        normal = np.sign(position) * np.abs(position) ** (2 / .45 - 1)
        normal /= max(np.linalg.norm(normal), 1e-8)
        return np.r_[position, normal]

    rows = []
    for longitude in range(24):
        for latitude in range(12):
            u0, u1 = longitude * 2 * pi / 24, (longitude + 1) * 2 * pi / 24
            v0, v1 = -pi / 2 + latitude * pi / 12, -pi / 2 + (latitude + 1) * pi / 12
            a, b, c, d = vertex(u0, v0), vertex(u1, v0), vertex(u1, v1), vertex(u0, v1)
            rows.extend((a, b, c, a, c, d))
    return np.asarray(rows, dtype="f4")


@lru_cache(maxsize=1)
def _templates() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    paths: dict[str, list[tuple[np.ndarray, float]]] = {"cage": [], "ribs": [], "weave": []}
    for axis in range(3):
        for a in (-.85, .85):
            for b in (-.85, .85):
                ends = np.zeros((2, 3))
                ends[:, axis] = [-.85, .85]
                other = [k for k in range(3) if k != axis]
                ends[:, other[0]], ends[:, other[1]] = a, b
                paths["cage"].append((ends, .09))
    theta = np.linspace(0, pi, 13)
    for z in (-.75, 0., .75):
        paths["ribs"].append((np.c_[.86 * np.cos(theta), 1.55 * np.sin(theta) - .75,
                                           np.full_like(theta, z)], .075))
    for x in (-.86, .86):
        paths["ribs"].append((np.array([[x, -.75, -.85], [x, -.75, .85]]), .08))
    t = np.linspace(-.88, .88, 7)
    for level in np.linspace(-.85, .85, 5):
        paths["weave"].append((np.c_[t, np.full_like(t, level), .65 * (t * t - level * level)], .043))
        paths["weave"].append((np.c_[np.full_like(t, level), t, .65 * (level * level - t * t)], .043))
    meshes = {"rounded": _rounded()}
    bounds = {"rounded": np.array([[-1., -1., -1., 1., 1., 1.]])}
    for name, lines in paths.items():
        pieces = [_tube(points, radius) for points, radius in lines]
        vertices = np.concatenate([piece[0] for piece in pieces])
        boxes = np.concatenate([piece[1] for piece in pieces])
        orders = [("cage", [0, 1, 2])] if name == "cage" else [
            (f"{name}{axis}", order) for axis, order in enumerate(([2, 0, 1], [0, 2, 1], [0, 1, 2]))]
        for key, order in orders:
            meshes[key] = np.ascontiguousarray(np.c_[vertices[:, :3][:, order], vertices[:, 3:][:, order]], dtype="f4")
            bounds[key] = np.ascontiguousarray(np.c_[boxes[:, :3][:, order], boxes[:, 3:][:, order]])
    for item in meshes.values():
        # Axis permutations can reverse handedness. Keep winding outward for
        # renderers that enable back-face culling.
        triangles = item.reshape(-1, 3, 6)
        normal = np.cross(triangles[:, 1, :3] - triangles[:, 0, :3],
                          triangles[:, 2, :3] - triangles[:, 0, :3])
        inward = np.sum(normal * triangles[:, :, 3:].mean(axis=1), axis=1) < 0
        triangles[inward] = triangles[inward][:, [0, 2, 1]]
    for item in (*meshes.values(), *bounds.values()):
        item.flags.writeable = False
    return meshes, bounds


def form_meshes() -> dict[str, np.ndarray]:
    """Interleaved position/normal triangle vertices for each instance family."""
    return _templates()[0]


def form_bounds(mesh: str) -> np.ndarray:
    """Local min/max bounds per tube segment, or one bound for a rounded solid."""
    return _templates()[1][mesh]
