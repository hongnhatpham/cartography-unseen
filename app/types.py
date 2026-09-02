from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, slots=True)
class CameraSnapshot:
    view_matrix: np.ndarray
    projection_matrix: np.ndarray
    position: np.ndarray
    rotation: np.ndarray


@dataclass(frozen=True, slots=True)
class ConditioningFrame:
    rgb: np.ndarray
    depth: np.ndarray
    edges: np.ndarray
    camera: CameraSnapshot
    timestamp: float
    sequence: int


@dataclass(frozen=True, slots=True)
class GeneratedFrame:
    image: np.ndarray
    depth: np.ndarray
    view_matrix: np.ndarray
    projection_matrix: np.ndarray
    camera_position: np.ndarray
    camera_rotation: np.ndarray
    generation_timestamp: float
    conditioning_timestamp: float
    sequence: int
    stats: dict[str, float | int | str] = field(default_factory=dict)
