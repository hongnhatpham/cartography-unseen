from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin, tan

import numpy as np

from app.types import CameraSnapshot

# Eye height above the walkable surface, in world units (a ground cell is 8).
EYE_HEIGHT = 2.2


def perspective(fov_y_degrees: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / tan(radians(fov_y_degrees) * 0.5)
    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[0, 0] = f / aspect
    matrix[1, 1] = f
    matrix[2, 2] = (far + near) / (near - far)
    matrix[2, 3] = (2.0 * far * near) / (near - far)
    matrix[3, 2] = -1.0
    return matrix


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward /= np.linalg.norm(forward)
    side = np.cross(forward, up)
    side /= np.linalg.norm(side)
    corrected_up = np.cross(side, forward)
    matrix = np.identity(4, dtype=np.float32)
    matrix[0, :3] = side
    matrix[1, :3] = corrected_up
    matrix[2, :3] = -forward
    matrix[:3, 3] = -matrix[:3, :3] @ eye
    return matrix


@dataclass(slots=True)
class Camera:
    """First-person walking camera; the renderer owns spawn placement and height."""

    position: np.ndarray
    yaw: float = 0.0
    pitch: float = 0.0
    # Wide enough that nearby slabs converge hard toward the vanishing point.
    fov: float = 82.0
    near: float = 0.05
    # Stay inside the nearest edge of the fixed 11x11 streamed chunk window.
    far: float = 280.0

    @classmethod
    def create_default(cls) -> "Camera":
        # Logical world coordinates stay double precision so a long walk never
        # loses sub-unit movement at large chunk indices.
        return cls(position=np.array([0.0, EYE_HEIGHT, 0.0], dtype=np.float64))

    def reset(self) -> None:
        self.position[:] = (0.0, EYE_HEIGHT, 0.0)
        self.yaw = 0.0
        self.pitch = 0.0

    @property
    def forward(self) -> np.ndarray:
        yaw = radians(self.yaw)
        pitch = radians(self.pitch)
        return np.array(
            [sin(yaw) * cos(pitch), sin(pitch), -cos(yaw) * cos(pitch)], dtype=np.float64
        )

    @property
    def right(self) -> np.ndarray:
        yaw = radians(self.yaw)
        return np.array([cos(yaw), 0.0, sin(yaw)], dtype=np.float64)

    def rotate(self, delta_x: float, delta_y: float, sensitivity: float) -> None:
        self.yaw = (self.yaw + delta_x * sensitivity) % 360.0
        self.pitch = float(np.clip(self.pitch - delta_y * sensitivity, -88.0, 88.0))

    def walk(self, local_x: float, local_z: float, distance: float) -> None:
        """Step along the heading and strafe; height is the renderer's job.

        Only the horizontal projection of the look direction moves the walker,
        so looking up at a monolith or down into a ravine never lifts or sinks
        the camera. ``constrain_camera`` then settles it onto the ground.
        """

        forward = self.forward
        heading = np.array([forward[0], 0.0, forward[2]], dtype=np.float64)
        length = float(np.linalg.norm(heading))
        if length < 1e-9:
            heading = np.array([sin(radians(self.yaw)), 0.0, -cos(radians(self.yaw))])
        else:
            heading /= length
        direction = self.right * local_x + heading * local_z
        length = float(np.linalg.norm(direction))
        if length > 0.0:
            self.position += direction / length * distance

    def snapshot(self, aspect: float = 1.0) -> CameraSnapshot:
        view = look_at(
            self.position,
            self.position + self.forward,
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )
        projection = perspective(self.fov, aspect, self.near, self.far)
        return CameraSnapshot(
            view_matrix=view,
            projection_matrix=projection,
            position=self.position.copy(),
            rotation=np.array([self.pitch, self.yaw, 0.0], dtype=np.float32),
        )
