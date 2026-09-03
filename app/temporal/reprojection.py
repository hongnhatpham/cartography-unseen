from __future__ import annotations

import numpy as np
from PIL import Image

from app.types import ConditioningFrame


def reproject_previous_image(
    image: np.ndarray,
    previous: ConditioningFrame,
    current: ConditioningFrame,
    occlusion_tolerance: float = 0.003,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp a previous generated image into the current conditioning camera.

    The returned confidence mask rejects sky, off-screen pixels, and geometry
    that is hidden behind the previous depth buffer.
    """
    height, width = current.depth.shape
    if image.shape[:2] != previous.depth.shape:
        previous_height, previous_width = previous.depth.shape
        image = np.asarray(
            Image.fromarray(image, mode="RGB").resize(
                (previous_width, previous_height), Image.Resampling.BILINEAR
            ),
            dtype=np.uint8,
        )
    if previous.depth.shape != current.depth.shape:
        raise ValueError("Previous and current conditioning dimensions must match")

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    clip = np.stack(
        (
            xx / max(width - 1, 1) * 2.0 - 1.0,
            1.0 - yy / max(height - 1, 1) * 2.0,
            current.depth * 2.0 - 1.0,
            np.ones_like(current.depth),
        ),
        axis=-1,
    ).reshape(-1, 4)

    current_vp = current.camera.projection_matrix @ current.camera.view_matrix
    previous_vp = previous.camera.projection_matrix @ previous.camera.view_matrix
    world = clip @ np.linalg.inv(current_vp).T
    world /= np.where(np.abs(world[:, 3:4]) > 1e-7, world[:, 3:4], 1.0)
    projected = world @ previous_vp.T
    projected_w = projected[:, 3]
    safe_w = np.where(np.abs(projected_w) > 1e-7, projected_w, 1.0)
    ndc = projected[:, :3] / safe_w[:, None]

    source_x = (ndc[:, 0] * 0.5 + 0.5) * (width - 1)
    source_y = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * (height - 1)
    # Matrix roundoff can turn an exact integer coordinate into 0.99999994.
    # Snap only near-integers so identity reprojection remains pixel-exact.
    rounded_x = np.rint(source_x)
    rounded_y = np.rint(source_y)
    source_x = np.where(np.abs(source_x - rounded_x) < 1e-4, rounded_x, source_x)
    source_y = np.where(np.abs(source_y - rounded_y) < 1e-4, rounded_y, source_y)
    valid = (
        (current.depth.reshape(-1) < 0.9998)
        & (projected_w > 1e-7)
        & (source_x >= 0.0)
        & (source_x <= width - 1)
        & (source_y >= 0.0)
        & (source_y <= height - 1)
    )

    x0 = np.clip(np.floor(source_x).astype(np.int32), 0, width - 1)
    y0 = np.clip(np.floor(source_y).astype(np.int32), 0, height - 1)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = (source_x - x0).astype(np.float32)
    wy = (source_y - y0).astype(np.float32)

    def sample(values: np.ndarray) -> np.ndarray:
        top = values[y0, x0] * (1.0 - wx)[:, None] + values[y0, x1] * wx[:, None]
        bottom = values[y1, x0] * (1.0 - wx)[:, None] + values[y1, x1] * wx[:, None]
        return top * (1.0 - wy)[:, None] + bottom * wy[:, None]

    warped = sample(image.astype(np.float32)).reshape(height, width, 3)
    sampled_depth = sample(previous.depth[:, :, None]).reshape(-1)
    valid &= sampled_depth < 0.9998
    expected_depth = ndc[:, 2] * 0.5 + 0.5
    behind_previous = np.maximum(expected_depth - sampled_depth, 0.0)
    confidence = np.clip(
        1.0 - behind_previous / max(occlusion_tolerance, 1e-6), 0.0, 1.0
    )
    confidence *= valid.astype(np.float32)
    return np.clip(warped, 0, 255).astype(np.uint8), confidence.reshape(height, width)
