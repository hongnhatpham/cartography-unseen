"""Holding a published frame reuses its pixels; changing view never reuses stale pixels."""
import numpy as np

from app.renderer.camera import Camera
from app.renderer.proxy_renderer import ProxyRenderer
from app.types import ConditioningFrame, GeneratedFrame


class Texture:
    def __init__(self):
        self.writes = []

    def write(self, pixels):
        self.writes.append(bytes(pixels))


def renderer():
    result = ProxyRenderer.__new__(ProxyRenderer)
    result.render_width, result.render_height = 8, 8
    result.display_texture = Texture()
    result.reproject_source_depth = Texture()
    result._display_frame = None
    result._reproject_frame = None
    result.reproject_ms = 0.
    result._world_contains = lambda position: False
    result._present_texture = lambda *args, **kwargs: None
    return result


def frame(value, sequence=1):
    camera = Camera.create_default().snapshot(1.)
    rgb = np.full((8, 8, 3), value, dtype=np.uint8)
    return GeneratedFrame(rgb, np.ones((8, 8), dtype='f4'), camera.view_matrix,
                          camera.projection_matrix, camera.position, camera.rotation,
                          0., 0., sequence)


def test_held_ai_image_uploads_once_for_many_display_refreshes():
    display = renderer()
    source = frame(124)
    for _ in range(120):
        display.display(source.image, trail_frame=source)
    assert len(display.display_texture.writes) == 1
    assert display.display_texture.writes[-1] == np.flipud(source.image).tobytes()


def test_proxy_frames_and_diagnostic_images_do_not_share_cached_pixels():
    display = renderer()
    rgb = np.arange(192, dtype=np.uint8).reshape(8, 8, 3)
    source = ConditioningFrame(rgb, np.ones((8, 8)), None, Camera.create_default().snapshot(1.), 0., 1)
    display.display(rgb, trail_frame=source)
    display.display(rgb, trail_frame=source)
    assert len(display.display_texture.writes) == 1
    diagnostic = np.zeros_like(rgb)
    display.display(diagnostic, trail_frame=source)
    display.display(rgb, trail_frame=source)
    assert len(display.display_texture.writes) == 3
    assert display.display_texture.writes[-1] == np.flipud(rgb).tobytes()


def test_mutable_unversioned_images_always_upload():
    display = renderer()
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    display.display(image)
    image[:] = 255
    display.display(image)
    assert len(display.display_texture.writes) == 2
    assert display.display_texture.writes[-1] == image.tobytes()


def test_raw_and_reprojected_views_restore_shared_texture_and_accept_reused_sequence():
    display = renderer()
    camera = Camera.create_default()
    first, second = frame(45), frame(190)
    display.display_reprojected(first, camera)
    display.display_reprojected(first, camera)
    assert len(display.display_texture.writes) == 1
    display.display(second.image, trail_frame=second)
    display.display_reprojected(first, camera)
    assert display.display_texture.writes[-1] == first.image.tobytes()
    # A backend or test can publish another frame with the same sequence.
    display.display_reprojected(second, camera)
    assert display.display_texture.writes[-1] == second.image.tobytes()
    assert len(display.reproject_source_depth.writes) == 2
