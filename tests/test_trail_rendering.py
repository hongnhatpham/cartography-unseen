"""The visible route follows presentation crop/depth and never enters AI input."""
from pathlib import Path

import moderngl
import numpy as np
import pytest

from app.renderer.camera import Camera, perspective
from app.renderer.player_trail import PlayerTrail
from app.renderer.proxy_renderer import ProxyRenderer
from app.renderer.trail_renderer import TrailRenderer
from app.types import CameraSnapshot, GeneratedFrame


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def trail_gpu():
    ctx = moderngl.create_standalone_context(require=330)
    color = ctx.texture((128, 128), 3)
    target = ctx.framebuffer(color)
    renderer = TrailRenderer(ctx, ROOT / "shaders")
    target.use()
    ctx.viewport = (0, 0, 128, 128)
    try:
        yield renderer, target
    finally:
        renderer.close()
        target.release()
        color.release()
        ctx.release()


def pixels(target):
    return np.frombuffer(target.read(components=3), dtype=np.uint8).reshape(128, 128, 3)


def snapshot(position=None):
    position = np.zeros(3) if position is None else position
    view = np.eye(4)
    view[:3, 3] = -position
    return CameraSnapshot(view, np.eye(4), position, np.zeros(3))


@pytest.mark.parametrize("axis", [0, 1])
def test_trail_is_fifteen_pixels_wide_and_occludes_across_its_width(trail_gpu, axis):
    renderer, target = trail_gpu
    direction = np.eye(3)[axis]
    trail = PlayerTrail()
    for at, distance in [(0., -.7), (.1, .7), (.2, 4.7)]:
        trail.update(np.array([0., .7, 0.]) + direction * distance, at)
    depth = np.ones((128, 128), dtype="f4")
    target.clear()
    renderer.draw(trail, .2, snapshot(), depth, (128, 128), (1., 1.), (0., 0.))
    image = pixels(target).max(axis=2)
    cross_section = image[:, 64] if axis == 0 else image[64, :]
    covered = np.flatnonzero(cross_section)
    assert len(covered) == 15

    # A foreground edge covers half of the width, not just the centerline.
    if axis == 0:
        depth[:64, :] = .25
    else:
        depth[:, 64:] = .25
    target.clear()
    renderer.draw(trail, .2, snapshot(), depth.copy(), (128, 128), (1., 1.), (0., 0.))
    image = pixels(target).max(axis=2)
    cross_section = image[:, 64] if axis == 0 else image[64, :]
    assert cross_section[covered[covered >= 64]].max() == 0
    assert cross_section[covered[covered < 64]].min() > 150


@pytest.mark.parametrize("reverse", [False, True])
def test_trail_crossing_camera_clips_at_near_plane(trail_gpu, monkeypatch, reverse):
    renderer, target = trail_gpu
    trail = PlayerTrail()
    vertices = np.array([[-.15, .7, -1., .8], [.15, .7, .2, .8]], dtype="f4")
    if reverse:
        vertices = vertices[::-1].copy()
    monkeypatch.setattr(trail, "vertices", lambda *args: vertices)
    camera = snapshot()
    camera = CameraSnapshot(camera.view_matrix, perspective(90., 1., .1, 10.),
                            camera.position, camera.rotation)
    depth = np.ones((128, 128), dtype="f4")
    target.clear()
    renderer.draw(trail, 0., camera, depth, (128, 128), (1., 1.), (0., 0.))
    image = pixels(target).max(axis=2)
    # The crossing endpoint clips to x=.075, z=-.1 (NDC x=.75).
    # Dividing its original negative W would mirror the line to the left.
    assert image[64, 60:110].min() > 150
    assert image[:, :52].max() == 0
    assert image[:, 114:].max() == 0
    assert np.count_nonzero(image[:, 80]) == 15

    # A route wholly behind the camera contributes no pixels.
    vertices[:, 2] = .2
    target.clear()
    renderer.draw(trail, 0., camera, depth, (128, 128), (1., 1.), (0., 0.))
    assert pixels(target).max() == 0


@pytest.mark.parametrize("scale", [(1., .5), (.5, 1.)])
def test_trail_occlusion_matches_cover_crop_and_depth_orientation(trail_gpu, scale):
    renderer, target = trail_gpu
    trail = PlayerTrail()
    # The shader lowers the path by .7; its source-image height is then .65.
    trail.update(np.array([-.6, 1., 0.]), 0.)
    trail.update(np.array([.6, 1., 0.]), .1)
    trail.update(np.array([4.6, 1., 0.]), .2)
    depth = np.ones((128, 128), dtype="f4")
    # CPU images use top-left origin. This box is above the image midpoint.
    depth[26:51, 51:77] = .25
    original = depth.copy()
    offset = tuple((1. - value) / 2 for value in scale)
    target.clear()
    renderer.draw(trail, .2, snapshot(), depth, (128, 128), scale, offset)
    result = pixels(target)
    y = int((.65 - offset[1]) / scale[1] * 128)
    visible_x = int((.65 - offset[0]) / scale[0] * 128)
    hidden_x = int((.5 - offset[0]) / scale[0] * 128)
    assert result[y - 2:y + 3, visible_x - 1:visible_x + 2].max() > 150
    assert result[y - 2:y + 3, hidden_x - 1:hidden_x + 2].max() == 0
    assert abs(np.where(result.max(axis=2) > 0)[0].mean() - y) < 2
    np.testing.assert_array_equal(depth, original)
    # Keeping the source reference also avoids uploading unchanged AI depth
    # on every screen refresh. Resized conditioning gets a correctly sized map.
    texture = renderer._depth(depth)
    assert renderer._depth(depth) is texture
    assert renderer._depth_source is depth
    assert renderer._depth(np.ones((64, 96), dtype="f4")).size == (96, 64)


def test_trail_fades_to_no_pixels_and_keeps_remote_coordinate_precision(trail_gpu):
    renderer, target = trail_gpu
    origin = np.array([1e9, -1e9, 1e9])
    trail = PlayerTrail()
    trail.update(origin + [-.7, .7, 0.], 0.)
    trail.update(origin + [.7, .7, 0.], .1)
    trail.update(origin + [4.7, .7, 0.], .2)
    depth = np.ones((128, 128), dtype="f4")
    energy = []
    for now in (.2, 5.2, 10.2):
        target.clear()
        renderer.draw(trail, now, snapshot(origin), depth, (128, 128), (1., 1.), (0., 0.))
        energy.append(int(pixels(target).sum()))
    assert energy[0] > energy[1] > energy[2] == 0


def test_trail_presentation_preserves_conditioning_and_uses_matching_camera(monkeypatch):
    renderer = ProxyRenderer(ROOT, (128, 128), False, window_size=(256, 128))
    try:
        camera = Camera.create_default()
        renderer.spawn_camera(camera)
        frame = renderer.render_proxy(camera, 0.)
        generated = GeneratedFrame(
            frame.rgb, frame.depth, frame.camera.view_matrix, frame.camera.projection_matrix,
            frame.camera.position, frame.camera.rotation, 0., 0., frame.sequence,
        )
        trail = PlayerTrail()
        trail.update(camera.position - camera.forward * 6., -.2)
        trail.update(camera.position - camera.forward * 4., -.1)
        trail.update(camera.position, 0.)
        calls = []
        draw = renderer.trail_renderer.draw

        def record(*args):
            calls.append(args)
            draw(*args)

        monkeypatch.setattr(renderer.trail_renderer, "draw", record)
        renderer.display(frame.rgb, trail=trail, trail_frame=frame, trail_time=.1)
        renderer.display(generated.image, trail=trail, trail_frame=generated, trail_time=.1)
        after = renderer.capture_conditioning(frame.camera, .1)
        np.testing.assert_array_equal(frame.rgb, after.rgb)
        np.testing.assert_array_equal(frame.depth, after.depth)
        assert all(call[3] is frame.depth for call in calls)
        width, height = renderer.window_size
        scale = (1., height / width) if width >= height else (width / height, 1.)
        offset = tuple((1. - value) / 2 for value in scale)
        assert all(call[5:7] == (scale, offset) for call in calls)
        assert all(call[7] == frame.timestamp for call in calls)

        camera.position += camera.forward * .4
        renderer.display_reprojected(generated, camera, trail=trail, trail_time=.2)
        warped_camera, warped_depth = calls[-1][2:4]
        np.testing.assert_allclose(warped_camera.position, camera.position)
        assert warped_depth is renderer.depth_texture
        assert calls[-1][7] == generated.conditioning_timestamp
        # An out-of-window warp falls back to the source image and source depth.
        monkeypatch.setattr(renderer, "_world_contains", lambda position: False)
        renderer.display_reprojected(generated, camera, trail=trail, trail_time=.3)
        source_camera, source_depth = calls[-1][2:4]
        np.testing.assert_array_equal(source_camera.position, generated.camera_position)
        assert source_depth is generated.depth
        assert calls[-1][7] == generated.conditioning_timestamp
    finally:
        renderer.close()
