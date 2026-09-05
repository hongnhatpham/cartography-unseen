"""Soft mottling stays visible and attached to surfaces across origin rebases."""
from pathlib import Path

import moderngl
import numpy as np
import pytest


@pytest.fixture
def render_face():
    """Render the production shaders on a constant-color face, without a window."""
    root = Path(__file__).resolve().parents[1]
    ctx = moderngl.create_standalone_context(require=330)
    program = ctx.program(
        vertex_shader=(root / "shaders/proxy.vert").read_text(),
        fragment_shader=(root / "shaders/proxy.frag").read_text(),
    )
    vertices = np.array([
        [-8., -8., 0., 0., 0., 1.],
        [8., -8., 0., 0., 0., 1.],
        [-8., 8., 0., 0., 0., 1.],
        [8., 8., 0., 0., 0., 1.],
    ], dtype="f4")
    vertex_buffer = ctx.buffer(vertices.tobytes())
    instance_buffer = ctx.buffer(reserve=19 * 4)
    vao = ctx.vertex_array(program, [
        (vertex_buffer, "3f 3f", "in_position", "in_normal"),
        (instance_buffer, "4f 4f 4f 4f 3f /i", "instance_model_0",
         "instance_model_1", "instance_model_2", "instance_model_3", "instance_color"),
    ])
    color = ctx.texture((128, 128), 3)
    depth = ctx.depth_texture((128, 128))
    framebuffer = ctx.framebuffer(color_attachments=[color], depth_attachment=depth)
    program["projection"].write(np.diag([1 / 8, 1 / 8, -1 / 16, 1]).astype("f4").tobytes())
    for name in ("fog_color", "zenith_color", "nadir_color"):
        program[name].value = (.2, .3, .4)
    program["fog_distance"].value = 100.
    ctx.enable(moderngl.DEPTH_TEST)

    def render(anchor, origin):
        anchor, origin = np.asarray(anchor, dtype="f8"), np.asarray(origin, dtype="f8")
        model = np.eye(4, dtype="f4")
        model[:3, 3] = anchor - origin
        instance_buffer.write(np.concatenate((model.T.ravel(), [.7, .7, .7])).astype("f4").tobytes())
        camera = anchor - origin + [0., 0., 8.]
        view = np.eye(4, dtype="f4")
        view[:3, 3] = -camera
        program["view"].write(view.T.tobytes())
        program["camera_position"].value = tuple(camera)
        program["surface_origin_coarse"].value = tuple(np.remainder(origin * .20, 256.))
        program["surface_origin_fine"].value = tuple(np.remainder(origin * .53, 256.))
        framebuffer.use()
        framebuffer.clear(depth=1.)
        vao.render(mode=moderngl.TRIANGLE_STRIP)
        rgb = np.frombuffer(color.read(), dtype=np.uint8).reshape(128, 128, 3).copy()
        z = np.frombuffer(depth.read(), dtype="f4").reshape(128, 128).copy()
        return rgb, z

    try:
        yield render
    finally:
        for resource in (framebuffer, depth, color, vao, instance_buffer, vertex_buffer, program):
            resource.release()
        ctx.release()


def test_flat_face_has_broad_visible_variation_without_changing_depth(render_face):
    rgb, depth = render_face([0., 0., 0.], [0., 0., 0.])
    brightness = rgb.astype(float).mean(axis=2)
    contrast = np.percentile(brightness, 95) - np.percentile(brightness, 5)
    assert contrast > 8
    # Broad patches change slowly between adjacent pixels, unlike fine grain.
    assert np.abs(np.diff(brightness, axis=0)).mean() < contrast / 20
    assert np.abs(np.diff(brightness, axis=1)).mean() < contrast / 20
    np.testing.assert_allclose(depth, .75, atol=1e-6, rtol=0)


@pytest.mark.parametrize("anchor", [
    [0., 0., 0.],
    [1280., -1280., 1280.],  # Coarse field crosses its 256-cell wrap.
    [239. / .53, -17. / .53, 239. / .53],  # Fine field wraps after its +17 offset.
    [1_000_000_000_000., -1_000_000_000_000., 1_000_000_000_000.],
    [-1_000_000_000_000., 1_000_000_000_000., -1_000_000_000_000.],
])
def test_surface_pattern_survives_positive_and_negative_origin_rebases(render_face, anchor):
    reference_rgb, reference_depth = render_face(anchor, anchor)
    for offset in ([64., -64., 64.], [-64., 64., -64.]):
        rgb, depth = render_face(anchor, np.asarray(anchor) + offset)
        difference = np.abs(rgb.astype(int) - reference_rgb.astype(int))
        assert difference.max() <= 1
        assert difference.mean() < .05
        np.testing.assert_array_equal(depth, reference_depth)
