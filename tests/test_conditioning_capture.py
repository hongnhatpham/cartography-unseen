"""Skipping unused edges leaves all AI conditioning and diagnostic pixels intact."""
import numpy as np

from app.renderer.camera import Camera
from app.renderer.proxy_renderer import ProxyRenderer


class Texture:
    def __init__(self, pixels):
        self.pixels = pixels

    def read(self, alignment):
        return self.pixels.tobytes()


def test_optional_edges_preserve_rgb_depth_and_edge_diagnostic(monkeypatch):
    renderer = ProxyRenderer.__new__(ProxyRenderer)
    renderer.render_width, renderer.render_height = 12, 8
    renderer.sequence = 0
    rng = np.random.default_rng(8)
    renderer.color_texture = Texture(rng.integers(0, 256, (8, 12, 3), dtype=np.uint8))
    renderer.depth_texture = Texture(rng.random((8, 12), dtype=np.float32))
    camera = Camera.create_default().snapshot(1.5)
    complete = renderer.capture_conditioning(camera, 0.)

    def unexpected_edges(*args):
        raise AssertionError('Unused edge calculation ran')

    with monkeypatch.context() as patch:
        patch.setattr(renderer, '_edges', unexpected_edges)
        minimal = renderer.capture_conditioning(camera, 1., include_edges=False)

    assert minimal.edges is None
    assert np.array_equal(minimal.rgb, complete.rgb)
    assert np.array_equal(minimal.depth, complete.depth)
    assert minimal.camera is camera
    assert minimal.sequence == complete.sequence + 1
    assert np.array_equal(renderer.diagnostic_image(minimal, 'edges'),
                          renderer.diagnostic_image(complete, 'edges'))
