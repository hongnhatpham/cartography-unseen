"""Visible-form uploads preserve pixels while flying and rebasing the world."""
from pathlib import Path

import numpy as np

from app.renderer.camera import Camera
from app.renderer.proxy_renderer import ProxyRenderer


def test_frustum_keeps_forms_crossing_planes_and_rejects_distant_forms():
    bounds = np.array([[0., 0., 0., .2], [1.2, 0., 0., .3],
                       [0., 0., -1.1, .2], [3., 0., 0., .2]])
    assert ProxyRenderer._visible_forms(bounds, np.eye(4)).tolist() == [True, True, True, False]


def test_culled_forms_match_full_geometry_during_camera_changes(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    renderer = ProxyRenderer(root, (384, 256), False, world_seed=934943880)
    try:
        for position, yaw, pitch in (
            ((12., 12., 12.), 35., 10.),
            ((76., 12., 12.), 145., -35.),
            ((12., 1_000_000_012., 12.), 255., 60.),
            ((12., -999_999_988., 12.), 0., -60.),
        ):
            camera = Camera(position=np.array(position), yaw=yaw, pitch=pitch)
            frame = renderer.render_proxy(camera, 0.)
            total_forms = sum(len(data) for data in renderer._form_instances.values())
            drawn_forms = sum(count for mesh, count in renderer._instance_counts.items() if mesh != 'cube')
            assert total_forms > 0
            assert drawn_forms < total_forms
            renderer._form_view_key = None
            with monkeypatch.context() as patch:
                patch.setattr(renderer, '_visible_forms', lambda bounds, clip: np.ones(len(bounds), dtype=bool))
                full = renderer.capture_conditioning(renderer.render_scene(camera, manage_chunks=False), 0.)
            assert np.array_equal(frame.rgb, full.rgb)
            assert np.array_equal(frame.depth, full.depth)
            renderer._form_view_key = None
        renderer.randomize_world(7)
        assert not renderer._form_instances
        assert not renderer._chunk_form_instances
        assert renderer._form_view_key is None
    finally:
        renderer.close()
