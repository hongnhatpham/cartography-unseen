"""Fog distance changes atmospheric depth without changing geometry."""
from pathlib import Path

import numpy as np
import pytest

from app.config import AppConfig, DEFAULT_FOG_DISTANCE
from app.renderer.camera import Camera
from app.renderer.proxy_renderer import ProxyRenderer


def test_fog_distance_defaults_and_validation(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{}')
    assert AppConfig.load(path).fog_distance == DEFAULT_FOG_DISTANCE
    for distance in (40., DEFAULT_FOG_DISTANCE, 300.):
        config = AppConfig(fog_distance=distance)
        config.validate()
        assert "fog_distance" not in config.backend_settings()
    for distance in (0., 39., 301., float('nan'), float('inf')):
        with pytest.raises(RuntimeError, match="fog_distance"):
            AppConfig(fog_distance=distance).validate()


def test_nearer_fog_reduces_color_contrast_without_changing_depth():
    root = Path(__file__).resolve().parents[1]
    renderer = ProxyRenderer(root, (384, 256), False, world_seed=934943880)
    camera = Camera(position=np.array([-178.4476876, 57.2336552, -176.1916768]), yaw=0., pitch=-30.)
    try:
        frames = []
        for distance in (40., DEFAULT_FOG_DISTANCE, 300.):
            renderer.fog_distance = distance
            frames.append(renderer.render_proxy(camera, 0.))
        # Draw the same atmosphere without structures, to measure how far the
        # structures' colors remain from their fog destination at each setting.
        renderer._instance_counts = dict.fromkeys(renderer._instance_counts, 0)
        sky = renderer.capture_conditioning(renderer.render_scene(camera, manage_chunks=False), 0.)
        contrast = [np.abs(frame.rgb.astype(float) - sky.rgb).mean() for frame in frames]
        assert contrast[0] < contrast[1] < contrast[2]
        assert contrast[2] - contrast[0] > 5
        assert all(np.array_equal(frame.depth, frames[0].depth) for frame in frames)
    finally:
        renderer.close()
