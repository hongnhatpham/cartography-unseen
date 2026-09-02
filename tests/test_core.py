from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from app.config import AppConfig, structure_lock_percent
from app.main import (
    apply_master_prefix,
    choose_different_prompt,
    load_prompt_library,
    persist_config_value,
    persist_prompt,
)
from app.renderer.camera import Camera
from app.renderer.proxy_renderer import ProxyRenderer
from app.utils.latest_value import LatestValue


def test_config_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"diffusion_resolution": 448}), encoding="utf-8")
    config = AppConfig.load(path)
    assert config.diffusion_resolution == 448


def test_rectangular_resolution_modes() -> None:
    assert AppConfig(diffusion_resolution="640x384").diffusion_size == (640, 384)
    assert AppConfig(diffusion_resolution="384x256").diffusion_size == (384, 256)
    assert AppConfig(diffusion_resolution="512x512").diffusion_size == (512, 512)
    assert AppConfig(diffusion_resolution="768x512").diffusion_size == (768, 512)
    assert AppConfig(diffusion_resolution="1024x768").diffusion_size == (1024, 768)


def test_structure_lock_applies_to_one_and_multi_step_modes() -> None:
    one_step = AppConfig(steps=1, one_step_timestep=400)
    multi_step = AppConfig(steps=2, img2img_strength=0.4)
    assert structure_lock_percent(one_step) == 60.0
    assert structure_lock_percent(multi_step) == 60.0


def test_latest_value_discards_stale_values() -> None:
    slot: LatestValue[int] = LatestValue()
    slot.publish(1)
    slot.publish(2)
    version, value = slot.wait_newer(0, threading.Event(), timeout=0.001)
    assert version == 2
    assert value == 2


def test_camera_snapshot_is_finite() -> None:
    snapshot = Camera.create_default().snapshot()
    assert snapshot.view_matrix.shape == (4, 4)
    assert snapshot.projection_matrix.shape == (4, 4)
    assert np.isfinite(snapshot.view_matrix).all()


def test_camera_reset_restores_start() -> None:
    camera = Camera.create_default()
    camera.move(1.0, 1.0, 0.0, 8.0)
    camera.rotate(100.0, -40.0, 0.15)
    camera.reset()
    assert np.allclose(camera.position, [0.0, 1.65, 5.5])
    assert camera.yaw == 0.0
    assert camera.pitch == 0.0


def test_persist_prompt_preserves_other_config(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"prompt": "old", "backend": "sd_turbo_stream"}), encoding="utf-8")
    persist_prompt(path, "new painted city")
    updated = json.loads(path.read_text(encoding="utf-8"))
    assert updated == {"prompt": "new painted city", "backend": "sd_turbo_stream"}


def test_persist_runtime_setting(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"reprojection_strength": 0.5, "seed_mode": "fixed"}), encoding="utf-8")
    persist_config_value(path, "reprojection_strength", 0.3)
    updated = json.loads(path.read_text(encoding="utf-8"))
    assert updated == {"reprojection_strength": 0.3, "seed_mode": "fixed"}


def test_prompt_library_is_editable_and_avoids_current_prompt(tmp_path: Path) -> None:
    path = tmp_path / "prompts.json"
    path.write_text(
        json.dumps(
            {
                "prompts": [
                    {"name": "First", "prompt": "first prompt"},
                    {"name": "Second", "prompt": "second prompt"},
                ]
            }
        ),
        encoding="utf-8",
    )
    entries = load_prompt_library(path)
    assert choose_different_prompt(entries, "first prompt") == {
        "name": "Second",
        "prompt": "second prompt",
    }


def test_master_prefix_is_applied_once() -> None:
    prefix = "Photoreal abstract rendering, highly detailed"
    prompt = "alien filament city"
    combined = apply_master_prefix(prefix, prompt)
    assert combined == f"{prefix}, {prompt}"
    assert apply_master_prefix(prefix, combined) == combined


def test_procedural_world_seed_changes_geometry_and_is_deep() -> None:
    first = ProxyRenderer._make_scene(12345)
    repeat = ProxyRenderer._make_scene(12345)
    second = ProxyRenderer._make_scene(54321)
    assert len(first) == len(repeat)
    assert np.allclose(first[40].model, repeat[40].model)
    assert not np.allclose(first[40].model, second[40].model)
    deepest = min(float(item.model[2, 3]) for item in first)
    assert deepest < -390.0
    assert all(item.mesh != "torus" for item in first)
