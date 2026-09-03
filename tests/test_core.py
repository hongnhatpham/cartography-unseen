from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from app.config import AppConfig, structure_lock_percent
from app.diffusion.sd_turbo_stream import classifier_free_guidance_enabled
from app.main import (
    apply_master_prefix,
    choose_different_prompt,
    load_prompt_library,
    persist_config_value,
    persist_prompt,
)
from app.renderer.camera import EYE_HEIGHT, Camera
from app.renderer.proxy_renderer import ProxyRenderer
from app.temporal.reprojection import reproject_previous_image
from app.types import CameraSnapshot, ConditioningFrame
from app.utils.latest_value import LatestValue


def test_config_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"diffusion_resolution": 448}), encoding="utf-8")
    config = AppConfig.load(path)
    assert config.diffusion_resolution == 448


def test_rectangular_resolution_modes() -> None:
    assert AppConfig(diffusion_resolution="384x216").diffusion_size == (384, 216)
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


def test_low_cfg_and_temporal_drift_config() -> None:
    config = AppConfig(
        guidance_scale=1.25,
        seed_mode="drift",
        noise_persistence=0.975,
        prompt_caption=True,
    )
    config.validate()
    assert config.prompt_caption is True
    assert classifier_free_guidance_enabled(1.0) is False
    assert classifier_free_guidance_enabled(1.25) is True


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


def test_camera_far_plane_stays_inside_streamed_city() -> None:
    from app.renderer.city import ACTIVE_CHUNK_RADIUS, CHUNK_SIZE

    assert Camera.create_default().far < ACTIVE_CHUNK_RADIUS * CHUNK_SIZE


def test_camera_reset_restores_start() -> None:
    camera = Camera.create_default()
    camera.move(1.0, 1.0, 8.0)
    camera.rotate(100.0, -40.0, 0.15)
    camera.reset()
    assert np.allclose(camera.position, [0.0, EYE_HEIGHT, 5.5])
    assert camera.yaw == 0.0
    assert camera.pitch == 0.0


def test_camera_follows_road_at_fixed_eye_height() -> None:
    camera = Camera.create_default()
    camera.follow_ground(12.75)
    assert np.isclose(camera.position[1], 12.75 + EYE_HEIGHT)


def test_camera_keeps_sub_unit_movement_at_large_world_coordinates() -> None:
    camera = Camera.create_default()
    camera.position[0] = 16_777_216.0
    camera.move(1.0, 0.0, 0.25)
    assert camera.position[0] == 16_777_216.25


def test_identity_reprojection_preserves_image() -> None:
    height, width = 4, 6
    image = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
    depth = np.full((height, width), 0.5, dtype=np.float32)
    camera = CameraSnapshot(
        view_matrix=np.eye(4, dtype=np.float32),
        projection_matrix=np.eye(4, dtype=np.float32),
        position=np.zeros(3, dtype=np.float32),
        rotation=np.zeros(2, dtype=np.float32),
    )
    frame = ConditioningFrame(
        rgb=image,
        depth=depth,
        edges=np.zeros((height, width), dtype=np.uint8),
        camera=camera,
        timestamp=0.0,
        sequence=1,
    )
    warped, confidence = reproject_previous_image(image, frame, frame)
    assert np.array_equal(warped, image)
    assert np.allclose(confidence, 1.0)


def test_reprojection_rejects_previous_sky() -> None:
    height, width = 4, 6
    image = np.full((height, width, 3), 127, dtype=np.uint8)
    camera = CameraSnapshot(
        view_matrix=np.eye(4, dtype=np.float32),
        projection_matrix=np.eye(4, dtype=np.float32),
        position=np.zeros(3, dtype=np.float32),
        rotation=np.zeros(3, dtype=np.float32),
    )
    previous = ConditioningFrame(
        rgb=image,
        depth=np.ones((height, width), dtype=np.float32),
        edges=np.zeros((height, width), dtype=np.uint8),
        camera=camera,
        timestamp=0.0,
        sequence=1,
    )
    current = ConditioningFrame(
        rgb=image,
        depth=np.full((height, width), 0.5, dtype=np.float32),
        edges=np.zeros((height, width), dtype=np.uint8),
        camera=camera,
        timestamp=1.0,
        sequence=2,
    )

    _, confidence = reproject_previous_image(image, previous, current)

    assert np.count_nonzero(confidence) == 0


def test_reprojection_resizes_fallback_output_to_the_depth_grid() -> None:
    height, width = 4, 6
    image = np.full((height // 2, width // 2, 3), 127, dtype=np.uint8)
    depth = np.full((height, width), 0.5, dtype=np.float32)
    camera = CameraSnapshot(
        view_matrix=np.eye(4, dtype=np.float32),
        projection_matrix=np.eye(4, dtype=np.float32),
        position=np.zeros(3, dtype=np.float32),
        rotation=np.zeros(3, dtype=np.float32),
    )
    frame = ConditioningFrame(
        rgb=np.full((height, width, 3), 127, dtype=np.uint8),
        depth=depth,
        edges=np.zeros((height, width), dtype=np.uint8),
        camera=camera,
        timestamp=0.0,
        sequence=1,
    )

    warped, confidence = reproject_previous_image(image, frame, frame)

    assert warped.shape == (height, width, 3)
    assert np.all(warped == 127)
    assert np.allclose(confidence, 1.0)


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


def test_city_instance_packing_preserves_transform_columns_and_color() -> None:
    from app.renderer.city import CityCube

    item = CityCube(
        role="building",
        position=(11.0, 13.0, -17.0),
        half_extents=(2.0, 3.0, 5.0),
        rotation=(0.0, 0.0, 0.0),
        color=(0.2, 0.4, 0.6),
    )
    packed = ProxyRenderer._pack_instances((item,))
    uploaded_model = packed[0, :16].reshape(4, 4).T

    assert np.allclose(np.diag(uploaded_model)[:3], item.half_extents)
    assert np.allclose(uploaded_model[:3, 3], item.position)
    assert np.allclose(packed[0, 16:], item.color)


def test_renderer_chunk_cache_stays_bounded_and_regenerates_evicted_chunks() -> None:
    from app.renderer.city import (
        CHUNK_SIZE,
        MAX_ACTIVE_CHUNKS,
        MAX_OBJECTS_PER_CHUNK,
    )

    class FakeBuffer:
        size = 19 * 4

        def orphan(self, size: int) -> None:
            self.size = size

        def write(self, data: bytes) -> None:
            assert len(data) <= self.size

    renderer = ProxyRenderer.__new__(ProxyRenderer)
    renderer.world_seed = 12345
    renderer._chunks = {}
    renderer._chunk_instances = {}
    renderer._active_chunk_coords = ()
    renderer._stream_center = None
    renderer.instance_buffers = {"cube": FakeBuffer()}
    renderer._instance_counts = {"cube": 0, "sphere": 0, "cylinder": 0}
    renderer._instance_buffer_revision = 0

    origin = np.array([0.0, 1.65, 0.0], dtype=np.float32)
    renderer._update_city(origin)
    original_chunk = renderer._chunks[(0, 0)]
    initial_revision = renderer._instance_buffer_revision
    renderer._update_city(origin)
    assert renderer._instance_buffer_revision == initial_revision

    for step in range(1, 12):
        renderer._update_city(
            np.array([CHUNK_SIZE * step, 1.65, 0.0], dtype=np.float32)
        )
        assert len(renderer._chunks) == MAX_ACTIVE_CHUNKS
        assert renderer._chunks.keys() == renderer._chunk_instances.keys()
        assert renderer._instance_counts["cube"] <= (
            MAX_ACTIVE_CHUNKS * MAX_OBJECTS_PER_CHUNK
        )

    assert (0, 0) not in renderer._chunks
    renderer._update_city(origin)
    assert renderer._chunks[(0, 0)] == original_chunk
