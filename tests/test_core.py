from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest

from app.config import RESOLUTION_MODES, AppConfig
from app.diffusion.latent_walk import classifier_free_guidance_enabled
from app.main import (
    PromptEntry,
    advance_prompt,
    apply_master_prefix,
    choose_family_prompt,
    compose_prompt,
    entry_for_prompt,
    hue_words,
    load_prompt_library,
    next_level,
    persist_config_value,
    persist_prompt,
)
from app.renderer.camera import Camera
from app.renderer.proxy_renderer import ProxyRenderer
from app.temporal.reprojection import reproject_previous_image
from app.types import CameraSnapshot, ConditioningFrame
from app.utils.latest_value import LatestValue


def test_config_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"diffusion_resolution": "512x384"}), encoding="utf-8")
    config = AppConfig.load(path)
    assert config.diffusion_size == (512, 384)
    assert config.backend == "latent_walk"


def test_every_resolution_mode_is_selectable_and_latent_aligned() -> None:
    assert len(RESOLUTION_MODES) == 6
    for width, height in RESOLUTION_MODES:
        assert width % 8 == 0 and height % 8 == 0
        AppConfig(diffusion_resolution=f"{width}x{height}").validate()


def test_latent_walk_keys_are_validated_and_forwarded() -> None:
    config = AppConfig(timestep_min=780, timestep_max=900, instability=0.5, guide_strength=0.6)
    config.validate()
    assert config.backend_settings() == {
        "steps": 1,
        "guidance_scale": 1.8,
        "timestep_min": 780,
        "timestep_max": 900,
        "instability": 0.5,
        "guide_strength": 0.6,
        "memory_match": 1.0,
        "memory_match_std": 0.5,
        "memory_leash": 0.7,
        "depth_guide": 0.5,
        "depth_shade": -0.6,
        "guide_wobble": 0.0,
        "noise_walk_seconds": 5.0,
        "noise_jitter": 0.16,
        "prompt_walk_seconds": 6.0,
        "feedback_reprojection": False,
    }
    assert classifier_free_guidance_enabled(1.0) is False
    assert classifier_free_guidance_enabled(1.25) is True

    with pytest.raises(RuntimeError, match="timestep_min"):
        AppConfig(timestep_min=900, timestep_max=800).validate()
    with pytest.raises(RuntimeError, match="instability"):
        AppConfig(instability=1.4).validate()
    with pytest.raises(RuntimeError, match="memory_match"):
        AppConfig(memory_match=1.4).validate()
    with pytest.raises(RuntimeError, match="memory_leash"):
        AppConfig(memory_leash=9.0).validate()
    with pytest.raises(RuntimeError, match="noise_walk_seconds"):
        AppConfig(noise_walk_seconds=-1.0).validate()
    with pytest.raises(RuntimeError, match="guide_wobble"):
        AppConfig(guide_wobble=1.2).validate()
    # depth_guide and depth_shade are signed: the sign is the direction of the
    # ramp, so only the magnitude is bounded.
    with pytest.raises(RuntimeError, match="depth_shade"):
        AppConfig(depth_shade=-1.4).validate()
    AppConfig(depth_guide=-1.0, depth_shade=1.0).validate()


def test_backend_dict_resolves_both_model_folders(tmp_path: Path) -> None:
    resolved = AppConfig().backend_dict(tmp_path)
    assert resolved["model_path"] == (tmp_path / "models/sd_turbo").resolve()
    assert resolved["taesd_path"] == (tmp_path / "models/taesd").resolve()
    assert (resolved["diffusion_width"], resolved["diffusion_height"]) == (512, 512)


def test_next_level_wraps_from_the_nearest_step() -> None:
    levels = (0.0, 0.25, 0.5, 1.0)
    assert next_level(levels, 0.26) == 0.5
    assert next_level(levels, 1.0) == 0.0


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


def test_camera_far_plane_stays_inside_the_streamed_window() -> None:
    from app.renderer.world import ACTIVE_CHUNK_RADIUS, CHUNK_SIZE

    assert Camera.create_default().far < ACTIVE_CHUNK_RADIUS * CHUNK_SIZE


def test_camera_reset_restores_start() -> None:
    camera = Camera.create_default()
    start = camera.position.copy()
    camera.fly(1.0, 0.0, 1.0, 8.0)
    camera.rotate(100.0, -40.0, 0.15)
    camera.reset()
    assert np.allclose(camera.position, start)
    assert camera.yaw == 0.0
    assert camera.pitch == 0.0


def test_flying_forward_follows_the_full_look_direction() -> None:
    """Holding forward while looking up climbs: there is no ground to walk on."""

    for pitch in (-85.0, -40.0, 0.0, 60.0):
        camera = Camera.create_default()
        camera.pitch = pitch
        for _ in range(60):
            camera.fly(0.0, 0.0, 1.0, 3.0)
        assert np.allclose(camera.position, camera.forward * 180.0)
        assert np.isclose(np.linalg.norm(camera.position), 180.0)


def test_world_up_and_strafe_stay_level_whatever_the_pitch() -> None:
    """Q/E is a deliberate climb, so it must not depend on where the gaze points."""

    camera = Camera.create_default()
    camera.pitch = -70.0
    camera.yaw = 33.0
    camera.fly(0.0, 1.0, 0.0, 5.0)
    assert np.allclose(camera.position, (0.0, 5.0, 0.0))
    camera.fly(1.0, 0.0, 0.0, 4.0)
    assert np.isclose(camera.position[1], 5.0)


def test_camera_keeps_sub_unit_movement_at_large_world_coordinates() -> None:
    camera = Camera.create_default()
    camera.position[0] = 16_777_216.0
    camera.fly(1.0, 0.0, 0.0, 0.25)
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
    path.write_text(json.dumps({"prompt": "old", "backend": "latent_walk"}), encoding="utf-8")
    persist_prompt(path, "new painted city")
    updated = json.loads(path.read_text(encoding="utf-8"))
    assert updated == {"prompt": "new painted city", "backend": "latent_walk"}


def test_persist_runtime_setting(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"reprojection_strength": 0.5, "steps": 1}), encoding="utf-8")
    persist_config_value(path, "reprojection_strength", 0.3)
    updated = json.loads(path.read_text(encoding="utf-8"))
    assert updated == {"reprojection_strength": 0.3, "steps": 1}


def _library(tmp_path: Path) -> Path:
    path = tmp_path / "prompts.json"
    path.write_text(
        json.dumps(
            {
                "master_prefix": "studio",
                "families": [
                    {
                        "name": "Alpha",
                        "base": "matte stone",
                        "variants": ["a ridge", "a cliff"],
                        "settings": {"timestep_min": 500, "timestep_max": 600},
                    },
                    {
                        "name": "Beta",
                        "base": "wet chrome",
                        "variants": ["a tube", "a coil"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_families_expand_into_variants_carrying_their_settings(tmp_path: Path) -> None:
    entries = load_prompt_library(_library(tmp_path))

    assert [entry.family for entry in entries] == ["Alpha", "Alpha", "Beta", "Beta"]
    assert entries[0].prompt == "studio, a ridge, matte stone"
    assert entries[0].settings == {"timestep_min": 500, "timestep_max": 600}
    # A family without an override must not inherit the previous family's.
    assert entries[2].settings == {}


def test_a_legacy_flat_prompt_list_still_loads(tmp_path: Path) -> None:
    path = tmp_path / "prompts.json"
    path.write_text(
        json.dumps({"prompts": [{"name": "First", "prompt": "first prompt"}]}), encoding="utf-8"
    )
    entries = load_prompt_library(path)

    assert entries == [PromptEntry(family="First", prompt="first prompt", settings={})]


def test_unknown_family_settings_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "prompts.json"
    path.write_text(
        json.dumps(
            {
                "families": [
                    {"name": "A", "base": "b", "variants": ["v"], "settings": {"nope": 1}}
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="nope"):
        load_prompt_library(path)


def test_space_and_auto_advance_leave_the_current_family(tmp_path: Path) -> None:
    """Space has to change the whole look, so it may never return a sibling."""
    entries = load_prompt_library(_library(tmp_path))
    alpha = entries[0]

    assert all(
        choose_family_prompt(entries, "Alpha").family == "Beta" for _ in range(20)
    )
    # A list of recent families is excluded as a whole; falling back to the
    # full library beats returning nothing when everything is recent.
    assert all(
        choose_family_prompt(entries, ["Beta"]).family == "Alpha" for _ in range(20)
    )
    assert choose_family_prompt(entries, ["Alpha", "Beta"]) in entries
    assert all(advance_prompt(entries, alpha).family == "Beta" for _ in range(20))
    assert advance_prompt(entries, alpha, ["Beta", "Alpha"]).family == "Beta"


def test_entry_for_prompt_falls_back_to_a_settings_free_custom_entry(tmp_path: Path) -> None:
    entries = load_prompt_library(_library(tmp_path))

    assert entry_for_prompt(entries, entries[1].prompt) is entries[1]
    assert entry_for_prompt(entries, "hand typed") == PromptEntry(
        family="Custom", prompt="hand typed", settings={}
    )


def test_master_prefix_is_applied_once() -> None:
    prefix = "Photoreal abstract rendering, highly detailed"
    prompt = "alien filament city"
    combined = apply_master_prefix(prefix, prompt)
    assert combined == f"{prefix}, {prompt}"
    assert apply_master_prefix(prefix, combined) == combined


def test_hue_words_come_from_the_world_label() -> None:
    assert hue_words("shards+voxels / violet-lime-cyan") == "violet and lime"
    assert hue_words("dunes / amber") == "amber"
    assert hue_words("") == ""


def test_hue_words_rotate_through_the_world_accents() -> None:
    """A long walk has to change colour, and each world carries three accents."""
    label = "shards+voxels / violet-lime-cyan"
    pairs = [hue_words(label, offset) for offset in range(4)]
    assert pairs == ["violet and lime", "lime and cyan", "cyan and violet", "violet and lime"]
    assert hue_words("dunes / amber", 3) == "amber"


def test_compose_prompt_appends_the_world_hue_once() -> None:
    """Local hues follow the original prompt and never stack."""
    composed = compose_prompt(
        "a voxel landscape, corrupted 3D render", "shards+voxels / violet-lime-cyan"
    )
    assert composed == "a voxel landscape, corrupted 3D render, violet and lime"
    # Idempotent, so auto-advance and Space cannot stack hues onto one prompt.
    assert compose_prompt(composed, "shards+voxels / violet-lime-cyan") == composed
    assert compose_prompt("a voxel landscape", "shards+voxels") == "a voxel landscape"
    assert compose_prompt("a voxel landscape", "shards / amber") == "a voxel landscape, amber"


def test_world_instance_packing_preserves_transform_columns_and_color() -> None:
    from app.renderer.world import WorldCube

    item = WorldCube(
        role="panel",
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
    from app.renderer.world import (
        CHUNK_SIZE,
        MAX_ACTIVE_CHUNKS,
        MAX_OBJECTS_PER_CHUNK,
    )

    class FakeBuffer:
        size = 19 * 4

        def orphan(self, size: int) -> None:
            self.size = size

        def write(self, data, offset=0) -> None:
            assert offset + data.nbytes <= self.size

    renderer = ProxyRenderer.__new__(ProxyRenderer)
    renderer.world_seed = 12345
    renderer._chunk_instances = {}
    renderer._chunk_form_instances = {}
    renderer._form_instances = {}
    renderer._form_bounds = {}
    renderer._chunk_colliders = {}
    renderer._stream_center = None
    renderer.instance_buffers = {"cube": FakeBuffer()}
    renderer._instance_counts = {"cube": 0}
    renderer._instance_buffer_revision = 0

    origin = np.array([0.0, 1.65, 0.0], dtype=np.float32)
    renderer._update_world(origin)
    original_chunk = renderer._chunk_instances[(0, 0, 0)].copy()
    initial_revision = renderer._instance_buffer_revision
    renderer._update_world(origin)
    assert renderer._instance_buffer_revision == initial_revision

    for step in range(1, 12):
        renderer._update_world(
            np.array([0.0, CHUNK_SIZE * step, 0.0], dtype=np.float64)
        )
        assert len(renderer._chunk_instances) <= MAX_ACTIVE_CHUNKS
        assert renderer._chunk_colliders.keys() == renderer._chunk_instances.keys()
        assert renderer._instance_counts["cube"] <= (
            MAX_ACTIVE_CHUNKS * MAX_OBJECTS_PER_CHUNK
        )

    assert (0, 0, 0) not in renderer._chunk_instances
    renderer._update_world(origin)
    np.testing.assert_array_equal(renderer._chunk_instances[(0, 0, 0)], original_chunk)

