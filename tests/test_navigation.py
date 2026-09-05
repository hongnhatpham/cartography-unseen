"""Exercise movement through the real application loop and renderer."""

from collections import defaultdict
from dataclasses import asdict
import json
import sys

import numpy as np
import pygame
import pytest

from app import main
from app.renderer.camera import Camera
from app.renderer.world import Autowalk


@pytest.mark.parametrize("trigger, view_key", [
    ("space", None), ("space", pygame.K_F2), ("space", pygame.K_F3),
    ("auto", None),
])
def test_prompt_selection_preserves_live_experience(monkeypatch, tmp_path, trigger, view_key):
    """Run real key handling and rendering with one deliberately slow AI frame."""
    from app.diffusion.worker import DiffusionWorker
    from app.renderer.proxy_renderer import ProxyRenderer
    from app.types import GeneratedFrame

    config = json.loads((main.project_root() / "config.json").read_text(encoding="utf-8"))
    config.update(backend="proxy_passthrough", fullscreen=False, debug_overlay=True,
                  reprojection=False, random_seed_on_launch=False,
                  prompt_auto_advance_seconds=30 if trigger == "auto" else 0,
                  autowalk_idle_seconds=0)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["app", "--config", str(path), "--windowed"])
    monkeypatch.setattr(pygame.mouse, "get_rel", lambda: (0, 0))
    monkeypatch.setattr(pygame.key, "get_pressed", lambda: defaultdict(bool))
    chosen = main.PromptEntry("Next", "a different imagined passage", {
        "guidance_scale": 3.0, "guide_strength": 0.1, "timestep_min": 1,
    })
    initial_prompt = main.apply_master_prefix(
        main.load_master_prefix(main.project_root() / "prompts.json"), config["prompt"]
    )
    monkeypatch.setattr(main, "load_prompt_library", lambda path: [
        main.PromptEntry("Current", initial_prompt, {}), chosen,
    ])
    monkeypatch.setattr(DiffusionWorker, "start", lambda self: None)
    monkeypatch.setattr(DiffusionWorker, "stop", lambda self: None)
    frame = -1
    worker = None
    states = []
    poses = []
    images = []
    settings = []
    reseeds = []
    prompts = []
    clock = main.perf_counter
    monkeypatch.setattr(main, "perf_counter", lambda: clock() + (60 if trigger == "auto" and frame >= 3 else 0))

    def publish_once(self, conditioning):
        nonlocal worker
        worker = self
        if self.generated.get()[1] is None:
            camera = conditioning.camera
            self.generated.publish(GeneratedFrame(
                np.full_like(conditioning.rgb, 123), conditioning.depth,
                camera.view_matrix, camera.projection_matrix, camera.position,
                camera.rotation, conditioning.timestamp, conditioning.timestamp,
                conditioning.sequence, {"prompt_revision": 0},
            ))
        self._set_status(state="ready")

    request_settings = DiffusionWorker.request_settings
    request_reseed = DiffusionWorker.request_reseed
    request_prompt = DiffusionWorker.request_prompt
    constrain_camera = ProxyRenderer.constrain_camera
    display = ProxyRenderer.display
    build_overlay = main.build_overlay

    def record_settings(self, **values):
        settings.append((frame, values))
        request_settings(self, **values)

    def record_reseed(self, seed=None):
        reseeds.append(frame)
        return request_reseed(self, seed)

    def record_prompt(self, prompt, negative=""):
        prompts.append((frame, prompt, negative))
        return request_prompt(self, prompt, negative)

    def record_pose(self, camera, dt=None):
        result = constrain_camera(self, camera, dt)
        if frame >= 0:
            poses.append((camera.position.copy(), camera.yaw, camera.pitch, self.world_label()))
        return result

    def record_display(self, image, *args, **kwargs):
        images.append(image.copy())
        return display(self, image, *args, **kwargs)

    def record_overlay(*args, **kwargs):
        states.append((asdict(args[8]), args[6], kwargs["hue_offset"]))
        return build_overlay(*args, **kwargs)

    def events():
        nonlocal frame
        frame += 1
        keys = []
        if frame == 0:
            keys = [pygame.K_c, pygame.K_l, pygame.K_y]
        elif frame == 1 and view_key is not None:
            keys = [view_key]
        elif frame == 3 and trigger == "space":
            keys = [pygame.K_SPACE]
        elif frame >= 5:
            return [pygame.event.Event(pygame.QUIT)]
        return [pygame.event.Event(pygame.KEYDOWN, key=key, mod=0) for key in keys]

    monkeypatch.setattr(DiffusionWorker, "publish", publish_once)
    monkeypatch.setattr(DiffusionWorker, "request_settings", record_settings)
    monkeypatch.setattr(DiffusionWorker, "request_reseed", record_reseed)
    monkeypatch.setattr(DiffusionWorker, "request_prompt", record_prompt)
    monkeypatch.setattr(ProxyRenderer, "constrain_camera", record_pose)
    monkeypatch.setattr(ProxyRenderer, "display", record_display)
    monkeypatch.setattr(main, "build_overlay", record_overlay)
    monkeypatch.setattr(pygame.event, "get", events)
    assert main.run() == 0
    before, view, hue = states[2]
    after, next_view, next_hue = states[4]
    assert before["prompt"] != chosen.prompt
    assert after == {**before, "prompt": chosen.prompt}
    assert settings and all(at == 0 for at, _ in settings)
    assert reseeds == []
    assert prompts == [(3, main.compose_prompt(chosen.prompt, poses[2][3], hue), config["negative_prompt"])]
    assert worker._requested_seed == config["seed"]
    assert np.array_equal(poses[2][0], poses[4][0])
    assert poses[2][1:] == poses[4][1:]
    assert (next_view, next_hue) == (view, hue)
    assert np.array_equal(images[2], images[4])
    persisted = json.loads(path.read_text(encoding="utf-8"))
    expected = {**before, "prompt": chosen.prompt if trigger == "space" else before["prompt"]}
    assert all(persisted[key] == expected[key] for key in config)


@pytest.mark.parametrize("idle", [False, True])
def test_application_accepts_manual_and_idle_flight(monkeypatch, tmp_path, idle):
    config = json.loads((main.project_root() / "config.json").read_text(encoding="utf-8"))
    config.update(backend="proxy_passthrough", fullscreen=False, debug_overlay=False,
                  prompt_auto_advance_seconds=0, autowalk_idle_seconds=0.001 if idle else 0)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["app", "--config", str(path), "--windowed"])
    monkeypatch.setattr(pygame.mouse, "get_rel", lambda: (0, 0))
    frame = -1
    calls = []
    progress = []
    motion_before = None
    fly = Camera.fly
    drift = Autowalk.step

    def record_flight(camera, x, y, z, distance):
        nonlocal motion_before
        camera.pitch = 60.0 if idle or frame < 18 else -60.0
        before = camera.position.copy()
        motion_before = before
        fly(camera, x, y, z, distance)
        calls.append((frame, (x, y, z), distance, camera.position.copy() - before))

    def record_drift(self, *args, **kwargs):
        progress.append(args[5])  # gained distance after collision
        assert args[5] == pytest.approx(np.linalg.norm(np.asarray(args[2]) - motion_before))
        return drift(self, *args, **kwargs)

    class KeyState(defaultdict):
        def __iter__(self):
            return iter(self.values())

    def keys():
        nonlocal frame
        frame += 1
        pressed = KeyState(bool)
        if not idle:
            key = (pygame.K_e, pygame.K_q, pygame.K_w, pygame.K_w, None, pygame.K_d)[min(frame // 6, 5)]
            if key is not None:
                pressed[key] = True
            if frame >= 30:
                pressed[pygame.K_LSHIFT] = True
        return pressed

    monkeypatch.setattr(Camera, "fly", record_flight)
    monkeypatch.setattr(Autowalk, "step", record_drift)
    monkeypatch.setattr(pygame.key, "get_pressed", keys)
    monkeypatch.setattr(pygame.event, "get", lambda: [pygame.event.Event(pygame.QUIT)] if frame >= 35 else [])
    assert main.run() == 0
    assert calls
    if idle:
        assert progress, "Idle input must reach the three-dimensional autopilot"
        assert any(gained > 0 for gained in progress)
        assert any(delta[1] > 0 for _, _, _, delta in calls)
    else:
        expected = ((0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, 1), (0, 0, 0), (1, 0, 0))
        for group, direction in enumerate(expected):
            rows = [row for row in calls if group * 6 <= row[0] < (group + 1) * 6]
            assert rows and all(row[1] == direction for row in rows)
            if group < 4:
                sign = 1 if group in (0, 2) else -1
                assert any(row[3][1] * sign > 0 for row in rows)
            elif group == 4:
                assert all(np.array_equal(row[3], np.zeros(3)) for row in rows)
