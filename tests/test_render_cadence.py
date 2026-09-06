"""Keep display-rate flight while drawing conditioning only when it is consumed."""

from collections import defaultdict
import json
import sys

import numpy as np
import pygame
import pytest

from app import main
from app.types import ConditioningFrame, GeneratedFrame


@pytest.mark.parametrize("backend", ["proxy_passthrough", "latent_walk"])
@pytest.mark.parametrize("view_key, key_presses", [
    (None, 0), (pygame.K_F2, 1), (pygame.K_F3, 1),
    (pygame.K_F3, 2), (pygame.K_F5, 1), (pygame.K_v, 1), (pygame.K_RETURN, 1),
])
def test_proxy_work_follows_consumers_without_slowing_flight(
    monkeypatch, tmp_path, backend, view_key, key_presses,
):
    """Drive 64 real app ticks with deterministic time and instrumented IO."""
    from app.diffusion.worker import DiffusionWorker
    from app.renderer import proxy_renderer

    config = json.loads((main.project_root() / "config.json").read_text(encoding="utf-8"))
    config.update(journey_map=False, backend=backend, fullscreen=False, reprojection=False,
                  target_display_fps=64, conditioning_fps=16, debug_overlay=False,
                  prompt_caption=False, prompt_auto_advance_seconds=0,
                  autowalk_idle_seconds=0, player_trail=True)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["app", "--config", str(path)])
    tick = 0
    captures = []
    displayed = []
    displayed_trails = []
    streamed = []
    collisions = []
    renders = []
    warps = []
    screenshot_ticks = []

    class Clock:
        def get_time(self):
            return 1000 / 64

        def tick(self, fps):
            nonlocal tick
            assert fps == 64
            tick += 1

    class Renderer:
        def set_operator_mode(self, enabled): pass

        def __init__(self, *args, **kwargs):
            self.sequence = 0
            self.reproject_ms = 0
            self.loading_image = np.zeros((2, 2, 3), dtype=np.uint8)

        def loading_screen(self, *args):
            pass

        def read_input(self):
            return (0, 0), defaultdict(bool, {pygame.K_e: True}), (False, False, False)

        def spawn_camera(self, camera):
            camera.position[:] = 0

        def poll_events(self):
            if tick == 63:
                return [pygame.event.Event(pygame.QUIT)]
            if tick == 1 and view_key is not None:
                unlock = [] if view_key == pygame.K_RETURN else [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F1, mod=0)]
                return unlock + [
                    pygame.event.Event(pygame.KEYDOWN, key=view_key, mod=0)
                    for _ in range(key_presses)
                ]
            return []

        def constrain_camera(self, camera, dt):
            collisions.append(camera.position.copy())

        def update_world(self, position):
            streamed.append((tick, position.copy()))

        def world_label(self):
            return "passage / cyan-lime"

        def render_scene(self, camera, *, manage_chunks=True):
            if manage_chunks:
                self.update_world(camera.position)
                renders.append(tick)
            else:
                warps.append(tick)
            return camera.snapshot(1.0)

        def capture_conditioning(self, snapshot, timestamp, *, include_edges=True):
            self.sequence += 1
            frame = ConditioningFrame(
                np.full((2, 2, 3), self.sequence, dtype=np.uint8),
                np.ones((2, 2), dtype=np.float32),
                np.zeros((2, 2), dtype=np.uint8) if include_edges else None,
                snapshot, timestamp, self.sequence,
            )
            captures.append((tick, frame))
            return frame

        def diagnostic_image(self, conditioning, mode):
            if mode == "edges":
                assert conditioning.edges is not None
            return conditioning.rgb

        def display(self, image, *args, **kwargs):
            displayed.append((tick, image.copy()))
            if kwargs.get("screenshot") is not None:
                screenshot_ticks.append(tick)
            displayed_trails.append(kwargs.get("trail"))

        def display_reprojected(self, frame, camera, *args, **kwargs):
            # The real renderer draws its own warped camera each display tick.
            assert np.array_equal(streamed[-1][1], camera.position)
            self.render_scene(camera, manage_chunks=False)
            self.display(frame.image)

        def close(self):
            pass

    def publish(self, conditioning):
        camera = conditioning.camera
        self.generated.publish(GeneratedFrame(
            conditioning.rgb, conditioning.depth, camera.view_matrix,
            camera.projection_matrix, camera.position, camera.rotation,
            conditioning.timestamp, conditioning.timestamp, conditioning.sequence,
            {"prompt_revision": 0},
        ))
        self._set_status(state="ready")

    monkeypatch.setattr(proxy_renderer, "ProxyRenderer", Renderer)
    monkeypatch.setattr(main, "perf_counter", lambda: 1 + tick / 64)
    monkeypatch.setattr(pygame.time, "Clock", Clock)
    monkeypatch.setattr(DiffusionWorker, "start", lambda self: None)
    monkeypatch.setattr(DiffusionWorker, "stop", lambda self: None)
    monkeypatch.setattr(DiffusionWorker, "publish", publish)

    assert main.run() == 0
    expected_ticks = list(range(64)) if view_key in (pygame.K_F2, pygame.K_F3) else list(range(0, 64, 4))
    assert [at for at, _ in captures] == expected_ticks
    assert renders == expected_ticks
    assert [at for at, _ in streamed] == list(range(64))
    assert len(collisions) == len(displayed) == 64
    assert screenshot_ticks == ([1] if view_key == pygame.K_RETURN else [])
    if view_key == pygame.K_v:
        assert displayed_trails[0] is not None
        assert all(trail is None for trail in displayed_trails[1:])
        assert json.loads(path.read_text())["player_trail"] is False
    assert np.all(np.diff([position[1] for position in collisions]) > 0)
    for at, frame in captures:
        assert np.array_equal(frame.camera.position, collisions[at])
        expected_edges = backend != "latent_walk" or (key_presses == 2 and at >= 1)
        assert (frame.edges is not None) == expected_edges
    assert warps == (list(range(1, 64)) if view_key == pygame.K_F5 else [])
    for at, image in displayed:
        latest_capture = next(frame for captured_at, frame in reversed(captures) if captured_at <= at)
        assert np.array_equal(image, latest_capture.rgb)
