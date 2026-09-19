"""Visitor instructions follow physical activity, independent of automatic motion."""

from collections import defaultdict
import json
import sys

import numpy as np
import pygame
import pytest

from app import main
from app.idle_instructions import IdleInstructions
from app.types import ConditioningFrame, GeneratedFrame


def test_idle_delay_and_fades_use_elapsed_time():
    instructions = IdleInstructions(100.)
    sample = lambda now, active=False: instructions.update(now, active=active, suppressed=False)
    assert not instructions.showing
    assert sample(109.9) == 0.
    assert sample(110.) == 0.
    assert instructions.showing
    assert sample(110.6) == pytest.approx(.5)
    assert sample(111.2) == pytest.approx(1.)
    assert sample(115., active=True) == pytest.approx(1.)
    assert not instructions.showing
    assert sample(115.125, active=True) == pytest.approx(.5)
    assert sample(115.25, active=True) == 0.
    assert sample(125.25) == 0.
    assert sample(126.45) == pytest.approx(1.)


def test_panels_hide_immediately_without_counting_as_physical_activity():
    instructions = IdleInstructions(0.)
    assert instructions.update(12., active=False, suppressed=False) == 1.
    assert instructions.update(13., active=False, suppressed=True) == 0.
    assert instructions.update(17., active=False, suppressed=True) == 0.
    assert instructions.last_input == 0.
    assert instructions.update(17.6, active=False, suppressed=False) == pytest.approx(.5)
    assert instructions.update(18.2, active=False, suppressed=False) == pytest.approx(1.)


@pytest.mark.parametrize("reprojection", [False, True])
@pytest.mark.parametrize("activity", [pygame.KEYUP, pygame.TEXTINPUT, pygame.MOUSEWHEEL])
def test_main_idle_controls_cover_both_presentations(monkeypatch, tmp_path, reprojection, activity):
    from app.diffusion.worker import DiffusionWorker
    from app.renderer import proxy_renderer

    config = json.loads((main.project_root() / "config.json").read_text(encoding="utf-8"))
    config.update(journey_map=False, backend="proxy_passthrough", fullscreen=False, debug_overlay=False,
                  reprojection=reprojection, prompt_caption=False, autowalk_idle_seconds=2,
                  prompt_auto_advance_seconds=3, random_seed_on_launch=False)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["app", "--config", str(path)])
    times = (0., 9.9, 10., 10.6, 11.2, 12., 12.125, 12.25,
             22., 22.6, 23.2, 24., 35.2, 36., 47.2, 48., 59.2)
    tick = 0
    revision = 0
    opacities = []
    presentations = []
    positions = []

    class Keys(defaultdict):
        def __iter__(self):
            return iter(self.values())

    class Clock:
        def get_time(self): return 16

        def tick(self, fps):
            nonlocal tick
            tick += 1

    class Renderer:
        def set_operator_mode(self, enabled): pass

        def __init__(self, *args, **kwargs):
            self.sequence = 0
            self.reproject_ms = 0.
            self.loading_image = np.zeros((2, 2, 3), dtype=np.uint8)

        def loading_screen(self, *args): pass
        def spawn_camera(self, camera): camera.position[:] = 0.
        def update_world(self, position): pass
        def constrain_camera(self, camera, dt): positions.append(camera.position.copy())
        def nearby_colliders(self, camera): return np.empty((0, 6))
        def close(self): pass
        def world_label(self): return "terrain / cyan-lime"

        def read_input(self):
            return (0, 0), Keys(bool, {pygame.K_q: tick == 11}), (tick == 13, False, False)

        def poll_events(self):
            if tick == 5:
                return [pygame.event.Event(activity, key=pygame.K_c, text="c", x=0, y=1)]
            if tick == 15:
                return [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F1, mod=0)]
            if tick == len(times) - 1:
                return [pygame.event.Event(pygame.QUIT)]
            return []

        def render_scene(self, camera): return camera.snapshot(1.)

        def capture_conditioning(self, snapshot, timestamp, **kwargs):
            self.sequence += 1
            return ConditioningFrame(self.loading_image, np.ones((2, 2)), None,
                                     snapshot, timestamp, self.sequence)

        def display(self, image, *args, **kwargs):
            opacities.append(kwargs["idle_opacity"])
            presentations.append("raw")

        def display_reprojected(self, frame, camera, *args, **kwargs):
            opacities.append(kwargs["idle_opacity"])
            presentations.append("reprojected")

    def publish(self, conditioning):
        camera = conditioning.camera
        self.generated.publish(GeneratedFrame(
            conditioning.rgb, conditioning.depth, camera.view_matrix,
            camera.projection_matrix, camera.position, camera.rotation,
            conditioning.timestamp, conditioning.timestamp, conditioning.sequence,
            {"prompt_revision": revision},
        ))
        self._set_status(state="ready")

    def request_prompt(self, prompt, negative, *, settings=None):
        nonlocal revision
        revision += 1
        return revision

    monkeypatch.setattr(proxy_renderer, "ProxyRenderer", Renderer)
    monkeypatch.setattr(main, "perf_counter", lambda: 100. + times[tick])
    monkeypatch.setattr(pygame.time, "Clock", Clock)
    monkeypatch.setattr(DiffusionWorker, "start", lambda self: None)
    monkeypatch.setattr(DiffusionWorker, "stop", lambda self: None)
    monkeypatch.setattr(DiffusionWorker, "publish", publish)
    monkeypatch.setattr(DiffusionWorker, "request_prompt", request_prompt)

    assert main.run() == 0
    assert opacities == pytest.approx([
        0., 0., 0., .5, 1., 1., .5, 0., 0., .5, 1., 1., 1., 1., 1., 0., 0.,
    ])
    assert revision > 4  # Timed prompt changes never count as visitor input.
    assert not np.array_equal(positions[0], positions[4])  # Autowalk remains visible behind instructions.
    assert set(presentations) == {"reprojected" if reprojection else "raw"}
