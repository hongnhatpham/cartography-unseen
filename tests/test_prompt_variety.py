"""Prompt subjects and palette cues must change as the experience progresses."""
from app import main
from app.renderer.world import world_label


def test_subject_changes_vary_tuning_while_travel_only_changes_color(monkeypatch, tmp_path):
    from collections import defaultdict
    import json
    import sys
    import numpy as np
    import pygame
    from app.diffusion.worker import DiffusionWorker
    from app.renderer import proxy_renderer
    from app.types import ConditioningFrame

    entries = main.load_prompt_library(main.project_root() / "prompts.json")
    config = json.loads((main.project_root() / "config.json").read_text())
    config.update(prompt=entries[0].prompt, backend="proxy_passthrough", fullscreen=False,
                  debug_overlay=False, prompt_caption=False, reprojection=False,
                  prompt_auto_advance_seconds=24, prompt_walk_seconds=6,
                  autowalk_idle_seconds=0, random_seed_on_launch=False)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(sys, "argv", ["app", "--config", str(path)])
    monkeypatch.setattr(main.secrets, "choice", lambda values: values[0])
    variation = {"timestep_min": 120, "timestep_max": 240, "display_sharpen": 1.7,
                 "instability": .23, "guide_strength": .91}
    samples = []

    def sample():
        samples.append(tick)
        return variation.copy()

    monkeypatch.setattr(main, "random_prompt_settings", sample)
    tick = 0
    prompts, settings, reseeds = [], [], []

    class Clock:
        def get_time(self):
            return 16

        def tick(self, fps):
            nonlocal tick
            tick += 1

    class Renderer:
        def __init__(self, *args, **kwargs):
            self.sequence = 0
            self.reproject_ms = 0
            self.loading_image = np.zeros((2, 2, 3), dtype=np.uint8)

        def loading_screen(self, *args): pass
        def spawn_camera(self, camera): camera.position[:] = 0
        def constrain_camera(self, camera, dt): pass
        def update_world(self, position): pass
        def display(self, *args, **kwargs): pass
        def close(self): pass

        def read_input(self):
            return (0, 0), defaultdict(bool), (False, False, False)

        def poll_events(self):
            if tick == 10:
                return [pygame.event.Event(pygame.QUIT)]
            if tick in (2, 6):
                return [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE, mod=0)]
            return []

        def world_label(self):
            return "reefs / " + ("cyan-violet" if tick < 3 else "magenta-yellow" if tick < 7 else "blue-lime")

        def render_scene(self, camera):
            return camera.snapshot(1.)

        def capture_conditioning(self, snapshot, timestamp, **kwargs):
            self.sequence += 1
            return ConditioningFrame(self.loading_image, np.ones((2, 2)), None,
                                     snapshot, timestamp, self.sequence)

    monkeypatch.setattr(proxy_renderer, "ProxyRenderer", Renderer)
    monkeypatch.setattr(main, "perf_counter", lambda: 1 + tick * 8)
    monkeypatch.setattr(pygame.time, "Clock", Clock)
    monkeypatch.setattr(DiffusionWorker, "start", lambda self: None)
    monkeypatch.setattr(DiffusionWorker, "stop", lambda self: None)
    monkeypatch.setattr(DiffusionWorker, "publish", lambda *args: None)
    monkeypatch.setattr(DiffusionWorker, "request_settings", lambda self, **values: settings.append(values))
    monkeypatch.setattr(DiffusionWorker, "request_reseed", lambda *args: reseeds.append(True))

    def request_prompt(self, prompt, negative, *, settings=None):
        prompts.append((tick, prompt))
        if settings is not None:
            assert settings == {key: value for key, value in variation.items() if key != "display_sharpen"}
        return len(prompts)

    monkeypatch.setattr(DiffusionWorker, "request_prompt", request_prompt)
    assert main.run() == 0
    # Palette updates do not postpone auto-advance, repeat each frame, or keep
    # the previous region's suffix. Both ways of selecting a subject share history.
    assert [at for at, _ in prompts] == [2, 3, 5, 6, 7, 9]
    assert prompts[1][1].endswith("magenta and yellow")
    assert "cyan and violet" not in prompts[1][1]
    assert prompts[4][1].endswith("blue and lime")
    subjects = [next(entry.family for entry in entries if text.startswith(entry.prompt))
                for at, text in prompts if at in (2, 5, 6, 9)]
    for index, family in enumerate(subjects):
        assert family not in subjects[max(0, index - main.RECENT_FAMILY_MEMORY):index]
    assert settings == reseeds == []
    assert samples == [2, 5, 6, 9]
    persisted = json.loads(path.read_text())
    assert {key: value for key, value in persisted.items() if key != "prompt"} == {
        key: value for key, value in {**config, **variation}.items() if key != "prompt"}


def test_auto_advance_changes_family_even_when_random_choice_favors_siblings(monkeypatch):
    entries = main.load_prompt_library(main.project_root() / "prompts.json")
    monkeypatch.setattr(main.secrets, "randbelow", lambda count: count - 1)
    monkeypatch.setattr(main.secrets, "choice", lambda values: values[0])
    current = entries[0]
    selected = main.advance_prompt(entries, current)
    assert selected.family != current.family


def test_travel_changes_palette_cues_in_each_axis():
    for axis in range(3):
        labels = set()
        for distance in range(-768, 769, 128):
            point = [16., 16., 16.]
            point[axis] += distance
            labels.add(main.hue_words(world_label(934943880, *point)))
        assert len(labels) >= 4
