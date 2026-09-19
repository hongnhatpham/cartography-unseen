"""Main-loop exhibition controls must preserve maps through reset and save errors."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import sys
from types import SimpleNamespace

import numpy as np
import pygame
import pytest

from app import journey as archive, journey_session, main, map_view
from app.diffusion.worker import DiffusionWorker
from app.renderer import proxy_renderer
from app.types import ConditioningFrame, GeneratedFrame


def key(value, *, repeat=False, released=False):
    return pygame.event.Event(pygame.KEYUP if released else pygame.KEYDOWN,
                              key=value, mod=0, repeat=repeat)


@pytest.fixture
def exhibition(monkeypatch, tmp_path):
    # Local save-failure patches must run in this process alongside the scripted UI.
    monkeypatch.setattr(archive, "ProcessPoolExecutor",
                        lambda **kwargs: ThreadPoolExecutor(max_workers=kwargs["max_workers"]))
    source = main.project_root()
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    config.update(journey_map=True, backend="proxy_passthrough", fullscreen=True,
                  debug_overlay=False, reprojection=False, prompt_caption=False,
                  autowalk_idle_seconds=0, prompt_auto_advance_seconds=0,
                  random_seed_on_launch=False, map_capture_distance=.1, map_export_svg=True,
                  map_sync_enabled=False)
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "prompts.json").write_bytes((source / "prompts.json").read_bytes())
    monkeypatch.setattr(main, "project_root", lambda: tmp_path)
    monkeypatch.setattr(main, "configure_local_environment", lambda *a, **k: None)
    monkeypatch.setattr(main, "configure_logging", lambda *a, **k: tmp_path / "test.log")
    monkeypatch.setattr(sys, "argv", ["app"])
    state = SimpleNamespace(tick=0, events={}, displays=[], windows=[], renderer=None,
                            moving=True, close_ticks=[], focus_ticks=[], root=tmp_path,
                            main_display=0, map_display=1)
    monkeypatch.setattr(main, "perf_counter", lambda: 10. + state.tick / 10.)
    monkeypatch.setattr(journey_session, "perf_counter", lambda: 10. + state.tick / 10.)

    class Clock:
        def get_time(self): return 100
        def tick(self, fps): state.tick += 1

    class Window:
        def __init__(self, root, **options):
            self.fullscreen = []
            self.operator = []
            self.updates = []
            self.closed = False
            self.options = options
            self.window_status = {"display": state.map_display,
                                  "fullscreen": bool(options.get("fullscreen", False))}
            self.ready = True
            state.windows.append(self)

        def set_fullscreen(self, enabled):
            self.fullscreen.append((state.tick, enabled))
            self.window_status["fullscreen"] = enabled
            self.ready = True
        def set_operator_mode(self, enabled): self.operator.append((state.tick, enabled))

        def update(self, *args): self.updates.append(args)
        def poll(self):
            if self.ready:
                self.ready = False
                return ['__map_display_ready__']
            return []
        def close(self): self.closed = True

    class Renderer:
        def __init__(self, root, resolution, fullscreen, **kwargs):
            self.sequence = 0
            self.reproject_ms = 0.
            self.loading_image = np.zeros((2, 2, 3), dtype=np.uint8)
            self.is_fullscreen = fullscreen
            self.initial_fullscreen = fullscreen
            self.fullscreen = []
            self.operator = []
            state.renderer = self

        def loading_screen(self, *args): pass
        def spawn_camera(self, camera): camera.position[:] = 0.
        def update_world(self, position): pass
        def world_label(self): return "terrain / cyan-lime"
        def constrain_camera(self, camera, dt): pass
        def set_prompt_editing(self, enabled): pass
        def close(self): state.close_ticks.append(state.tick)
        def focus(self): state.focus_ticks.append(state.tick)
        def set_operator_mode(self, enabled): self.operator.append((state.tick, enabled))

        @property
        def window_status(self):
            return {"display": state.main_display, "fullscreen": self.is_fullscreen}

        def set_fullscreen(self, enabled):
            self.is_fullscreen = enabled
            self.fullscreen.append((state.tick, enabled))
            return enabled

        def poll_events(self):
            assert state.tick < 20, "Application failed to exit after scripted save retry"
            return state.events.get(state.tick, [])

        def read_input(self):
            return (0, 0), defaultdict(bool, {pygame.K_w: state.moving}), (False, False, False)

        def render_scene(self, camera): return camera.snapshot(1.)

        def capture_conditioning(self, camera, timestamp, **kwargs):
            self.sequence += 1
            return ConditioningFrame(self.loading_image, np.ones((2, 2)), None,
                                     camera, timestamp, self.sequence)

        def display(self, image, overlay_lines=None, **kwargs):
            state.displays.append((state.tick, overlay_lines or []))

    def publish(worker, frame):
        worker.generated.publish(GeneratedFrame(
            frame.rgb, frame.depth, frame.camera.view_matrix, frame.camera.projection_matrix,
            frame.camera.position, frame.camera.rotation, frame.timestamp + .01,
            frame.timestamp, frame.sequence, {"prompt_revision": worker._prompt_revision}))

    monkeypatch.setattr(map_view, "MapWindow", Window)
    monkeypatch.setattr(proxy_renderer, "ProxyRenderer", Renderer)
    monkeypatch.setattr(pygame.time, "Clock", Clock)
    monkeypatch.setattr(DiffusionWorker, "start", lambda self: None)
    monkeypatch.setattr(DiffusionWorker, "stop", lambda self: None)
    monkeypatch.setattr(DiffusionWorker, "publish", publish)
    return state


def manifests(root):
    return [json.loads(path.read_text(encoding="utf-8"))
            for path in root.glob("journeys/*/manifest.json")]


def test_f_requires_overlay_and_space_hold_archives_once_then_quit_saves(exhibition):
    state = exhibition
    state.events = {
        0: [key(pygame.K_F1), key(pygame.K_f)],
        1: [key(pygame.K_F1), key(pygame.K_f), key(pygame.K_f, repeat=True)],
        2: [key(pygame.K_f), key(pygame.K_SPACE), key(pygame.K_SPACE)],
        3: [key(pygame.K_SPACE), key(pygame.K_SPACE, repeat=True)],
        4: [key(pygame.K_SPACE, released=True), key(pygame.K_ESCAPE)],
    }
    assert main.run() == 0
    assert state.renderer.initial_fullscreen is False
    assert state.focus_ticks == [0]
    expected_modes = [(1, True), (2, False)]
    assert state.renderer.fullscreen == expected_modes
    assert state.windows[0].fullscreen == expected_modes
    assert state.windows[0].closed
    archived = manifests(state.root)
    assert len(archived) == 2
    previous = next(item for item in archived if item["completion_reason"] == "space")
    current = next(item for item in archived if item["completion_reason"] == "quit")
    assert previous["segments"] and current["segments"]
    assert previous["prompts"][0]["trigger"] == "initial"
    assert [event["trigger"] for event in current["prompts"]] == ["space"]
    assert current["prompts"][0]["prompt"] != previous["prompts"][0]["prompt"]
    assert current["preceding_archive"] == previous["id"]
    assert all((state.root / "journeys" / item["id"] / "map.svg").exists() for item in archived)


def test_placement_fullscreen_persists_both_selected_monitors(exhibition):
    state = exhibition
    state.main_display = 2
    state.map_display = 3
    state.events = {0: [key(pygame.K_f)], 1: [key(pygame.K_ESCAPE)]}
    assert main.run() == 0
    config = json.loads((state.root / "config.json").read_text(encoding="utf-8"))
    assert config["fullscreen"] is True
    assert config["display_monitor"] == 2
    assert config["map_display_monitor"] == 3


def test_configured_displays_start_both_fullscreen_with_overlay_closed(exhibition):
    state = exhibition
    config_path = state.root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(display_monitor=0, map_display_monitor=1, fullscreen=True,
                  debug_overlay=True)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    state.events = {0: [key(pygame.K_ESCAPE)]}
    assert main.run() == 0
    assert state.renderer.initial_fullscreen is True
    assert state.renderer.operator[0] == (0, False)
    assert state.windows[0].options == {"display_monitor": 1, "fullscreen": True}
    assert state.windows[0].operator[0] == (0, False)


def test_failed_quit_save_keeps_windows_open_and_retries_same_journey(exhibition, monkeypatch):
    state = exhibition
    state.events = {0: [key(pygame.K_F1)], 1: [key(pygame.K_ESCAPE)],
                    3: [key(pygame.K_ESCAPE)]}
    original = archive._export_svg
    attempts = []

    def failing_once(path, data):
        attempts.append((state.tick, data["id"]))
        if len(attempts) == 1:
            raise OSError("disk unavailable")
        return original(path, data)

    monkeypatch.setattr(archive, "_export_svg", failing_once)
    assert main.run() == 0
    assert len(attempts) == 2
    assert attempts[0][1] == attempts[1][1]
    assert state.close_ticks == [4]
    assert state.windows[0].closed
    assert any("MAP SAVE FAILED" in line for tick, lines in state.displays if tick == 2 for line in lines)
    assert any(tick == 2 and enabled for tick, enabled in state.renderer.operator)
    archived = manifests(state.root)
    assert len(archived) == 1
    assert archived[0]["completion_reason"] == "quit"


def test_failed_space_save_preserves_prompt_and_route_until_new_press(exhibition, monkeypatch):
    state = exhibition
    state.events = {1: [key(pygame.K_SPACE)], 2: [key(pygame.K_SPACE)],
                    3: [key(pygame.K_SPACE, released=True), key(pygame.K_SPACE)],
                    4: [key(pygame.K_ESCAPE)]}
    original = archive._export_svg
    attempts = []

    def failing_once(path, data):
        attempts.append((state.tick, data["id"], data["completion_reason"]))
        if len(attempts) == 1:
            raise OSError("disk unavailable")
        return original(path, data)

    monkeypatch.setattr(archive, "_export_svg", failing_once)
    assert main.run() == 0
    assert [tick for tick, _, _ in attempts] == [1, 3, 5]
    assert attempts[0][1] == attempts[1][1]
    archived = manifests(state.root)
    previous = next(item for item in archived if item["completion_reason"] == "space")
    current = next(item for item in archived if item["completion_reason"] == "quit")
    assert [event["trigger"] for event in previous["prompts"]] == ["initial"]
    assert [event["trigger"] for event in current["prompts"]] == ["space"]
    assert previous["segments"][0]["points"][-1]["timestamp"] >= 10.2
    assert any("NEW JOURNEY NOT STARTED" in line for tick, lines in state.displays if tick == 1 for line in lines)
