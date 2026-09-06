"""Only an explicitly opened F1 panel permits operator setting changes."""
from collections import defaultdict
import json
import sys

import numpy as np
import pygame

from app import main
from app.diffusion.worker import DiffusionWorker
from app.renderer import proxy_renderer
from app.types import ConditioningFrame


def test_f1_unlocks_settings_in_event_order_without_unlocking_for_notices(monkeypatch, tmp_path):
    config = json.loads((main.project_root() / "config.json").read_text(encoding="utf-8"))
    config.update(journey_map=False, backend="proxy_passthrough", fullscreen=False, debug_overlay=False,
                  reprojection=False, prompt_caption=False, autowalk_idle_seconds=0,
                  prompt_auto_advance_seconds=0, random_seed_on_launch=False,
                  player_trail=True, fog_distance=196.)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["app", "--config", str(path)])
    tick = 0
    variation = {"timestep_min": 120, "timestep_max": 240, "display_sharpen": 1.7,
                 "instability": .23, "guide_strength": .91}
    monkeypatch.setattr(main, "random_prompt_settings", lambda: variation.copy())
    writes, settings, prompts, edits, other_actions, poses, panels = [], [], [], [], [], [], []
    saved = []
    persist = main.persist_config_values

    def record_write(config_path, values):
        writes.append((tick, values.copy()))
        return persist(config_path, values)

    def key(value, mod=0):
        return pygame.event.Event(pygame.KEYDOWN, key=value, mod=mod)

    locked_keys = [
        pygame.K_F2, pygame.K_F3, pygame.K_F4, pygame.K_F5, pygame.K_F6,
        pygame.K_F7, pygame.K_F8, pygame.K_F9, pygame.K_F10, pygame.K_F11, pygame.K_F12,
        pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET, pygame.K_MINUS, pygame.K_KP_MINUS,
        pygame.K_EQUALS, pygame.K_KP_PLUS, pygame.K_COMMA, pygame.K_PERIOD,
        pygame.K_i, pygame.K_o, pygame.K_h, pygame.K_j, pygame.K_v, pygame.K_k,
        pygame.K_l, pygame.K_t, pygame.K_y, pygame.K_n, pygame.K_m, pygame.K_c,
    ]

    def blocked():
        return [key(value) for value in locked_keys] + [
            key(pygame.K_r, pygame.KMOD_SHIFT), key(pygame.K_p), key(pygame.K_RETURN),
        ]

    class Clock:
        def get_time(self):
            return 16

        def tick(self, fps):
            nonlocal tick
            saved.append(json.loads(path.read_text(encoding="utf-8")))
            tick += 1

    class Renderer:
        def set_operator_mode(self, enabled): pass

        def __init__(self, *args, **kwargs):
            self.sequence = 0
            self.reproject_ms = 0
            self.loading_image = np.zeros((2, 2, 3), dtype=np.uint8)

        def loading_screen(self, *args): pass
        def spawn_camera(self, camera): camera.position[:] = 0.
        def update_world(self, position): pass
        def close(self): pass
        def world_label(self): return "terrain / cyan-lime"

        def read_input(self):
            return (2, -1), defaultdict(bool, {pygame.K_w: True, pygame.K_e: True}), (False, False, False)

        def set_prompt_editing(self, editing):
            edits.append((tick, "start" if editing else "stop"))

        def poll_events(self):
            if tick == 0:
                return blocked()
            if tick == 1:
                # Open, edit, close and attempt more edits in one event batch.
                return [key(value) for value in (
                    pygame.K_F1, pygame.K_c, pygame.K_h, pygame.K_v,
                    pygame.K_p, pygame.K_ESCAPE, pygame.K_F1,
                    pygame.K_c, pygame.K_h, pygame.K_v, pygame.K_p, pygame.K_RETURN,
                )]
            if tick == 2:
                return blocked() + [key(pygame.K_SPACE)]
            return [key(pygame.K_ESCAPE)]

        def constrain_camera(self, camera, dt):
            poses.append((camera.position.copy(), camera.yaw))

        def render_scene(self, camera):
            return camera.snapshot(1.)

        def capture_conditioning(self, camera, timestamp, **kwargs):
            self.sequence += 1
            return ConditioningFrame(self.loading_image, np.ones((2, 2)), None,
                                     camera, timestamp, self.sequence)

        def diagnostic_image(self, frame, mode):
            other_actions.append((tick, mode))
            return frame.rgb

        def display(self, image, overlay_lines=None, **kwargs):
            panels.append(bool(overlay_lines))

        def set_resolution(self, size):
            other_actions.append((tick, "resize"))

        def toggle_fullscreen(self):
            other_actions.append((tick, "fullscreen"))
            return True

    def request_prompt(self, prompt, negative, *, settings=None):
        prompts.append((tick, prompt))
        return len(prompts)

    monkeypatch.setattr(proxy_renderer, "ProxyRenderer", Renderer)
    monkeypatch.setattr(main, "persist_config_values", record_write)
    monkeypatch.setattr(main, "perf_counter", lambda: 1. + tick / 60.)
    monkeypatch.setattr(pygame.time, "Clock", Clock)
    monkeypatch.setattr(DiffusionWorker, "start", lambda self: None)
    monkeypatch.setattr(DiffusionWorker, "stop", lambda self: None)
    monkeypatch.setattr(DiffusionWorker, "publish", lambda *args: None)
    monkeypatch.setattr(DiffusionWorker, "request_settings", lambda self, **values: settings.append((tick, values)))
    monkeypatch.setattr(DiffusionWorker, "request_prompt", request_prompt)
    monkeypatch.setattr(DiffusionWorker, "request_resolution", lambda *args: other_actions.append((tick, "resolution")))
    monkeypatch.setattr(DiffusionWorker, "set_frozen", lambda *args: other_actions.append((tick, "frozen")))
    monkeypatch.setattr(DiffusionWorker, "request_reseed", lambda *args: other_actions.append((tick, "reseed")) or 123)

    assert main.run() == 0
    assert saved[0] == config
    expected_cfg = main.next_level(main.CFG_LEVELS, config["guidance_scale"])
    unlocked = {**config, "guidance_scale": expected_cfg, "fog_distance": 186., "player_trail": False}
    assert saved[1] == unlocked
    assert saved[2] == {**unlocked, **variation, "prompt": saved[2]["prompt"]}
    assert saved[2]["prompt"] != config["prompt"]
    assert saved[3] == saved[2]
    assert writes == [
        (1, {"debug_overlay": True}), (1, {"guidance_scale": expected_cfg}),
        (1, {"fog_distance": 186.}), (1, {"player_trail": False}),
        (1, {"debug_overlay": False}), (2, {"prompt": saved[2]["prompt"], **variation}),
    ]
    assert settings == [(1, {"guidance_scale": expected_cfg})]
    assert edits == [(1, "start"), (1, "stop")]
    assert [at for at, _ in prompts] == [2]
    assert other_actions == []
    # Loading status and a later cancellation notice both produce visible panels
    # while settings remain locked. Mouse look and flight continue throughout.
    assert all(panels)
    assert np.all(np.diff([position[1] for position, _ in poses]) > 0)
    assert np.all(np.diff([yaw for _, yaw in poses]) > 0)
