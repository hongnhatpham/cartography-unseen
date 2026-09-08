import json

import pytest

from tools.replay_performance import Metrics, NoKeys, measurement_config, route_pose


def test_route_does_not_depend_on_previous_frames_or_startup():
    origin = (12, 8, -31)
    assert route_pose(-300, origin, 12, 8, 8) == (origin, 12, 8)
    expected = route_pose(40, origin, 12, 8, 8)
    for time in (1, 2, 10, 20, 39.95):
        route_pose(time, origin, 12, 8, 8)
    assert route_pose(40, origin, 12, 8, 8) == expected
    assert all(a != b for a, b in zip(expected[0], origin))


@pytest.mark.parametrize("normal", [False, True])
def test_performance_runs_never_start_live_archive_uploads(normal):
    source = {"map_sync_enabled": True}
    assert measurement_config(source, normal)["map_sync_enabled"] is False
    assert source["map_sync_enabled"] is True


@pytest.mark.parametrize("stall,passes", [(16, True), (120, False)])
def test_replay_threshold_is_red_capable_and_writes_every_timing(tmp_path, stall, passes):
    metrics = Metrics(tmp_path, 120)
    metrics.writer.start()
    metrics.displayed(1)
    metrics.started = 100
    metrics.displayed(100.01)
    metrics.displayed(100.01 + stall / 1000)
    metrics.record("generate", 101, 101.08)
    metrics.ended = 220
    summary = metrics.finish(0, "", 100)
    assert summary["completed"]
    assert summary["passed"] is passes
    assert summary["timings"]["display_interval"]["count"] == 1
    assert summary["timings"]["display_interval"]["max_ms"] == pytest.approx(stall)
    assert len((tmp_path / "timings.csv").read_text().splitlines()) == 3
    assert json.loads((tmp_path / "summary.json").read_text())["passed"] is passes


def test_partial_run_and_dropped_measurements_cannot_pass(tmp_path):
    metrics = Metrics(tmp_path, 120)
    metrics.writer.start()
    metrics.started = 100
    metrics.record("display_interval", 100, 100.016)
    metrics.ended = 110
    assert not metrics.finish(0, "", 100)["passed"]
    metrics.ended = 220
    metrics.dropped = 1
    assert not metrics.finish(0, "", 100)["passed"]


def test_held_keys_are_neutral_even_with_large_pygame_key_codes():
    keys = NoKeys()
    assert not keys[1073742049]
    assert not any(keys)


@pytest.mark.parametrize("gap,passes", [(100, True), (300, False)])
def test_ai_gap_threshold_detects_frozen_image_even_with_smooth_display(tmp_path, gap, passes):
    metrics = Metrics(tmp_path, .3)
    metrics.writer.start()
    metrics.published(100)
    assert metrics.started == 100
    assert "ai_publication_interval" not in metrics.histograms
    metrics.record("display_interval", 100, 100.016)
    metrics.published(100 + gap / 1000)
    metrics.ended = 100.31
    summary = metrics.finish(0, "", 100, 250)
    assert summary["passed"] is passes
    assert summary["ai_gaps"]["count"] == 1
    assert summary["ai_gaps"]["max_ms"] == pytest.approx(gap)


def test_ai_publications_cannot_stop_silently_before_the_run_ends(tmp_path):
    metrics = Metrics(tmp_path, 1)
    metrics.writer.start()
    metrics.published(100)
    metrics.published(100.1)
    metrics.record("display_interval", 100, 100.016)
    metrics.ended = 101
    summary = metrics.finish(0, "", 100, 250)
    assert summary["completed"]
    assert not summary["passed"]
    assert summary["ai_gaps"]["final_unpublished_gap_ms"] == pytest.approx(900)


def test_normal_config_retains_image_and_prompt_settings_without_mutating_source():
    config = {"backend": "latent_walk", "diffusion_resolution": "640x384", "guidance_scale": 2,
              "prompt_auto_advance_seconds": 24, "random_seed_on_launch": True, "debug_overlay": True,
              "autowalk_idle_seconds": 60, "reprojection": False, "prompt": "distant figure", "seed": 123}
    original = dict(config)
    normal = measurement_config(config, normal=True)
    assert normal == {**config, "autowalk_idle_seconds": 2.0}
    fixed = measurement_config(config)
    assert fixed == {**config, "backend": "latent_walk", "diffusion_resolution": "384x256",
                     "guidance_scale": 1.5, "prompt_auto_advance_seconds": 0,
                     "random_seed_on_launch": False, "debug_overlay": False, "autowalk_idle_seconds": 0}
    assert config == original


@pytest.mark.parametrize("normal", [False, True])
def test_normal_mode_retains_real_input_navigation_and_palette_without_gpu(monkeypatch, tmp_path, normal):
    from tools import replay_performance
    from app import main
    from app.renderer.camera import Camera
    from app.renderer.proxy_renderer import ProxyRenderer
    import pygame

    source = tmp_path / "source.json"
    source.write_text(json.dumps({"movement_speed": 8, "debug_overlay": True}), encoding="utf-8")
    source_text = source.read_text()
    output = tmp_path / "measurement"
    methods = [(Camera, "fly"), (Camera, "rotate"), (ProxyRenderer, "read_input"),
               (ProxyRenderer, "constrain_camera"), (ProxyRenderer, "render_scene"),
               (ProxyRenderer, "capture_conditioning"), (ProxyRenderer, "_update_world"), (main, "hue_words")]
    originals = [getattr(target, name) for target, name in methods]
    events = [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE),
              pygame.event.Event(pygame.MOUSEMOTION), pygame.event.Event(pygame.QUIT)]
    monkeypatch.setattr(ProxyRenderer, "poll_events", staticmethod(lambda: list(events)))
    observed = []

    def application():
        observed.append(True)
        for (target, name), original in zip(methods, originals):
            assert (getattr(target, name) is original) is normal
        assert ProxyRenderer.poll_events() == (events if normal else [events[-1]])
        return 0

    monkeypatch.setattr(main, "run", application)
    # No frames were generated by this fake application, so duration is incomplete.
    assert replay_performance.run(600, output, source, 100, normal=normal) == 1
    assert observed == [True]
    assert source.read_text() == source_text
    metadata = json.loads((output / "run.json").read_text())
    assert metadata["mode"] == ("normal" if normal else "fixed_route")
    assert metadata["source_sha256"]
    assert json.loads((output / "config.json").read_text()) == measurement_config(json.loads(source_text), normal)


def test_normal_cli_forwards_duration_and_thresholds_without_loading_app(monkeypatch, tmp_path):
    from tools import replay_performance
    calls = []
    monkeypatch.setattr(replay_performance.sys, "argv", ["replay", "--normal", "--seconds", "600",
                                                       "--output", str(tmp_path)])
    monkeypatch.setattr(replay_performance, "run", lambda *args: calls.append(args) or 0)
    assert replay_performance.cli() == 0
    assert calls[0][0] == 600
    assert calls[0][3:] == (100, 250, True, None)


@pytest.mark.parametrize("map_check", ["off", "active", "idle", "cycle", "recorder", "soak", "soak-off"])
def test_map_comparison_matches_controls_and_windows_without_gpu(monkeypatch, tmp_path, map_check):
    from tools import replay_performance
    from app import main
    from app.journey import JourneyRecorder
    from app.renderer.proxy_renderer import ProxyRenderer
    import pygame
    map_enabled = map_check not in ("off", "soak-off")

    source = tmp_path / "source.json"
    source.write_text(json.dumps({"movement_speed": 8, "journey_map": True,
                                  "fullscreen": True, "debug_overlay": True,
                                  "prompt_auto_advance_seconds": 61}), encoding="utf-8")
    source_text = source.read_text()
    output = tmp_path / map_check
    renderer_arguments, archive_roots = [], []
    monkeypatch.setattr(ProxyRenderer, "__init__",
                        lambda self, *args, **kwargs: renderer_arguments.append(kwargs))
    monkeypatch.setattr(JourneyRecorder, "__init__",
                        lambda self, root, *args, **kwargs: archive_roots.append(root))
    monkeypatch.setattr(ProxyRenderer, "poll_events", staticmethod(lambda: []))

    def application():
        renderer = ProxyRenderer(window_size=(384, 256), window_position=(60, 60))
        mouse, keys, buttons = renderer.read_input()
        assert keys[pygame.K_w] is (map_check != "idle")
        assert not any(keys[key] for key in (pygame.K_a, pygame.K_s, pygame.K_d, pygame.K_LSHIFT))
        assert mouse == (0, 0)
        assert buttons == (False, False, False)
        events = renderer.poll_events()
        assert [(event.type, event.key) for event in events] == (
            [(pygame.KEYDOWN, pygame.K_F1)] if map_enabled else [])
        assert renderer.poll_events() == []
        if map_enabled:
            JourneyRecorder(tmp_path / "real-journeys", 123)
        return 0

    monkeypatch.setattr(main, "run", application)
    assert replay_performance.run(600, output, source, 100, map_check=map_check) == 1
    summary = json.loads((output / "summary.json").read_text())
    assert summary["error"] == ("" if not map_enabled else
                                "Map child did not return rendering measurements")
    assert renderer_arguments == [{"window_size": (1920, 1080), "window_position": (0, 0)}]
    assert archive_roots == ([output / "journeys"] if map_enabled else [])
    copied_config = json.loads((output / "config.json").read_text())
    assert copied_config["journey_map"] is map_enabled
    assert copied_config["prompt_auto_advance_seconds"] == (61 if map_check in ("soak", "soak-off") else 0)
    assert not copied_config["fullscreen"]
    assert not copied_config["debug_overlay"]
    assert source.read_text() == source_text


def test_map_cli_cannot_silently_use_unmatched_normal_route(monkeypatch):
    from tools import replay_performance
    monkeypatch.setattr(replay_performance.sys, "argv", ["replay", "--normal", "--map-check", "active"])
    with pytest.raises(SystemExit) as error:
        replay_performance.cli()
    assert error.value.code == 2
