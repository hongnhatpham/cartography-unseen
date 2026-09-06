import base64
from io import BytesIO
import json
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import pytest

from app.journey import JourneyRecorder
from app.types import GeneratedFrame


def pose(recorder, position, timestamp, active=True):
    recorder.update_pose(position, (12., 35., 0.), timestamp, active)


def prompt(recorder, timestamp=0., revision=1, active=True):
    recorder.record_prompt('glass <forest> & "rain"', revision, "initial", (0., 0., 0.),
                           (12., 35., 0.), timestamp, active,
                           {"submitted_prompt": "the exact backend composition"})


def frame(timestamp, sequence, position=(0., 0., 0.), revision=1):
    pixels = np.arange(48 * 32 * 3, dtype=np.uint8).reshape(32, 48, 3)
    return GeneratedFrame(pixels, np.zeros((32, 48)), np.eye(4), np.eye(4),
                          np.asarray(position, dtype=float), np.array([12., 35., 0.]),
                          timestamp + .1, timestamp, sequence, {"prompt_revision": revision})


def test_accumulated_3d_distance_captures_loops_but_not_turning(tmp_path):
    recorder = JourneyRecorder(tmp_path, 7, capture_distance=12)
    pose(recorder, (0, 0, 0), 0.)
    prompt(recorder)
    assert recorder.offer_frame(frame(0., 1))
    recorder.flush()
    # Return to the same XYZ after moving 12 units, including vertical travel.
    for timestamp, point in enumerate(((0, 3, 0), (3, 3, 0), (3, 0, 0), (0, 0, 0)), 1):
        pose(recorder, point, float(timestamp))
    assert recorder.offer_frame(frame(4., 2))
    recorder.flush()
    pose(recorder, (0, 0, 0), 4.2)
    assert not recorder.offer_frame(frame(4.2, 3))
    assert len(recorder.snapshot()["images"]) == 2
    recorder.close()


def test_idle_resume_preserves_exact_player_pose_and_disconnected_segments(tmp_path):
    recorder = JourneyRecorder(tmp_path, 7)
    pose(recorder, (1, 2, 3), 0.)
    prompt(recorder)
    pose(recorder, (1.2, 2, 3), .1)
    pose(recorder, (90, 40, -20), .2, active=False)
    assert not recorder.offer_frame(frame(.2, 1, (90, 40, -20)))
    pose(recorder, (900, 80, -40), .3, active=False)
    pose(recorder, (901, 82, -40), .4)
    state = recorder.snapshot()
    assert state["current_pose"]["position"] == [901., 82., -40.]
    assert len(state["segments"]) == 2
    assert state["segments"][0]["points"][-1]["position"] == [1.2, 2., 3.]
    assert state["segments"][1]["points"][0]["position"] == [901., 82., -40.]
    assert recorder.offer_frame(frame(.1, 2, (1.2, 2, 3)))
    recorder.flush()
    assert recorder.snapshot()["images"][0]["segment_id"] == "segment-1"
    assert recorder.offer_frame(frame(.4, 3, (901, 82, -40)))
    recorder.close()


def test_delayed_frame_uses_source_distance_and_pose_not_live_distance(tmp_path):
    recorder = JourneyRecorder(tmp_path, 7)
    pose(recorder, (0, 0, 0), 0.)
    prompt(recorder)
    assert recorder.offer_frame(frame(0., 1))
    recorder.flush()
    pose(recorder, (5, 0, 0), .1)
    pose(recorder, (20, 0, 0), .2)
    assert not recorder.offer_frame(frame(.1, 2, (5, 0, 0)))
    assert recorder.offer_frame(frame(.2, 3, (20, 0, 0)))
    recorder.close()


def test_reset_archives_before_clear_and_rejects_old_frames(tmp_path):
    recorder = JourneyRecorder(tmp_path, 42)
    pose(recorder, (10, 20, 30), 0.)
    prompt(recorder)
    assert recorder.offer_frame(frame(0., 1, (10, 20, 30)))
    old_id = recorder.id
    archive = recorder.reset(1.)
    assert archive.name == old_id
    manifest = json.loads((archive / "manifest.json").read_text())
    assert manifest["completion_reason"] == "space"
    assert manifest["images"][0]["position"] == [10., 20., 30.]
    assert recorder.id != old_id
    assert recorder.snapshot()["preceding_archive"] == old_id
    assert not recorder.nonempty
    pose(recorder, (20, 20, 30), 1.)
    prompt(recorder, 1., revision=2)
    assert not recorder.offer_frame(frame(.9, 2, revision=1))
    assert not recorder.offer_frame(frame(1., 3, revision=1))
    assert recorder.offer_frame(frame(1., 4, revision=2))
    recorder.close()
    assert len(list(tmp_path.glob("*/manifest.json"))) == 2


def test_save_failure_retains_journey_and_retry_archives_same_data(tmp_path, monkeypatch):
    import app.journey as module

    recorder = JourneyRecorder(tmp_path, 42)
    pose(recorder, (1, 2, 3), 0.)
    prompt(recorder)
    original = module._atomic_json

    def fail(*args):
        raise OSError("disk unavailable")

    monkeypatch.setattr(module, "_atomic_json", fail)
    old_id = recorder.id
    with pytest.raises(OSError, match="disk unavailable"):
        recorder.reset(1.)
    assert recorder.id == old_id and recorder.nonempty
    monkeypatch.setattr(module, "_atomic_json", original)
    archive = recorder.reset(2.)
    assert json.loads((archive / "manifest.json").read_text())["segments"][0]["points"][0]["position"] == [1., 2., 3.]
    assert not recorder.nonempty
    recorder.close()


def test_failed_png_task_retains_original_and_retry_saves_it(tmp_path, monkeypatch):
    recorder = JourneyRecorder(tmp_path, 42)
    pose(recorder, (0, 0, 0), 0.)
    prompt(recorder)
    original = Image.Image.save

    def fail(*args, **kwargs):
        raise OSError("image write failed")

    monkeypatch.setattr(Image.Image, "save", fail)
    assert recorder.offer_frame(frame(0., 1))
    with pytest.raises(OSError, match="image write failed"):
        recorder.flush()
    assert "image write failed" in recorder.poll()
    assert not recorder.offer_frame(frame(0., 2))
    assert len(recorder.snapshot()["images"]) == 1
    monkeypatch.setattr(Image.Image, "save", original)
    archive = recorder.close()
    with Image.open(archive / "images/000001.png") as image:
        np.testing.assert_array_equal(np.asarray(image), frame(0., 1).image)


def test_svg_preserves_original_pixels_vectors_and_exact_prompt(tmp_path):
    recorder = JourneyRecorder(tmp_path, 42)
    pose(recorder, (0, 0, 0), 0.)
    prompt(recorder)
    generated = frame(0., 1)
    generated.stats.update(active_seed=93, timestep_now=315, prompt_walk_t=.25)
    assert recorder.offer_frame(generated)
    generated.stats["active_seed"] = 94
    pose(recorder, (200, 40, -500), .1)
    archive = recorder.close()
    svg = ET.parse(archive / "map.svg")
    ns = {"svg": "http://www.w3.org/2000/svg"}
    assert svg.find("svg:polyline", ns) is not None
    element = svg.find("svg:image", ns)
    assert "matrix(" in element.attrib["transform"]
    encoded = element.attrib["{http://www.w3.org/1999/xlink}href"].split(",", 1)[1]
    with Image.open(BytesIO(base64.b64decode(encoded))) as image:
        assert image.size == (48, 32)
        np.testing.assert_array_equal(np.asarray(image), frame(0., 1).image)
    assert 'glass <forest> & "rain"' in " ".join(svg.getroot().itertext())
    manifest = json.loads((archive / "manifest.json").read_text())
    assert manifest["prompts"][0]["settings"]["submitted_prompt"] == "the exact backend composition"
    assert manifest["prompts"][0]["first_frame"]["sequence"] == 1
    assert manifest["images"][0]["generation_stats"]["active_seed"] == 93
    assert manifest["images"][0]["generation_stats"]["prompt_walk_t"] == .25


def test_idle_only_prompt_history_does_not_create_archive(tmp_path):
    recorder = JourneyRecorder(tmp_path, 42)
    pose(recorder, (0, 0, 0), 0., active=False)
    prompt(recorder, active=False)
    assert not recorder.offer_frame(frame(0., 1))
    assert recorder.close() is None
    assert not list(tmp_path.iterdir())


def test_checkpoint_is_durable_and_only_one_task_is_queued(tmp_path):
    recorder = JourneyRecorder(tmp_path, 42)
    pose(recorder, (0, 0, 0), 0.)
    prompt(recorder)
    pose(recorder, (0, 12, 0), 5.)
    recorder.flush()
    manifest = json.loads((recorder.archive_dir / "manifest.json").read_text())
    assert "completion_reason" not in manifest
    assert manifest["segments"][0]["points"][-1]["position"] == [0., 12., 0.]
    assert recorder._task is None
    recorder.close()


def test_prompt_change_tracks_first_frame_without_triggering_capture(tmp_path):
    recorder = JourneyRecorder(tmp_path, 42)
    pose(recorder, (0, 0, 0), 0.)
    prompt(recorder)
    assert recorder.offer_frame(frame(0., 1))
    recorder.flush()
    recorder.record_prompt("new prompt </script>", 2, "editor", (0, 0, 0),
                           (0, 45, 0), .1, True, {"submitted_prompt": "new prompt </script>"})
    pose(recorder, (0, 0, 0), .2)
    assert not recorder.offer_frame(frame(.2, 2, revision=2))
    event = recorder.snapshot()["prompts"][1]
    assert event["previous_prompt"] == 'glass <forest> & "rain"'
    assert event["first_frame"]["timestamp"] == .2
    assert event["timestamp"] == .1
    archive = recorder.close()
    assert len(json.loads((archive / "manifest.json").read_text())["images"]) == 1
    # The inline manifest cannot terminate its script element through prompt text.
    viewer = archive / "index.html"
    if viewer.exists():
        content = viewer.read_text(encoding="utf-8")
        assert "new prompt </script>" not in content
        assert "new prompt <\\/script>" in content


def test_snapshot_stays_consistent_while_route_and_prompt_frames_advance(tmp_path):
    recorder = JourneyRecorder(tmp_path, 42)
    pose(recorder, (0, 0, 0), 0.)
    settings = {"generation": {"seed": 42}}
    recorder.record_prompt("forest", 1, "initial", (0, 0, 0), (0, 0, 0), 0., True, settings)
    settings["generation"]["seed"] = 99
    pose(recorder, (.2, 0, 0), .1)
    initial = recorder.snapshot()
    initial_json = json.dumps(initial)
    assert initial["segments"][0]["points"][-1]["position"] == [.2, 0., 0.]
    assert initial["prompts"][0]["first_frame"] is None
    assert initial["prompts"][0]["settings"]["generation"]["seed"] == 42
    assert recorder.offer_frame(frame(.1, 1, (.2, 0, 0)))
    recorder.flush()
    captured = recorder.snapshot()
    captured_json = json.dumps(captured)
    pose(recorder, (12, 0, 0), .2)
    pose(recorder, (50, 0, 0), .3, active=False)
    pose(recorder, (100, 0, 0), .4)
    prompt(recorder, .4, revision=2)
    assert recorder.offer_frame(frame(.4, 2, (100, 0, 0), revision=2))
    recorder.flush()
    assert json.dumps(initial) == initial_json
    assert json.dumps(captured) == captured_json
    assert initial["segments"][0]["ended"] is None
    assert recorder.snapshot()["segments"][0]["ended"] == .3
    assert len(initial["images"]) == 0
    assert len(captured["images"]) == 1
    assert len(recorder.snapshot()["images"]) == 2
    recorder.close()
