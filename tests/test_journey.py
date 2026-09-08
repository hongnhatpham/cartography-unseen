import base64
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from io import BytesIO
import hashlib
import json
import os
import pickle
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import pytest

from app.journey import JourneyRecorder, _atomic_json
from app.types import GeneratedFrame


@pytest.fixture
def archive_thread(monkeypatch):
    """Keep local fault-injection patches visible inside the archive worker."""
    from app import journey
    monkeypatch.setattr(journey, "ProcessPoolExecutor",
                        lambda **kwargs: ThreadPoolExecutor(max_workers=kwargs["max_workers"]))


def test_manifest_invalid_update_preserves_previous_file_and_allows_retry(tmp_path):
    path = tmp_path / "manifest.json"
    original = {"prompt": "forêt 雨", "position": [1.5, 2, 3]}
    _atomic_json(path, original)
    previous = path.read_bytes()
    with pytest.raises(ValueError):
        _atomic_json(path, {"position": [float("nan"), 2, 3]})
    assert path.read_bytes() == previous
    replacement = {"prompt": "nước", "archive_dir": str(tmp_path)}
    _atomic_json(path, replacement)
    assert json.loads(path.read_text(encoding="utf-8")) == {"prompt": "nước"}
    assert replacement["archive_dir"] == str(tmp_path)


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


def test_save_failure_retains_journey_and_retry_archives_same_data(tmp_path, monkeypatch, archive_thread):
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


def test_failed_image_task_retains_original_and_retry_saves_it(tmp_path, monkeypatch, archive_thread):
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
    with Image.open(archive / "images/000001.webp") as image:
        np.testing.assert_array_equal(np.asarray(image), frame(0., 1).image)


def test_dead_archive_worker_retries_accepted_frame_without_losing_original(tmp_path):
    recorder = JourneyRecorder(tmp_path, 42)
    pose(recorder, (0, 0, 0), 0.)
    prompt(recorder)
    assert recorder.offer_frame(frame(0., 1))
    recorder.flush()
    # Kill the actual spawned worker between jobs, then submit through the real caller.
    with pytest.raises(BrokenProcessPool):
        recorder._executor.submit(os._exit, 1).result(timeout=15)
    pose(recorder, (12, 0, 0), .2)
    generated = frame(.2, 2, (12, 0, 0))
    expected = generated.image.copy()
    assert recorder.offer_frame(generated)
    generated.image[:] = 255
    assert recorder.poll() is not None
    recorder.flush()
    checkpoint = json.loads((recorder.archive_dir / "manifest.json").read_text(encoding="utf-8"))
    assert [image["id"] for image in checkpoint["images"]] == ["image-1", "image-2"]
    archive = recorder.close()
    manifest = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
    assert [image["id"] for image in manifest["images"]] == ["image-1", "image-2"]
    with Image.open(archive / "images/000002.webp") as image:
        np.testing.assert_array_equal(np.asarray(image), expected)


def test_failed_final_export_then_worker_restart_preserves_resumed_live_history(tmp_path):
    recorder = JourneyRecorder(tmp_path, 42, export_svg=True)
    pose(recorder, (0, 0, 0), 0.)
    prompt(recorder)
    assert recorder.offer_frame(frame(0., 1))
    recorder.flush()
    with pytest.raises(BrokenProcessPool):
        recorder._executor.submit(os._exit, 1).result(timeout=15)
    with pytest.raises(BrokenProcessPool):
        recorder.reset(.1)

    # The replacement worker saves the manifest but cannot complete the export.
    blocked = recorder.archive_dir / "map.svg.tmp"
    blocked.mkdir()
    with pytest.raises(OSError):
        recorder.reset(.2)
    blocked.rmdir()
    pose(recorder, (12, 0, 0), .3)
    assert recorder.offer_frame(frame(.3, 2, (12, 0, 0)))
    recorder.flush()
    checkpoint = json.loads((recorder.archive_dir / "manifest.json").read_text(encoding="utf-8"))
    assert [image["id"] for image in checkpoint["images"]] == ["image-1", "image-2"]
    assert checkpoint["segments"][0]["points"][0]["position"] == [0., 0., 0.]
    assert "completion_reason" not in checkpoint
    recorder.close()


def test_failed_incremental_image_job_resets_worker_before_retry_without_duplicate_images(tmp_path):
    recorder = JourneyRecorder(tmp_path, 42)
    pose(recorder, (0, 0, 0), 0.)
    prompt(recorder)
    assert recorder.offer_frame(frame(0., 1))
    recorder.flush()

    # The second job merges an append delta before this real filesystem failure.
    blocked = recorder.archive_dir / "images/000002.webp.tmp"
    blocked.mkdir()
    pose(recorder, (12, 0, 0), .2)
    assert recorder.offer_frame(frame(.2, 2, (12, 0, 0)))
    with pytest.raises(OSError):
        recorder.flush()
    manifest_path = recorder.archive_dir / "manifest.json"
    checkpoint = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [image["id"] for image in checkpoint["images"]] == ["image-1"]

    blocked.rmdir()
    recorder.flush()
    checkpoint = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [image["id"] for image in checkpoint["images"]] == ["image-1", "image-2"]
    with Image.open(recorder.archive_dir / "images/000002.webp") as image:
        np.testing.assert_array_equal(np.asarray(image), frame(.2, 2).image)

    # Subsequent append deltas must continue from the successfully retried snapshot.
    pose(recorder, (24, 0, 0), .4)
    assert recorder.offer_frame(frame(.4, 3, (24, 0, 0)))
    recorder.flush()
    checkpoint = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [image["id"] for image in checkpoint["images"]] == ["image-1", "image-2", "image-3"]
    assert checkpoint["segments"][0]["points"][-1]["position"] == [24., 0., 0.]
    recorder.close()


def test_live_archive_submission_transfers_new_records_instead_of_accumulated_route(tmp_path, archive_thread):
    recorder = JourneyRecorder(tmp_path, 42)
    pose(recorder, (0, 0, 0), 0.)
    prompt(recorder)
    for index in range(1, 1001):
        pose(recorder, (index, 0, 0), index / 1000)
    assert recorder.offer_frame(frame(1., 1, (1000, 0, 0)))
    recorder.flush()

    original_submit = recorder._executor.submit
    submitted = []

    def capture_submit(task, *args):
        submitted.append(args[-1])
        return original_submit(task, *args)

    recorder._executor.submit = capture_submit
    pose(recorder, (1012, 0, 0), 1.1)
    assert recorder.offer_frame(frame(1.1, 2, (1012, 0, 0)))
    recorder.flush()
    update = submitted[0]
    assert update["reset"] is False
    assert [image["id"] for image in update["images"]] == ["image-2"]
    assert len(update["segments"][0]["points"]) == 2
    assert len(pickle.dumps(update)) < 4096
    checkpoint = json.loads((recorder.archive_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(checkpoint["segments"][0]["points"]) == 1002
    recorder.close()


def test_worker_dying_after_io_failure_can_retry_again_after_failed_submission(tmp_path):
    recorder = JourneyRecorder(tmp_path, 42)
    pose(recorder, (0, 0, 0), 0.)
    prompt(recorder)
    blocked = recorder.archive_dir / "images/000001.webp.tmp"
    blocked.mkdir(parents=True)
    assert recorder.offer_frame(frame(0., 1))
    with pytest.raises(OSError):
        recorder.flush()

    # The retained future reports the I/O error, so it does not reveal this later death.
    with pytest.raises(BrokenProcessPool):
        recorder._executor.submit(os._exit, 1).result(timeout=15)
    blocked.rmdir()
    with pytest.raises(BrokenProcessPool):
        recorder.flush()

    # A failed retry submission must allow the next explicit retry to replace the pool.
    recorder.flush()
    checkpoint = json.loads((recorder.archive_dir / "manifest.json").read_text(encoding="utf-8"))
    assert [image["id"] for image in checkpoint["images"]] == ["image-1"]
    with Image.open(recorder.archive_dir / "images/000001.webp") as image:
        np.testing.assert_array_equal(np.asarray(image), frame(0., 1).image)
    recorder.close()


@pytest.mark.parametrize("image_format", ["png", "webp"])
def test_svg_preserves_original_pixels_vectors_and_exact_prompt(tmp_path, image_format):
    recorder = JourneyRecorder(tmp_path, 42, image_format=image_format, export_svg=True)
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
    href = element.attrib["{http://www.w3.org/1999/xlink}href"]
    assert href.startswith(f"data:image/{image_format};base64,")
    encoded = href.split(",", 1)[1]
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
    assert not (recorder.archive_dir / "complete.json").exists()
    recorder.close()


@pytest.mark.parametrize("image_format", ["webp", "png"])
def test_compact_archive_preserves_pixels_and_exports_svg_on_demand(tmp_path, image_format):
    from tools.export_journey_svg import export_journey_svg

    options = {} if image_format == "webp" else {"image_format": "png"}
    recorder = JourneyRecorder(tmp_path / "journeys", 42, **options)
    pose(recorder, (0, 0, 0), 0.)
    prompt(recorder)
    generated = frame(0., 1)
    generated.image[:] = np.random.default_rng(42).integers(0, 256, generated.image.shape, dtype=np.uint8)
    assert recorder.offer_frame(generated)
    archive = recorder.close()
    manifest_bytes = (archive / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    assert manifest["images"][0]["path"] == f"images/000001.{image_format}"
    with Image.open(archive / manifest["images"][0]["path"]) as image:
        assert image.format == image_format.upper()
        np.testing.assert_array_equal(np.asarray(image), generated.image)
    assert not (archive / "map.svg").exists()
    assert (archive / "index.html").is_file()
    marker_bytes = (archive / "complete.json").read_bytes()
    assert json.loads(marker_bytes) == {
        "schema_version": 1, "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}

    # Legacy PNG archives did not have completion markers.
    if image_format == "png":
        (archive / "complete.json").unlink()
    output = export_journey_svg(archive, tmp_path / "exports" / "map.svg")
    element = ET.parse(output).find("{http://www.w3.org/2000/svg}image")
    href = element.attrib["{http://www.w3.org/1999/xlink}href"]
    assert href.startswith(f"data:image/{image_format};base64,")
    with Image.open(BytesIO(base64.b64decode(href.split(",", 1)[1]))) as image:
        np.testing.assert_array_equal(np.asarray(image), generated.image)
    assert (archive / "manifest.json").read_bytes() == manifest_bytes
    if image_format == "webp":
        assert (archive / "complete.json").read_bytes() == marker_bytes


@pytest.mark.parametrize("failed_artifact", ["manifest.json", "map.svg", "index.html", "complete.json"])
def test_completion_marker_is_removed_before_retry_and_published_only_after_success(
        tmp_path, monkeypatch, archive_thread, failed_artifact):
    import app.journey as module

    recorder = JourneyRecorder(tmp_path, 42, export_svg=True)
    pose(recorder, (0, 0, 0), 0.)
    prompt(recorder)
    assert recorder.offer_frame(frame(0., 1))
    recorder.flush()
    marker = recorder.archive_dir / "complete.json"
    marker.write_text('{"stale":true}', encoding="utf-8")
    original_json, original_svg = module._atomic_json, module._export_svg

    def save_json(path, data):
        if path.name == failed_artifact:
            raise OSError("archive failure")
        original_json(path, data)

    def save_svg(*args, **kwargs):
        if failed_artifact == "map.svg":
            raise OSError("archive failure")
        original_svg(*args, **kwargs)

    monkeypatch.setattr(module, "_atomic_json", save_json)
    monkeypatch.setattr(module, "_export_svg", save_svg)
    blocked = recorder.archive_dir / "index.html.tmp"
    if failed_artifact == "index.html":
        blocked.mkdir()
    with pytest.raises(OSError):
        recorder.close()
    assert not marker.exists()
    assert recorder.nonempty
    monkeypatch.setattr(module, "_atomic_json", original_json)
    monkeypatch.setattr(module, "_export_svg", original_svg)
    if failed_artifact == "index.html":
        blocked.rmdir()
    archive = recorder.close()
    assert json.loads(marker.read_text())["manifest_sha256"] == hashlib.sha256(
        (archive / "manifest.json").read_bytes()).hexdigest()
    assert (archive / "index.html").is_file()
    assert (archive / "map.svg").is_file()


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
