import math
from queue import Queue
from threading import Event

import numpy as np
import pytest

from app.map_view import (MapWindow, _Scene, _merge_snapshot, _snapshot_delta,
                          image_corners, representative_images)
from app.map_idle import MapFade


def test_fade_reverses_without_jumping_and_settles_at_both_endpoints():
    fade = MapFade()
    assert fade.value(0) == 0
    assert not fade.running(0)
    fade.set_active(True, 1)
    assert fade.value(1.3) == pytest.approx(.5)
    assert fade.value(1.6) == 1
    fade.set_active(False, 2)
    halfway = fade.value(2.6)
    assert halfway == pytest.approx(.5)
    fade.set_active(True, 2.6)
    assert fade.value(2.6) == halfway
    assert fade.value(3.2) == 1
    assert not fade.running(3.2)
    fade.set_active(False, 4)
    assert fade.value(5.2) == 0
    assert not fade.running(5.2)


def test_image_plane_matches_capture_pitch_yaw_and_position():
    item = {"position": [12, 34, 56], "rotation": [35, 71, 0],
            "plane_width": 8, "width": 1024, "height": 512}
    corners = image_corners(item)
    np.testing.assert_allclose(corners.mean(axis=0), item["position"])
    np.testing.assert_allclose(np.linalg.norm(corners[1]-corners[0]), 8)
    np.testing.assert_allclose(np.linalg.norm(corners[3]-corners[0]), 4)
    pitch, yaw = map(math.radians, item["rotation"][:2])
    forward = np.array([math.sin(yaw)*math.cos(pitch), math.sin(pitch),
                        -math.cos(yaw)*math.cos(pitch)])
    np.testing.assert_allclose((corners-np.asarray(item["position"])) @ forward,
                               np.zeros(4), atol=1e-12)


def test_pose_channel_stays_bounded_and_pending_snapshot_retries():
    window = MapWindow.__new__(MapWindow)
    window._stop = Event()
    window._maps = Queue(maxsize=1)
    window._poses = Queue(maxsize=2)
    window._snapshot = None
    window._fullscreen = False
    window._operator = False
    first, second = {"id": "first"}, {"id": "second"}
    window.update(first, [0, 0, 0], [0, 0, 0], True)
    for _ in range(100):
        window.update(second, [10, 20, 30], [0, 45, 0], False)
    assert window._snapshot is first
    assert window._poses.qsize() == 2
    assert window._maps.get_nowait()["metadata"] == first
    window.update(second, [10, 20, 30], [0, 45, 0], False)
    assert window._maps.get_nowait()["metadata"] == second
    window.update(second, [10, 20, 30], [0, 45, 0], False)
    assert window._maps.empty()


def test_deltas_replace_or_remove_live_tail_append_images_and_reset():
    from copy import deepcopy

    def point(x):
        return {"position": [x, 0, 0], "timestamp": x}

    first = {"id": "a", "segments": [{"id": "s1", "ended": None,
             "points": [point(0), point(.2)]}], "images": [],
             "prompts": [{"id": "p1", "first_frame": None}]}
    snapshots = [first]
    replaced = deepcopy(first)
    replaced["segments"][0]["points"][-1] = point(.7)
    snapshots.append(replaced)
    returned = deepcopy(first)
    returned["segments"][0]["points"].pop()
    snapshots.append(returned)
    appended = deepcopy(first)
    appended["segments"][0]["points"] = [point(0), point(1.2), point(1.5)]
    appended["images"] = [{"id": "image1"}]
    appended["prompts"][0]["first_frame"] = {"timestamp": 1.2}
    snapshots.append(appended)
    resumed = deepcopy(appended)
    resumed["segments"][0]["ended"] = 2
    resumed["segments"].append({"id": "s2", "ended": None, "points": [point(100)]})
    resumed["images"].append({"id": "image2"})
    snapshots.append(resumed)
    new_map = {"id": "b", "segments": [], "images": [], "prompts": []}
    snapshots.append(new_map)
    previous = merged = None
    for snapshot in snapshots:
        merged = _merge_snapshot(merged, _snapshot_delta(previous, snapshot))
        assert merged == snapshot
        previous = snapshot


def test_long_route_delta_payload_stays_small():
    import pickle
    points = [{"position": [float(i), 0, 0], "timestamp": i} for i in range(100_000)]
    previous = {"id": "long", "segments": [{"id": "s", "ended": None, "points": points}],
                "images": [], "prompts": []}
    snapshot = dict(previous, segments=[dict(previous["segments"][0], points=points+[
        {"position": [100000.,0,0], "timestamp": 100000}])])
    delta = _snapshot_delta(previous, snapshot)
    assert len(delta["segments"][0]["points"]) == 2
    assert len(pickle.dumps(delta)) < 1024


def test_bounded_image_selection_keeps_nearby_and_older_coverage():
    images = [{"id": i, "position": [i, 0, 0]} for i in range(2000)]
    selected = representative_images(images, [1999, 0, 0])
    assert len(selected) == 512
    assert len({item["id"] for item in selected}) == 512
    assert selected[0]["id"] == 1999
    assert any(item["id"] < 10 for item in selected)


def test_render_crossfades_map_to_cached_title_and_keeps_operator_controls(tmp_path, monkeypatch):
    """Exercise GL composition and ensure settled idle never loads map images."""
    import moderngl
    import pygame
    from PIL import Image
    try:
        context = moderngl.create_standalone_context(require=330)
    except Exception as error:
        pytest.skip(f"OpenGL 3.3 unavailable: {error}")
    pygame.font.init()
    framebuffer = context.simple_framebuffer((640, 480))
    framebuffer.use()
    Image.new("RGB", (128, 64), (50, 160, 240)).save(tmp_path / "frame.png")
    scene = _Scene(context)
    scene.update({"id": "fixture", "archive_dir": str(tmp_path),
        "current_pose": {"position": [0, 0, 0]},
        "segments": [{"points": [{"position": [-20, 0, 0]}, {"position": [20, 0, 0]}]},
                     {"points": [{"position": [40, 8, 0]}, {"position": [50, 12, 0]}]}],
        "images": [{"id": "frame", "path": "frame.png", "position": [0,0,0],
                    "rotation": [20,0,0], "width": 128, "height": 64, "plane_width": 24}],
        "prompts": []})
    assert len(scene.lines) == 2  # Idle gaps never become connecting line segments.
    pose = {"position": [0,0,0], "active": True}
    scene.draw(pose, -math.pi/2, (640,480), False)
    rgb = np.frombuffer(framebuffer.read(components=3), dtype=np.uint8).reshape(480,640,3)
    assert np.count_nonzero((rgb[:,:,2] > 150) & (rgb[:,:,0] < 100)) > 100
    pose["active"] = False
    scene.draw(pose, -math.pi/2, (640,480), False)
    idle = np.frombuffer(framebuffer.read(components=3), dtype=np.uint8).reshape(480,640,3)
    assert np.count_nonzero(idle[:,:,0] > 180) > 1000
    assert np.count_nonzero((idle[:,:,2] > 150) & (idle[:,:,0] < 100)) == 0
    scene.draw(pose, -math.pi/2, (640,480), False, .5)
    blend = np.frombuffer(framebuffer.read(components=3), dtype=np.uint8).reshape(480,640,3)
    np.testing.assert_allclose(blend, (rgb.astype(float)+idle)/2, atol=1)
    texture = scene.title_texture
    monkeypatch.setattr(scene, "_load_textures", lambda *_: pytest.fail("Idle loaded map textures"))
    scene.draw(pose, 0, (640,480), False)
    assert scene.title_texture is texture
    scene.draw(pose, 0, (640,480), True)
    assert framebuffer.read(components=3) != idle.tobytes()
    framebuffer.release()
    context.release()
