"""Exercise live map visibility and source-frame association with real archives."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from app import map_view
from app.journey_session import JourneySession
from app.types import GeneratedFrame


@pytest.fixture
def session(monkeypatch, tmp_path):
    class Window:
        def __init__(self, root):
            self.updates = []

        def update(self, snapshot, position, rotation, active):
            self.updates.append((snapshot, position, rotation, active))

        def poll(self): return []
        def close(self): pass

    monkeypatch.setattr(map_view, "MapWindow", Window)
    session = JourneySession(tmp_path, SimpleNamespace(
        world_seed=42, map_capture_distance=12., map_idle_seconds=1.))
    yield session
    session.close()


def observe(session, position, timestamp, *, interacting=False, automatic=False, suppressed=False):
    session.observe(position, [12., 35., 0.], timestamp,
                    interacting=interacting, autowalking=automatic, suppressed=suppressed)


def frame(timestamp, sequence, position, revision):
    return GeneratedFrame(np.zeros((8, 12, 3), dtype=np.uint8), np.ones((8, 12)),
                          np.eye(4), np.eye(4), np.asarray(position, dtype=float),
                          np.array([12., 35., 0.]), timestamp + .1, timestamp,
                          sequence, {"prompt_revision": revision})


def test_idle_start_and_automatic_resume_keep_exact_pose_without_connecting_route(session):
    observe(session, [0., 0., 0.], 0.)
    assert session.window.updates[-1][-1] is False
    assert not session.recorder.nonempty
    observe(session, [1., 2., 3.], .1, interacting=True)
    observe(session, [1.2, 2., 3.], .2)
    observe(session, [90., 40., -20.], .3, automatic=True)
    assert session.window.updates[-1][-1] is False
    observe(session, [900., 80., -40.], .4, automatic=True)
    observe(session, [901., 82., -40.], .5, interacting=True)
    snapshot, position, rotation, active = session.window.updates[-1]
    assert active is True
    assert position == [901., 82., -40.]
    assert rotation == [12., 35., 0.]
    assert len(snapshot["segments"]) == 2
    assert snapshot["segments"][0]["points"][-1]["position"] == [1.2, 2., 3.]
    assert snapshot["segments"][1]["points"][0]["position"] == position
    observe(session, position, 1.6)
    assert session.window.updates[-1][-1] is False
    observe(session, position, 1.7, interacting=True, suppressed=True)
    assert session.window.updates[-1][-1] is False


def test_prompt_markers_and_delayed_frames_follow_source_revision_across_reset(session):
    camera = SimpleNamespace(position=np.array([1., 2., 3.]), pitch=12., yaw=35.)
    session.prompt("first world", 0, "initial", camera, {}, timestamp=0.)
    observe(session, camera.position, .1, interacting=True)
    session.prompt("glass <forest>", 1, "editor", camera,
                   {"submitted_prompt": "glass <forest>, cyan"}, timestamp=.15)
    observe(session, [20., 2., 3.], .2, interacting=True)
    session.offer_frame(frame(.1, 1, [1., 2., 3.], 0))
    session.recorder.flush()
    session.offer_frame(frame(.2, 2, [20., 2., 3.], 1))
    session.recorder.flush()
    snapshot = session.recorder.snapshot()
    assert [image["prompt_revision"] for image in snapshot["images"]] == [0, 1]
    event = snapshot["prompts"][1]
    assert (event["prompt"], event["trigger"], event["active"]) == ("glass <forest>", "editor", True)
    assert event["position"] == [1., 2., 3.]
    assert event["first_frame"]["position"] == [20., 2., 3.]

    saved = session.reset(.3)
    archived = json.loads((saved / "manifest.json").read_text(encoding="utf-8"))
    assert archived["completion_reason"] == "space"
    session.prompt("new world", 2, "space", camera, {}, timestamp=.3)
    observe(session, [22., 2., 3.], .4, interacting=True)
    session.offer_frame(frame(.2, 3, [20., 2., 3.], 1))
    assert session.recorder.snapshot()["images"] == []
    session.offer_frame(frame(.4, 4, [22., 2., 3.], 2))
    session.recorder.flush()
    assert session.recorder.snapshot()["images"][0]["position"] == [22., 2., 3.]
    observe(session, [23., 2., 3.], .9, interacting=True)
    assert session.window.updates[-1][0]["id"] != archived["id"]
