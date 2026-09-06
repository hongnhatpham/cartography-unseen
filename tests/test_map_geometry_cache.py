"""Check cached map geometry without opening a window or uploading textures."""
from copy import deepcopy

import numpy as np

from app import map_view


class FakeResource:
    def __init__(self, data=b""):
        self.data = data
        self.releases = 0

    def release(self):
        self.releases += 1


class FakeGL:
    def __init__(self):
        self.buffers = []

    def program(self, **_):
        return {}

    def enable(self, *_):
        pass

    def buffer(self, data):
        resource = FakeResource(data)
        self.buffers.append(resource)
        return resource

    def vertex_array(self, *_):
        return FakeResource()


def sample_snapshot(images=600, points=5):
    return {
        "id": "first", "current_pose": {"position": [32, 10, -8]},
        "prompts": [],
        "segments": [{"id": "s1", "ended": None, "points": [
            {"position": [float(i), float(i % 3), -float(i)], "timestamp": i}
            for i in range(points)]}],
        "images": [{"id": str(i), "path": f"images/{i}.png",
            "position": [float(i), float(i % 7), -float(i)],
            "rotation": [i % 40, i % 360, i % 15], "width": 384, "height": 256}
            for i in range(images)],
    }


def vertices(resource):
    return np.frombuffer(resource.data, dtype="f4").reshape(-1, 5)


def assert_matches_uncached_geometry(scene, snapshot):
    selected = map_view.representative_images(snapshot["images"],
                                               snapshot["current_pose"]["position"])
    wanted = {str(item["id"]) for item in selected}
    assert [str(plane[2]["id"]) for plane in scene.planes] == [
        str(item["id"]) for item in snapshot["images"] if str(item["id"]) in wanted]
    assert len(scene.planes) <= 512
    expected_outlines = []
    for item in snapshot["images"]:
        corners = map_view.image_corners(item)-scene.anchor
        if str(item["id"]) not in wanted:
            expected_outlines.extend(corners[[0, 1, 1, 2, 2, 3, 3, 0]])
    if expected_outlines:
        np.testing.assert_array_equal(vertices(scene.outlines[0])[:, :3],
                                      np.asarray(expected_outlines, dtype="f4"))
    else:
        assert scene.outlines is None
    for buffer, _, item in scene.planes:
        expected = np.column_stack((map_view.image_corners(item)-scene.anchor,
                                    [[0, 0], [1, 0], [1, 1], [0, 1]])).astype("f4")
        np.testing.assert_array_equal(vertices(buffer), expected)
    expected_lines = [segment["points"] for segment in snapshot["segments"]
                      if len(segment["points"]) > 1]
    assert len(scene.lines) == len(expected_lines)
    for (buffer, _, count), points in zip(scene.lines, expected_lines):
        assert count == len(points)
        np.testing.assert_array_equal(vertices(buffer)[:, :3],
            (np.asarray([point["position"] for point in points])-scene.anchor).astype("f4"))


def test_geometry_survives_merged_tail_updates_and_only_new_images_compute_corners(monkeypatch):
    calls = []
    original_corners = map_view.image_corners

    def counted(item):
        calls.append(item["id"])
        return original_corners(item)

    monkeypatch.setattr(map_view, "image_corners", counted)
    first = sample_snapshot()
    merged = map_view._merge_snapshot(None, map_view._snapshot_delta(None, first))
    scene = map_view._Scene(FakeGL())
    scene.update(merged)
    assert len(calls) == 600
    prior_planes = {str(plane[2]["id"]): plane for plane in scene.planes}
    prior_outlines = scene.outlines
    scene.nearest_at = 123.0

    tail_changed = deepcopy(first)
    tail_changed["segments"][0]["points"][-1]["position"] = [3.5, 8, -9]
    assert map_view._merge_snapshot(merged, map_view._snapshot_delta(first, tail_changed)) is merged
    scene.update(merged)
    assert len(calls) == 600
    assert scene.outlines is prior_outlines
    assert scene.nearest_at == 123.0
    assert all(plane is prior_planes[str(plane[2]["id"])] for plane in scene.planes)
    assert all(resource.releases == 0 for plane in scene.planes for resource in plane[:2])

    appended = deepcopy(tail_changed)
    appended["images"].append(sample_snapshot(images=601)["images"][-1])
    map_view._merge_snapshot(merged, map_view._snapshot_delta(tail_changed, appended))
    scene.update(merged)
    assert calls == [str(i) for i in range(601)]
    next_planes = {str(plane[2]["id"]): plane for plane in scene.planes}
    for key, plane in prior_planes.items():
        if key in next_planes:
            assert next_planes[key] is plane
            assert plane[0].releases == plane[1].releases == 0
        else:
            assert plane[0].releases == plane[1].releases == 1
    assert all(resource.releases == 1 for resource in prior_outlines)
    assert_matches_uncached_geometry(scene, merged)


def test_journey_reset_releases_cached_resources_and_reanchors_reused_image_ids():
    scene = map_view._Scene(FakeGL())
    first = sample_snapshot()
    scene.update(first)
    old_planes = list(scene.planes)
    old_outlines = scene.outlines
    old_texture = FakeResource()
    scene.textures["0"] = old_texture
    scene.retry["0"] = 100
    scene.nearest = list(scene.planes[:256])
    scene.nearest_at = 123

    next_map = sample_snapshot(images=1)
    next_map["id"] = "second"
    next_map["current_pose"]["position"] = [10000, 4, 3]
    next_map["images"][0]["position"] = [10010, 9, 8]
    scene.update(next_map)
    np.testing.assert_array_equal(scene.anchor, [10000, 4, 3])
    assert scene._image_count == 1
    assert set(scene._image_corners) == {"0"}
    assert not scene.textures and not scene.retry and not scene.nearest
    assert scene.nearest_at == 0
    assert old_texture.releases == 1
    assert all(resource.releases == 1 for plane in old_planes for resource in plane[:2])
    assert all(resource.releases == 1 for resource in old_outlines)
    assert_matches_uncached_geometry(scene, next_map)

    scene.update({"id": "third", "images": [], "segments": [], "prompts": []})
    assert not scene.planes and not scene.lines and scene.outlines is None
    assert not scene._image_corners and scene._image_count == 0
    assert scene.bounds is None


def test_moving_selection_reuses_surviving_planes_and_keeps_texture_budget():
    scene = map_view._Scene(FakeGL())
    snapshot = sample_snapshot(images=900)
    scene.update(snapshot)
    previous = {str(plane[2]["id"]): plane for plane in scene.planes}
    scene.nearest_at = float("inf")
    snapshot["current_pose"]["position"] = [880, 0, -880]
    scene.update(snapshot)
    assert scene.nearest_at == 0
    assert any(str(plane[2]["id"]) not in previous for plane in scene.planes)
    for plane in scene.planes:
        key = str(plane[2]["id"])
        if key in previous:
            assert plane is previous[key]
    assert_matches_uncached_geometry(scene, snapshot)

    # Seed fake resident textures so this exercises eviction without image I/O.
    scene.textures = {str(plane[2]["id"]): FakeResource() for plane in scene.planes}
    expected = map_view.representative_images([plane[2] for plane in scene.planes],
        snapshot["current_pose"]["position"], limit=256, nearby=192)
    scene._load_textures(snapshot["current_pose"]["position"])
    assert set(scene.textures) == {str(item["id"]) for item in expected}
    assert len(scene.textures) == len(scene.nearest) == 256
