"""Vertical streaming, world-coordinate continuity and the approved spacing."""

import numpy as np
import pytest

from app.renderer import world
from app.renderer.camera import Camera
from app.renderer.proxy_renderer import ProxyRenderer


def test_vertical_chunk_window_moves_and_evicts_in_both_directions():
    original = world.active_chunk_coords(0.0, 0.0, 0.0)
    for height in (-100_000.0, 100_000.0):
        plan = world.plan_chunk_cache(original, 0.0, height, 0.0)
        assert plan.center == (0, int(np.floor(height / world.CHUNK_SIZE)), 0)
        assert len(plan.desired) == world.MAX_ACTIVE_CHUNKS
        assert set(plan.evict) == set(original)
        assert not set(plan.desired) & set(original)
        assert min(c[1] for c in plan.desired) < plan.center[1] < max(c[1] for c in plan.desired)


def test_geometry_extends_above_and_below_the_previous_slab():
    for cy in (-10_000, -10, 10, 10_000):
        chunk = world.generate_chunk((0, cy, 0), 12345)
        assert chunk.objects
        assert all(abs(cube.position[1] - (cy + .5) * world.CHUNK_SIZE) < world.CHUNK_SIZE
                   for cube in chunk.objects)
        assert chunk == world.generate_chunk((0, cy, 0), 12345)
        cells = world.chunk_cells((0, cy, 0))
        assert len(cells) == world.CELLS_PER_CHUNK ** 3


def test_clear_passages_preserve_panels_and_match_the_selected_filter():
    seed = 934943880
    removed = 0
    for coord in ((0, 0, 0), (-3, -1, 0), (2, 1, 0)):
        panels, masses = [], []
        for cell in world.chunk_cells(coord):
            for axis in range(3):
                if not world.face_open(cell, axis, seed):
                    panels.extend(world._face_panels(cell, axis, seed))
            masses.extend(world._cell_interior(cell, seed))
        actual = world.generate_chunk(coord, seed)
        assert [c.position for c in actual.objects if c.role == "panel"] == [c.position for c in panels]
        expected = [c.position for c in masses if world.keep_interior(c.position, seed)]
        assert [c.position for c in actual.objects if c.role == "mass"] == expected
        removed += len(masses) - len(expected)
    assert removed > 0


def test_local_instance_packing_keeps_detail_far_from_origin():
    origin = np.array([0.0, 1_000_000_000.0, 0.0])
    cube = world.WorldCube("mass", (1.0, origin[1] + .25, 2.0), (1, 1, 1), (0, 0, 0), (1, 1, 1))
    packed = ProxyRenderer._pack_instances((cube,), origin=origin)
    assert packed[0, 12:15] == pytest.approx((1.0, .25, 2.0))
    camera = Camera(position=origin + (0.0, .125, 0.0))
    view = camera.snapshot().view_matrix
    assert view.dtype == np.float64
    assert view[1, 3] == -(origin[1] + .125)


def test_collision_does_not_clamp_free_flight_at_large_altitudes():
    renderer = ProxyRenderer.__new__(ProxyRenderer)
    renderer._chunk_colliders = {}
    for height in (-1_000_000_000.0, 1_000_000_000.0):
        camera = Camera(position=np.array((0.0, height, 0.0)))
        renderer._chunks = {world.world_to_chunk(*camera.position): None}
        renderer.constrain_camera(camera)
        assert camera.position[1] == height


def test_interior_thinning_follows_channels_and_preserves_dense_regions():
    retained = {"channel": [], "dense": []}
    for x in range(-320, 321, 16):
        for z in range(-320, 321, 16):
            position = (float(x), 0.0, float(z))
            weight = world.channel_weight(*position, 934943880)
            if weight > .9:
                retained["channel"].append(world.keep_interior(position, 934943880))
            elif weight < .05:
                retained["dense"].append(world.keep_interior(position, 934943880))
    assert .25 < np.mean(retained["channel"]) < .50
    assert np.mean(retained["dense"]) > .85
