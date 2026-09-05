"""Exact caches must retain world geometry and stay bounded during endless flight."""
from random import Random

import pytest

from app.renderer import world


def test_channel_cache_preserves_complete_chunks(monkeypatch):
    cached = world.channel_weight
    uncached = cached.__wrapped__
    try:
        for seed in (12345, 934943880):
            for coord in ((0, 0, 0), (-1, 2, -3), (23, -17, 11), (0, 15_625_000, 0)):
                cached.cache_clear()
                with monkeypatch.context() as patch:
                    patch.setattr(world, "channel_weight", uncached)
                    expected = world.generate_chunk.__wrapped__(coord, seed)
                assert world.generate_chunk.__wrapped__(coord, seed) == expected
    finally:
        cached.cache_clear()


def test_channel_cache_reuses_exact_points_and_evicts(monkeypatch):
    calls = []

    def ridge(*args):
        calls.append(args)
        return .8

    world.channel_weight.cache_clear()
    monkeypatch.setattr(world, "ridge", ridge)
    try:
        sample = (16., -16., 48., 12345)
        expected = world.channel_weight(*sample)
        assert world.channel_weight(*sample) == expected
        assert len(calls) == len(world._CHANNEL_CELLS)
        world.channel_weight(*sample[:3], 9876)
        assert len(calls) == 2 * len(world._CHANNEL_CELLS)
        limit = world.channel_weight.cache_info().maxsize
        assert limit is not None and limit <= 32768
        for index in range(limit + 1):
            world.channel_weight(float(index), 0., 0., 42)
        assert world.channel_weight.cache_info().currsize == limit
        before = len(calls)
        world.channel_weight(*sample)
        assert len(calls) == before + len(world._CHANNEL_CELLS)
    finally:
        world.channel_weight.cache_clear()


@pytest.mark.parametrize("radius", (0, 1, 5))
def test_cached_window_plan_matches_original_order_for_partial_and_moved_caches(radius):
    rng = Random(173)
    existing = set()
    for center in ((0, 0, 0), (0, 0, 0), (1, -1, 0), (1, -1, 0), (-9, 15_625_000, 3)):
        position = tuple(axis * world.CHUNK_SIZE + 12 for axis in center)
        plan = world.plan_chunk_cache(existing, *position, radius)
        offsets = sorted(
            ((x, y, z) for x in range(-radius, radius + 1)
             for y in range(-radius, radius + 1) for z in range(-radius, radius + 1)),
            key=lambda coord: (max(map(abs, coord)), sum(map(abs, coord)), coord),
        )
        desired = tuple(tuple(center[i] + offset[i] for i in range(3)) for offset in offsets)
        order = {coord: index for index, coord in enumerate(desired)}
        assert plan == world.ChunkCachePlan(
            center, desired, tuple(coord for coord in desired if coord not in existing),
            tuple(sorted(existing & set(desired), key=order.__getitem__)),
            tuple(sorted(existing - set(desired))),
        )
        existing = {coord for coord in desired if rng.random() < .7}
