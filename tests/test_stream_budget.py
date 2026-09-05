"""Slow chunks yield before the next chunk instead of consuming a whole frame."""
from types import SimpleNamespace

import numpy as np

from app.renderer import proxy_renderer as module
from app.renderer.world import WorldChunk
from tests.test_stream_uploads import make_renderer


def test_streaming_yields_after_expensive_chunk_and_finishes_same_world(monkeypatch):
    renderer = make_renderer()
    renderer.instance_buffers["cube"].write = lambda data, offset=0: None
    origin = (0, 0, 0)
    renderer._chunk_instances[origin] = np.empty((0, 19), dtype="f4")
    renderer._chunk_form_instances[origin] = {}
    renderer._stream_center = origin
    desired = (origin, (1, 0, 0), (2, 0, 0), (3, 0, 0))
    elapsed = [0.]
    def plan(existing, *args):
        return SimpleNamespace(center=origin, desired=desired, evict=(),
                               load=tuple(coord for coord in desired if coord not in existing))
    def generate(coord, seed):
        elapsed[0] += .004
        return WorldChunk(coord, (), ())
    monkeypatch.setattr(module, "plan_chunk_cache", plan)
    monkeypatch.setattr(module, "generate_chunk", generate)
    monkeypatch.setattr(module, "perf_counter", lambda: elapsed[0])
    renderer._update_world(np.zeros(3))
    assert len(renderer._chunk_instances) == 2, "A slow chunk must yield before building the next one"
    for _ in range(2):
        renderer._update_world(np.zeros(3))
    assert tuple(renderer._chunk_instances) == desired
