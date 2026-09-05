"""Streaming uploads retain the complete world without copying it each tick."""
import numpy as np

from app.renderer.proxy_renderer import ProxyRenderer
from app.renderer.world import MAX_ACTIVE_CHUNKS


class RecordingBuffer:
    def __init__(self):
        self.data = bytearray()
        self.transferred = 0

    @property
    def size(self):
        return len(self.data)

    def orphan(self, size):
        self.data = bytearray(size)

    def write(self, data, offset=0):
        raw = memoryview(data).cast('B')
        assert offset + len(raw) <= self.size
        self.data[offset:offset + len(raw)] = raw
        self.transferred += len(raw)


def make_renderer():
    r = ProxyRenderer.__new__(ProxyRenderer)
    r.world_seed = 934943880
    r._chunk_instances = {}
    r._chunk_form_instances = {}
    r._form_instances = {}
    r._form_bounds = {}
    r._chunk_colliders = {}
    r._stream_center = None
    r._render_origin = np.zeros(3, dtype=np.float64)
    r._instance_counts = {"cube": 0}
    r._instance_buffer_revision = 0
    r.instance_buffers = {"cube": RecordingBuffer()}
    return r


def assert_complete_geometry(r):
    count = r._instance_counts['cube']
    actual = np.frombuffer(r.instance_buffers['cube'].data, dtype='f4', count=count*19).reshape(-1, 19)
    expected = []
    for coord, local in r._chunk_instances.items():
        part = local.copy()
        part[:, 12:15] += np.array([coord[i] - r._stream_center[i] for i in range(3)]) * 64
        expected.extend(row.tobytes() for row in part)
    assert sorted(row.tobytes() for row in actual) == sorted(expected)
    expected_forms = {}
    for coord, batches in r._chunk_form_instances.items():
        for mesh, local in batches.items():
            part = local.copy()
            part[:, 12:15] += np.array([coord[i] - r._stream_center[i] for i in range(3)]) * 64
            expected_forms.setdefault(mesh, []).extend(row.tobytes() for row in part)
    assert set(r._form_instances) == set(expected_forms)
    for mesh, rows in expected_forms.items():
        assert sorted(row.tobytes() for row in r._form_instances[mesh]) == sorted(rows)


def test_partial_load_uploads_only_new_geometry_and_preserves_retained_chunks():
    r = make_renderer()
    r._update_world(np.array([12., 12., 12.]))
    for position in ([76., 12., 12.], [76., -52., 12.], [12., 12., 12.],
                     [12., 1_000_000_012., 12.], [12., -999_999_988., 12.]):
        r._update_world(np.array(position))
        assert_complete_geometry(r)
        buffer = r.instance_buffers['cube']
        buffer.transferred = 0
        for _ in range(450):
            r._update_world(np.array(position))
            if len(r._chunk_instances) == MAX_ACTIVE_CHUNKS:
                break
        assert len(r._chunk_instances) == MAX_ACTIVE_CHUNKS
        assert_complete_geometry(r)
        # A boundary exposes one 11x11 chunk face. Refilling it should transfer
        # far less than even two complete worlds, including a buffer resize.
        assert buffer.transferred < r._instance_counts['cube'] * 19 * 4 * 2
