"""Route marks follow movement, age out, and stay bounded during endless flight."""
import numpy as np

from app.renderer.player_trail import MAX_TRAIL_SAMPLES, PlayerTrail


def test_trail_fades_oldest_first_and_expires_while_standing():
    trail = PlayerTrail()
    for time in (0., 1., 2.):
        trail.update(np.array([time, 0., 0.]), time)
    origin = np.zeros(3)
    before = trail.vertices(origin, 2.)
    after = trail.vertices(origin, 3.)
    assert before.shape == (4, 4)
    assert before[0, 3] < before[-1, 3]
    assert np.all(after[:, 3] < before[:, 3])
    for time in (3., 4., 5., 6., 7.):
        trail.update(np.array([2., 0., 0.]), time)
    assert trail.vertices(origin, 7.).size == 0


def test_disabling_and_relocations_do_not_resurrect_or_bridge_routes():
    trail = PlayerTrail()
    trail.update(np.zeros(3), 0.)
    trail.update(np.ones(3), 1.)
    trail.update(np.ones(3), 1.1, enabled=False)
    assert trail.vertices(np.zeros(3), 1.1).size == 0
    trail.update(np.ones(3), 1.2)
    assert trail.vertices(np.zeros(3), 1.2).size == 0
    trail.update(np.ones(3) * 200., 2.)
    assert trail.vertices(np.zeros(3), 2.).size == 0


def test_long_high_refresh_flight_is_bounded_and_preserves_remote_detail():
    trail = PlayerTrail()
    origin = np.array([1e9, -1e9, 1e9])
    for frame in range(14400):
        time = frame / 240.
        trail.update(origin + np.array([0., time, 0.]), time)
    vertices = trail.vertices(origin, time)
    assert len(trail._samples) <= MAX_TRAIL_SAMPLES
    assert len(vertices) <= 2 * (MAX_TRAIL_SAMPLES - 1)
    assert vertices.dtype == np.float32 and vertices.flags.c_contiguous
    assert vertices[-1, 1] > 59.9
    assert np.all(np.diff(vertices[::2, 1]) > 0.)
    assert np.all(vertices[:, (0, 2)] == 0.)
