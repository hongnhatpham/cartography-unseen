"""Additive geometry preserves the existing world and the openings between cells."""
from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest

from app.renderer import world
from app.renderer.form_meshes import form_meshes


@pytest.mark.parametrize("seed,coord,expected", [
    (0, (0, 0, 0), "111e5532c58467ced4adc8a77be052ce1b725e9c703747b799daaa04ee258d90"),
    (42, (-2, 1, 3), "bf9a331a5d62428c102a32dbcc0b72a3e0dae1ae613711acbf760083e584aa2b"),
    (123, (5, -8, 9), "a9cf688aacf8b05b675df869039c1a627746b3db155dd45aed2a9bf876adf5bd"),
    (987654, (2, 15625000, -3), "aae81542ccb55811dd6db7b9409bf9773867dc43e063e20194190fbaaab5b5f0"),
])
def test_original_geometry_matches_checkpoint(seed, coord, expected):
    # Digests captured from checkpoint-before-additive-forms, excluding colour.
    chunk = world.generate_chunk(coord, seed)
    geometry = [(o.role, o.position, o.half_extents, o.rotation, o.tone) for o in chunk.objects]
    assert sha256(repr(geometry).encode()).hexdigest() == expected


def test_templates_have_outward_normals_and_bounded_complexity():
    for vertices in form_meshes().values():
        assert vertices.dtype == np.float32 and vertices.flags.c_contiguous
        assert len(vertices) <= 2160
        assert np.isfinite(vertices).all()
        assert np.abs(vertices[:, :3]).max() <= 1.000001
        np.testing.assert_allclose(np.linalg.norm(vertices[:, 3:], axis=1), 1, atol=1e-6)
        triangles = vertices.reshape(-1, 3, 6)
        normal = np.cross(triangles[:, 1, :3] - triangles[:, 0, :3],
                          triangles[:, 2, :3] - triangles[:, 0, :3])
        assert np.min(np.sum(normal * triangles[:, :, 3:].mean(axis=1), axis=1)) >= -1e-7


def test_form_families_repeat_deterministically_at_remote_altitudes():
    seen = set()
    for height in (-15625000, 0, 15625000):
        for x in range(-3, 4):
            chunk = world.generate_chunk((x, height, 0), 42)
            seen.update(form.mesh for form in chunk.forms)
            assert len(chunk.forms) <= world.MAX_FORMS_PER_CHUNK
            world.generate_chunk.cache_clear()
            assert chunk == world.generate_chunk(chunk.coord, 42)
    assert seen == set(form_meshes())


def test_companions_leave_cell_boundaries_and_strong_channels_clear():
    checked = 0
    for x in range(-3, 4):
        for y in range(-2, 3):
            chunk = world.generate_chunk((x, y, 0), 42)
            for form in chunk.forms:
                checked += 1
                rows = world.form_colliders(form)
                low, high = (rows[:, :3] - rows[:, 5:]).min(axis=0), (rows[:, :3] + rows[:, 5:]).max(axis=0)
                owner = np.floor(np.asarray(form.position) / world.VOLUME_CELL)
                assert np.all(low >= owner * world.VOLUME_CELL + world.WALKER_RADIUS + .99)
                assert np.all(high <= (owner + 1) * world.VOLUME_CELL - world.WALKER_RADIUS - .99)
                assert world.channel_weight(*form.position, 42) < .95
    assert checked > 30


def test_cage_and_ribs_have_camera_sized_openings():
    cage = world.WorldForm("cage", (0., 0., 0.), (10., 10., 10.), (0., 0., 0.), (1., 1., 1.))
    rows = world.form_colliders(cage)
    for z in np.linspace(-15, 15, 121):
        assert not world.is_blocked(0, 0, z, rows)
    assert world.is_blocked(8.5, 8.5, 0, rows)
    ribs = replace(cage, mesh="ribs2")
    rows = world.form_colliders(ribs)
    for z in np.linspace(-15, 15, 121):
        assert not world.is_blocked(0, 0, z, rows)
    assert world.is_blocked(0, 8, 0, rows)


def test_wire_collision_follows_rendered_segments_after_rotation():
    for name, vertices in form_meshes().items():
        form = world.WorldForm(name, (14., -8., 21.), (11., 6., 8.), (17., 38., 23.), (1., 1., 1.))
        rows = world.form_colliders(form)
        linear = world._form_rotation(form.rotation) * np.asarray(form.half_extents)[None, :]
        points = vertices[::7, :3] @ linear.T + np.asarray(form.position)
        # Every sampled surface vertex belongs to a collision piece even when
        # the mesh is tilted and scaled differently along its three axes.
        assert all(world.is_blocked(*point, rows, radius=1e-4) for point in points)
    weave = replace(form, mesh="weave2", position=(0., 0., 0.), half_extents=(12., 12., 12.), rotation=(0., 0., 0.))
    assert not world.is_blocked(2.55, 2.55, 0., world.form_colliders(weave), radius=.1)


def test_heading_intervals_match_sampled_collision_with_open_frames():
    chunk = world.generate_chunk((0, 0, 0), 42)
    rows = world.chunk_colliders(chunk)
    directions = tuple(world.direction_of(yaw, pitch)
                       for yaw in range(0, 360, 45) for pitch in (-90., -30., 0., 30., 90.))
    step, reach = 1.25, 70.
    for origin in ((16., 16., 16.), (0., 0., 0.), (32., 32., 32.)):
        expected = []
        for direction in directions:
            distance = reach
            for sample in range(1, int(reach / step) + 1):
                point = np.asarray(origin) + np.asarray(direction) * sample * step
                if world.is_blocked(*point, rows):
                    distance = (sample - 1) * step
                    break
            expected.append(distance)
        np.testing.assert_array_equal(world.clear_distances(origin, directions, rows), expected)
