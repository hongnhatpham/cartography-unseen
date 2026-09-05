from __future__ import annotations

from colorsys import rgb_to_hsv
from random import Random

import numpy as np
import pytest

from app.renderer.world import (
    SPAWN_HEIGHT_RANGE,
    BIOME_CELL,
    BIOME_NAMES,
    CELLS_PER_CHUNK,
    CHUNK_SIZE,
    LEGIBLE_OPEN,
    MAX_OBJECTS_PER_CHUNK,
    SPAWN_PITCHES,
    VOLUME_CELL,
    WALKER_RADIUS,
    Autowalk,
    WorldChunk,
    WorldCube,
    _BIOME_GENERATOR,
    _stable_seed,
    active_chunk_coords,
    biome_roster,
    biome_weights,
    channel_weight,
    chunk_cells,
    chunk_colliders,
    direction_of,
    dominant_biome,
    face_open,
    generate_chunk,
    is_accent_region,
    is_blocked,
    legibility,
    nadir_color,
    occupancy_grid,
    open_heading,
    plan_chunk_cache,
    resolve_collisions,
    sky_color,
    spawn_pose,
    walkable_components,
    world_label,
    world_palette,
    world_to_chunk,
    zenith_color,
)

SEEDS = (12345, 7, 9, 1, 17, 10, 1439185252)
COORDS = ((0, 0, 0), (-1, -1, -1), (-7, 18, 3), (413, -1200, -908))


def _wall(centre, half) -> WorldCube:
    return WorldCube("panel", centre, half, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def test_chunk_generation_is_deterministic_including_negative_coordinates() -> None:
    for coord in COORDS:
        first = generate_chunk(coord, 12345)
        assert first == generate_chunk(coord, 12345)
        assert first != generate_chunk(coord, 999)


def test_every_chunk_respects_the_hard_object_cap() -> None:
    for seed in SEEDS:
        for chunk_z in range(-2, 3):
            for chunk_x in range(-2, 3):
                chunk = generate_chunk((chunk_x, 0, chunk_z), seed)
                assert 0 < len(chunk.objects) <= MAX_OBJECTS_PER_CHUNK


def test_each_volume_cell_belongs_to_exactly_one_chunk() -> None:
    seen: set[tuple[int, int, int]] = set()
    for chunk_x in range(-2, 2):
        for chunk_y in range(-2, 2):
            for chunk_z in range(-2, 2):
                cells = chunk_cells((chunk_x, chunk_y, chunk_z))
                assert len(cells) == CELLS_PER_CHUNK**3
                assert not seen & set(cells)
                seen.update(cells)


def test_the_volume_is_one_connected_space() -> None:
    """Broad passages remain connected after thinning interior groups."""

    for seed in SEEDS:
        grid = occupancy_grid(seed)
        survey = walkable_components(seed, grid=grid)
        assert survey.open_cells > 0
        assert survey.reachable_fraction >= 0.95, (seed, survey.reachable_fraction)


def test_channel_field_opens_the_faces_it_runs_through() -> None:
    """Openings string into routes instead of scattering, which is connectivity."""

    for seed in SEEDS[:3]:
        inside_open = inside = outside_open = outside = 0
        for index in range(1500):
            cell = (index % 17 - 8, index // 17 % 18 - 9, index // 306 - 2)
            centre = tuple((value + 0.5) * VOLUME_CELL for value in cell)
            channel = channel_weight(*centre, seed)
            for axis in range(3):
                if channel > 0.8:
                    inside += 1
                    inside_open += face_open(cell, axis, seed)
                elif channel < 0.05:
                    outside += 1
                    outside_open += face_open(cell, axis, seed)
        assert inside > 0 and outside > 0
        assert inside_open / inside > outside_open / outside + 0.2


def test_biome_field_blends_in_three_dimensions_and_changes_over_a_few_hundred_units() -> None:
    for seed in SEEDS:
        assert len(set(biome_roster(seed))) == 3
        for point in ((0.0, 0.0, 0.0), (137.0, -60.0, -412.0), (BIOME_CELL, 80.0, 0.0)):
            weights = biome_weights(*point, seed)
            assert abs(sum(weights.values()) - 1.0) < 1e-9
            assert all(weight >= 0.0 for weight in weights.values())

    # Flying a few hundred units off the origin, or straight up, must reach
    # another biome, and at least one sampled step must be a genuine blend.
    crossed = blended = climbed = False
    for seed in SEEDS:
        start = dominant_biome(0.0, 0.0, 0.0, seed)
        for distance in range(0, 900, 25):
            weights = biome_weights(float(distance), 0.0, float(distance) * 0.4, seed)
            blended = blended or 0.25 < max(weights.values()) < 0.85
            crossed = crossed or dominant_biome(
                float(distance), 0.0, float(distance) * 0.4, seed
            ) != start
        for height in range(-int(SPAWN_HEIGHT_RANGE), int(SPAWN_HEIGHT_RANGE), 20):
            climbed = climbed or dominant_biome(0.0, float(height), 0.0, seed) != start
    assert crossed and blended and climbed


def test_biome_generators_produce_distinct_form_signatures() -> None:
    """The shell is shared, so a biome's identity is what it hangs in the cells."""

    signatures: dict[str, tuple[float, float, float]] = {}
    for name in BIOME_NAMES:
        cubes: list[WorldCube] = []
        for index in range(40):
            rng = Random(_stable_seed(12345, index, BIOME_NAMES.index(name)))
            cubes += _BIOME_GENERATOR[name]((0.0, 0.0, 0.0), rng)
        assert all(cube.role == "mass" for cube in cubes)
        tilted = sum(
            1
            for cube in cubes
            if abs(cube.rotation[0]) > 3.0 or abs(cube.rotation[2]) > 3.0
        ) / len(cubes)
        volumes = sorted(
            8.0 * cube.half_extents[0] * cube.half_extents[1] * cube.half_extents[2]
            for cube in cubes
        )
        signatures[name] = (
            round(tilted, 1),
            round(volumes[len(volumes) // 2], -1),
            round(len(cubes) / 40.0),
        )

    assert len(set(signatures.values())) == len(BIOME_NAMES)
    assert signatures["shards"][0] > 0.9
    assert signatures["reefs"][1] < signatures["overhangs"][1]


def test_chunks_hold_several_orders_of_object_size() -> None:
    """Wall-sized panels next to wire-thin bars; one mid scale reads as debris."""

    for seed in SEEDS:
        cubes = [
            cube
            for chunk_x in range(-1, 2)
            for chunk_y in range(-4, 5)
            for chunk_z in range(-1, 2)
            for cube in generate_chunk((chunk_x, chunk_y, chunk_z), seed).objects
        ]
        volumes = sorted(
            8.0 * cube.half_extents[0] * cube.half_extents[1] * cube.half_extents[2]
            for cube in cubes
        )
        assert volumes[-1] / volumes[0] > 50.0
        assert np.percentile(volumes, 90) / np.percentile(volumes, 10) > 15.0
        # Something thin enough to read as a bar, something wide enough to be a
        # wall the flier cannot see past.
        assert min(min(cube.half_extents) for cube in cubes) < 2.0
        assert max(max(cube.half_extents) for cube in cubes) > 12.0


def test_object_hues_snap_to_the_world_palette() -> None:
    for seed in SEEDS:
        palette = world_palette(seed)
        for cube in generate_chunk((2, 0, -3), seed).objects:
            hue = rgb_to_hsv(*cube.color)[0]
            assert min(abs(hue - option) for option in palette) < 1e-6


def test_the_accent_is_a_contiguous_three_dimensional_vein() -> None:
    """A per-cube coin flip does not survive diffusion; a whole wall does."""

    for seed in SEEDS:
        rng = Random(seed)
        samples = []
        for _ in range(600):
            x, y, z = rng.uniform(-400, 400), rng.uniform(-260, 260), rng.uniform(-400, 400)
            samples.append(
                (
                    is_accent_region(x, y, z, seed),
                    is_accent_region(x + 12.0, y, z, seed),
                    is_accent_region(x, y + 12.0, z, seed),
                )
            )
        agreement = sum(
            (here == beside) + (here == above) for here, beside, above in samples
        ) / (2 * len(samples))
        coverage = sum(here for here, _, _ in samples) / len(samples)
        assert agreement > 0.85
        assert 0.02 < coverage < 0.30


def test_objects_stay_legible_against_the_fog_they_share_a_hue_with() -> None:
    for seed in SEEDS:
        sky_value = rgb_to_hsv(*sky_color(seed))[2]
        primary = world_palette(seed)[0]
        for cube in generate_chunk((0, 0, 0), seed).objects:
            hue, _, value = rgb_to_hsv(*cube.color)
            if abs(hue - primary) < 1e-6:
                assert abs(value - sky_value) >= 0.24


def test_the_background_darkens_below_and_saturates_above() -> None:
    """No ground plane, so the only cue for which way is up is this gradient."""

    for seed in SEEDS:
        fog_value = rgb_to_hsv(*sky_color(seed))[2]
        zenith_hue, zenith_sat, zenith_value = rgb_to_hsv(*zenith_color(seed))
        nadir_value = rgb_to_hsv(*nadir_color(seed))[2]
        assert nadir_value < 0.2
        assert zenith_value < fog_value - 0.2
        assert zenith_sat > 0.4
        assert abs(zenith_hue - world_palette(seed)[1]) < 1e-6


def test_collision_pushes_out_along_the_shallowest_axis_in_three_dimensions() -> None:
    block = _wall((0.0, 0.0, 0.0), (2.0, 6.0, 8.0))
    colliders = chunk_colliders(WorldChunk((0, 0, 0), ("reefs",), (block,)))
    radius = 0.5
    # Dead centre on the +x face: pushed clear along x, no sideways nudge.
    assert resolve_collisions(0.5, 0.0, 0.0, colliders, radius) == pytest.approx(
        (2.5, 0.0, 0.0)
    )
    # Nearer the top than the side: resolved upward instead.
    x, y, z = resolve_collisions(0.0, 5.9, 0.0, colliders, radius)
    assert y == pytest.approx(6.5)
    assert (x, z) == pytest.approx((0.0, 0.0), abs=1e-9)
    # And downward on the other face, which gravity used to make impossible.
    assert resolve_collisions(0.0, -5.9, 0.0, colliders, radius)[1] == pytest.approx(-6.5)
    # Clear of the box on every axis: untouched.
    assert resolve_collisions(0.0, 9.0, 0.0, colliders, radius) == (0.0, 9.0, 0.0)
    # A yawed box resolves in its own frame: the point ends up outside a box
    # whose 2-unit half extent now runs along world z.
    turned = WorldCube("panel", (0.0, 0.0, 0.0), (2.0, 6.0, 8.0), (0.0, 90.0, 0.0), (0, 0, 0))
    colliders = chunk_colliders(WorldChunk((0, 0, 0), ("reefs",), (turned,)))
    assert is_blocked(0.0, 0.0, 1.0, colliders, radius)
    assert not is_blocked(*resolve_collisions(0.0, 0.0, 1.0, colliders, radius), colliders, radius)


def test_a_tilted_plate_is_a_solid_box_not_a_surface_to_slip_through() -> None:
    """Tilt is folded into the vertical half extent, so a shard still blocks."""

    flat = WorldCube("mass", (0.0, 0.0, 0.0), (8.0, 0.5, 8.0), (0.0, 0.0, 0.0), (0, 0, 0))
    tilted = WorldCube("mass", (0.0, 0.0, 0.0), (8.0, 0.5, 8.0), (0.0, 0.0, 40.0), (0, 0, 0))
    rows_flat = chunk_colliders(WorldChunk((0, 0, 0), ("shards",), (flat,)))
    rows_tilted = chunk_colliders(WorldChunk((0, 0, 0), ("shards",), (tilted,)))
    assert rows_tilted[0, 6] > rows_flat[0, 6] + 4.0
    assert not is_blocked(0.0, 3.0, 0.0, rows_flat, 0.5)
    assert is_blocked(0.0, 3.0, 0.0, rows_tilted, 0.5)


def test_open_heading_reports_a_pitch_as_well_as_a_yaw() -> None:
    """Boxed in on every side but the top, the search has to look up."""

    walls = (
        _wall((24.0, 20.0, 0.0), (4.0, 60.0, 30.0)),
        _wall((-24.0, 20.0, 0.0), (4.0, 60.0, 30.0)),
        _wall((0.0, 20.0, 24.0), (30.0, 60.0, 4.0)),
        _wall((0.0, 20.0, -24.0), (30.0, 60.0, 4.0)),
        _wall((0.0, -14.0, 0.0), (30.0, 10.0, 30.0)),
    )
    colliders = chunk_colliders(WorldChunk((0, 0, 0), ("reefs",), walls))
    yaw, pitch, distance = open_heading(0.0, 0.0, 0.0, colliders)
    assert 0.0 <= yaw < 360.0
    assert pitch > 60.0
    assert distance > 40.0


def test_spawn_floats_in_an_open_pocket_with_somewhere_to_fly() -> None:
    for seed in SEEDS:
        (x, y, z), yaw, pitch = spawn_pose(seed)
        assert 0.0 <= yaw < 360.0
        assert pitch in SPAWN_PITCHES
        assert abs(y) <= SPAWN_HEIGHT_RANGE
        colliders = np.concatenate(
            [
                chunk_colliders(generate_chunk((world_to_chunk(x, y, z)[0] + dx,
                                                world_to_chunk(x, y, z)[1] + dy,
                                                world_to_chunk(x, y, z)[2] + dz), seed))
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                for dz in (-1, 0, 1)
            ]
        )
        assert not is_blocked(x, y, z, colliders)
        # The spawn faces a real run, not a wall a step away.
        forward = direction_of(yaw, pitch)
        clear = 0.0
        while clear < LEGIBLE_OPEN:
            clear += 2.0
            if is_blocked(
                x + forward[0] * clear, y + forward[1] * clear, z + forward[2] * clear,
                colliders,
            ):
                break
        assert clear >= 12.0, (seed, clear)


def test_autopilot_steers_yaw_and_pitch_away_from_a_blocked_line() -> None:
    """Blocked on every side but above, the drift has to climb out."""

    walls = (
        _wall((24.0, 20.0, 0.0), (4.0, 60.0, 30.0)),
        _wall((-24.0, 20.0, 0.0), (4.0, 60.0, 30.0)),
        _wall((0.0, 20.0, 24.0), (30.0, 60.0, 4.0)),
        _wall((0.0, 20.0, -24.0), (30.0, 60.0, 4.0)),
        _wall((0.0, -14.0, 0.0), (30.0, 10.0, 30.0)),
    )
    colliders = chunk_colliders(WorldChunk((0, 0, 0), ("reefs",), walls))
    walker = Autowalk(seed=3)
    yaw, pitch = 180.0, 0.0
    for frame in range(360):
        yaw, pitch = walker.step(
            yaw, pitch, (0.0, 0.0, 0.0), frame / 60.0, 1 / 60.0,
            gained=0.0, step_length=1.0 / 60.0, colliders=colliders,
        )
    assert pitch > 40.0
    assert -88.0 <= pitch <= 88.0
    assert not walker.turning(360 / 60.0)


def test_world_label_names_the_local_biome_and_palette() -> None:
    label = world_label(12345, 0.0, 0.0, 0.0)
    assert dominant_biome(0.0, 0.0, 0.0, 12345) in label
    assert "/" in label


def test_active_window_cache_evicts_and_regenerates_identically() -> None:
    original = generate_chunk((0, 0, 0), 12345)
    cache = {coord: generate_chunk(coord, 12345) for coord in active_chunk_coords(0.0, 0.0, 0.0)}
    plan = plan_chunk_cache(cache, 0.0, CHUNK_SIZE * 13, 0.0)
    assert set(plan.evict) == set(cache) - set(plan.desired)
    assert (0, 0, 0) in plan.evict
    for coord in plan.evict:
        del cache[coord]
    for coord in plan.load:
        cache[coord] = generate_chunk(coord, 12345)
    assert len(cache) == len(plan.desired)
    assert generate_chunk((0, 0, 0), 12345) == original


def test_world_to_chunk_handles_negative_boundaries() -> None:
    assert world_to_chunk(-0.001, -0.001, -0.001) == (-1, -1, -1)
    assert world_to_chunk(0.0, 0.0, 0.0) == (0, 0, 0)
    assert world_to_chunk(-CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE) == (-1, 1, 1)


def test_negative_cache_radius_is_rejected() -> None:
    with pytest.raises(ValueError):
        active_chunk_coords(0.0, 0.0, 0.0, -1)


def test_the_body_is_wide_enough_to_keep_the_camera_off_the_panels() -> None:
    """At 82 degrees a panel one unit away is the frame, not a surface."""

    assert WALKER_RADIUS >= 2.0
    assert WALKER_RADIUS < VOLUME_CELL * 0.25
