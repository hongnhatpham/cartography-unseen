from __future__ import annotations

from colorsys import rgb_to_hsv
from math import hypot
from random import Random

import pytest

from app.renderer.world import (
    _BIOME_GENERATOR,
    _blocks_walker,
    MIN_FORM_GAP,
    WALKER_RADIUS,
    walkable_components,
    BIOME_NAMES,
    BIOME_CELL,
    CLIMB_RATE,
    EYE_HEIGHT,
    CHUNK_SIZE,
    MAX_OBJECTS_PER_CHUNK,
    _chunk_grid,
    _sky_hsv,
    _stable_seed,
    active_chunk_coords,
    biome_roster,
    biome_weights,
    chunk_colliders,
    resolve_collisions,
    dominant_biome,
    generate_chunk,
    is_accent_region,
    plan_chunk_cache,
    ravine_depth,
    settle_height,
    STEP_MAX,
    step_blocked,
    pass_weight,
    WorldChunk,
    WorldCube,
    spawn_pose,
    surface_height,
    terrain_height,
    walk_height,
    world_label,
    world_palette,
    world_to_chunk,
)

SEEDS = (12345, 7, 9, 1, 17, 10, 1439185252)
COORDS = ((0, 0), (-1, -1), (-7, 3), (413, -908))


def test_chunk_generation_is_deterministic_including_negative_coordinates() -> None:
    for coord in COORDS:
        first = generate_chunk(coord, 12345)
        assert first == generate_chunk(coord, 12345)
        assert first != generate_chunk(coord, 999)


def test_height_field_has_no_seam_at_chunk_boundaries() -> None:
    """The landform is a pure function of world position, so chunks line up."""

    for seed in SEEDS:
        for boundary in (-CHUNK_SIZE, 0.0, CHUNK_SIZE * 5):
            left = terrain_height(boundary - 1e-4, 12.5, seed)
            right = terrain_height(boundary + 1e-4, 12.5, seed)
            assert abs(left - right) < 1e-3


def test_a_400_unit_flight_crosses_more_than_40_units_of_relief() -> None:
    """Guards the run-2 failure: a level horizon reads as a city after diffusion."""

    for seed in SEEDS:
        # Two diagonals: a single line can run along one terrace in a seed.
        relief = max(
            max(line) - min(line)
            for line in (
                [surface_height(step * 4.0, step * 2.0, seed) for step in range(101)],
                [surface_height(step * -2.0, step * 4.0, seed) for step in range(101)],
            )
        )
        assert relief > 40.0


def test_ground_columns_close_the_drop_onto_their_neighbours() -> None:
    """A column has to reach past its lowest neighbour or the cliff face is hollow."""

    for seed in (12345, 1439185252):
        for coord in ((0, 0), (-3, 5)):
            tops, _ = _chunk_grid(coord, seed)
            columns = [
                cube for cube in generate_chunk(coord, seed).objects
                if cube.role == "ground"
            ]
            # _ground_columns walks iz then ix over an 8x8 grid, in that order.
            assert len(columns) == 64
            for index, cube in enumerate(columns):
                ix, iz = index % 8, index // 8
                bottom = cube.position[1] - cube.half_extents[1]
                top = cube.position[1] + cube.half_extents[1]
                assert top == pytest.approx(tops[(ix, iz)])
                lowest = min(
                    tops[(ix + dx, iz + dz)]
                    for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1))
                )
                assert bottom <= lowest


def test_every_chunk_respects_the_hard_object_cap() -> None:
    for seed in SEEDS:
        for chunk_z in range(-2, 3):
            for chunk_x in range(-2, 3):
                chunk = generate_chunk((chunk_x, chunk_z), seed)
                assert 0 < len(chunk.objects) <= MAX_OBJECTS_PER_CHUNK


def test_biome_field_blends_and_changes_over_a_few_hundred_units() -> None:
    for seed in SEEDS:
        assert len(set(biome_roster(seed))) == 3
        # Weights are a partition of unity everywhere.
        for x, z in ((0.0, 0.0), (137.0, -412.0), (BIOME_CELL, BIOME_CELL * 0.5)):
            weights = biome_weights(x, z, seed)
            assert abs(sum(weights.values()) - 1.0) < 1e-9
            assert all(weight >= 0.0 for weight in weights.values())

    # Flying a few hundred units off the origin must reach another biome, and
    # at least one sampled step must be a genuine blend rather than a jump.
    crossed = False
    blended = False
    for seed in SEEDS:
        start = dominant_biome(0.0, 0.0, seed)
        for distance in range(0, 900, 25):
            weights = biome_weights(float(distance), float(distance) * 0.4, seed)
            top = max(weights.values())
            blended = blended or 0.25 < top < 0.85
            if dominant_biome(float(distance), float(distance) * 0.4, seed) != start:
                crossed = True
                break
    assert crossed and blended


def test_biome_generators_produce_distinct_form_signatures() -> None:
    """Ground is shared, so a biome's identity is the forms it adds on top."""

    signatures: dict[str, tuple[float, float, float]] = {}
    for name in BIOME_NAMES:
        cubes = []
        for chunk_x in range(3):
            coord = (chunk_x, 0)
            tops, cuts = _chunk_grid(coord, 12345)
            rng = Random(_stable_seed(12345, chunk_x, 0, BIOME_NAMES.index(name)))
            cubes += _BIOME_GENERATOR[name](coord, 12345, 1.0, rng, tops, cuts)
        assert all(cube.role == "form" for cube in cubes)
        tilted = sum(
            1
            for cube in cubes
            if abs(cube.rotation[0]) > 3.0 or abs(cube.rotation[2]) > 3.0
        ) / len(cubes)
        volumes = sorted(
            8.0 * cube.half_extents[0] * cube.half_extents[1] * cube.half_extents[2]
            for cube in cubes
        )
        lifts = sorted(
            cube.position[1] - terrain_height(cube.position[0], cube.position[2], 12345)
            for cube in cubes
        )
        signatures[name] = (
            round(tilted, 1),
            round(volumes[len(volumes) // 2], -1),
            round(lifts[len(lifts) // 2], -1),
        )

    assert len(set(signatures.values())) == len(BIOME_NAMES)
    assert signatures["shards"][0] > 0.5
    assert signatures["voxels"][1] < signatures["strata"][1]
    assert signatures["monoliths"][2] > signatures["voxels"][2]


def test_chunks_hold_several_orders_of_object_size() -> None:
    """Giant slabs next to gravel; a uniform mid-scale field reads as debris."""

    for seed in SEEDS:
        widths = sorted(
            max(cube.half_extents)
            for chunk_x in range(-1, 2)
            for chunk_z in range(-1, 2)
            for cube in generate_chunk((chunk_x, chunk_z), seed).objects
            if cube.role == "form"
        )
        # Smallest forms are now standing blocks at least a few units wide;
        # the giants must still be an order of magnitude bigger.
        assert widths[len(widths) // 20] < 5.0
        assert widths[-1] > 12.0


def test_object_hues_snap_to_the_world_palette() -> None:
    for seed in SEEDS:
        palette = world_palette(seed)
        for cube in generate_chunk((2, -3), seed).objects:
            hue = rgb_to_hsv(*cube.color)[0]
            assert min(abs(hue - option) for option in palette) < 1e-6


def test_the_accent_forms_contiguous_regions_not_scattered_cubes() -> None:
    """A per-cube coin flip does not survive diffusion; a whole terrace does."""

    for seed in SEEDS:
        samples = [
            (
                is_accent_region(index * 13.0, index * 29.0, seed),
                is_accent_region(index * 13.0 + 12.0, index * 29.0, seed),
            )
            for index in range(400)
        ]
        agreement = sum(here == nearby for here, nearby in samples) / len(samples)
        coverage = sum(here for here, _ in samples) / len(samples)
        assert agreement > 0.85
        assert 0.03 < coverage < 0.55


def test_every_seed_is_walkable_end_to_end_without_backtracking() -> None:
    """A walker must reach nearly all of a seed's ground and meet few pockets.

    The probe floods an 8-unit grid over a 640-unit window around the spawn,
    treating risers and standing forms exactly as the walker does. Before the
    corridor mesh and the form spacing this measured 0.00-0.58 reachable with
    5.6-9.8 per cent dead ends.
    """

    for seed in SEEDS:
        survey = walkable_components(seed)
        assert survey.open_cells
        assert survey.reachable_fraction >= 0.9, (seed, survey.reachable_fraction)
        assert survey.dead_end_fraction <= 0.03, (seed, survey.dead_end_fraction)


def test_standing_forms_leave_a_gap_wide_enough_to_walk_between() -> None:
    """Forms that span the eye line keep MIN_FORM_GAP clear of one another."""

    clearance = MIN_FORM_GAP + 2.0 * WALKER_RADIUS
    for seed in SEEDS[:3]:
        for coord in ((0, 0), (-1, 2)):
            forms = [
                cube
                for cube in generate_chunk(coord, seed).objects
                if _blocks_walker(cube, seed)
            ]
            for index, first in enumerate(forms):
                for second in forms[index + 1 :]:
                    gap = hypot(
                        first.position[0] - second.position[0],
                        first.position[2] - second.position[2],
                    ) - hypot(first.half_extents[0], first.half_extents[2]) - hypot(
                        second.half_extents[0], second.half_extents[2]
                    )
                    assert gap >= clearance - 1e-6, (seed, coord, gap)


def test_pass_corridors_form_one_connected_network() -> None:
    """Corridors are ridge lines, so they branch and meet instead of ringing."""

    for seed in SEEDS:
        cell = 16.0
        span = 20
        inside = {
            (ix, iz)
            for iz in range(-span, span + 1)
            for ix in range(-span, span + 1)
            if pass_weight(ix * cell, iz * cell, seed) > 0.85
        }
        assert inside
        seen: set[tuple[int, int]] = set()
        largest = 0
        for start in inside:
            if start in seen:
                continue
            stack, size = [start], 0
            seen.add(start)
            while stack:
                current = stack.pop()
                size += 1
                for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
                    other = (current[0] + dx, current[1] + dz)
                    if other in inside and other not in seen:
                        seen.add(other)
                        stack.append(other)
            largest = max(largest, size)
        assert largest / len(inside) > 0.85, (seed, largest, len(inside))


def test_ravine_floors_are_darker_than_the_plateau() -> None:
    """The conditioning needs real darks next to the pale slabs, not a 60-90% band.

    Floors sit around a third of the plateau value: dark enough to read as a
    channel, light enough that a walker standing in one still has a frame.
    """

    for seed in SEEDS:
        plateau: list[float] = []
        floors: list[float] = []
        for chunk_x in range(-2, 3):
            for cube in generate_chunk((chunk_x, 1), seed).objects:
                if cube.role != "ground":
                    continue
                value = rgb_to_hsv(*cube.color)[2]
                cut = ravine_depth(cube.position[0], cube.position[2], seed)
                (floors if cut > 40.0 else plateau).append(value)
        assert plateau
        if floors:
            assert max(floors) < 0.55
            assert sum(floors) / len(floors) < sum(plateau) / len(plateau) * 0.65


def test_ground_objects_stay_legible_against_the_sky() -> None:
    for seed in SEEDS:
        _, _, sky_value = _sky_hsv(seed)
        primary = world_palette(seed)[0]
        for cube in generate_chunk((0, 0), seed).objects:
            hue, _, value = rgb_to_hsv(*cube.color)
            if abs(hue - primary) < 1e-6:
                assert abs(value - sky_value) >= 0.24


def test_walker_settles_on_the_terrain_at_eye_height() -> None:
    for seed in SEEDS:
        for x, z in ((0.0, 0.0), (-311.0, 742.0), (1290.0, -88.0)):
            eye = walk_height(x, z, seed)
            # The ring average may dip below the cell underfoot beside a drop,
            # but never by more than a step.
            assert eye >= surface_height(x, z, seed) + EYE_HEIGHT - STEP_MAX
            # A snap (dt None) lands exactly on the walk height from anywhere.
            assert settle_height(x, eye - 500.0, z, seed, None) == pytest.approx(eye)
            assert settle_height(x, eye + 500.0, z, seed, None) == pytest.approx(eye)
            # A timed settle moves at most CLIMB_RATE * dt toward it.
            climbed = settle_height(x, eye - 100.0, z, seed, 0.1)
            assert climbed == pytest.approx(
                max(eye - 100.0 + CLIMB_RATE * 0.1, surface_height(x, z, seed) + EYE_HEIGHT * 0.5)
            )
            assert settle_height(x, eye + 100.0, z, seed, 0.1) == pytest.approx(eye + 100.0 - CLIMB_RATE * 0.1)


def test_walker_is_never_pushed_below_the_ground_underfoot() -> None:
    for seed in SEEDS:
        for x, z in ((0.0, 0.0), (-311.0, 742.0), (1290.0, -88.0)):
            floor = surface_height(x, z, seed)
            assert settle_height(x, floor - 50.0, z, seed, 0.01) >= floor + EYE_HEIGHT * 0.5


def test_spawn_stands_on_open_ground_inside_the_relief() -> None:
    for seed in SEEDS:
        (x, y, z), yaw, pitch = spawn_pose(seed)
        assert y == pytest.approx(walk_height(x, z, seed))
        assert 0.0 <= yaw < 360.0
        assert -13.0 < pitch < -3.0
        # The ring around the walker must rise or fall by a real terrace, so
        # the first frame has structure around the eye rather than a flat plate.
        ring = [
            surface_height(x + 40.0 * dx, z + 40.0 * dz, seed)
            for dx, dz in ((1, 0), (0, 1), (-1, 0), (0, -1), (0.7, 0.7), (-0.7, -0.7))
        ]
        here = surface_height(x, z, seed)
        assert max(ring) - here > 12.0 or here - min(ring) > 12.0


def test_world_label_names_the_local_biome_and_palette() -> None:
    label = world_label(12345, 0.0, 0.0)
    assert dominant_biome(0.0, 0.0, 12345) in label
    assert "/" in label


def test_active_window_cache_evicts_and_regenerates_identically() -> None:
    original = generate_chunk((0, 0), 12345)
    cache = {coord: generate_chunk(coord, 12345) for coord in active_chunk_coords(0.0, 0.0)}
    plan = plan_chunk_cache(cache, CHUNK_SIZE * 13, 0.0)
    assert set(plan.evict) == set(cache) - set(plan.desired)
    assert (0, 0) in plan.evict
    for coord in plan.evict:
        del cache[coord]
    for coord in plan.load:
        cache[coord] = generate_chunk(coord, 12345)
    assert len(cache) == len(plan.desired)
    assert generate_chunk((0, 0), 12345) == original


def test_world_to_chunk_handles_negative_boundaries() -> None:
    assert world_to_chunk(-0.001, -0.001) == (-1, -1)
    assert world_to_chunk(0.0, 0.0) == (0, 0)
    assert world_to_chunk(-CHUNK_SIZE, CHUNK_SIZE) == (-1, 1)


def test_negative_cache_radius_is_rejected() -> None:
    with pytest.raises(ValueError):
        active_chunk_coords(0.0, 0.0, -1)


def test_walker_slides_out_of_standing_forms_along_the_shallow_axis() -> None:
    block = WorldCube("form", (0.0, 3.0, 0.0), (2.0, 3.0, 4.0), (0.0, 0.0, 0.0), (0, 0, 0))
    colliders = chunk_colliders(WorldChunk((0, 0), ("strata",), (block,)))
    # Dead centre on the face: pushed clear along x, no sideways nudge.
    assert resolve_collisions(0.5, 2.2, 0.0, colliders) == pytest.approx((2.8, 0.0))
    # Off centre: pushed clear along the shallow axis and nudged toward the
    # nearer edge along the face, which is what makes head-on contact slide.
    x, z = resolve_collisions(0.0, 2.2, 3.5, colliders)
    assert z == pytest.approx(4.8) and x == pytest.approx(0.0, abs=1e-9)
    x, z = resolve_collisions(1.0, 2.2, 3.5, colliders)
    assert z == pytest.approx(4.8) and x > 1.0
    # Above the form there is nothing to hit.
    assert resolve_collisions(0.0, 9.0, 0.0, colliders) == (0.0, 0.0)
    # A yawed footprint resolves in its own frame.
    turned = WorldCube("form", (0.0, 3.0, 0.0), (2.0, 3.0, 4.0), (0.0, 90.0, 0.0), (0, 0, 0))
    colliders = chunk_colliders(WorldChunk((0, 0), ("strata",), (turned,)))
    x, z = resolve_collisions(3.0, 2.2, 0.0, colliders)
    assert x == pytest.approx(4.8) and z == pytest.approx(0.0, abs=1e-9)
    # Ground columns are not colliders; the walker stands on them.
    assert chunk_colliders(generate_chunk((0, 0), 7)).shape[1] == 8


def test_autowalk_turns_toward_open_ground_when_blocked() -> None:
    """A blocked step must retarget to the longest open heading, then ease to it."""

    from app.renderer.world import Autowalk, WorldChunk, chunk_colliders

    # A wall of blocks across +z; open ground lies toward -z (yaw 0 faces -z).
    wall = tuple(
        WorldCube("form", (x, 4.0, 6.0), (3.0, 4.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        for x in range(-30, 31, 6)
    )
    colliders = chunk_colliders(WorldChunk((0, 0), ("strata",), wall))
    walker = Autowalk(seed=3)
    yaw, pitch = 180.0, -4.0
    for frame in range(360):
        # Stationary against the wall: the windowed progress check must fire.
        yaw, pitch = walker.step(
            yaw, pitch, (0.0, 2.2, 0.0), frame / 60.0, 1 / 60.0,
            gained=0.0, step_length=1.0 / 60.0, colliders=colliders,
        )
    # Faces away from the wall: anywhere in the open half-plane (yaw within
    # 90 degrees of 0), since the pocket escape may leave along an edge.
    assert min(yaw, 360.0 - yaw) < 90.0
    # And it is walking again, not frozen mid-turn.
    assert not walker.turning(360 / 60.0)
    assert -15.0 <= pitch <= 7.0


def test_pass_corridors_smooth_the_terraces_and_risers_block_elsewhere() -> None:
    """Inside a pass the ground changes gently; on the terraces a riser is a wall."""

    for seed in SEEDS:
        gentle = 0
        passes = 0
        walls = 0
        for index in range(400):
            x, z = index * 7.3, index * 3.1
            if pass_weight(x, z, seed) > 0.9:
                passes += 1
                if not step_blocked(x, z, x + 4.0, z, seed):
                    gentle += 1
            elif step_blocked(x, z, x + 4.0, z, seed):
                walls += 1
        assert passes > 10, "every world needs pass corridors"
        assert gentle / passes > 0.9
        assert walls > 0
