from __future__ import annotations

from collections import defaultdict
from math import atan2, cos, degrees, radians, sin

import pytest

from app.renderer.city import (
    ACTIVE_CHUNK_RADIUS,
    BUILDING_ROAD_SETBACK,
    CHUNK_SIZE,
    JUNCTION_HEIGHT_TOLERANCE,
    MAX_ACTIVE_CHUNKS,
    MAX_OBJECTS_PER_CHUNK,
    MAX_WALK_STEP,
    MIN_ALLEY_WIDTH,
    ROAD_HALF_THICKNESS,
    ROAD_SEGMENT_LENGTH,
    TERRACE_STEP,
    WIRE_SEGMENTS_PER_RUN,
    _bridge_lift,
    _contact_from_projection,
    _nearby_roads,
    _nearest_road,
    _road_curve,
    _road_point,
    _road_projection,
    _rendered_road_height,
    _terrain_plate_height,
    _wire_cube,
    active_chunk_coords,
    generate_chunk,
    move_road_contact,
    plan_chunk_cache,
    sky_color,
    spawn_road_contact,
    terrain_height,
    walkable_height,
    world_to_chunk,
)


def _road_endpoints(road):
    assert road.basis is not None
    axis = road.basis[0]
    half_length = road.half_extents[0]
    return tuple(
        road.position[index] - axis[index] * half_length for index in range(3)
    ), tuple(road.position[index] + axis[index] * half_length for index in range(3))


def test_chunk_generation_is_deterministic_for_negative_coordinates() -> None:
    expected = generate_chunk((-3, -4), 98123)
    generate_chunk((100, -75), 7)
    assert generate_chunk((-3, -4), 98123) == expected
    assert generate_chunk((-3, -4), 98124) != expected


def test_terrain_is_continuous_across_chunk_seams_and_has_relief() -> None:
    seed = 2026
    epsilon = 1e-5
    for seam in (-CHUNK_SIZE, 0.0, CHUNK_SIZE):
        for offset in (-47.5, -1.0, 19.25, 61.0):
            assert terrain_height(seam - epsilon, offset, seed) == pytest.approx(
                terrain_height(seam + epsilon, offset, seed), abs=1e-4
            )
            assert terrain_height(offset, seam - epsilon, seed) == pytest.approx(
                terrain_height(offset, seam + epsilon, seed), abs=1e-4
            )
    samples = [
        terrain_height(x, z, seed)
        for x in range(-256, 257, 32)
        for z in range(-256, 257, 32)
    ]
    assert 12.0 <= max(samples) - min(samples) <= 32.0
    assert terrain_height(0.0, 0.0, seed) == 0.0


def test_road_segments_share_endpoints_across_chunk_boundaries() -> None:
    chunks = [
        generate_chunk((x, z), 2718)
        for z in range(-2, 3)
        for x in range(-2, 3)
    ]
    by_road = defaultdict(dict)
    owners = {}
    for chunk in chunks:
        for road in (item for item in chunk.surfaces if item.role == "road"):
            assert road.cell is not None
            by_road[road.cell[0]][road.cell[1]] = road
            owners[road.cell] = chunk.coord
    seam_pairs = 0
    for road_tag, segments in by_road.items():
        for index, road in segments.items():
            following = segments.get(index + 1)
            if following is None:
                continue
            _, end = _road_endpoints(road)
            start, _ = _road_endpoints(following)
            assert end == pytest.approx(start, abs=1e-10)
            for basis in (road.basis, following.basis):
                assert basis is not None
                for axis in basis:
                    assert sum(value * value for value in axis) == pytest.approx(1.0)
                assert sum(basis[0][i] * basis[1][i] for i in range(3)) == pytest.approx(0.0)
                cross = (
                    basis[0][1] * basis[1][2] - basis[0][2] * basis[1][1],
                    basis[0][2] * basis[1][0] - basis[0][0] * basis[1][2],
                    basis[0][0] * basis[1][1] - basis[0][1] * basis[1][0],
                )
                assert sum(cross[i] * basis[2][i] for i in range(3)) > 0.999
            if owners[(road_tag, index)] != owners[(road_tag, index + 1)]:
                seam_pairs += 1
    assert seam_pairs > 10


def test_roads_actually_curve_and_change_elevation() -> None:
    chunks = [
        generate_chunk((x, z), 314159)
        for z in range(1, 5)
        for x in range(1, 5)
    ]
    by_road = defaultdict(list)
    for chunk in chunks:
        for road in (item for item in chunk.surfaces if item.role == "road"):
            assert road.cell is not None and road.basis is not None
            by_road[road.cell[0]].append(road)
    curved = False
    graded = False
    for roads in by_road.values():
        if len(roads) < 5:
            continue
        headings = [degrees(atan2(road.basis[0][2], road.basis[0][0])) for road in roads]
        elevations = [road.position[1] for road in roads]
        curved |= max(headings) - min(headings) > 5.0
        graded |= max(elevations) - min(elevations) > 1.5
    assert curved
    assert graded


def test_buildings_clear_roads_and_vary_yaw_height_and_base() -> None:
    chunks = [
        generate_chunk((x, z), 2026)
        for z in range(-3, 4)
        for x in range(-3, 4)
    ]
    buildings = [building for chunk in chunks for building in chunk.buildings]
    assert len(buildings) > 500
    for building in buildings:
        angle = -radians(building.rotation[1])
        local_x = (cos(angle), sin(angle))
        local_z = (-sin(angle), cos(angle))
        for road in _nearby_roads(
            building.position[0], building.position[2], 2026, include_height=False
        ):
            normal = (-road.tangent[1], road.tangent[0])
            extent = (
                abs(normal[0] * local_x[0] + normal[1] * local_x[1])
                * building.half_extents[0]
                + abs(normal[0] * local_z[0] + normal[1] * local_z[1])
                * building.half_extents[2]
            )
            assert road.distance >= road.width * 0.5 + extent + BUILDING_ROAD_SETBACK - 1e-9
    yaws = {round(building.rotation[1] / 5.0) for building in buildings}
    heights = {round(building.half_extents[1] * 2.0) for building in buildings}
    bases = {
        round((building.position[1] - building.half_extents[1]) / TERRACE_STEP)
        for building in buildings
    }
    assert len(yaws) >= 12
    assert len(heights) >= 25
    assert len(bases) >= 8

    by_cell = {building.cell: building for building in buildings}
    for building in buildings:
        assert building.cell is not None
        for offset_x, offset_z in ((1, 0), (0, 1), (1, 1), (-1, 1)):
            neighbor = by_cell.get(
                (building.cell[0] + offset_x, building.cell[1] + offset_z)
            )
            if neighbor is None:
                continue
            distance = (
                (building.position[0] - neighbor.position[0]) ** 2
                + (building.position[2] - neighbor.position[2]) ** 2
            ) ** 0.5
            radius = (
                building.half_extents[0] ** 2 + building.half_extents[2] ** 2
            ) ** 0.5
            neighbor_radius = (
                neighbor.half_extents[0] ** 2 + neighbor.half_extents[2] ** 2
            ) ** 0.5
            assert distance >= radius + neighbor_radius + MIN_ALLEY_WIDTH - 1e-9


def test_rooftop_cables_are_present_and_cross_different_levels() -> None:
    chunks = [generate_chunk((x, z), 2026) for z in range(3) for x in range(3)]
    wires = [wire for chunk in chunks for wire in chunk.wires]
    assert wires
    assert len(wires) % WIRE_SEGMENTS_PER_RUN == 0
    assert all(wire.role == "wire" for wire in wires)
    assert any(abs(wire.rotation[2]) > 25.0 for wire in wires)


def test_every_chunk_respects_hard_object_cap() -> None:
    for seed in (0, 2026, 999999):
        for z in range(-2, 3):
            for x in range(-2, 3):
                assert len(generate_chunk((x, z), seed).objects) <= MAX_OBJECTS_PER_CHUNK


def test_boundary_ownership_has_no_duplicates() -> None:
    chunks = [
        generate_chunk((x, z), 12345)
        for z in range(-2, 3)
        for x in range(-2, 3)
    ]
    building_cells = [building.cell for chunk in chunks for building in chunk.buildings]
    road_cells = [
        item.cell
        for chunk in chunks
        for item in chunk.surfaces
        if item.role == "road"
    ]
    assert len(building_cells) == len(set(building_cells))
    assert len(road_cells) == len(set(road_cells))
    for chunk in chunks:
        assert all(
            world_to_chunk(building.position[0], building.position[2]) == chunk.coord
            for building in chunk.buildings
        )


def test_contact_cannot_walk_off_road_and_follows_grade() -> None:
    seed = 412
    contact = spawn_road_contact(seed)
    start_y = contact.surface_y
    start_along = contact.along
    for _ in range(80):
        _, derivative = _road_curve(contact.family, contact.road_id, contact.along, seed)
        tangent = (derivative, 1.0) if contact.family == 0 else (1.0, derivative)
        length = (tangent[0] ** 2 + tangent[1] ** 2) ** 0.5
        proposed_x = contact.x + tangent[0] / length * 1.0
        proposed_z = contact.z + tangent[1] / length * 1.0
        contact = move_road_contact(contact, proposed_x, proposed_z, seed)
    assert abs(contact.along - start_along) > 50.0
    assert abs(contact.surface_y - start_y) > 0.3
    road = _nearest_road(contact.x, contact.z, seed)
    assert road.distance <= road.width * 0.5

    normal = (-road.tangent[1], road.tangent[0])
    pushed = move_road_contact(
        contact,
        contact.x + normal[0] * 50.0,
        contact.z + normal[1] * 50.0,
        seed,
    )
    same_road = _nearest_road(pushed.x, pushed.z, seed)
    assert (pushed.family, pushed.road_id) == (contact.family, contact.road_id)
    assert same_road.distance <= same_road.width * 0.5


def test_contact_height_matches_rendered_deck_and_spawn_is_valid() -> None:
    for seed in range(12):
        contact = spawn_road_contact(seed)
        road = _road_projection(
            contact.family, contact.road_id, contact.x, contact.z, seed
        )
        assert road.distance == pytest.approx(0.0, abs=1e-6)
        assert 0.0 <= contact.heading < 360.0
        assert contact.surface_y == pytest.approx(
            _rendered_road_height(contact.family, contact.road_id, contact.along, seed)
        )

    chunk = generate_chunk((-2, -2), 2718)
    road = next(item for item in chunk.surfaces if item.role == "road")
    assert road.cell is not None and road.basis is not None
    segment_midpoint = (road.cell[1] + 0.5) * ROAD_SEGMENT_LENGTH
    point = _road_point(road.cell[0] & 1, road.cell[0] // 2, segment_midpoint, 2718)
    projection = _road_projection(
        road.cell[0] & 1, road.cell[0] // 2, point[0], point[2], 2718
    )
    assert projection.surface_y == pytest.approx(
        road.position[1] + road.basis[1][1] * ROAD_HALF_THICKNESS
    )

    roadbeds = {
        item.cell
        for item in chunk.surfaces
        if item.role == "retaining" and item.cell is not None
    }
    for item in (surface for surface in chunk.surfaces if surface.role == "road"):
        assert item.cell is not None and item.basis is not None
        ground = _terrain_plate_height(item.position[0], item.position[2], 2718)
        gap = item.position[1] - item.basis[1][1] * item.half_extents[1] - ground
        family, road_id = item.cell[0] & 1, item.cell[0] // 2
        along = (item.cell[1] + 0.5) * ROAD_SEGMENT_LENGTH
        if gap > 0.08 and _bridge_lift(family, road_id, along, 2718) <= 1.5:
            assert item.cell in roadbeds


def test_same_level_crossing_transfers_but_overpass_does_not() -> None:
    seed = 0
    # This deterministic crossing is a same-height, connected junction.
    x, z = -175.2073308131835, -173.0
    current = _road_projection(0, -2, x, z, seed)
    branch = _road_projection(1, -2, x, z, seed)
    contact = _contact_from_projection(current, 0.0)
    turned = move_road_contact(
        contact,
        x + branch.tangent[0] * 2.0,
        z + branch.tangent[1] * 2.0,
        seed,
    )
    assert (turned.family, turned.road_id) == (1, -2)

    # Here the same two-dimensional crossing is separated by more than 6 units.
    x, z = -337.64948867116476, -166.0
    upper = _road_projection(0, -4, x, z, seed)
    lower = _road_projection(1, -2, x, z, seed)
    contact = _contact_from_projection(upper, 0.0)
    attempted = move_road_contact(
        contact,
        x + lower.tangent[0] * 2.0,
        z + lower.tangent[1] * 2.0,
        seed,
    )
    assert abs(upper.surface_y - lower.surface_y) > 6.0
    assert (attempted.family, attempted.road_id) == (0, -4)


def test_same_level_crossing_transfers_between_different_road_ids() -> None:
    seed = 1178648799
    x, z = -318.191, -93.138
    current = _road_projection(0, -4, x, z, seed)
    branch = _road_projection(1, -1, x, z, seed)
    contact = _contact_from_projection(current, 0.0)

    turned = move_road_contact(
        contact,
        x + branch.tangent[0] * 2.0,
        z + branch.tangent[1] * 2.0,
        seed,
    )

    assert abs(current.surface_y - branch.surface_y) < 0.01
    assert (turned.family, turned.road_id) == (1, -1)


def test_connected_branch_can_change_grade_after_the_turn() -> None:
    seed = 1178648799
    x, z = -258.686, 331.612
    current = _road_projection(0, -3, x, z, seed)
    branch = _road_projection(1, 4, x, z, seed)
    contact = _contact_from_projection(current, 0.0)
    proposed_x = x + branch.tangent[0] * 2.0
    proposed_z = z + branch.tangent[1] * 2.0
    branch_ahead = _road_projection(1, 4, proposed_x, proposed_z, seed)
    current_ahead = _road_projection(0, -3, proposed_x, proposed_z, seed)

    turned = move_road_contact(contact, proposed_x, proposed_z, seed)

    assert abs(current.surface_y - branch.surface_y) < 0.01
    assert (
        abs(branch_ahead.surface_y - current_ahead.surface_y)
        > JUNCTION_HEIGHT_TOLERANCE
    )
    assert (turned.family, turned.road_id) == (1, 4)


def test_contact_does_not_jump_onto_unreachable_bridge() -> None:
    seed = 2026
    bridge = None
    for family in (0, 1):
        for road_id in range(-5, 6):
            for along in range(-4096, 4097, 4):
                if _bridge_lift(family, road_id, float(along), seed) > MAX_WALK_STEP + 2.0:
                    bridge = (family, road_id, float(along))
                    break
            if bridge:
                break
        if bridge:
            break
    assert bridge is not None
    family, road_id, along = bridge
    point = _road_point(family, road_id, along, seed)
    ground = terrain_height(point[0], point[2], seed)
    assert walkable_height(point[0], point[2], seed, ground) < point[1] - 2.0


def test_negative_contact_queries_are_deterministic() -> None:
    seed = -8801
    contact = spawn_road_contact(seed)
    first = move_road_contact(contact, -145.25, -93.75, seed)
    second = move_road_contact(contact, -145.25, -93.75, seed)
    assert first == second
    assert abs(first.along - contact.along) <= ROAD_SEGMENT_LENGTH * 2.0


def test_every_proxy_element_and_sky_receive_seeded_random_colors() -> None:
    first = generate_chunk((0, 0), 12345)
    repeat = generate_chunk((0, 0), 12345)
    different_world = generate_chunk((0, 0), 54321)
    colors = tuple(item.color for item in first.objects)

    assert first == repeat
    assert len(set(colors)) == len(colors)
    assert colors != tuple(item.color for item in different_world.objects)
    assert sky_color(12345) == sky_color(12345)
    assert sky_color(12345) != sky_color(54321)
    assert all(
        sum(
            (channel - sky_channel) ** 2
            for channel, sky_channel in zip(color, sky_color(12345))
        )
        >= 0.12
        for color in colors
    )


def test_active_window_cache_evicts_and_regenerates_identically() -> None:
    initial_coords = active_chunk_coords(-0.001, -0.001)
    initial = generate_chunk(initial_coords[0], 77)
    assert len(initial_coords) == MAX_ACTIVE_CHUNKS
    assert initial_coords[0] == (-1, -1)
    assert MAX_ACTIVE_CHUNKS == (ACTIVE_CHUNK_RADIUS * 2 + 1) ** 2
    plan = plan_chunk_cache(initial_coords, CHUNK_SIZE * 20, -CHUNK_SIZE * 15)
    assert len(plan.desired) == MAX_ACTIVE_CHUNKS
    assert len(plan.evict) == MAX_ACTIVE_CHUNKS
    assert not plan.keep
    assert generate_chunk(initial_coords[0], 77) == initial


def test_world_to_chunk_handles_negative_boundaries() -> None:
    assert world_to_chunk(0.0, 0.0) == (0, 0)
    assert world_to_chunk(CHUNK_SIZE - 0.001, CHUNK_SIZE - 0.001) == (0, 0)
    assert world_to_chunk(-0.001, -0.001) == (-1, -1)
    assert world_to_chunk(-CHUNK_SIZE, -CHUNK_SIZE) == (-1, -1)


def test_wire_rotation_aligns_the_cube_with_its_segment() -> None:
    start = (1.0, 2.0, 3.0)
    end = (4.0, 6.0, 15.0)
    wire = _wire_cube(start, end)
    _, yaw_degrees, roll_degrees = wire.rotation
    yaw = radians(yaw_degrees)
    roll = radians(roll_degrees)
    rotated_local_x = (
        cos(roll) * cos(yaw),
        sin(roll) * cos(yaw),
        -sin(yaw),
    )
    length = wire.half_extents[0] * 2.0
    expected = tuple((end[index] - start[index]) / length for index in range(3))
    assert rotated_local_x == pytest.approx(expected)


def test_negative_cache_radius_is_rejected() -> None:
    with pytest.raises(ValueError, match="radius"):
        active_chunk_coords(0.0, 0.0, radius=-1)
