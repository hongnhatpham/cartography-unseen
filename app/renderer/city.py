from __future__ import annotations

from colorsys import hsv_to_rgb
from dataclasses import dataclass, replace
from functools import lru_cache
from math import atan2, cos, degrees, floor, pi, sin, sqrt
from random import Random
from typing import Iterable, Literal, TypeAlias


ChunkCoord: TypeAlias = tuple[int, int]
Vec3: TypeAlias = tuple[float, float, float]
Basis3: TypeAlias = tuple[Vec3, Vec3, Vec3]
Color: TypeAlias = tuple[float, float, float]

CHUNK_SIZE = 64.0
ACTIVE_CHUNK_RADIUS = 3
MAX_ACTIVE_CHUNKS = (ACTIVE_CHUNK_RADIUS * 2 + 1) ** 2
MAX_OBJECTS_PER_CHUNK = 256

TERRACE_SIZE = 16.0
TERRACE_STEP = 2.0
ROAD_SPACING = 82.0
ROAD_SEGMENT_LENGTH = 8.0
BUILDING_CELL_SIZE = 8.0
BUILDING_ROAD_SETBACK = 0.9
MIN_ALLEY_WIDTH = 0.55
WIRE_RUNS_PER_CHUNK = 12
WIRE_SEGMENTS_PER_RUN = 3
MAX_WALK_STEP = 1.35
JUNCTION_HEIGHT_TOLERANCE = 0.28
ROAD_HALF_THICKNESS = 0.11

_MASK_64 = (1 << 64) - 1
_BUILDING_SALT = 0xB17D1A6
_ROAD_SALT = 0x70AD5
_WIRE_SALT = 0xC4B1E
_BRIDGE_SALT = 0xB71D6E
_OBJECT_COLOR_SALT = 0xC01045
_SKY_COLOR_SALT = 0x5A7C010
_HEIGHT_RANGES = ((3.5, 7.5), (7.5, 14.0), (14.0, 24.0), (24.0, 39.0))
_PALETTE: tuple[Color, ...] = (
    (0.25, 0.36, 0.37),
    (0.34, 0.45, 0.42),
    (0.47, 0.35, 0.39),
    (0.55, 0.38, 0.25),
    (0.48, 0.44, 0.26),
    (0.30, 0.36, 0.52),
    (0.40, 0.31, 0.25),
)


@dataclass(frozen=True, slots=True)
class CityCube:
    """Renderer-neutral cube using the renderer's half-extent convention."""

    role: Literal[
        "terrain", "road", "retaining", "support", "building", "wire"
    ]
    position: Vec3
    half_extents: Vec3
    rotation: Vec3
    color: Color
    cell: tuple[int, int] | None = None
    height_tier: int | None = None
    basis: Basis3 | None = None


@dataclass(frozen=True, slots=True)
class CityChunk:
    coord: ChunkCoord
    surfaces: tuple[CityCube, ...]
    buildings: tuple[CityCube, ...]
    wires: tuple[CityCube, ...]

    @property
    def objects(self) -> tuple[CityCube, ...]:
        return self.surfaces + self.buildings + self.wires


@dataclass(frozen=True, slots=True)
class ChunkCachePlan:
    """A fixed-size target set plus the exact cache mutations needed."""

    center: ChunkCoord
    desired: tuple[ChunkCoord, ...]
    load: tuple[ChunkCoord, ...]
    keep: tuple[ChunkCoord, ...]
    evict: tuple[ChunkCoord, ...]


@dataclass(frozen=True, slots=True)
class _RoadInfo:
    family: int
    road_id: int
    along: float
    distance: float
    signed_distance: float
    width: float
    tangent: tuple[float, float]
    point: tuple[float, float]
    surface_y: float


@dataclass(frozen=True, slots=True)
class RoadContact:
    """Stable road identity and camera-foot position used during traversal."""

    family: int
    road_id: int
    segment: int
    along: float
    lateral: float
    x: float
    z: float
    surface_y: float
    heading: float


@dataclass(frozen=True, slots=True)
class _BuildingCandidate:
    cell: tuple[int, int]
    x: float
    z: float
    width: float
    depth: float
    height: float
    height_tier: int
    yaw: float
    base: float
    color: Color
    priority: int
    roads: tuple[_RoadInfo, ...]

    @property
    def radius(self) -> float:
        return sqrt(self.width * self.width + self.depth * self.depth) * 0.5


def _mix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _MASK_64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK_64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK_64
    return value ^ (value >> 31)


def _stable_seed(world_seed: int, *parts: int) -> int:
    value = _mix64(world_seed & _MASK_64)
    for part in parts:
        value = _mix64(value ^ (part & _MASK_64))
    return value


def _unit_float(world_seed: int, *parts: int) -> float:
    return (_stable_seed(world_seed, *parts) >> 11) / float(1 << 53)


def sky_color(world_seed: int) -> Color:
    """Return the seeded clear and fog color shared by one world."""

    hue = _unit_float(world_seed, _SKY_COLOR_SALT, 1)
    saturation = 0.45 + 0.5 * _unit_float(world_seed, _SKY_COLOR_SALT, 2)
    value = 0.38 + 0.5 * _unit_float(world_seed, _SKY_COLOR_SALT, 3)
    return hsv_to_rgb(hue, saturation, value)


def _abstract_object_color(
    world_seed: int,
    coord: ChunkCoord,
    group: int,
    index: int,
    role: str,
) -> Color:
    """Give each proxy element a vivid deterministic segmentation color."""

    role_id = ("terrain", "road", "retaining", "support", "building", "wire").index(role)
    parts = (_OBJECT_COLOR_SALT, coord[0], coord[1], group, index, role_id)
    hue = _unit_float(world_seed, *parts, 1)
    saturation = 0.62 + 0.36 * _unit_float(world_seed, *parts, 2)
    value = 0.55 + 0.43 * _unit_float(world_seed, *parts, 3)
    color = hsv_to_rgb(hue, saturation, value)

    # Keep silhouettes legible when an object happens to land near the sky.
    sky = sky_color(world_seed)
    distance_squared = sum((channel - sky_channel) ** 2 for channel, sky_channel in zip(color, sky))
    if distance_squared < 0.12:
        color = hsv_to_rgb((hue + 0.5) % 1.0, saturation, value)
    return color


def _recolor_objects(
    objects: tuple[CityCube, ...],
    world_seed: int,
    coord: ChunkCoord,
    group: int,
) -> tuple[CityCube, ...]:
    return tuple(
        replace(
            item,
            color=_abstract_object_color(
                world_seed, coord, group, index, item.role
            ),
        )
        for index, item in enumerate(objects)
    )


def terrain_height(x: float, z: float, world_seed: int) -> float:
    """Continuous world terrain with a calm, traversable origin."""

    phase_x, phase_z, phase_cross = _terrain_phases(world_seed)

    def raw(px: float, pz: float) -> float:
        return (
            5.2 * sin(px / 76.0 + phase_x)
            + 4.4 * sin(pz / 63.0 + phase_z)
            + 2.8 * sin((px + pz) / 47.0 + phase_cross)
            + 1.4 * sin((px - pz) / 29.0 + phase_z - phase_x)
        )

    radius = sqrt(x * x + z * z)
    amount = min(1.0, max(0.0, (radius - 10.0) / 38.0))
    smooth_amount = amount * amount * (3.0 - 2.0 * amount)
    return (raw(x, z) - raw(0.0, 0.0)) * smooth_amount


@lru_cache(maxsize=16)
def _terrain_phases(world_seed: int) -> tuple[float, float, float]:
    return (
        _unit_float(world_seed, 11) * 2.0 * pi,
        _unit_float(world_seed, 13) * 2.0 * pi,
        _unit_float(world_seed, 17) * 2.0 * pi,
    )


def _terrace_height(x: float, z: float, world_seed: int) -> float:
    return round(terrain_height(x, z, world_seed) / TERRACE_STEP) * TERRACE_STEP


def _terrain_plate_height(x: float, z: float, world_seed: int) -> float:
    tile_x = floor(x / TERRACE_SIZE)
    tile_z = floor(z / TERRACE_SIZE)
    return _terrace_height(
        (tile_x + 0.5) * TERRACE_SIZE,
        (tile_z + 0.5) * TERRACE_SIZE,
        world_seed,
    )


def world_to_chunk(x: float, z: float) -> ChunkCoord:
    return floor(x / CHUNK_SIZE), floor(z / CHUNK_SIZE)


def active_chunk_coords(
    x: float, z: float, radius: int = ACTIVE_CHUNK_RADIUS
) -> tuple[ChunkCoord, ...]:
    """Return the finite chunk window around a world-space position."""

    if radius < 0:
        raise ValueError("radius must be zero or greater")
    center_x, center_z = world_to_chunk(x, z)
    coords = (
        (chunk_x, chunk_z)
        for chunk_z in range(center_z - radius, center_z + radius + 1)
        for chunk_x in range(center_x - radius, center_x + radius + 1)
    )
    return tuple(
        sorted(
            coords,
            key=lambda coord: (
                max(abs(coord[0] - center_x), abs(coord[1] - center_z)),
                abs(coord[0] - center_x) + abs(coord[1] - center_z),
                coord[1],
                coord[0],
            ),
        )
    )


def plan_chunk_cache(
    existing: Iterable[ChunkCoord],
    x: float,
    z: float,
    radius: int = ACTIVE_CHUNK_RADIUS,
) -> ChunkCachePlan:
    """Plan loads and evictions without retaining generated chunks here."""

    desired = active_chunk_coords(x, z, radius)
    desired_set = set(desired)
    existing_set = set(existing)
    order = {coord: index for index, coord in enumerate(desired)}
    return ChunkCachePlan(
        center=world_to_chunk(x, z),
        desired=desired,
        load=tuple(coord for coord in desired if coord not in existing_set),
        keep=tuple(sorted(existing_set & desired_set, key=order.__getitem__)),
        evict=tuple(sorted(existing_set - desired_set)),
    )


@lru_cache(maxsize=256)
def _road_parameters(
    family: int, road_id: int, world_seed: int
) -> tuple[float, float, float, float, float, float]:
    rng = Random(_stable_seed(world_seed, family, road_id, _ROAD_SALT))
    return (
        rng.uniform(-11.0, 11.0),
        rng.uniform(7.0, 11.5),
        rng.uniform(31.0, 45.0),
        rng.uniform(0.0, 2.0 * pi),
        rng.uniform(2.2, 4.2),
        rng.uniform(15.0, 23.0),
    )


def _road_curve(
    family: int, road_id: int, along: float, world_seed: int
) -> tuple[float, float]:
    offset, amplitude, wavelength, phase, ripple, ripple_wave = _road_parameters(
        family, road_id, world_seed
    )
    ripple_phase = phase * 1.71 + road_id * 0.37
    across = (
        road_id * ROAD_SPACING
        + offset
        + amplitude * sin(along / wavelength + phase)
        + ripple * sin(along / ripple_wave + ripple_phase)
    )
    derivative = (
        amplitude / wavelength * cos(along / wavelength + phase)
        + ripple / ripple_wave * cos(along / ripple_wave + ripple_phase)
    )
    return across, derivative


@lru_cache(maxsize=256)
def _road_width(family: int, road_id: int, world_seed: int) -> float:
    return 6.4 + 2.4 * _unit_float(world_seed, family, road_id, _ROAD_SALT, 91)


def _bridge_lift(family: int, road_id: int, along: float, world_seed: int) -> float:
    block = floor(along / 512.0)
    if _unit_float(world_seed, family, road_id, block, _BRIDGE_SALT) > 0.16:
        return 0.0
    rng = Random(_stable_seed(world_seed, family, road_id, block, _BRIDGE_SALT, 3))
    center = (block + 0.5) * 512.0 + rng.uniform(-120.0, 120.0)
    half_length = rng.uniform(34.0, 48.0)
    distance = abs(along - center)
    if distance >= half_length:
        return 0.0
    wave = sin(pi * (1.0 - distance / half_length) * 0.5)
    return rng.uniform(5.5, 8.5) * wave * wave


def _road_point(family: int, road_id: int, along: float, world_seed: int) -> Vec3:
    across, _ = _road_curve(family, road_id, along, world_seed)
    x, z = (across, along) if family == 0 else (along, across)
    y = terrain_height(x, z, world_seed) + 1.25
    y += _bridge_lift(family, road_id, along, world_seed)
    return x, y, z


def _road_projection(
    family: int,
    road_id: int,
    x: float,
    z: float,
    world_seed: int,
    include_height: bool = True,
) -> _RoadInfo:
    along_value = z if family == 0 else x
    across_value = x if family == 0 else z
    along = along_value
    for _ in range(3):
        across, derivative = _road_curve(family, road_id, along, world_seed)
        along += (
            (across_value - across) * derivative + along_value - along
        ) / (1.0 + derivative * derivative)
    across, derivative = _road_curve(family, road_id, along, world_seed)
    point = (across, along) if family == 0 else (along, across)
    tangent = (derivative, 1.0) if family == 0 else (1.0, derivative)
    tangent_length = sqrt(tangent[0] ** 2 + tangent[1] ** 2)
    tangent = (tangent[0] / tangent_length, tangent[1] / tangent_length)
    normal = (-tangent[1], tangent[0])
    signed_distance = (x - point[0]) * normal[0] + (z - point[1]) * normal[1]
    surface_y = (
        _rendered_road_height(family, road_id, along, world_seed)
        if include_height
        else 0.0
    )
    return _RoadInfo(
        family=family,
        road_id=road_id,
        along=along,
        distance=abs(signed_distance),
        signed_distance=signed_distance,
        width=_road_width(family, road_id, world_seed),
        tangent=tangent,
        point=point,
        surface_y=surface_y,
    )


def _nearby_roads(
    x: float, z: float, world_seed: int, include_height: bool = True
) -> tuple[_RoadInfo, ...]:
    roads: list[_RoadInfo] = []
    for family, across_value in ((0, x), (1, z)):
        guessed_id = round(across_value / ROAD_SPACING)
        for road_id in range(guessed_id - 1, guessed_id + 2):
            roads.append(
                _road_projection(
                    family, road_id, x, z, world_seed, include_height=include_height
                )
            )
    return tuple(sorted(roads, key=lambda road: (road.distance, road.family, road.road_id)))


def _nearest_road(x: float, z: float, world_seed: int) -> _RoadInfo:
    return _nearby_roads(x, z, world_seed)[0]


def _rendered_road_height(
    family: int, road_id: int, along: float, world_seed: int
) -> float:
    """Top of the exact graded cube segment used by the renderer."""

    segment = floor(along / ROAD_SEGMENT_LENGTH)
    start = _road_point(family, road_id, segment * ROAD_SEGMENT_LENGTH, world_seed)
    end = _road_point(family, road_id, (segment + 1) * ROAD_SEGMENT_LENGTH, world_seed)
    amount = (along / ROAD_SEGMENT_LENGTH) - segment
    center_y = start[1] + (end[1] - start[1]) * amount
    basis = _basis_between(start, end)
    return center_y + basis[1][1] * ROAD_HALF_THICKNESS


def walkable_height(
    x: float,
    z: float,
    world_seed: int,
    current_feet_y: float | None = None,
) -> float:
    """Return reachable terrain or road height without jumping onto bridges."""

    ground = _terrain_plate_height(x, z, world_seed)
    for road in _nearby_roads(x, z, world_seed):
        if road.distance > road.width * 0.5:
            continue
        if current_feet_y is not None and abs(road.surface_y - current_feet_y) <= MAX_WALK_STEP:
            return road.surface_y
        if current_feet_y is None and road.surface_y - ground <= MAX_WALK_STEP:
            return road.surface_y
    return ground


def _contact_from_projection(road: _RoadInfo, lateral: float | None = None) -> RoadContact:
    margin = max(0.0, road.width * 0.5 - 0.22)
    clamped_lateral = min(
        margin, max(-margin, road.signed_distance if lateral is None else lateral)
    )
    normal = (-road.tangent[1], road.tangent[0])
    return RoadContact(
        family=road.family,
        road_id=road.road_id,
        segment=floor(road.along / ROAD_SEGMENT_LENGTH),
        along=road.along,
        lateral=clamped_lateral,
        x=road.point[0] + normal[0] * clamped_lateral,
        z=road.point[1] + normal[1] * clamped_lateral,
        surface_y=road.surface_y,
        heading=degrees(atan2(road.tangent[0], -road.tangent[1])) % 360.0,
    )


def spawn_road_contact(world_seed: int) -> RoadContact:
    """Place a new player on the road nearest the configured origin spawn."""

    return _contact_from_projection(
        _nearest_road(0.0, 5.5, world_seed), lateral=0.0
    )


def move_road_contact(
    contact: RoadContact,
    proposed_x: float,
    proposed_z: float,
    world_seed: int,
) -> RoadContact:
    """Move on one road, transferring only through a reachable intersection."""

    current = _road_projection(
        contact.family, contact.road_id, proposed_x, proposed_z, world_seed
    )
    adjacent_min = (contact.segment - 1) * ROAD_SEGMENT_LENGTH
    adjacent_max = (contact.segment + 2) * ROAD_SEGMENT_LENGTH
    along = min(
        adjacent_max,
        max(adjacent_min, current.along),
    )
    if along != current.along:
        center = _road_point(contact.family, contact.road_id, along, world_seed)
        current = _road_projection(
            contact.family, contact.road_id, center[0], center[2], world_seed
        )

    # A crossing becomes a junction only when both decks meet at walking height.
    branch_family = 1 - contact.family
    branch_across = current.point[0] if branch_family == 0 else current.point[1]
    guessed_branch = round(branch_across / ROAD_SPACING)
    chosen = current
    move_x = proposed_x - contact.x
    move_z = proposed_z - contact.z
    move_length = sqrt(move_x * move_x + move_z * move_z)
    current_alignment = (
        abs(move_x * current.tangent[0] + move_z * current.tangent[1]) / move_length
        if move_length > 1e-9
        else 1.0
    )
    for branch_id in range(guessed_branch - 1, guessed_branch + 2):
        branch_at_proposal = _road_projection(
            branch_family, branch_id, proposed_x, proposed_z, world_seed
        )
        branch_at_current = _road_projection(
            branch_family, branch_id, contact.x, contact.z, world_seed
        )
        branch_alignment = (
            abs(
                move_x * branch_at_proposal.tangent[0]
                + move_z * branch_at_proposal.tangent[1]
            )
            / move_length
            if move_length > 1e-9
            else 0.0
        )
        if (
            branch_at_current.distance <= branch_at_current.width * 0.5 + 0.35
            and branch_at_proposal.distance <= branch_at_proposal.width * 0.5
            and abs(branch_at_current.surface_y - contact.surface_y)
            <= JUNCTION_HEIGHT_TOLERANCE
            and branch_alignment > current_alignment + 0.15
            and branch_at_proposal.distance + 0.15 < current.distance
        ):
            chosen = branch_at_proposal
            break

    if abs(chosen.surface_y - contact.surface_y) > MAX_WALK_STEP:
        return contact
    return _contact_from_projection(chosen)


def constrain_to_road(
    previous_x: float,
    previous_z: float,
    proposed_x: float,
    proposed_z: float,
    world_seed: int,
    current_feet_y: float | None = None,
) -> tuple[float, float, float]:
    """Clamp a movement step to its road ribbon and slide along the curb."""

    previous_options = _nearby_roads(previous_x, previous_z, world_seed)
    reachable = [
        road
        for road in previous_options
        if road.distance <= road.width * 0.5 + 0.35
        and (
            current_feet_y is None
            or abs(road.surface_y - current_feet_y) <= MAX_WALK_STEP
        )
    ]
    current = reachable[0] if reachable else previous_options[0]

    proposed_options = _nearby_roads(proposed_x, proposed_z, world_seed)
    same = next(
        road
        for road in proposed_options
        if (road.family, road.road_id) == (current.family, current.road_id)
    )

    # Transfer only while both centerlines share the same ground-level junction.
    for branch in proposed_options:
        if (branch.family, branch.road_id) == (current.family, current.road_id):
            continue
        at_junction = _road_projection(
            branch.family, branch.road_id, *current.point, world_seed
        )
        same_level = abs(at_junction.surface_y - current.surface_y) <= MAX_WALK_STEP
        branch_contains_previous = _road_projection(
            branch.family, branch.road_id, previous_x, previous_z, world_seed
        ).distance <= branch.width * 0.5 + 0.35
        if (
            same_level
            and branch_contains_previous
            and branch.distance <= branch.width * 0.5
            and branch.distance + 0.15 < same.distance
        ):
            same = branch
            break

    reference_y = current_feet_y if current_feet_y is not None else current.surface_y
    if abs(same.surface_y - reference_y) > MAX_WALK_STEP:
        return previous_x, previous_z, current.surface_y

    margin = max(0.0, same.width * 0.5 - 0.22)
    lateral = min(margin, max(-margin, same.signed_distance))
    normal = (-same.tangent[1], same.tangent[0])
    return (
        same.point[0] + normal[0] * lateral,
        same.point[1] + normal[1] * lateral,
        same.surface_y,
    )


def _basis_between(start: Vec3, end: Vec3) -> Basis3:
    delta = tuple(end[index] - start[index] for index in range(3))
    length = sqrt(sum(value * value for value in delta))
    x_axis = tuple(value / length for value in delta)
    horizontal = sqrt(x_axis[0] * x_axis[0] + x_axis[2] * x_axis[2])
    z_axis = (-x_axis[2] / horizontal, 0.0, x_axis[0] / horizontal)
    y_axis = (
        -x_axis[1] * z_axis[2],
        x_axis[0] * z_axis[2] - x_axis[2] * z_axis[0],
        x_axis[1] * z_axis[0],
    )
    return x_axis, y_axis, z_axis


def _generate_roads(coord: ChunkCoord, world_seed: int) -> tuple[CityCube, ...]:
    origin_x = coord[0] * CHUNK_SIZE
    origin_z = coord[1] * CHUNK_SIZE
    curve_reach = 28.0
    roads: list[CityCube] = []
    for family in (0, 1):
        across_min = origin_x if family == 0 else origin_z
        across_max = across_min + CHUNK_SIZE
        first_id = floor((across_min - curve_reach) / ROAD_SPACING) - 1
        last_id = floor((across_max + curve_reach) / ROAD_SPACING) + 1
        along_min = origin_z if family == 0 else origin_x
        along_max = along_min + CHUNK_SIZE
        first_segment = floor(along_min / ROAD_SEGMENT_LENGTH) - 1
        last_segment = floor(along_max / ROAD_SEGMENT_LENGTH) + 1
        for road_id in range(first_id, last_id + 1):
            width = _road_width(family, road_id, world_seed)
            for segment in range(first_segment, last_segment + 1):
                start = _road_point(family, road_id, segment * ROAD_SEGMENT_LENGTH, world_seed)
                end = _road_point(
                    family, road_id, (segment + 1) * ROAD_SEGMENT_LENGTH, world_seed
                )
                midpoint = tuple((start[index] + end[index]) * 0.5 for index in range(3))
                if world_to_chunk(midpoint[0], midpoint[2]) != coord:
                    continue
                basis = _basis_between(start, end)
                length = sqrt(sum((end[index] - start[index]) ** 2 for index in range(3)))
                road_tag = road_id * 2 + family
                roads.append(
                    CityCube(
                        role="road",
                        position=midpoint,
                        half_extents=(length * 0.5, ROAD_HALF_THICKNESS, width * 0.5),
                        rotation=(0.0, 0.0, 0.0),
                        color=(0.075, 0.085, 0.095),
                        cell=(road_tag, segment),
                        basis=basis,
                    )
                )
    return tuple(sorted(roads, key=lambda road: road.cell or (0, 0)))


def _generate_roadbeds(
    roads: tuple[CityCube, ...], world_seed: int
) -> tuple[CityCube, ...]:
    beds: list[CityCube] = []
    for road in roads:
        assert road.basis is not None and road.cell is not None
        ground = _terrain_plate_height(road.position[0], road.position[2], world_seed)
        deck_bottom = road.position[1] - road.basis[1][1] * road.half_extents[1]
        gap = deck_bottom - ground
        family = road.cell[0] & 1
        road_id = road.cell[0] // 2
        along = (road.cell[1] + 0.5) * ROAD_SEGMENT_LENGTH
        is_raised = _bridge_lift(family, road_id, along, world_seed) > 1.5
        if gap <= 0.08 or is_raised:
            continue
        direction_x = road.basis[0][0]
        direction_z = road.basis[0][2]
        horizontal_length = sqrt(direction_x * direction_x + direction_z * direction_z)
        x_axis = (direction_x / horizontal_length, 0.0, direction_z / horizontal_length)
        z_axis = (-x_axis[2], 0.0, x_axis[0])
        beds.append(
            CityCube(
                role="retaining",
                position=(road.position[0], ground + gap * 0.5, road.position[2]),
                half_extents=(
                    road.half_extents[0] * horizontal_length,
                    gap * 0.5,
                    max(0.1, road.half_extents[2] - 0.12),
                ),
                rotation=(0.0, 0.0, 0.0),
                color=(0.20, 0.21, 0.19),
                cell=road.cell,
                basis=(x_axis, (0.0, 1.0, 0.0), z_axis),
            )
        )
    return tuple(beds)


def _generate_terrain(coord: ChunkCoord, world_seed: int) -> tuple[CityCube, ...]:
    origin_x = coord[0] * CHUNK_SIZE
    origin_z = coord[1] * CHUNK_SIZE
    tiles: list[CityCube] = []
    first_x = floor(origin_x / TERRACE_SIZE) - 1
    first_z = floor(origin_z / TERRACE_SIZE) - 1
    tile_count = round(CHUNK_SIZE / TERRACE_SIZE)
    for tile_z in range(first_z, first_z + tile_count + 2):
        for tile_x in range(first_x, first_x + tile_count + 2):
            center_x = (tile_x + 0.5) * TERRACE_SIZE
            center_z = (tile_z + 0.5) * TERRACE_SIZE
            top = _terrace_height(center_x, center_z, world_seed)
            if world_to_chunk(center_x, center_z) == coord:
                tiles.append(
                    CityCube(
                        role="terrain",
                        position=(center_x, top - 20.0, center_z),
                        half_extents=(TERRACE_SIZE * 0.5, 20.0, TERRACE_SIZE * 0.5),
                        rotation=(0.0, 0.0, 0.0),
                        color=(0.19, 0.23, 0.20),
                        cell=(tile_x, tile_z),
                    )
                )
            for axis, neighbor in ((0, (tile_x + 1, tile_z)), (1, (tile_x, tile_z + 1))):
                other_x = (neighbor[0] + 0.5) * TERRACE_SIZE
                other_z = (neighbor[1] + 0.5) * TERRACE_SIZE
                other_top = _terrace_height(other_x, other_z, world_seed)
                if abs(top - other_top) < TERRACE_STEP * 0.75:
                    continue
                if axis == 0:
                    wall_x, wall_z = (tile_x + 1.0) * TERRACE_SIZE, center_z
                    half_extents = (0.18, abs(top - other_top) * 0.5, TERRACE_SIZE * 0.5)
                else:
                    wall_x, wall_z = center_x, (tile_z + 1.0) * TERRACE_SIZE
                    half_extents = (TERRACE_SIZE * 0.5, abs(top - other_top) * 0.5, 0.18)
                if world_to_chunk(wall_x, wall_z) != coord:
                    continue
                tiles.append(
                    CityCube(
                        role="retaining",
                        position=(wall_x, min(top, other_top) + half_extents[1], wall_z),
                        half_extents=half_extents,
                        rotation=(0.0, 0.0, 0.0),
                        color=(0.26, 0.25, 0.22),
                    )
                )
    return tuple(tiles)


def _candidate(cell_x: int, cell_z: int, world_seed: int) -> _BuildingCandidate:
    rng = Random(_stable_seed(world_seed, cell_x, cell_z, _BUILDING_SALT))
    x = (cell_x + 0.5) * BUILDING_CELL_SIZE + rng.uniform(-1.65, 1.65)
    z = (cell_z + 0.5) * BUILDING_CELL_SIZE + rng.uniform(-1.65, 1.65)
    # These six projections are stored on the candidate and reused by the
    # clearance pass. The memoization stays local to one bounded chunk build.
    roads = _nearby_roads(x, z, world_seed, include_height=False)
    road = roads[0]
    yaw = -degrees(atan2(road.tangent[1], road.tangent[0])) + rng.uniform(-11.0, 11.0)
    width = rng.uniform(3.5, 6.0)
    depth = rng.uniform(3.0, 5.2)
    roll = rng.randrange(100)
    height_tier = 0 if roll < 23 else 1 if roll < 61 else 2 if roll < 89 else 3
    low, high = _HEIGHT_RANGES[height_tier]
    return _BuildingCandidate(
        cell=(cell_x, cell_z),
        x=x,
        z=z,
        width=width,
        depth=depth,
        height=rng.uniform(low, high),
        height_tier=height_tier,
        yaw=yaw,
        base=_terrain_plate_height(x, z, world_seed),
        color=_PALETTE[rng.randrange(len(_PALETTE))],
        priority=_stable_seed(world_seed, cell_x, cell_z, _BUILDING_SALT, 77),
        roads=roads,
    )


def _candidate_clears_roads(candidate: _BuildingCandidate) -> bool:
    yaw = -candidate.yaw * pi / 180.0
    local_x = (cos(yaw), sin(yaw))
    local_z = (-sin(yaw), cos(yaw))
    half_x = candidate.width * 0.5
    half_z = candidate.depth * 0.5
    for road in candidate.roads:
        normal = (-road.tangent[1], road.tangent[0])
        footprint_extent = (
            abs(normal[0] * local_x[0] + normal[1] * local_x[1]) * half_x
            + abs(normal[0] * local_z[0] + normal[1] * local_z[1]) * half_z
        )
        if road.distance < road.width * 0.5 + footprint_extent + BUILDING_ROAD_SETBACK:
            return False
    return True


def _candidate_is_accepted(
    candidate: _BuildingCandidate,
    viable: dict[tuple[int, int], _BuildingCandidate],
) -> bool:
    if candidate.cell not in viable:
        return False
    cell_x, cell_z = candidate.cell
    for neighbor_z in range(cell_z - 1, cell_z + 2):
        for neighbor_x in range(cell_x - 1, cell_x + 2):
            if (neighbor_x, neighbor_z) == candidate.cell:
                continue
            neighbor = viable.get((neighbor_x, neighbor_z))
            if neighbor is None:
                continue
            if (neighbor.priority, neighbor.cell) >= (candidate.priority, candidate.cell):
                continue
            distance = sqrt((candidate.x - neighbor.x) ** 2 + (candidate.z - neighbor.z) ** 2)
            if distance < candidate.radius + neighbor.radius + MIN_ALLEY_WIDTH:
                return False
    return True


def _building_from_candidate(candidate: _BuildingCandidate) -> CityCube:
    return CityCube(
        role="building",
        position=(candidate.x, candidate.base + candidate.height * 0.5, candidate.z),
        half_extents=(candidate.width * 0.5, candidate.height * 0.5, candidate.depth * 0.5),
        rotation=(0.0, candidate.yaw, 0.0),
        color=candidate.color,
        cell=candidate.cell,
        height_tier=candidate.height_tier,
    )


def _accepted_buildings_near(
    min_x: float, max_x: float, min_z: float, max_z: float, world_seed: int
) -> dict[tuple[int, int], CityCube]:
    first_x = floor(min_x / BUILDING_CELL_SIZE) - 1
    last_x = floor(max_x / BUILDING_CELL_SIZE) + 1
    first_z = floor(min_z / BUILDING_CELL_SIZE) - 1
    last_z = floor(max_z / BUILDING_CELL_SIZE) + 1
    candidates = {
        (cell_x, cell_z): _candidate(cell_x, cell_z, world_seed)
        for cell_z in range(first_z - 1, last_z + 2)
        for cell_x in range(first_x - 1, last_x + 2)
    }
    viable = {
        cell: candidate
        for cell, candidate in candidates.items()
        if _candidate_clears_roads(candidate)
    }
    buildings: dict[tuple[int, int], CityCube] = {}
    for cell_z in range(first_z, last_z + 1):
        for cell_x in range(first_x, last_x + 1):
            candidate = candidates[(cell_x, cell_z)]
            if _candidate_is_accepted(candidate, viable):
                buildings[candidate.cell] = _building_from_candidate(candidate)
    return buildings


def _generate_supports(
    roads: tuple[CityCube, ...], world_seed: int
) -> tuple[CityCube, ...]:
    supports: list[CityCube] = []
    for road in roads:
        if road.cell is None or road.cell[1] % 2:
            continue
        x, road_y, z = road.position
        ground = _terrain_plate_height(x, z, world_seed)
        bottom = road_y - road.half_extents[1]
        if bottom - ground < 2.4:
            continue
        supports.append(
            CityCube(
                role="support",
                position=(x, ground + (bottom - ground) * 0.5, z),
                half_extents=(0.48, (bottom - ground) * 0.5, 0.48),
                rotation=(0.0, 0.0, 0.0),
                color=(0.22, 0.22, 0.20),
                cell=road.cell,
            )
        )
    return tuple(supports)


def _generate_wires(
    coord: ChunkCoord,
    buildings: dict[tuple[int, int], CityCube],
    world_seed: int,
) -> tuple[CityCube, ...]:
    values = tuple(buildings.values())
    candidates: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for index, start in enumerate(values):
        if start.cell is None:
            continue
        for end in values[index + 1 :]:
            if end.cell is None:
                continue
            planar_distance = sqrt(
                (end.position[0] - start.position[0]) ** 2
                + (end.position[2] - start.position[2]) ** 2
            )
            if not 5.0 <= planar_distance <= 17.0:
                continue
            midpoint = (
                (start.position[0] + end.position[0]) * 0.5,
                (start.position[2] + end.position[2]) * 0.5,
            )
            if world_to_chunk(*midpoint) != coord:
                continue
            start_cell, end_cell = sorted((start.cell, end.cell))
            score = _stable_seed(world_seed, *start_cell, *end_cell, _WIRE_SALT)
            candidates.append((score, start_cell, end_cell))
    candidates.sort()
    wires: list[CityCube] = []
    used: set[tuple[int, int]] = set()
    for _, start_cell, end_cell in candidates:
        if len(wires) >= WIRE_RUNS_PER_CHUNK * WIRE_SEGMENTS_PER_RUN:
            break
        if start_cell in used and end_cell in used:
            continue
        start, end = buildings[start_cell], buildings[end_cell]
        rng = Random(_stable_seed(world_seed, *start_cell, *end_cell, _WIRE_SALT, 9))
        start_point = _roof_anchor(start, rng)
        end_point = _roof_anchor(end, rng)
        sag = rng.uniform(0.45, 1.35)
        points = (
            start_point,
            _lerp_with_sag(start_point, end_point, 1.0 / 3.0, sag),
            _lerp_with_sag(start_point, end_point, 2.0 / 3.0, sag),
            end_point,
        )
        wires.extend(
            _wire_cube(points[index], points[index + 1])
            for index in range(WIRE_SEGMENTS_PER_RUN)
        )
        used.update((start_cell, end_cell))
    return tuple(wires)


def generate_chunk(coord: ChunkCoord, world_seed: int) -> CityChunk:
    """Generate one chunk independently of cache state or generation order."""

    origin_x = coord[0] * CHUNK_SIZE
    origin_z = coord[1] * CHUNK_SIZE
    terrain = _generate_terrain(coord, world_seed)
    roads = _generate_roads(coord, world_seed)
    roadbeds = _generate_roadbeds(roads, world_seed)
    supports = _generate_supports(roads, world_seed)
    nearby = _accepted_buildings_near(
        origin_x - 9.0,
        origin_x + CHUNK_SIZE + 9.0,
        origin_z - 9.0,
        origin_z + CHUNK_SIZE + 9.0,
        world_seed,
    )
    buildings = tuple(
        sorted(
            (
                building
                for building in nearby.values()
                if world_to_chunk(building.position[0], building.position[2]) == coord
            ),
            key=lambda building: building.cell or (0, 0),
        )
    )
    wires = _generate_wires(coord, nearby, world_seed)
    chunk = CityChunk(
        coord=coord,
        surfaces=_recolor_objects(
            terrain + roadbeds + roads + supports, world_seed, coord, 0
        ),
        buildings=_recolor_objects(buildings, world_seed, coord, 1),
        wires=_recolor_objects(wires, world_seed, coord, 2),
    )
    if len(chunk.objects) > MAX_OBJECTS_PER_CHUNK:
        raise AssertionError("city chunk exceeded its object budget")
    return chunk


def _roof_anchor(building: CityCube, rng: Random) -> Vec3:
    x, y, z = building.position
    half_x, half_y, half_z = building.half_extents
    return (
        x + rng.uniform(-half_x * 0.28, half_x * 0.28),
        y + half_y + 0.18,
        z + rng.uniform(-half_z * 0.28, half_z * 0.28),
    )


def _lerp_with_sag(start: Vec3, end: Vec3, amount: float, sag: float) -> Vec3:
    x = start[0] + (end[0] - start[0]) * amount
    y = start[1] + (end[1] - start[1]) * amount
    z = start[2] + (end[2] - start[2]) * amount
    curve = 4.0 * amount * (1.0 - amount)
    return x, y - sag * curve, z


def _wire_cube(start: Vec3, end: Vec3) -> CityCube:
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    delta_z = end[2] - start[2]
    xy_length = sqrt(delta_x * delta_x + delta_y * delta_y)
    length = sqrt(xy_length * xy_length + delta_z * delta_z)
    yaw = -degrees(atan2(delta_z, xy_length))
    roll = degrees(atan2(delta_y, delta_x))
    return CityCube(
        role="wire",
        position=(
            (start[0] + end[0]) * 0.5,
            (start[1] + end[1]) * 0.5,
            (start[2] + end[2]) * 0.5,
        ),
        half_extents=(length * 0.5, 0.045, 0.045),
        rotation=(0.0, yaw, roll),
        color=(0.025, 0.03, 0.035),
    )
