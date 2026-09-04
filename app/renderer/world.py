"""Procedural chaotic landscape streamed as instanced cubes.

One world seed picks a palette and a roster of biomes; a low-frequency biome
field decides which generators add forms to each patch of ground and blends
neighbours at the borders, so flying a few hundred units crosses into a
different landscape. Everything is a deterministic function of (chunk, seed).

The landform carries the drama. Quantised terraces over a carved ravine network
give more than 100 units of relief across a short flight, and every ground cube
is a column sized to the drop onto its neighbours, so cliff faces are solid and
the silhouette and horizon keep changing instead of reading as one level field
of debris. Ground is emitted once per chunk, not once per biome, so blend zones
stay solid; a biome's identity is the forms it stacks on top.
"""

from __future__ import annotations

from colorsys import hsv_to_rgb
from dataclasses import dataclass, replace
from functools import lru_cache
from math import atan2, cos, degrees, floor, radians, sin, sqrt
from random import Random
from typing import Callable, Iterable, Literal, TypeAlias

import numpy as np

ChunkCoord: TypeAlias = tuple[int, int]
Vec3: TypeAlias = tuple[float, float, float]
Color: TypeAlias = tuple[float, float, float]
Role: TypeAlias = Literal["ground", "form"]
HeightGrid: TypeAlias = dict[tuple[int, int], float]

CHUNK_SIZE = 64.0
ACTIVE_CHUNK_RADIUS = 5
MAX_ACTIVE_CHUNKS = (ACTIVE_CHUNK_RADIUS * 2 + 1) ** 2
MAX_OBJECTS_PER_CHUNK = 512
# Standing blocks per chunk shared by every biome; roughly one per 60 square
# units, so a walker always has forms within a few steps on every side.
FIELD_BLOCKS_PER_CHUNK = 18

# Ground grid shared by every biome in a chunk: 8-unit cells plus a one-cell
# margin, so a column can be sized against neighbours across a chunk border.
GRID_DIVISIONS = 8
GRID_CELL = CHUNK_SIZE / GRID_DIVISIONS
GROUND_MIN_DEPTH = 16.0

# One biome cell is four chunks across, so a 200-400 unit flight crosses a
# border and the blend band is roughly half a cell wide.
BIOME_CELL = 256.0
BIOME_NAMES: tuple[str, ...] = ("strata", "shards", "canyons", "voxels", "monoliths")
BIOMES_PER_WORLD = 3
MIN_BIOME_WEIGHT = 0.06

# Walker height. The eye follows the terrain sampled a short ring ahead, so a
# terrace riser starts lifting the walker before the wall is reached, and the
# climb is rate-limited in world units per second so a 16-unit step reads as a
# half-second rise rather than a cut. The hard floor below the eye is what
# keeps the camera out of the ground columns during a sprint.
EYE_HEIGHT = 2.2
# Ground changes taller than this between two footfalls are walls, not steps,
# and so is any grade steeper than MAX_SLOPE (rise per unit of horizontal
# travel), which is what keeps a tiny per-frame step off a near-cliff ramp.
STEP_MAX = 4.5
MAX_SLOPE = 1.2
# Radius of the ring the eye height is averaged over, so 8-unit ground cells
# on a slope read as a ramp rather than a staircase.
EYE_SMOOTH_RADIUS = 5.0
CLIMB_RATE = 14.0
# Walker body radius for collision against standing forms, and the share of a
# push that is redirected along the face so head-on contact slides.
WALKER_RADIUS = 0.8
SLIDE_NUDGE = 0.35

_MASK_64 = (1 << 64) - 1
_TERRAIN_SALT = 0x7E44A1
_BIOME_SALT = 0xB10E5
_OBJECT_COLOR_SALT = 0xC01045
_ACCENT_SALT = 0xACCE27
_SKY_COLOR_SALT = 0x5A7C010
_SPAWN_SALT = 0x5AA17
_GROUND_SALT = 0x6120D

# Landform scales. Three smooth octaves plus the ravine cut are summed and then
# quantised once: quantising the sum is what produces terraces several ground
# cells wide with a 16-unit riser between them, instead of the per-cell stepping
# that quantising each octave separately gives.
_MACRO_CELL, _MACRO_AMPLITUDE = 420.0, 120.0
_MID_CELL, _MID_AMPLITUDE = 190.0, 48.0
_FINE_CELL, _FINE_AMPLITUDE = 90.0, 6.0
_TERRACE_STEP = 24.0
_RAVINE_CELL, _RAVINE_DEPTH, _RAVINE_EDGE = 160.0, 60.0, 0.70
# Pass corridors: a band of a low-frequency field where the terraces give way
# to the smooth landform and the ravine cut fades, so a walker can change
# level on a slope. Everywhere else a terrace riser is a cliff and a wall.
_PASS_CELL, _PASS_CENTER, _PASS_HALF_WIDTH = 260.0, 0.5, 0.26

_HUE_OFFSETS = (0.17, 0.33, 0.5)
_LEGIBILITY_MARGIN = 0.25
_ROLE_HUE_SLOT: dict[str, int] = {"ground": 0, "form": 1}
# Base values before shading and tone; the banded key light in proxy.frag
# drops away-facing sides to roughly a sixth of these.
_ROLE_VALUE_RANGE: dict[str, tuple[float, float]] = {
    "ground": (0.62, 0.90),
    "form": (0.70, 0.96),
}
# The accent is a low-frequency ridge field, not a per-cube coin flip, so a
# whole terrace or ravine floor takes the hue and it survives diffusion at
# 512x384. The ridge form is used because plain thresholded noise swings from
# 5% to 68% coverage between seeds, which floods some worlds entirely.
_ACCENT_CELL = 110.0
_ACCENT_THRESHOLD = 0.86
_NEUTRAL_SATURATION = (0.04, 0.13)
_ACCENT_SATURATION = (0.74, 0.96)
_ACCENT_VALUE = (0.60, 0.92)
_MIN_TONE = 0.10
_HUE_NAMES = (
    (0.04, "red"), (0.10, "amber"), (0.19, "lime"), (0.42, "green"),
    (0.53, "cyan"), (0.66, "cobalt"), (0.79, "violet"), (0.92, "magenta"),
    (1.01, "red"),
)


@dataclass(frozen=True, slots=True)
class WorldCube:
    """Renderer-neutral cube using the renderer's half-extent convention.

    ``tone`` scales the colour value down; ravine floors and column bodies use
    it to put true blacks next to the pale slabs.
    """

    role: Role
    position: Vec3
    half_extents: Vec3
    rotation: Vec3
    color: Color
    tone: float = 1.0


@dataclass(frozen=True, slots=True)
class WorldChunk:
    coord: ChunkCoord
    biomes: tuple[str, ...]
    objects: tuple[WorldCube, ...]


@dataclass(frozen=True, slots=True)
class ChunkCachePlan:
    """A fixed-size target set plus the exact cache mutations needed."""

    center: ChunkCoord
    desired: tuple[ChunkCoord, ...]
    load: tuple[ChunkCoord, ...]
    keep: tuple[ChunkCoord, ...]
    evict: tuple[ChunkCoord, ...]


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


def _smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


@lru_cache(maxsize=8192)
def _lattice(world_seed: int, salt: int, ix: int, iz: int) -> float:
    """Cached noise value at one lattice corner; chunks reuse corners heavily."""

    return _unit_float(world_seed, salt, ix, iz)


def _value_noise(x: float, z: float, cell: float, world_seed: int, salt: int) -> float:
    """Smooth lattice noise in [0, 1), continuous in x and z."""

    fx, fz = x / cell, z / cell
    ix, iz = floor(fx), floor(fz)
    tx, tz = _smoothstep(fx - ix), _smoothstep(fz - iz)
    c00 = _lattice(world_seed, salt, ix, iz)
    c10 = _lattice(world_seed, salt, ix + 1, iz)
    c01 = _lattice(world_seed, salt, ix, iz + 1)
    c11 = _lattice(world_seed, salt, ix + 1, iz + 1)
    low = c00 + (c10 - c00) * tx
    high = c01 + (c11 - c01) * tx
    return low + (high - low) * tz


def _ridge(x: float, z: float, cell: float, world_seed: int, salt: int) -> float:
    """Ridged variant of the value noise, in [0, 1], peaking on ridge lines."""

    return 1.0 - abs(2.0 * _value_noise(x, z, cell, world_seed, salt) - 1.0)


def ravine_depth(x: float, z: float, world_seed: int) -> float:
    """How far the ravine network cuts below the plateau here; 0 on the plateau.

    Ridge lines of the noise are the channel centres, so the ravines form a
    connected branching network rather than isolated pits.
    """

    channel = _ridge(x, z, _RAVINE_CELL, world_seed, _TERRAIN_SALT + 4)
    t = (channel - _RAVINE_EDGE) / (1.0 - _RAVINE_EDGE)
    if t <= 0.0:
        return 0.0
    return _RAVINE_DEPTH * _smoothstep(min(t, 1.0))


def pass_weight(x: float, z: float, world_seed: int) -> float:
    """1 inside a pass corridor, 0 on the terraces, smooth in between."""

    field = _value_noise(x, z, _PASS_CELL, world_seed, _TERRAIN_SALT + 7)
    t = 1.0 - abs(field - _PASS_CENTER) / _PASS_HALF_WIDTH
    # The inner part of the corridor is fully smooth; only its walls blend.
    return _smoothstep(min(1.0, max(0.0, t * 1.6)))


def terrain_height(x: float, z: float, world_seed: int) -> float:
    """Continuous landform shared by every biome, so seams and clamps agree.

    Mesa, bench and grain octaves sum into a smooth field. On the terraces it
    is quantised into hard steps and the ravine network cuts through it; inside
    a pass corridor the smooth field shows through and the cut fades, which is
    the slope a walker uses to change level.
    """

    smooth = (
        _MACRO_AMPLITUDE * _value_noise(x, z, _MACRO_CELL, world_seed, _TERRAIN_SALT + 1)
        + _MID_AMPLITUDE * _value_noise(x, z, _MID_CELL, world_seed, _TERRAIN_SALT + 2)
        + _FINE_AMPLITUDE * _value_noise(x, z, _FINE_CELL, world_seed, _TERRAIN_SALT + 3)
    )
    passing = pass_weight(x, z, world_seed)
    value = smooth - ravine_depth(x, z, world_seed) * (1.0 - passing)
    # Soft quantiser: on the terraces (passing 0) this is a hard floor with a
    # riser at every step; inside a pass the top ``passing`` share of each
    # terrace becomes a ramp up to the next, so the surface is continuous and
    # the walker can change level. Blending a hard floor with the smooth field
    # instead would leave a scaled-down jump at every riser.
    terraced = floor(value / _TERRACE_STEP) * _TERRACE_STEP
    # Inside a pass the smooth field shows through. The corridor walls, where
    # ``passing`` is fractional, keep a scaled-down riser, which the slope
    # test still treats as a wall; the corridor centre is a clean ramp.
    return terraced * (1.0 - passing) + value * passing


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


# --- palette ---------------------------------------------------------------


@lru_cache(maxsize=16)
def _sky_hsv(world_seed: int) -> tuple[float, float, float]:
    """A near-white overcast sky with only a faint seeded tint."""

    hue = _unit_float(world_seed, _SKY_COLOR_SALT, 1)
    saturation = 0.03 + 0.09 * _unit_float(world_seed, _SKY_COLOR_SALT, 2)
    value = 0.88 + 0.12 * _unit_float(world_seed, _SKY_COLOR_SALT, 3)
    return hue, saturation, value


def sky_color(world_seed: int) -> Color:
    """Return the seeded clear and fog color shared by one world."""

    return hsv_to_rgb(*_sky_hsv(world_seed))


def _hue_distance(a: float, b: float) -> float:
    """Shortest distance between two hues on the wrap-around [0, 1) wheel."""

    diff = abs(a - b) % 1.0
    return min(diff, 1.0 - diff)


@lru_cache(maxsize=16)
def world_palette(world_seed: int) -> tuple[float, float, float]:
    """Two or three hues shared by every object in one world.

    The primary hue matches the sky and the secondary steps around the wheel
    by a per-seed offset. Only the accent slot is ever drawn at full
    saturation, so these read as one pale field plus a bold region. An offset
    of 0.5 folds accent = primary + 2*0.5 back onto the primary; that is still
    a deliberate two-tone world, but the accent slot would then share the sky
    hue and vanish, so it maps to the secondary hue instead.
    """

    primary_hue, _, _ = _sky_hsv(world_seed)
    pick = min(2, int(_unit_float(world_seed, _OBJECT_COLOR_SALT, 50) * len(_HUE_OFFSETS)))
    offset = _HUE_OFFSETS[pick]
    secondary_hue = (primary_hue + offset) % 1.0
    accent_hue = secondary_hue if offset == 0.5 else (primary_hue + 2.0 * offset) % 1.0
    return primary_hue, secondary_hue, accent_hue


def is_accent_region(x: float, z: float, world_seed: int) -> bool:
    """Whether this patch of ground belongs to a contiguous accent region."""

    return _ridge(x, z, _ACCENT_CELL, world_seed, _ACCENT_SALT) > _ACCENT_THRESHOLD


def object_color(world_seed: int, position: Vec3, role: Role, tone: float) -> Color:
    """Colour one cube from its world position: pale neutral, or accent hue.

    The accent comes from a low-frequency spatial field, so contiguous
    regions - a whole terrace, a whole ravine floor - carry the hue. ``tone``
    then scales the value down, which is what puts blacks in the frame.
    """

    x, y, z = position
    parts = (_OBJECT_COLOR_SALT, int(x * 8.0), int(y * 8.0), int(z * 8.0))
    palette = world_palette(world_seed)
    # Flat plateau floors stay neutral: a saturated plate at eye level reads
    # as a tabletop. Forms and ravine floors carry the accent.
    accent = is_accent_region(x, z, world_seed) and (
        role == "form" or ravine_depth(x, z, world_seed) > 8.0
    )
    hue = palette[2 if accent else _ROLE_HUE_SLOT[role]]
    sat_low, sat_high = _ACCENT_SATURATION if accent else _NEUTRAL_SATURATION
    saturation = sat_low + (sat_high - sat_low) * _unit_float(world_seed, *parts, 2)
    low, high = _ACCENT_VALUE if accent else _ROLE_VALUE_RANGE[role]
    value = low + (high - low) * _unit_float(world_seed, *parts, 3)

    # Ground carries the sky hue by design. At these saturations the tint is
    # barely visible, but a ground cube whose value also matches the sky would
    # vanish into the fog, so separate it by value instead of leaving the
    # palette. Every object band sits below the sky, so the push is downward
    # and cannot overshoot past white.
    if _hue_distance(hue, palette[0]) <= 1e-6:
        _, _, sky_value = _sky_hsv(world_seed)
        if abs(value - sky_value) < _LEGIBILITY_MARGIN:
            midpoint = (low + high) * 0.5
            pushed = (
                sky_value - _LEGIBILITY_MARGIN
                if midpoint <= sky_value
                else sky_value + _LEGIBILITY_MARGIN
            )
            value = min(1.0, max(0.0, pushed))
    return hsv_to_rgb(hue, saturation, value * max(_MIN_TONE, min(1.0, tone)))


def _hue_name(hue: float) -> str:
    for limit, name in _HUE_NAMES:
        if hue < limit:
            return name
    return "red"


# --- biome field -----------------------------------------------------------


@lru_cache(maxsize=32)
def biome_roster(world_seed: int) -> tuple[str, ...]:
    """The subset of biomes that exist in one world, so seeds differ."""

    rng = Random(_stable_seed(world_seed, _BIOME_SALT))
    return tuple(rng.sample(BIOME_NAMES, BIOMES_PER_WORLD))


@lru_cache(maxsize=4096)
def _biome_at_cell(cell_x: int, cell_z: int, world_seed: int) -> int:
    roster = biome_roster(world_seed)
    return min(
        len(roster) - 1,
        int(_unit_float(world_seed, _BIOME_SALT, cell_x, cell_z, 7) * len(roster)),
    )


def biome_weights(x: float, z: float, world_seed: int) -> dict[str, float]:
    """Blend weights over the world's biomes, summing to one."""

    roster = biome_roster(world_seed)
    fx, fz = x / BIOME_CELL - 0.5, z / BIOME_CELL - 0.5
    ix, iz = floor(fx), floor(fz)
    tx, tz = _smoothstep(fx - ix), _smoothstep(fz - iz)
    weights = dict.fromkeys(roster, 0.0)
    for offset_z, weight_z in ((0, 1.0 - tz), (1, tz)):
        for offset_x, weight_x in ((0, 1.0 - tx), (1, tx)):
            name = roster[_biome_at_cell(ix + offset_x, iz + offset_z, world_seed)]
            weights[name] += weight_x * weight_z
    return weights


def dominant_biome(x: float, z: float, world_seed: int) -> str:
    weights = biome_weights(x, z, world_seed)
    return max(weights, key=lambda name: (weights[name], name))


def surface_height(x: float, z: float, world_seed: int) -> float:
    """Top of the solid ground layer, which every biome shares.

    Ground is emitted once per chunk rather than once per biome, so blend zones
    keep full coverage instead of showing sky through a half-thinned floor, and
    the camera clamp can trust a single height field.
    """

    return terrain_height(x, z, world_seed)


def walk_height(x: float, z: float, world_seed: int) -> float:
    """Eye height here: the ground averaged over a small ring, plus eye height.

    Risers are walls (``step_blocked``), so the only ground changes a walker
    crosses are slope cells; averaging over a ring turns those 8-unit cells
    into a ramp. Samples more than a step above the ground underfoot are
    ignored so a wall the walker stands beside does not lift the eye.
    """

    here = surface_height(x, z, world_seed)
    total, weight = here, 1.0
    for step in range(8):
        angle = radians(step * 45.0)
        sample = surface_height(
            x + EYE_SMOOTH_RADIUS * cos(angle), z + EYE_SMOOTH_RADIUS * sin(angle), world_seed
        )
        # Soft weight: a sample fades out as it approaches a step above or
        # below, so a wall entering the ring never jolts the eye.
        share = max(0.0, 1.0 - abs(sample - here) / STEP_MAX)
        total += sample * share
        weight += share
    return total / weight + EYE_HEIGHT


def step_blocked(from_x: float, from_z: float, to_x: float, to_z: float, world_seed: int) -> bool:
    """Whether the ground between two footfalls is a wall: too tall or too steep."""

    rise = abs(surface_height(to_x, to_z, world_seed) - surface_height(from_x, from_z, world_seed))
    run = sqrt((to_x - from_x) ** 2 + (to_z - from_z) ** 2)
    return rise > STEP_MAX or rise > MAX_SLOPE * max(run, 1e-6)


def settle_height(x: float, y: float, z: float, world_seed: int, dt: float | None) -> float:
    """Move the eye toward ``walk_height`` at ``CLIMB_RATE``; ``dt`` None snaps.

    Descents and climbs both ease, but the eye is never allowed below the
    ground directly underfoot, so a sprint into a cliff cannot clip inside it.
    """

    target = walk_height(x, z, world_seed)
    if dt is None:
        return target
    limit = CLIMB_RATE * max(0.0, dt)
    y = y + max(-limit, min(limit, target - y))
    return max(y, surface_height(x, z, world_seed) + EYE_HEIGHT * 0.5)


def chunk_colliders(chunk: "WorldChunk") -> np.ndarray:
    """Standing forms of one chunk as yaw-aligned footprints for collision.

    Rows are (cx, cz, cos_yaw, sin_yaw, half_x, half_z, y_low, y_high). Tilt is
    ignored: the forms a walker can hit are the upright ones, and the tilted
    giants hang above head height by construction.
    """

    rows = [
        (
            cube.position[0],
            cube.position[2],
            cos(radians(cube.rotation[1])),
            sin(radians(cube.rotation[1])),
            cube.half_extents[0],
            cube.half_extents[2],
            cube.position[1] - cube.half_extents[1],
            cube.position[1] + cube.half_extents[1],
        )
        for cube in chunk.objects
        if cube.role == "form"
    ]
    return np.asarray(rows, dtype=np.float64).reshape(-1, 8)


def resolve_collisions(
    x: float, y: float, z: float, colliders: np.ndarray, radius: float = WALKER_RADIUS
) -> tuple[float, float]:
    """Push (x, z) out of every footprint whose vertical span contains ``y``.

    Each overlap is resolved along its shallower axis in the form's own frame,
    which makes the walker slide along a wall instead of stopping dead. The
    loop runs until nothing overlaps, so standing between two forms settles to
    a point both accept instead of bouncing between them every frame.
    """

    if colliders.shape[0] == 0:
        return x, z
    for _ in range(6):
        hit = (colliders[:, 6] < y) & (y < colliders[:, 7])
        if not hit.any():
            break
        rows = colliders[hit]
        dx, dz = x - rows[:, 0], z - rows[:, 1]
        c, s_ = rows[:, 2], rows[:, 3]
        # The model rotates local x/z by yaw; undo it to get footprint coords.
        local_x = dx * c - dz * s_
        local_z = dx * s_ + dz * c
        pen_x = rows[:, 4] + radius - np.abs(local_x)
        pen_z = rows[:, 5] + radius - np.abs(local_z)
        inside = (pen_x > 0.0) & (pen_z > 0.0)
        if not inside.any():
            break
        index = int(np.argmax(np.where(inside, np.minimum(pen_x, pen_z), -np.inf)))
        # Resolve along the shallow axis, plus a nudge along the face toward the
        # nearer edge, so a head-on walk into a block slides off it instead of
        # pinning the walker against the wall.
        # The nudge scales with how far off centre the contact is, so a dead-on
        # hit is a clean push and an off-centre one curls around the corner.
        if pen_x[index] < pen_z[index]:
            push_x = np.copysign(pen_x[index], local_x[index])
            push_z = SLIDE_NUDGE * pen_x[index] * local_z[index] / (rows[index, 5] + radius)
        else:
            push_z = np.copysign(pen_z[index], local_z[index])
            push_x = SLIDE_NUDGE * pen_z[index] * local_x[index] / (rows[index, 4] + radius)
        # Back to world axes: the inverse rotation of the one above.
        x += push_x * c[index] + push_z * s_[index]
        z += -push_x * s_[index] + push_z * c[index]
    return x, z


def open_heading(
    x: float, y: float, z: float, colliders: np.ndarray, candidates: int = 16,
    world_seed: int | None = None,
) -> tuple[float, float]:
    """Yaw with the longest unobstructed walk from (x, z), and that distance.

    Steps 1.5 units at a time along each candidate heading, resolving
    collisions as the walker would and, when a seed is given, treating a
    riser taller than a step as a wall. Stops when a step gains under a third
    of its length. A spawn facing a wall would otherwise pin the walker.
    """

    best_yaw, best_distance = 0.0, -1.0
    # Start from a settled spot: a walker wedged between forms is inside an
    # overlap, and probing from inside one reports every heading as open.
    x, z = resolve_collisions(x, y, z, colliders)
    for index in range(candidates):
        yaw = index * 360.0 / candidates
        heading = (sin(radians(yaw)), -cos(radians(yaw)))
        px, pz = x, z
        travelled = 0.0
        for _ in range(40):
            nx, nz = resolve_collisions(px + heading[0] * 1.5, y, pz + heading[1] * 1.5, colliders)
            if world_seed is not None and step_blocked(px, pz, nx, nz, world_seed):
                break
            gained = (nx - px) * heading[0] + (nz - pz) * heading[1]
            # A step that slides sideways more than it advances is a wall.
            side = abs((nx - px) * heading[1] - (nz - pz) * heading[0])
            if gained < 0.5 or side > gained:
                break
            travelled += gained
            px, pz = nx, nz
        if travelled > best_distance:
            best_yaw, best_distance = yaw, travelled
    return best_yaw, best_distance


class Autowalk:
    """Idle stroll: wander at walking pace, drifting the gaze, turning at walls.

    Heading and gaze both ease toward slowly re-rolled targets so the motion
    reads as a person looking around rather than a camera on rails. A blocked
    step (the walker gained under a third of it) picks the longest open heading
    instead; ``open_heading`` is the same search the spawn uses.
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng = Random(seed)
        self._world_seed = seed
        self._target_yaw: float | None = None
        self._target_pitch = -4.0
        self._retarget_at = 0.0
        self._turning_until = 0.0
        self._anchor: tuple[float, float, float] | None = None
        self._low_frames = 0

    def reset(self) -> None:
        self._target_yaw = None
        self._retarget_at = 0.0
        self._turning_until = 0.0
        self._anchor = None

    def turning(self, now: float) -> bool:
        """True while the walker stands and turns after a blocked step."""
        return now < self._turning_until

    def step(
        self,
        yaw: float,
        pitch: float,
        position: Vec3,
        now: float,
        dt: float,
        gained: float,
        step_length: float,
        colliders: np.ndarray,
    ) -> tuple[float, float]:
        """Advance the stroll one frame; returns the new (yaw, pitch).

        ``gained`` is the horizontal distance the previous step actually moved
        after collision, against the ``step_length`` it asked for.
        """
        if self._target_yaw is None or now >= self._retarget_at:
            self._target_yaw = (yaw + self._rng.uniform(-70.0, 70.0)) % 360.0
            self._target_pitch = self._rng.uniform(-14.0, 6.0)
            self._retarget_at = now + self._rng.uniform(4.0, 9.0)
        # Progress is judged over a one-second window, not per frame: a walker
        # oscillating against a riser still "gains" a step every frame.
        stuck = False
        if self.turning(now) or step_length <= 0.0:
            self._anchor = None
            self._low_frames = 0
            # Turning has an end: once the heading has swung round, walk,
            # even if the target was reached early.
            if abs((self._target_yaw - yaw + 180.0) % 360.0 - 180.0) < 4.0:
                self._turning_until = 0.0
        else:
            self._low_frames = self._low_frames + 1 if gained < step_length * 0.35 else 0
            stuck = self._low_frames >= 10
        if self.turning(now) or step_length <= 0.0:
            pass
        elif self._anchor is None:
            self._anchor = (position[0], position[2], now)
        elif now - self._anchor[2] >= 1.0:
            moved = sqrt((position[0] - self._anchor[0]) ** 2 + (position[2] - self._anchor[1]) ** 2)
            stuck = stuck or moved < step_length / max(dt, 1e-6) * 0.35
            self._anchor = (position[0], position[2], now)
        if stuck:
            open_yaw, distance = open_heading(
                position[0], position[1], position[2], colliders, world_seed=self._world_seed
            )
            self._low_frames = 0
            if distance <= 3.0:
                # Boxed in on every heading: turn well away and try again.
                open_yaw = (yaw + self._rng.uniform(120.0, 240.0)) % 360.0
            elif distance < 25.0 and abs((open_yaw - yaw + 180.0) % 360.0 - 180.0) < 15.0:
                # A pocket: every heading is short and the best one is the one
                # just tried. Leave along a different edge instead.
                open_yaw = (open_yaw + self._rng.choice((-90.0, 90.0, 180.0))) % 360.0
            self._target_yaw = open_yaw
            self._retarget_at = now + self._rng.uniform(3.0, 6.0)
            # Stand and turn for a moment: pushing into the wall while the
            # heading eases round is what made the stroll jitter in place.
            self._turning_until = now + 1.2
        # Ease toward the targets along the shortest arc so a turn never spins.
        delta = (self._target_yaw - yaw + 180.0) % 360.0 - 180.0
        rate = 2.6 if self.turning(now) else 1.4
        new_yaw = (yaw + delta * min(1.0, rate * dt)) % 360.0
        new_pitch = pitch + (self._target_pitch - pitch) * min(1.0, 0.9 * dt)
        return new_yaw, new_pitch


def spawn_pose(world_seed: int) -> tuple[Vec3, float, float]:
    """Stand on open ground inside the relief, looking out along a drop.

    Candidates are scored by how much the ring around them rises and falls, so
    the walker starts surrounded by risers and ravines rather than on a flat
    plate. The yaw faces the lowest neighbour, which puts depth in the first
    frame, and the pitch is level with a slight downward tilt.
    """

    best: tuple[float, float, float, float] | None = None
    for index in range(20):
        x = (_unit_float(world_seed, _SPAWN_SALT, index, 1) - 0.5) * 700.0
        z = (_unit_float(world_seed, _SPAWN_SALT, index, 2) - 0.5) * 700.0
        here = surface_height(x, z, world_seed)
        if ravine_depth(x, z, world_seed) > 20.0:
            continue
        # Reject a pocket: if every direction rises within a few steps the
        # walker starts face-first in a riser.
        close = [
            surface_height(x + 12.0 * cos(radians(a)), z + 12.0 * sin(radians(a)), world_seed)
            for a in range(0, 360, 45)
        ]
        if min(close) > here + 8.0:
            continue
        ring = [
            (
                surface_height(
                    x + 40.0 * cos(radians(step * 45.0)),
                    z + 40.0 * sin(radians(step * 45.0)),
                    world_seed,
                ),
                step * 45.0,
            )
            for step in range(8)
        ]
        lowest, lowest_angle = min(ring)
        highest = max(height for height, _ in ring)
        score = min(highest - here, 48.0) + min(here - lowest, 48.0)
        if best is None or score > best[0]:
            # Camera forward is (sin yaw, ., -cos yaw); the ring offset is
            # (cos a, sin a) in x/z, so yaw = a + 90 points at that ring point.
            yaw = (90.0 + lowest_angle) % 360.0
            best = (score, x, z, yaw)

    _, x, z, yaw = best or (0.0, 0.0, 0.0, 0.0)
    pitch = -4.0 - 8.0 * _unit_float(world_seed, _SPAWN_SALT, 3)
    return (x, walk_height(x, z, world_seed), z), yaw, pitch


def world_label(world_seed: int, x: float = 0.0, z: float = 0.0) -> str:
    """Short biome and palette label for the on-screen overlay."""

    weights = biome_weights(x, z, world_seed)
    ranked = sorted(weights.items(), key=lambda item: (-item[1], item[0]))
    here = "+".join(name for name, weight in ranked if weight >= 0.2) or ranked[0][0]
    primary, secondary, accent = world_palette(world_seed)
    hues = [_hue_name(primary), _hue_name(secondary)]
    if _hue_distance(accent, secondary) > 1e-6:
        hues.append(_hue_name(accent))
    return f"{here} / {'-'.join(dict.fromkeys(hues))}"


# --- chunk scaffolding -----------------------------------------------------


def _chunk_grid(coord: ChunkCoord, world_seed: int) -> tuple[HeightGrid, HeightGrid]:
    """Landform top and ravine cut per cell of one chunk plus a one-cell margin.

    Sampled once and shared by every biome generator in the chunk. The margin
    is what lets a generator size a column against its neighbours across a
    chunk border, so cliff faces come out solid instead of floating.
    """

    origin_x, origin_z = coord[0] * CHUNK_SIZE, coord[1] * CHUNK_SIZE
    centers = {
        (ix, iz): (origin_x + (ix + 0.5) * GRID_CELL, origin_z + (iz + 0.5) * GRID_CELL)
        for iz in range(-1, GRID_DIVISIONS + 1)
        for ix in range(-1, GRID_DIVISIONS + 1)
    }
    tops = {key: terrain_height(x, z, world_seed) for key, (x, z) in centers.items()}
    cuts = {key: ravine_depth(x, z, world_seed) for key, (x, z) in centers.items()}
    return tops, cuts


def _cell_center(coord: ChunkCoord, ix: int, iz: int) -> tuple[float, float]:
    return (
        coord[0] * CHUNK_SIZE + (ix + 0.5) * GRID_CELL,
        coord[1] * CHUNK_SIZE + (iz + 0.5) * GRID_CELL,
    )



def _ground_columns(
    coord: ChunkCoord, tops: HeightGrid, cuts: HeightGrid, rng: Random
) -> list[WorldCube]:
    """The solid ground layer of one chunk: one column per grid cell.

    Each column reaches down past its lowest neighbour, which is what turns a
    terrace step into a real cliff face - a fixed-thickness plate leaves the
    wall hollow and the landform reads flat from the air. Columns inside the
    ravine network fade towards black so the channels stay legible.
    """

    cubes: list[WorldCube] = []
    for iz in range(GRID_DIVISIONS):
        for ix in range(GRID_DIVISIONS):
            top = tops[(ix, iz)]
            depth = max(GROUND_MIN_DEPTH, _neighbour_drop(tops, ix, iz) + 5.0)
            center_x, center_z = _cell_center(coord, ix, iz)
            # Ravine floors darken to about half, not black: the walker
            # stands in them now, and a black frame gives the sampler nothing.
            # Per-cell tone variance gives the floor a coarse checker, so a
            # flat plate carries perspective lines instead of reading as a desk.
            tone = rng.uniform(0.58, 1.0) * (
                1.0 - 0.55 * min(1.0, cuts[(ix, iz)] / 40.0)
            )
            cubes.append(
                WorldCube(
                    "ground",
                    (center_x, top - depth * 0.5, center_z),
                    (GRID_CELL * 0.5, depth * 0.5, GRID_CELL * 0.5),
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                    tone,
                )
            )
    return cubes


def _standing_blocks(
    coord: ChunkCoord, world_seed: int, rng: Random
) -> list[WorldCube]:
    """Human-scale blocks and pillars scattered over the whole chunk.

    This is what puts the walker among structures: a few dozen standing forms
    between waist height and three storeys, dense enough that something is
    always within a few steps in every direction but never so tight that a
    heading is blocked. Every biome shares it; the biome generators add their
    own giants on top.
    """

    cubes: list[WorldCube] = []
    origin_x, origin_z = coord[0] * CHUNK_SIZE, coord[1] * CHUNK_SIZE
    for _ in range(FIELD_BLOCKS_PER_CHUNK):
        x = origin_x + rng.uniform(0.0, CHUNK_SIZE)
        z = origin_z + rng.uniform(0.0, CHUNK_SIZE)
        base = terrain_height(x, z, world_seed)
        height = rng.choice((5.0, 9.0, 14.0, 22.0, 30.0)) * rng.uniform(0.7, 1.3)
        # Slender: a form wider than a few steps fills the frame at eye level.
        width = rng.uniform(1.2, 3.0) * (1.0 + height / 60.0)
        cubes.append(
            WorldCube(
                "form",
                (x, base + height * 0.5, z),
                (width, height * 0.5, width * rng.uniform(0.5, 1.4)),
                (rng.uniform(-6.0, 6.0), rng.uniform(0.0, 90.0), rng.uniform(-6.0, 6.0)),
                (0.0, 0.0, 0.0),
                rng.uniform(0.6, 1.0),
            )
        )
    return cubes


def _neighbour_drop(tops: HeightGrid, ix: int, iz: int) -> float:
    """Height difference from this cell down to its lowest 4-neighbour."""

    top = tops[(ix, iz)]
    return top - min(
        tops.get((ix + dx, iz + dz), top)
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1))
    )


def _fragment_cluster(
    rng: Random, center: Vec3, scale: float, count: int
) -> list[WorldCube]:
    """A floating cluster of slabs at one size class.

    Clusters are drawn at wildly different scales - gravel, plate, chunk-sized
    slab - so a frame holds several orders of size at once instead of uniform
    confetti.
    """

    spread = scale * 3.4
    return [
        WorldCube(
            "form",
            (
                center[0] + rng.uniform(-spread, spread),
                center[1] + rng.uniform(-spread * 0.5, spread * 0.9),
                center[2] + rng.uniform(-spread, spread),
            ),
            (
                scale * rng.uniform(0.6, 1.9),
                scale * rng.uniform(0.12, 0.45),
                scale * rng.uniform(0.5, 1.6),
            ),
            (rng.uniform(-48.0, 48.0), rng.uniform(0.0, 360.0), rng.uniform(-48.0, 48.0)),
            (0.0, 0.0, 0.0),
            rng.uniform(0.75, 1.0),
        )
        for _ in range(count)
    ]


# --- biome generators ------------------------------------------------------
#
# Generators add forms on top of the shared ground layer, thinned by their
# blend weight, so a border chunk holds a mix of two landscapes over one solid
# floor. They read only the shared grid, which is a pure function of the chunk
# coord and seed, so generation stays deterministic and order independent.
# Structural forms come first because the per-chunk cap truncates the tail.


def _gen_strata(
    coord: ChunkCoord,
    world_seed: int,
    weight: float,
    rng: Random,
    tops: HeightGrid,
    cuts: HeightGrid,
) -> list[WorldCube]:
    """Terrace caps: chunk-sized flat slabs on the plateaus, rim overhangs, grit."""

    cubes: list[WorldCube] = []
    origin_x, origin_z = coord[0] * CHUNK_SIZE, coord[1] * CHUNK_SIZE

    for _ in range(int(round(5 * weight))):
        ix, iz = rng.randrange(GRID_DIVISIONS), rng.randrange(GRID_DIVISIONS)
        center_x, center_z = _cell_center(coord, ix, iz)
        # Caps hang above head height so the walker passes under them, not
        # through them: a slab across the eye line is a blank frame for seconds.
        half_height = rng.uniform(1.2, 3.6)
        cubes.append(
            WorldCube(
                "form",
                (center_x, tops[(ix, iz)] + half_height + rng.uniform(7.0, 16.0), center_z),
                (rng.uniform(14.0, 30.0), half_height, rng.uniform(9.0, 22.0)),
                (0.0, rng.uniform(0.0, 360.0), rng.uniform(-4.0, 4.0)),
                (0.0, 0.0, 0.0),
                rng.uniform(0.85, 1.0),
            )
        )

    for iz in range(GRID_DIVISIONS):
        for ix in range(GRID_DIVISIONS):
            if rng.random() > weight * 0.45 or _neighbour_drop(tops, ix, iz) < 12.0:
                continue
            center_x, center_z = _cell_center(coord, ix, iz)
            reach = GRID_CELL * rng.uniform(0.9, 1.8)
            cubes.append(
                WorldCube(
                    "form",
                    (
                        center_x + rng.uniform(-reach, reach),
                        tops[(ix, iz)] + rng.uniform(0.4, 2.6),
                        center_z + rng.uniform(-reach, reach),
                    ),
                    (reach, rng.uniform(0.5, 1.6), GRID_CELL * rng.uniform(0.4, 0.9)),
                    (rng.uniform(-10.0, 10.0), rng.uniform(0.0, 360.0), rng.uniform(-10.0, 10.0)),
                    (0.0, 0.0, 0.0),
                    rng.uniform(0.8, 1.0),
                )
            )

    for _ in range(int(round(8 * weight))):
        x = origin_x + rng.uniform(0.0, CHUNK_SIZE)
        z = origin_z + rng.uniform(0.0, CHUNK_SIZE)
        size = rng.uniform(1.5, 4.0)
        cubes.append(
            WorldCube(
                "form",
                (x, terrain_height(x, z, world_seed) + size, z),
                (size, size, size),
                (0.0, rng.uniform(0.0, 45.0), 0.0),
                (0.0, 0.0, 0.0),
                rng.uniform(0.5, 1.0),
            )
        )
    return cubes


def _gen_shards(
    coord: ChunkCoord,
    world_seed: int,
    weight: float,
    rng: Random,
    tops: HeightGrid,
    cuts: HeightGrid,
) -> list[WorldCube]:
    """Steeply tilted giant slabs and floating fragment clusters at three scales."""

    cubes: list[WorldCube] = []
    origin_x, origin_z = coord[0] * CHUNK_SIZE, coord[1] * CHUNK_SIZE

    for _ in range(int(round(4 * weight))):
        ix, iz = rng.randrange(GRID_DIVISIONS), rng.randrange(GRID_DIVISIONS)
        center_x, center_z = _cell_center(coord, ix, iz)
        # Tilted giants clear head height at their centre; their tilt still
        # brings one edge down to the ground, which is the shard look.
        cubes.append(
            WorldCube(
                "form",
                (center_x, tops[(ix, iz)] + rng.uniform(12.0, 26.0), center_z),
                (rng.uniform(12.0, 30.0), rng.uniform(1.0, 3.4), rng.uniform(7.0, 18.0)),
                (rng.uniform(-28.0, 28.0), rng.uniform(0.0, 360.0), rng.uniform(-28.0, 28.0)),
                (0.0, 0.0, 0.0),
                rng.uniform(0.8, 1.0),
            )
        )

    for _ in range(int(round(3 * weight))):
        x = origin_x + rng.uniform(0.0, CHUNK_SIZE)
        z = origin_z + rng.uniform(0.0, CHUNK_SIZE)
        scale = rng.choice((2.6, 2.6, 4.5, 7.0))
        # Big clusters stay low; at cruise altitude a high one fills the frame.
        lift = rng.uniform(8.0, 46.0) if scale < 3.0 else rng.uniform(5.0, 20.0)
        cubes += _fragment_cluster(
            rng, (x, terrain_height(x, z, world_seed) + lift, z), scale, rng.randrange(3, 7)
        )
    return cubes


def _gen_canyons(
    coord: ChunkCoord,
    world_seed: int,
    weight: float,
    rng: Random,
    tops: HeightGrid,
    cuts: HeightGrid,
) -> list[WorldCube]:
    """Ravine furniture: spires off the dark floors, slabs jutting over the rims."""

    cubes: list[WorldCube] = []
    for iz in range(GRID_DIVISIONS):
        for ix in range(GRID_DIVISIONS):
            if rng.random() > weight:
                continue
            center_x, center_z = _cell_center(coord, ix, iz)
            top = tops[(ix, iz)]
            if cuts[(ix, iz)] > 20.0:
                if rng.random() < 0.3:
                    spire = rng.uniform(12.0, 40.0)
                    cubes.append(
                        WorldCube(
                            "form",
                            (center_x, top + spire * 0.5, center_z),
                            (rng.uniform(1.0, 3.2), spire * 0.5, rng.uniform(1.0, 3.2)),
                            (0.0, rng.uniform(0.0, 90.0), 0.0),
                            (0.0, 0.0, 0.0),
                            rng.uniform(0.25, 0.7),
                        )
                    )
                continue
            if _neighbour_drop(tops, ix, iz) < 11.0 or rng.random() > 0.5:
                continue
            reach = GRID_CELL * rng.uniform(1.1, 2.4)
            cubes.append(
                WorldCube(
                    "form",
                    (center_x + rng.uniform(-reach, reach), top - rng.uniform(0.5, 5.0), center_z),
                    (reach, rng.uniform(0.5, 1.5), GRID_CELL * rng.uniform(0.5, 1.2)),
                    (0.0, rng.uniform(0.0, 360.0), rng.uniform(-7.0, 7.0)),
                    (0.0, 0.0, 0.0),
                    rng.uniform(0.7, 1.0),
                )
            )
    return cubes


def _gen_voxels(
    coord: ChunkCoord,
    world_seed: int,
    weight: float,
    rng: Random,
    tops: HeightGrid,
    cuts: HeightGrid,
) -> list[WorldCube]:
    """Dense ridges of small cubes plus a handful of house-sized boulders."""

    cubes: list[WorldCube] = []
    origin_x, origin_z = coord[0] * CHUNK_SIZE, coord[1] * CHUNK_SIZE

    for _ in range(int(round(3 * weight))):
        ix, iz = rng.randrange(GRID_DIVISIONS), rng.randrange(GRID_DIVISIONS)
        center_x, center_z = _cell_center(coord, ix, iz)
        size = rng.uniform(5.0, 13.0)
        cubes.append(
            WorldCube(
                "form",
                (center_x, tops[(ix, iz)] + size, center_z),
                (size, size, size * rng.uniform(0.6, 1.0)),
                (0.0, rng.uniform(0.0, 90.0), 0.0),
                (0.0, 0.0, 0.0),
                rng.uniform(0.8, 1.0),
            )
        )

    for _ in range(int(round(60 * weight))):
        x = origin_x + rng.uniform(0.0, CHUNK_SIZE)
        z = origin_z + rng.uniform(0.0, CHUNK_SIZE)
        if _ridge(x, z, 47.0, world_seed, _TERRAIN_SALT + 21) < 0.62:
            continue
        base = terrain_height(x, z, world_seed)
        size = rng.uniform(1.6, 3.6)
        for level in range(rng.randrange(1, 4)):
            cubes.append(
                WorldCube(
                    "form",
                    (x, base + size * (2 * level + 1), z),
                    (size, size, size),
                    (0.0, rng.uniform(0.0, 45.0), 0.0),
                    (0.0, 0.0, 0.0),
                    rng.uniform(0.45, 1.0),
                )
            )
    return cubes


def _gen_monoliths(
    coord: ChunkCoord,
    world_seed: int,
    weight: float,
    rng: Random,
    tops: HeightGrid,
    cuts: HeightGrid,
) -> list[WorldCube]:
    """Upright slabs, block bridges strung between them, and high fragments."""

    cubes: list[WorldCube] = []
    origin_x, origin_z = coord[0] * CHUNK_SIZE, coord[1] * CHUNK_SIZE

    towers: list[Vec3] = []
    for _ in range(int(round(5 * weight))):
        x = origin_x + rng.uniform(5.0, CHUNK_SIZE - 5.0)
        z = origin_z + rng.uniform(5.0, CHUNK_SIZE - 5.0)
        height = rng.uniform(18.0, 48.0)
        base = terrain_height(x, z, world_seed)
        cubes.append(
            WorldCube(
                "form",
                (x, base + height * 0.5, z),
                (rng.uniform(3.0, 9.0), height * 0.5, rng.uniform(2.0, 6.0)),
                (0.0, rng.uniform(0.0, 90.0), rng.uniform(-6.0, 6.0)),
                (0.0, 0.0, 0.0),
                rng.uniform(0.6, 1.0),
            )
        )
        towers.append((x, base + height, z))

    for index in range(len(towers) - 1):
        if rng.random() > 0.55:
            continue
        start, end = towers[index], towers[index + 1]
        span = sqrt((end[0] - start[0]) ** 2 + (end[2] - start[2]) ** 2)
        blocks = max(4, min(12, int(span / 6.0)))
        yaw = -degrees(atan2(end[2] - start[2], end[0] - start[0]))
        for step in range(blocks):
            amount = (step + 0.5) / blocks
            arc = 4.0 * amount * (1.0 - amount)
            cubes.append(
                WorldCube(
                    "form",
                    (
                        start[0] + (end[0] - start[0]) * amount,
                        start[1] + (end[1] - start[1]) * amount - arc * rng.uniform(1.0, 6.0),
                        start[2] + (end[2] - start[2]) * amount,
                    ),
                    (span / blocks * 0.55, rng.uniform(0.5, 1.4), rng.uniform(1.0, 2.6)),
                    (0.0, yaw, 0.0),
                    (0.0, 0.0, 0.0),
                    rng.uniform(0.5, 0.95),
                )
            )

    for _ in range(int(round(3 * weight))):
        x = origin_x + rng.uniform(0.0, CHUNK_SIZE)
        z = origin_z + rng.uniform(0.0, CHUNK_SIZE)
        base = terrain_height(x, z, world_seed) + rng.uniform(52.0, 108.0)
        cubes += _fragment_cluster(rng, (x, base, z), rng.choice((1.0, 2.4)), rng.randrange(3, 7))
    return cubes


_BIOME_GENERATOR: dict[
    str, Callable[[ChunkCoord, int, float, Random, HeightGrid, HeightGrid], list[WorldCube]]
] = {
    "strata": _gen_strata,
    "shards": _gen_shards,
    "canyons": _gen_canyons,
    "voxels": _gen_voxels,
    "monoliths": _gen_monoliths,
}


def generate_chunk(coord: ChunkCoord, world_seed: int) -> WorldChunk:
    """Generate one chunk independently of cache state or generation order."""

    center_x = (coord[0] + 0.5) * CHUNK_SIZE
    center_z = (coord[1] + 0.5) * CHUNK_SIZE
    weights = biome_weights(center_x, center_z, world_seed)
    present = sorted(
        (name for name, weight in weights.items() if weight >= MIN_BIOME_WEIGHT),
        key=lambda name: (-weights[name], name),
    )
    tops, cuts = _chunk_grid(coord, world_seed)
    ground_rng = Random(_stable_seed(world_seed, coord[0], coord[1], _GROUND_SALT))
    cubes = _ground_columns(coord, tops, cuts, ground_rng)
    cubes.extend(_standing_blocks(coord, world_seed, ground_rng))
    for name in present:
        rng = Random(_stable_seed(world_seed, coord[0], coord[1], BIOME_NAMES.index(name)))
        cubes.extend(_BIOME_GENERATOR[name](coord, world_seed, weights[name], rng, tops, cuts))
    cubes = cubes[:MAX_OBJECTS_PER_CHUNK]
    return WorldChunk(
        coord=coord,
        biomes=tuple(present),
        objects=tuple(
            replace(cube, color=object_color(world_seed, cube.position, cube.role, cube.tone))
            for cube in cubes
        ),
    )
