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

The walker must never have to retrace. Terraces are separated by walls, so
every level change happens on a pass corridor, and the corridors are ridge
lines of two value-noise fields at different scales: ridge lines branch and
meet, so the network is connected by construction and reaches every terrace.
Ravine floors are corridors as well, un-terraced along the channel and opened
at the crossings where a corridor widens their mouth. Standing forms are then
thinned so each keeps a walkable gap from its neighbours and none stands on a
deep ravine floor. ``walkable_components`` measures the result: it floods an
8-unit grid over the streamed window exactly as the walker moves, and the tests
hold every seed above 90 per cent reachable with under 3 per cent pockets.
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
FIELD_BLOCKS_PER_CHUNK = 46
# Skyline giants: a few very tall, wide monoliths per chunk so the horizon is
# a jagged silhouette from anywhere at eye level. Without them a flat horizon
# under a pale sky reads as a lobby with a window wall.
SKYLINE_PER_CHUNK = 2

# Ground grid shared by every biome in a chunk: 8-unit cells plus a one-cell
# margin, so a column can be sized against neighbours across a chunk border.
GRID_DIVISIONS = 8
GRID_CELL = CHUNK_SIZE / GRID_DIVISIONS
GROUND_MIN_DEPTH = 16.0
# Ground checker tones, before the ravine falloff and the per-cell jitter. The
# spread is wide on purpose: a floor inside one narrow pale band reads as a
# tabletop to the sampler.
GROUND_TONE_LIGHT = 0.92
GROUND_TONE_DARK = 0.34

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
# Shortest run the grade test is applied over.
STEP_RUN = 3.0
# Radius of the ring the eye height is averaged over, so 8-unit ground cells
# on a slope read as a ramp rather than a staircase.
EYE_SMOOTH_RADIUS = 5.0
CLIMB_RATE = 14.0
# Walker body radius for collision against standing forms, and the share of a
# push that is redirected along the face so head-on contact slides.
WALKER_RADIUS = 0.8
SLIDE_NUDGE = 0.35
# Standing forms must leave a walkable gap: no form whose body spans the
# walker's eye line may come this close to another, measured between
# footprint circles and after the walker's own width, and none may stand on a
# ravine floor, where the channel is only a few cells wide.
MIN_FORM_GAP = 3.0
FORM_FREE_RAVINE = 28.0
# How far the ground around a standing form has to stay walkable for the
# form to be kept, so forms never wedge a gap shut against a riser.
FORM_CLEAR_REACH = 6.0

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
_MACRO_CELL, _MACRO_AMPLITUDE = 560.0, 132.0
_MID_CELL, _MID_AMPLITUDE = 250.0, 52.0
_FINE_CELL, _FINE_AMPLITUDE = 110.0, 6.0
_TERRACE_STEP = 24.0
_RAVINE_CELL, _RAVINE_DEPTH, _RAVINE_EDGE = 160.0, 60.0, 0.70
# How deep the cut has to get before the floor counts as a corridor, and how
# far a pass corridor widens the ravine mouth and shallows the cut where it
# crosses, which is the only way in or out of a channel.
_RAVINE_FLOOR_SHARE = 0.55
_RAVINE_MOUTH, _RAVINE_CORRIDOR_DEPTH = 0.50, 0.40
# Pass corridors: ridge lines of two value-noise fields at different scales,
# where the terraces give way to the smooth landform and the ravine cut fades,
# so a walker can change level on a slope. Everywhere else a terrace riser is a
# cliff and a wall. Ridge lines are continuous curves that branch and meet, so
# each field is a connected network and the two crossed are a mesh; an isoline band of one field gave closed rings that
# never reached each other.
_PASS_CELLS = (340.0, 220.0, 150.0)
_PASS_EDGE, _PASS_PEAK = 0.64, 0.86

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
    connected branching network rather than isolated pits. Where a pass
    corridor crosses, the mouth widens and the cut shallows, so the wall
    becomes a ramp: the floors are connected to each other by construction but
    without a mouth they are sealed off from the plateau, which was the single
    largest source of unreachable ground.
    """

    return _ravine_cut(x, z, world_seed, pass_weight(x, z, world_seed))


def _ravine_cut(x: float, z: float, world_seed: int, passing: float) -> float:
    """The ravine cut for an already-computed corridor weight.

    ``terrain_height`` needs both fields at the same point, and the corridor
    weight is three ridge lookups, so it is computed once and shared.
    """

    channel = _ridge(x, z, _RAVINE_CELL, world_seed, _TERRAIN_SALT + 4)
    edge = _RAVINE_EDGE - _RAVINE_MOUTH * passing
    t = (channel - edge) / (1.0 - edge)
    if t <= 0.0:
        return 0.0
    depth = _RAVINE_DEPTH * (1.0 - (1.0 - _RAVINE_CORRIDOR_DEPTH) * passing)
    return depth * _smoothstep(min(t, 1.0))


def pass_weight(x: float, z: float, world_seed: int) -> float:
    """1 inside a pass corridor, 0 on the terraces, smooth in between.

    The union of ridge networks at several scales, so corridors branch and
    cross instead of forming isolated bands. ``_PASS_EDGE`` sets how much of
    the world a corridor touches and ``_PASS_PEAK`` how quickly the weight
    saturates, so the corridor has a fully smooth interior with the blend
    confined to its walls. A corridor that only reached weight 1 on the ridge
    line itself was a wall along its whole length.
    """

    best = 0.0
    for index, cell in enumerate(_PASS_CELLS):
        ridge = _ridge(x, z, cell, world_seed, _TERRAIN_SALT + 7 + index)
        best = max(best, (ridge - _PASS_EDGE) / (_PASS_PEAK - _PASS_EDGE))
    return _smoothstep(min(1.0, max(0.0, best)))


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
    cut = _ravine_cut(x, z, world_seed, passing)
    value = smooth - cut
    # The ravine floor is a corridor as well: the network is connected by
    # construction, but terracing its floor would drop a riser across the
    # channel every 24 units and turn it back into a chain of pits.
    channel = _smoothstep(min(1.0, cut / (_RAVINE_DEPTH * _RAVINE_FLOOR_SHARE)))
    smoothness = max(passing, channel)
    # Soft quantiser: the top ``smoothness`` share of each terrace becomes a
    # ramp to the next one. At smoothness 0 that is a hard floor with a riser
    # at every step; at 1 the whole terrace is the ramp, so the corridor is a
    # continuous slope no steeper than the smooth field itself. Blending a
    # hard floor with the smooth field instead left a scaled-down riser at
    # every step of the corridor, which the slope test still calls a wall.
    steps = value / _TERRACE_STEP + 0.5
    level = floor(steps)
    frac = steps - level
    width = max(1e-3, smoothness)
    # The ramp sits in the middle of each tread, so the surface is identical
    # at every tread edge whatever ``smoothness`` is and the corridor never
    # rises more than half a terrace above the terrace beside it. Anchoring
    # the ramp at the top of the tread instead made the corridor a raised
    # ridge with a full 24-unit wall along both of its sides.
    ramp = min(1.0, max(0.0, (frac - 0.5 * (1.0 - width)) / width))
    return (level - 0.5 + ramp) * _TERRACE_STEP


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


def zenith_color(world_seed: int) -> Color:
    """Sky colour overhead: the secondary hue, deeper and darker than the fog.

    The gradient from pale horizon to a saturated zenith is what keeps the top
    of an eye-level frame reading as open sky rather than a white ceiling.
    """

    _, secondary, _ = world_palette(world_seed)
    _, _, sky_value = _sky_hsv(world_seed)
    return hsv_to_rgb(secondary, 0.55, max(0.30, sky_value - 0.42))


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
    # The grade limit is measured over a full step, so a per-frame move of a
    # few centimetres onto a terrace edge is not a wall the way a whole step
    # onto it would be.
    return rise > STEP_MAX or rise > MAX_SLOPE * max(run, STEP_RUN)


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


# --- connectivity probe ----------------------------------------------------

PROBE_CELL = 8.0
PROBE_RADIUS = 320.0
# The walker moves a fraction of a unit per frame, so the grade test is
# sampled at this stride rather than across a whole probe cell.
PROBE_SUBSTEP = 2.0
# Window and threshold the spawn search uses to reject pockets.
_SPAWN_PROBE_RADIUS = 112.0
_SPAWN_MIN_REACHABLE = 0.9
# Window of the form-aware spawn check; small, since it is the slow probe.
_SPAWN_FORM_PROBE_RADIUS = 48.0


@dataclass(frozen=True, slots=True)
class Walkability:
    """Flood-fill survey of an 8-unit grid over one streamed window.

    ``open_cells`` are cells a walker can stand in; ``reachable`` is the subset
    connected to the origin cell without leaving the window; ``dead_ends`` are
    open cells with exactly one walkable neighbour in any of the eight
    directions, the pockets a walker has to back out of.
    """

    cell: float
    origin: tuple[int, int]
    bounds: tuple[int, int, int, int]
    open_cells: frozenset[tuple[int, int]]
    reachable: frozenset[tuple[int, int]]
    dead_ends: frozenset[tuple[int, int]]

    @property
    def reachable_fraction(self) -> float:
        """Share of walkable cells connected to the origin."""

        return len(self.reachable) / max(1, len(self.open_cells))

    @property
    def dead_end_fraction(self) -> float:
        """Share of walkable cells with exactly one walkable neighbour."""

        return len(self.dead_ends) / max(1, len(self.open_cells))


@lru_cache(maxsize=512)
def _chunk_collider_rows(coord: ChunkCoord, world_seed: int) -> np.ndarray:
    return chunk_colliders(generate_chunk(coord, world_seed))


@lru_cache(maxsize=256)
def _neighbourhood_colliders(coord: ChunkCoord, world_seed: int) -> np.ndarray:
    """Form footprints of one chunk and its eight neighbours.

    A form near a chunk border still blocks cells on the far side, so the
    probe tests against the 3x3 neighbourhood rather than the chunk alone.
    """

    return np.concatenate(
        [
            _chunk_collider_rows((coord[0] + dx, coord[1] + dz), world_seed)
            for dx in (-1, 0, 1)
            for dz in (-1, 0, 1)
        ]
    )


def _form_blocked(x: float, y: float, z: float, colliders: np.ndarray) -> bool:
    """Whether a walker disc at (x, z, eye ``y``) overlaps any standing form."""

    if colliders.shape[0] == 0:
        return False
    rows = colliders[(colliders[:, 6] < y) & (y < colliders[:, 7])]
    if rows.shape[0] == 0:
        return False
    dx, dz = x - rows[:, 0], z - rows[:, 1]
    local_x = dx * rows[:, 2] - dz * rows[:, 3]
    local_z = dx * rows[:, 3] + dz * rows[:, 2]
    return bool(
        np.any(
            (np.abs(local_x) < rows[:, 4] + WALKER_RADIUS)
            & (np.abs(local_z) < rows[:, 5] + WALKER_RADIUS)
        )
    )


def walkable_components(
    world_seed: int,
    origin: tuple[float, float] | None = None,
    radius: float = PROBE_RADIUS,
    cell: float = PROBE_CELL,
) -> Walkability:
    """Survey which cells of a window a walker can reach from ``origin``.

    Cells are open when no standing form covers their centre; two open
    neighbours are linked when neither the riser between them is a wall
    (``step_blocked``) nor a form sits in the gap. The flood fill runs on the
    four orthogonal strides and starts at the open cell nearest ``origin`` (the
    spawn when omitted); diagonals are only checked where they decide whether a
    cell is a pocket.
    """

    if origin is None:
        position, _, _ = spawn_pose(world_seed)
        origin = (position[0], position[2])
    span = int(radius / cell)
    base_x, base_z = origin

    def centre(ix: int, iz: int) -> tuple[float, float]:
        return base_x + ix * cell, base_z + iz * cell

    heights: dict[tuple[int, int], float] = {}
    open_cells: set[tuple[int, int]] = set()
    for iz in range(-span, span + 1):
        for ix in range(-span, span + 1):
            x, z = centre(ix, iz)
            top = surface_height(x, z, world_seed)
            heights[(ix, iz)] = top
            colliders = _neighbourhood_colliders(world_to_chunk(x, z), world_seed)
            if not _form_blocked(x, top + EYE_HEIGHT, z, colliders):
                open_cells.add((ix, iz))

    def linked(a: tuple[int, int], b: tuple[int, int]) -> bool:
        """Whether a walker can stride from cell ``a`` to ``b``.

        The segment is sampled at ``PROBE_SUBSTEP`` so the grade test matches
        the short steps the walker actually takes; testing the 8-unit stride
        in one go would read every slope as a wall.
        """

        ax, az = centre(*a)
        bx, bz = centre(*b)
        steps = max(1, int(round(cell / PROBE_SUBSTEP)))
        px, pz, ph = ax, az, heights[a]
        for index in range(1, steps + 1):
            t = index / steps
            nx, nz = ax + (bx - ax) * t, az + (bz - az) * t
            nh = heights[b] if index == steps else surface_height(nx, nz, world_seed)
            run = sqrt((nx - px) ** 2 + (nz - pz) ** 2)
            if abs(nh - ph) > STEP_MAX or abs(nh - ph) > MAX_SLOPE * run:
                return False
            colliders = _neighbourhood_colliders(world_to_chunk(nx, nz), world_seed)
            if _form_blocked(nx, nh + EYE_HEIGHT, nz, colliders):
                # The walker slides round a form rather than stopping at it.
                # Accept the stride if the slid-out point is still on the way
                # to ``b`` and on walkable ground; a form sealing the whole
                # gap leaves no such point.
                sx, sz = resolve_collisions(nx, nh + EYE_HEIGHT, nz, colliders)
                if _form_blocked(sx, nh + EYE_HEIGHT, sz, colliders):
                    return False
                if (sx - px) * (bx - ax) + (sz - pz) * (bz - az) <= 0.0:
                    return False
                sh = surface_height(sx, sz, world_seed)
                if abs(sh - ph) > STEP_MAX:
                    return False
                nx, nz, nh = sx, sz, sh
            px, pz, ph = nx, nz, nh
        return True

    neighbours: dict[tuple[int, int], list[tuple[int, int]]] = {c: [] for c in open_cells}
    for a in open_cells:
        for step in ((1, 0), (0, 1)):
            b = (a[0] + step[0], a[1] + step[1])
            if b in open_cells and linked(a, b):
                neighbours[a].append(b)
                neighbours[b].append(a)

    # A cell with one orthogonal neighbour is only a pocket if it has no
    # diagonal one either: the walker turns freely, so leaving on the diagonal
    # is not retracing. Cells with two orthogonal neighbours already have a
    # way through, so only the thin tail needs the extra four strides.
    for a in (c for c in open_cells if len(neighbours[c]) < 2):
        for step in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            b = (a[0] + step[0], a[1] + step[1])
            if b in open_cells and b not in neighbours[a] and linked(a, b):
                neighbours[a].append(b)

    start = min(open_cells, key=lambda c: c[0] * c[0] + c[1] * c[1], default=None)
    reachable: set[tuple[int, int]] = set()
    if start is not None:
        stack = [start]
        reachable.add(start)
        while stack:
            current = stack.pop()
            for other in neighbours[current]:
                if other not in reachable:
                    reachable.add(other)
                    stack.append(other)
    dead_ends = frozenset(c for c in open_cells if len(neighbours[c]) == 1)
    return Walkability(
        cell=cell,
        origin=(0, 0),
        bounds=(-span, -span, span, span),
        open_cells=frozenset(open_cells),
        reachable=frozenset(reachable),
        dead_ends=dead_ends,
    )


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


def _stride_points(
    ax: float, az: float, bx: float, bz: float
) -> list[tuple[float, float]]:
    """Sample points along one probe stride, ending at (bx, bz)."""

    run = sqrt((bx - ax) ** 2 + (bz - az) ** 2)
    steps = max(1, int(round(run / PROBE_SUBSTEP)))
    return [
        (ax + (bx - ax) * index / steps, az + (bz - az) * index / steps)
        for index in range(1, steps + 1)
    ]


def _landform_open(ax: float, az: float, bx: float, bz: float, world_seed: int) -> bool:
    """Whether the landform along one probe stride stays walkable throughout.

    Sampled at ``PROBE_SUBSTEP`` so the grade test matches the short steps the
    walker takes; judging a whole probe cell in one stride reads every corridor
    ramp as a wall.
    """

    px, pz = ax, az
    ph = surface_height(ax, az, world_seed)
    for nx, nz in _stride_points(ax, az, bx, bz):
        nh = surface_height(nx, nz, world_seed)
        rise = abs(nh - ph)
        if rise > STEP_MAX or rise > MAX_SLOPE * sqrt((nx - px) ** 2 + (nz - pz) ** 2):
            return False
        px, pz, ph = nx, nz, nh
    return True


def _largest_component(neighbours: dict[tuple[int, int], list[tuple[int, int]]]) -> int:
    """Size of the biggest connected group in an adjacency map."""

    seen: set[tuple[int, int]] = set()
    best = 0
    for cell in neighbours:
        if cell in seen:
            continue
        stack, size = [cell], 0
        seen.add(cell)
        while stack:
            current = stack.pop()
            size += 1
            for other in neighbours[current]:
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        best = max(best, size)
    return best


def landform_reachable(
    x: float,
    z: float,
    world_seed: int,
    radius: float = _SPAWN_PROBE_RADIUS,
    cell: float = PROBE_CELL,
) -> float:
    """Share of the landform within ``radius`` a walker can reach from (x, z).

    Landform only: standing forms are thinned to leave gaps between them, so it
    is the terraces and ravine walls that decide whether a spot is a pocket,
    and skipping chunk generation keeps this cheap enough to run per spawn
    candidate.
    """

    span = int(radius / cell)
    cells = [
        (ix, iz) for iz in range(-span, span + 1) for ix in range(-span, span + 1)
    ]
    neighbours: dict[tuple[int, int], list[tuple[int, int]]] = {c: [] for c in cells}
    start = (0, 0)
    reachable = {start}
    stack = [start]
    linked: dict[tuple[int, int], list[tuple[int, int]]] = {}

    def edges(cell_key: tuple[int, int]) -> list[tuple[int, int]]:
        if cell_key not in linked:
            ax, az = x + cell_key[0] * cell, z + cell_key[1] * cell
            linked[cell_key] = [
                other
                for other in (
                    (cell_key[0] + dx, cell_key[1] + dz)
                    for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1))
                )
                if other in neighbours
                and _landform_open(
                    ax, az, x + other[0] * cell, z + other[1] * cell, world_seed
                )
            ]
        return linked[cell_key]

    while stack:
        current = stack.pop()
        for other in edges(current):
            if other not in reachable:
                reachable.add(other)
                stack.append(other)
    return len(reachable) / len(cells)


def spawn_pose(world_seed: int) -> tuple[Vec3, float, float]:
    """Stand on open ground inside the relief, looking out along a drop.

    Candidates are scored by how much the ring around them rises and falls, so
    the walker starts surrounded by risers and ravines rather than on a flat
    plate, and then taken best-first until one whose neighbourhood is at least
    ``_SPAWN_MIN_REACHABLE`` connected is found: a dramatic spot the walker
    cannot leave without retracing is worse than a plain one. The yaw faces
    the lowest neighbour, which puts depth in the first frame.
    """

    candidates: list[tuple[float, float, float, float]] = []
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
        # Camera forward is (sin yaw, ., -cos yaw); the ring offset is
        # (cos a, sin a) in x/z, so yaw = a + 90 points at that ring point.
        candidates.append((score, x, z, (90.0 + lowest_angle) % 360.0))

    candidates.sort(key=lambda item: -item[0])
    chosen = candidates[0] if candidates else (0.0, 0.0, 0.0, 0.0)
    for candidate in candidates:
        if landform_reachable(candidate[1], candidate[2], world_seed) < _SPAWN_MIN_REACHABLE:
            continue
        # The landform check ignores forms. Run the form-aware probe over a
        # short window too, so a spot the standing forms box in is skipped.
        survey = walkable_components(
            world_seed, (candidate[1], candidate[2]), _SPAWN_FORM_PROBE_RADIUS
        )
        if survey.reachable_fraction >= _SPAWN_MIN_REACHABLE:
            chosen = candidate
            break

    _, x, z, yaw = chosen
    # The landform check ignores forms; the chunk's standing forms may still
    # box the spot in. Slide out of them the way the walker does, then walk
    # the probe ring to the nearest cell a stride can leave from.
    colliders = _neighbourhood_colliders(world_to_chunk(x, z), world_seed)
    eye = surface_height(x, z, world_seed) + EYE_HEIGHT
    x, z = resolve_collisions(x, eye, z, colliders)
    if _form_blocked(x, eye, z, colliders) or open_heading(x, eye, z, colliders, world_seed=world_seed)[1] < 6.0:
        for ring in (8.0, 16.0, 24.0):
            for step in range(8):
                nx = x + ring * cos(radians(step * 45.0))
                nz = z + ring * sin(radians(step * 45.0))
                ne = surface_height(nx, nz, world_seed) + EYE_HEIGHT
                if not _form_blocked(nx, ne, nz, colliders) and open_heading(
                    nx, ne, nz, colliders, world_seed=world_seed
                )[1] >= 6.0:
                    x, z = nx, nz
                    break
            else:
                continue
            break
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
            # A flat pale floor plane is the sampler's strongest attractor at
            # eye level: it completes it as a desk with a game controller on
            # it, whatever the prompt says. A hard checker on global cell
            # parity gives that plane converging grid lines instead, and the
            # dark half supplies the true blacks the reference has. Parity is
            # global so the pattern runs unbroken across chunk seams.
            parity = (coord[0] * GRID_DIVISIONS + ix + coord[1] * GRID_DIVISIONS + iz) % 2
            tone = (GROUND_TONE_DARK if parity else GROUND_TONE_LIGHT) * rng.uniform(
                0.88, 1.12
            )
            # Ravine floors darken to about a third, not black: the walker
            # stands in them now, and a black frame gives the sampler nothing.
            tone = min(1.0, tone) * (1.0 - 0.62 * min(1.0, cuts[(ix, iz)] / 40.0))
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
    for _ in range(SKYLINE_PER_CHUNK):
        x = origin_x + rng.uniform(4.0, CHUNK_SIZE - 4.0)
        z = origin_z + rng.uniform(4.0, CHUNK_SIZE - 4.0)
        base = terrain_height(x, z, world_seed)
        height = rng.uniform(55.0, 110.0)
        width = rng.uniform(2.0, 4.5)
        cubes.append(
            WorldCube(
                "form",
                (x, base + height * 0.5, z),
                (width, height * 0.5, width * rng.uniform(0.6, 1.5)),
                (rng.uniform(-3.0, 3.0), rng.uniform(0.0, 90.0), rng.uniform(-3.0, 3.0)),
                (0.0, 0.0, 0.0),
                rng.uniform(0.7, 1.0),
            )
        )
    for _ in range(FIELD_BLOCKS_PER_CHUNK):
        x = origin_x + rng.uniform(0.0, CHUNK_SIZE)
        z = origin_z + rng.uniform(0.0, CHUNK_SIZE)
        base = terrain_height(x, z, world_seed)
        height = rng.choice((5.0, 9.0, 14.0, 22.0, 30.0)) * rng.uniform(0.7, 1.3)
        # Slender: a form wider than a few steps fills the frame at eye level,
        # and a narrow footprint is what lets the spacing rule keep a dense
        # field instead of thinning it down to a handful per chunk.
        width = rng.uniform(0.8, 2.0) * (1.0 + height / 90.0)
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


def _blocks_walker(cube: WorldCube, world_seed: int) -> bool:
    """Whether a form's body spans the walker's eye line on the ground below."""

    if cube.role != "form":
        return False
    eye = terrain_height(cube.position[0], cube.position[2], world_seed) + EYE_HEIGHT
    return (
        cube.position[1] - cube.half_extents[1]
        < eye
        < cube.position[1] + cube.half_extents[1]
    )


def _thin_forms(cubes: list[WorldCube], world_seed: int) -> list[WorldCube]:
    """Drop standing forms that would seal a walkable gap.

    A form whose body spans the walker's eye line is kept only when it stands
    on open ground off the ravine floors and clears every form already kept by
    ``MIN_FORM_GAP`` plus the walker's own width. Overhead slabs and giants are
    untouched, so the silhouette keeps its scale while the ground stays open.
    """

    blocks = [_blocks_walker(cube, world_seed) for cube in cubes]
    radii = [
        sqrt(cube.half_extents[0] ** 2 + cube.half_extents[2] ** 2) for cube in cubes
    ]
    # Biggest first: the giants carry the silhouette, so the field blocks are
    # the ones thinned around them rather than the other way round.
    order = sorted(
        (index for index, blocking in enumerate(blocks) if blocking),
        key=lambda index: (-radii[index], index),
    )
    keep = set()
    footprints: list[tuple[float, float, float]] = []
    clear = MIN_FORM_GAP + 2.0 * WALKER_RADIUS
    for index in order:
        cube = cubes[index]
        x, z = cube.position[0], cube.position[2]
        if ravine_depth(x, z, world_seed) > FORM_FREE_RAVINE:
            continue
        radius = radii[index]
        if any(
            (x - fx) ** 2 + (z - fz) ** 2 < (radius + fr + clear) ** 2
            for fx, fz, fr in footprints
        ):
            continue
        # A form standing against a riser wedges the gap beside it shut, which
        # is where most of the remaining pockets came from; only ground that
        # stays within a step on all four sides keeps one. This runs last
        # because it is four extra height samples and the spacing test has
        # already dropped most candidates.
        here = terrain_height(x, z, world_seed)
        if any(
            abs(
                terrain_height(
                    x + dx * FORM_CLEAR_REACH, z + dz * FORM_CLEAR_REACH, world_seed
                )
                - here
            )
            > STEP_MAX
            for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1))
        ):
            continue
        keep.add(index)
        footprints.append((x, z, radius))
    return [
        cube
        for index, cube in enumerate(cubes)
        if not blocks[index] or index in keep
    ]


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
    cubes = _thin_forms(cubes, world_seed)[:MAX_OBJECTS_PER_CHUNK]
    return WorldChunk(
        coord=coord,
        biomes=tuple(present),
        objects=tuple(
            replace(cube, color=object_color(world_seed, cube.position, cube.role, cube.tone))
            for cube in cubes
        ),
    )
