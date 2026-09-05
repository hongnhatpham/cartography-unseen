"""Volumetric field of solid structures and open curved forms.

There is no ground and no gravity. Space is an endless field of cubic cells; every
cell face is either an opening or a wall built from a few thick panels, and
every cell interior may carry one biome's vocabulary - drifting reefs, lattice
bars, tilted shards, monolith columns, giant overhangs. A flier moves along the
full look direction and can climb or dive as freely as it turns.

The cell shell is what makes the volume legible. A wall is a real barrier, an
opening is a real gap a cell wide. ``legibility`` measures the balance of near
surfaces and long clear headings. Connectivity comes from a 3D channel field:
ridge surfaces of value noise force the faces they touch open, so the open
space is one connected sheet network instead of a lattice of sealed rooms;
``walkable_components`` floods an 8-unit grid to check it.

Everything is a deterministic function of (cell, seed), so chunks generate
independently and in any order. A bounded 3D window follows the camera on every
axis. Interior groups thin along open channels while panels retain the sense
of enclosure. There is no ceiling, floor or altitude clamp.
"""

from __future__ import annotations

from collections import deque
from colorsys import hsv_to_rgb, rgb_to_hsv
from dataclasses import dataclass, replace
from functools import lru_cache
from math import cos, floor, radians, sin, sqrt
from random import Random
from typing import Callable, Iterable, Literal, TypeAlias

import numpy as np

from .form_meshes import form_bounds

ChunkCoord: TypeAlias = tuple[int, int, int]
Cell: TypeAlias = tuple[int, int, int]
Vec3: TypeAlias = tuple[float, float, float]
Color: TypeAlias = tuple[float, float, float]
Role: TypeAlias = Literal["panel", "mass"]

CHUNK_SIZE = 64.0
ACTIVE_CHUNK_RADIUS = 5
MAX_ACTIVE_CHUNKS = (ACTIVE_CHUNK_RADIUS * 2 + 1) ** 3
MAX_OBJECTS_PER_CHUNK = 512
MAX_FORMS_PER_CHUNK = 64

# One volume cell. Chambers this wide read as a room's worth of open space at
# flight speed while still putting a surface within a couple of body lengths of
# almost every point, which is what the legibility statistic measures.
VOLUME_CELL = 32.0
CELLS_PER_CHUNK = int(CHUNK_SIZE / VOLUME_CELL)
# Initial positions sample a familiar region; this never limits flight.
SPAWN_HEIGHT_RANGE = 90.0
# The longest clear line out of a spawn pocket is often straight up or straight
# down; the flier is free to take it, but opening on a frame of ceiling reads as
# a mistake, so the spawn only considers headings inside this band.
SPAWN_PITCH_LIMIT = 30.0
# Body radius used for collision and for the volumetric probe. Wide on purpose:
# at 82 degrees of field of view a panel one unit from the eye is the whole
# frame, so the body has to hold the camera a couple of metres off every
# surface for the proxy to keep showing a space rather than a texture.
WALKER_RADIUS = 2.2
SLIDE_NUDGE = 0.35

_MASK_64 = (1 << 64) - 1
_BIOME_SALT = 0xB10E5
_SPAWN_SALT = 0x5AA17
_FACE_SALT = 0xFACE05
_PANEL_SALT = 0x9A4E1
_INTERIOR_SALT = 0xE73A5
_CHANNEL_SALT = 0xC4A77E
_LEGIBLE_SALT = 0x1E6187

# Share of cell faces that are open before the channel field is added. Vertical
# travel is deliberately freer than horizontal: the point of the volume is that
# up and down are directions, not a fall.
_FACE_OPEN_SIDE = 0.44
_FACE_OPEN_UPDOWN = 0.58
_FACE_OPEN_CHANNEL = 0.52
# Channel field: ridge surfaces of 3D value noise at two scales. Where they peak
# every face opens, so the open space is a connected sheet network.
_CHANNEL_CELLS = (280.0, 150.0)
_CHANNEL_EDGE, _CHANNEL_PEAK = 0.72, 0.93

# One biome cell is a few chunks across, so a couple of hundred units of flight
# crosses a border.
BIOME_CELL = 256.0
BIOME_NAMES: tuple[str, ...] = ("reefs", "lattice", "shards", "columns", "overhangs")
BIOMES_PER_WORLD = 3
MIN_BIOME_WEIGHT = 0.06
# Share of cells left empty, which is what keeps the chambers flyable.
_EMPTY_CELL_SHARE = 0.30

_MIN_TONE = 0.10
# Panel tone checker: neighbouring walls alternate light and dark, so a frame
# filled by the shell still carries a value break instead of one flat band.
_TONE_LIGHT = 0.94
_TONE_DARK = 0.42
_HUE_NAMES = (
    (0.04, "red"), (0.10, "amber"), (0.18, "yellow"), (0.29, "lime"), (0.42, "green"),
    (0.53, "cyan"), (0.66, "cobalt"), (0.79, "violet"), (0.92, "magenta"), (0.98, "rose"),
    (1.01, "red"),
)


@dataclass(frozen=True, slots=True)
class WorldCube:
    """Renderer-neutral cube using the renderer's half-extent convention.

    ``tone`` scales the colour value down, which is what puts true blacks next
    to the pale panels.
    """

    role: Role
    position: Vec3
    half_extents: Vec3
    rotation: Vec3
    color: Color
    tone: float = 1.0


@dataclass(frozen=True, slots=True)
class WorldForm:
    """An additional curved solid or open frame, separate from the cell shell."""

    mesh: str
    position: Vec3
    half_extents: Vec3
    rotation: Vec3
    color: Color


@dataclass(frozen=True, slots=True)
class WorldChunk:
    coord: ChunkCoord
    biomes: tuple[str, ...]
    objects: tuple[WorldCube, ...]
    forms: tuple[WorldForm, ...] = ()


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


@lru_cache(maxsize=1 << 17)
def _lattice(world_seed: int, salt: int, ix: int, iy: int, iz: int) -> float:
    """Cached noise value at one lattice corner; neighbours reuse corners heavily."""

    return _unit_float(world_seed, salt, ix, iy, iz)


def value_noise(
    x: float, y: float, z: float, cell: float, world_seed: int, salt: int
) -> float:
    """Smooth trilinear lattice noise in [0, 1), continuous in x, y and z."""

    fx, fy, fz = x / cell, y / cell, z / cell
    ix, iy, iz = floor(fx), floor(fy), floor(fz)
    tx, ty, tz = _smoothstep(fx - ix), _smoothstep(fy - iy), _smoothstep(fz - iz)
    plane = []
    for dz in (0, 1):
        row = []
        for dy in (0, 1):
            low = _lattice(world_seed, salt, ix, iy + dy, iz + dz)
            high = _lattice(world_seed, salt, ix + 1, iy + dy, iz + dz)
            row.append(low + (high - low) * tx)
        plane.append(row[0] + (row[1] - row[0]) * ty)
    return plane[0] + (plane[1] - plane[0]) * tz


def ridge(x: float, y: float, z: float, cell: float, world_seed: int, salt: int) -> float:
    """Ridged variant of the value noise, in [0, 1], peaking on ridge surfaces."""

    return 1.0 - abs(2.0 * value_noise(x, y, z, cell, world_seed, salt) - 1.0)


@lru_cache(maxsize=1 << 15)
def channel_weight(x: float, y: float, z: float, world_seed: int) -> float:
    """1 inside an open channel, 0 in the plain cell shell, smooth in between.

    Ridge surfaces of 3D value noise at two scales. In three dimensions a ridge
    peak is a curved sheet rather than a line, so the channels are wide, they
    branch and meet, and the open space they carve is connected by construction.
    Thresholded plain noise gives isolated bubbles instead.

    Adjacent faces repeatedly sample the same cell centres. Reuse their exact
    field values, with a bounded cache so endless exploration cannot grow it.
    """

    best = 0.0
    for index, cell in enumerate(_CHANNEL_CELLS):
        value = ridge(x, y, z, cell, world_seed, _CHANNEL_SALT + index)
        best = max(best, (value - _CHANNEL_EDGE) / (_CHANNEL_PEAK - _CHANNEL_EDGE))
    return _smoothstep(min(1.0, max(0.0, best)))


def world_to_chunk(x: float, y: float, z: float) -> ChunkCoord:
    return floor(x / CHUNK_SIZE), floor(y / CHUNK_SIZE), floor(z / CHUNK_SIZE)


@lru_cache(maxsize=8)
def _chunk_offsets(radius: int) -> tuple[ChunkCoord, ...]:
    """Centre-first load order, shared by every streaming position."""
    if radius < 0:
        raise ValueError("radius must be zero or greater")
    coords = ((x, y, z) for x in range(-radius, radius + 1)
              for y in range(-radius, radius + 1) for z in range(-radius, radius + 1))
    return tuple(sorted(coords, key=lambda c: (max(map(abs, c)), sum(map(abs, c)), c)))


@lru_cache(maxsize=8)
def _chunk_window(
    center: ChunkCoord, radius: int
) -> tuple[tuple[ChunkCoord, ...], frozenset[ChunkCoord]]:
    """Reuse the immutable target window while its small batches stream in."""
    cx, cy, cz = center
    desired = tuple((cx + dx, cy + dy, cz + dz) for dx, dy, dz in _chunk_offsets(radius))
    return desired, frozenset(desired)


def active_chunk_coords(
    x: float, y: float, z: float, radius: int = ACTIVE_CHUNK_RADIUS
) -> tuple[ChunkCoord, ...]:
    """Return the finite chunk window around a world-space position."""

    return _chunk_window(world_to_chunk(x, y, z), radius)[0]


def plan_chunk_cache(
    existing: Iterable[ChunkCoord],
    x: float,
    y: float,
    z: float,
    radius: int = ACTIVE_CHUNK_RADIUS,
) -> ChunkCachePlan:
    """Plan loads and evictions without retaining generated chunks here."""

    center = world_to_chunk(x, y, z)
    desired, desired_set = _chunk_window(center, radius)
    existing_set = set(existing)
    return ChunkCachePlan(
        center=center,
        desired=desired,
        load=tuple(coord for coord in desired if coord not in existing_set),
        keep=tuple(coord for coord in desired if coord in existing_set),
        evict=tuple(sorted(existing_set - desired_set)),
    )


# --- palette ---------------------------------------------------------------


# Each region keeps a dominant surface hue and two smaller accents. The swatches
# combine the older pale/black structures with the newer wire-study colours.
PALETTE_CELL = 128.0
_PALETTE_BLEND = 24.0
_TEAL, _VIOLET, _YELLOW = (.04, .69, .68), (.32, .25, .84), (.92, .78, .12)
_ROSE, _PALE = (.87, .30, .48), (.72, .77, .82)
_COBALT, _LIME = (.10, .30, .91), (.57, .85, .10)
_REGIONAL_PALETTES: tuple[tuple[Color, Color, Color], ...] = (
    (_TEAL, _VIOLET, _YELLOW),
    (_COBALT, _PALE, _ROSE),
    (_ROSE, _VIOLET, _LIME),
    (_LIME, _TEAL, _PALE),
    (_YELLOW, _COBALT, _ROSE),
    (_VIOLET, _ROSE, _YELLOW),
)
_REGIONAL_HUES = tuple(tuple(rgb_to_hsv(*color)[0] for color in palette)
                       for palette in _REGIONAL_PALETTES)
# Horizon follows the dominant hue; the upper and lower gradient follow its
# secondary hue. The lower value stays dark even in yellow and lime regions.
_REGIONAL_ATMOSPHERES = tuple(
    (hsv_to_rgb(hues[0], .66, .46), hsv_to_rgb(hues[1], .57, .58),
     hsv_to_rgb(hues[1], .42, .11))
    for hues in _REGIONAL_HUES
)


@lru_cache(maxsize=1 << 15)
def _palette_at_cell(world_seed: int, x: int, y: int, z: int) -> int:
    return _stable_seed(world_seed, 0xC010B, x, y, z) % len(_REGIONAL_PALETTES)


def world_palette(
    world_seed: int, x: float = 0.0, y: float = 0.0, z: float = 0.0,
) -> tuple[float, float, float]:
    """Dominant, secondary and accent hues of the current spatial region."""
    cell = tuple(floor(value / PALETTE_CELL) for value in (x, y, z))
    return _REGIONAL_HUES[_palette_at_cell(world_seed, *cell)]


def atmosphere_colors(
    world_seed: int, x: float = 0.0, y: float = 0.0, z: float = 0.0,
) -> tuple[Color, Color, Color]:
    """Blend local horizon, zenith and nadir across region borders in all axes."""
    axes = []
    for value in (x, y, z):
        cell = floor(value / PALETTE_CELL)
        offset = value - cell * PALETTE_CELL
        if offset < _PALETTE_BLEND:
            weight = _smoothstep((offset + _PALETTE_BLEND) / (2 * _PALETTE_BLEND))
            axes.append(((cell - 1, 1 - weight), (cell, weight)))
        elif offset > PALETTE_CELL - _PALETTE_BLEND:
            weight = _smoothstep((offset - PALETTE_CELL + _PALETTE_BLEND) / (2 * _PALETTE_BLEND))
            axes.append(((cell, 1 - weight), (cell + 1, weight)))
        else:
            axes.append(((cell, 1.0),))
    colors = [[0.0, 0.0, 0.0] for _ in range(3)]
    for ix, wx in axes[0]:
        for iy, wy in axes[1]:
            for iz, wz in axes[2]:
                atmosphere = _REGIONAL_ATMOSPHERES[_palette_at_cell(world_seed, ix, iy, iz)]
                weight = wx * wy * wz
                for target, source in zip(colors, atmosphere):
                    for channel in range(3):
                        target[channel] += source[channel] * weight
    return tuple(tuple(color) for color in colors)


def sky_color(world_seed: int, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Color:
    """Local horizon colour, shared by the sky and distant geometry."""
    return atmosphere_colors(world_seed, x, y, z)[0]


def zenith_color(world_seed: int, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Color:
    return atmosphere_colors(world_seed, x, y, z)[1]


def nadir_color(world_seed: int, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Color:
    """Dark lower atmosphere supplies orientation without a ground plane."""
    return atmosphere_colors(world_seed, x, y, z)[2]


def _hue_distance(a: float, b: float) -> float:
    """Shortest distance between two hues on the wrap-around [0, 1) wheel."""
    diff = abs(a - b) % 1.0
    return min(diff, 1.0 - diff)


def object_color(world_seed: int, position: Vec3, role: Role, tone: float) -> Color:
    """Fixed local material colours, with smaller shared accents and dark cuts."""
    group = tuple(floor(value / PALETTE_CELL) for value in position)
    palette = _REGIONAL_PALETTES[_palette_at_cell(world_seed, *group)]
    cell = tuple(floor(value / VOLUME_CELL) for value in position)
    accent = _unit_float(world_seed, 0xC010D, *cell)
    slot = 0 if accent < .64 else 1 if accent < .89 else 2
    chosen = palette[slot]
    strong = value_noise(*position, 100.0, world_seed, 0xC010C) > .15
    scale = .6 + .4 * max(_MIN_TONE, min(1.0, tone))
    if tone < .3:
        scale *= max(_MIN_TONE, tone) / .3
    return tuple((channel if strong else neutral * .75 + channel * .25) * scale
                 for channel, neutral in zip(chosen, (.77, .80, .82)))


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


@lru_cache(maxsize=1 << 15)
def _biome_at_cell(cell_x: int, cell_y: int, cell_z: int, world_seed: int) -> int:
    roster = biome_roster(world_seed)
    return min(
        len(roster) - 1,
        int(_unit_float(world_seed, _BIOME_SALT, cell_x, cell_y, cell_z, 7) * len(roster)),
    )


def biome_weights(x: float, y: float, z: float, world_seed: int) -> dict[str, float]:
    """Blend weights over the world's biomes at one point, summing to one."""

    roster = biome_roster(world_seed)
    fx = x / BIOME_CELL - 0.5
    fy = y / (BIOME_CELL * 0.75) - 0.5
    fz = z / BIOME_CELL - 0.5
    ix, iy, iz = floor(fx), floor(fy), floor(fz)
    tx, ty, tz = _smoothstep(fx - ix), _smoothstep(fy - iy), _smoothstep(fz - iz)
    weights = dict.fromkeys(roster, 0.0)
    for offset_z, weight_z in ((0, 1.0 - tz), (1, tz)):
        for offset_y, weight_y in ((0, 1.0 - ty), (1, ty)):
            for offset_x, weight_x in ((0, 1.0 - tx), (1, tx)):
                name = roster[
                    _biome_at_cell(ix + offset_x, iy + offset_y, iz + offset_z, world_seed)
                ]
                weights[name] += weight_x * weight_y * weight_z
    return weights


def dominant_biome(x: float, y: float, z: float, world_seed: int) -> str:
    weights = biome_weights(x, y, z, world_seed)
    return max(weights, key=lambda name: (weights[name], name))


def world_label(world_seed: int, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> str:
    """Short biome and palette label for the on-screen overlay."""

    weights = biome_weights(x, y, z, world_seed)
    ranked = sorted(weights.items(), key=lambda item: (-item[1], item[0]))
    here = "+".join(name for name, weight in ranked if weight >= 0.2) or ranked[0][0]
    primary, secondary, accent = world_palette(world_seed, x, y, z)
    hues = [_hue_name(primary), _hue_name(secondary)]
    if _hue_distance(accent, secondary) > 1e-6:
        hues.append(_hue_name(accent))
    return f"{here} / {'-'.join(dict.fromkeys(hues))}"


# --- cell shell ------------------------------------------------------------


def cell_center(cell: Cell) -> Vec3:
    """World-space centre of one volume cell."""

    return (
        (cell[0] + 0.5) * VOLUME_CELL,
        (cell[1] + 0.5) * VOLUME_CELL,
        (cell[2] + 0.5) * VOLUME_CELL,
    )


def face_open(cell: Cell, axis: int, world_seed: int) -> bool:
    """Whether the face on the positive ``axis`` side of ``cell`` is an opening.

    A per-face coin flip - vertical faces open more often, so climbing and
    diving stay as available as turning - lifted wherever the channel field runs
    through, which is what strings openings into continuous routes instead of
    scattering them.
    """

    here = cell_center(cell)
    step = [0, 0, 0]
    step[axis] = 1
    there = cell_center((cell[0] + step[0], cell[1] + step[1], cell[2] + step[2]))
    channel = max(channel_weight(*here, world_seed), channel_weight(*there, world_seed))
    base = _FACE_OPEN_UPDOWN if axis == 1 else _FACE_OPEN_SIDE
    chance = min(0.97, base + _FACE_OPEN_CHANNEL * channel)
    return _unit_float(world_seed, _FACE_SALT, cell[0], cell[1], cell[2], axis) < chance


def _panel_tone(cell: Cell, axis: int, rng: Random) -> float:
    """Checker tone for one panel, on global cell parity.

    One pale band filling the frame is the sampler's strongest attractor toward
    flat interiors, so neighbouring walls alternate light and dark.
    """

    parity = (cell[0] + cell[1] + cell[2] + axis) % 2
    return (_TONE_DARK if parity else _TONE_LIGHT) * rng.uniform(0.88, 1.12)


def _face_panels(cell: Cell, axis: int, world_seed: int) -> list[WorldCube]:
    """The wall on one closed cell face: one to three thick panels.

    A single wide panel is the common case, which is what makes a wall read as a
    wall. The rest are partial and the plane itself is jittered off the cell
    boundary, so the shell stays ragged rather than architectural.
    """

    rng = Random(_stable_seed(world_seed, _PANEL_SALT, cell[0], cell[1], cell[2], axis))
    centre = list(cell_center(cell))
    centre[axis] += VOLUME_CELL * 0.5
    tangents = [index for index in range(3) if index != axis]
    count = 1 if rng.random() < 0.62 else rng.randrange(2, 4)
    coverage = 1.0 if count == 1 else 0.72
    cubes: list[WorldCube] = []
    for _ in range(count):
        half = [0.0, 0.0, 0.0]
        position = list(centre)
        half[axis] = rng.uniform(2.0, 4.4)
        position[axis] += rng.uniform(-1.0, 1.0) * VOLUME_CELL * 0.20
        for tangent in tangents:
            share = min(1.0, coverage * rng.uniform(0.72, 1.14))
            half[tangent] = VOLUME_CELL * 0.5 * share
            slack = VOLUME_CELL * 0.5 - half[tangent]
            position[tangent] += rng.uniform(-slack, slack)
        # Yaw is free on a horizontal panel (its thin axis stays vertical) and
        # small on an upright one, so a wall never turns edge-on to its cell.
        yaw = rng.uniform(0.0, 360.0) if axis == 1 else rng.uniform(-9.0, 9.0)
        cubes.append(
            WorldCube(
                "panel",
                (position[0], position[1], position[2]),
                (half[0], half[1], half[2]),
                (rng.uniform(-7.0, 7.0), yaw, rng.uniform(-7.0, 7.0)),
                (0.0, 0.0, 0.0),
                _panel_tone(cell, axis, rng),
            )
        )
    return cubes


# --- biome vocabulary ------------------------------------------------------
#
# Each generator fills one cell interior. They read only the cell centre and a
# seeded rng, so generation stays a pure function of (cell, seed) and chunks can
# be built in any order. Interiors are sparse on purpose: the shell carries the
# structure, these carry the character and the near-field detail.


def _gen_reefs(centre: Vec3, rng: Random) -> list[WorldCube]:
    """A drifting clump of small blocks, a voxel ridge torn loose."""

    anchor = tuple(value + rng.uniform(-10.0, 10.0) for value in centre)
    spread = rng.uniform(4.0, 9.0)
    return [
        WorldCube(
            "mass",
            (
                anchor[0] + rng.uniform(-spread, spread),
                anchor[1] + rng.uniform(-spread, spread),
                anchor[2] + rng.uniform(-spread, spread),
            ),
            (rng.uniform(2.2, 5.0), rng.uniform(2.2, 5.0), rng.uniform(2.2, 5.0)),
            (0.0, rng.uniform(0.0, 45.0), 0.0),
            (0.0, 0.0, 0.0),
            rng.uniform(0.45, 1.0),
        )
        for _ in range(rng.randrange(3, 6))
    ]


def _gen_lattice(centre: Vec3, rng: Random) -> list[WorldCube]:
    """Thin bars strung across the cell, a wire frame hanging in the air."""

    cubes: list[WorldCube] = []
    for _ in range(rng.randrange(2, 5)):
        axis = rng.randrange(3)
        half = [rng.uniform(0.9, 2.4), rng.uniform(0.9, 2.4), rng.uniform(0.9, 2.4)]
        half[axis] = VOLUME_CELL * rng.uniform(0.34, 0.5)
        position = [value + rng.uniform(-11.0, 11.0) for value in centre]
        position[axis] = centre[axis] + rng.uniform(-3.0, 3.0)
        cubes.append(
            WorldCube(
                "mass",
                (position[0], position[1], position[2]),
                (half[0], half[1], half[2]),
                (rng.uniform(-5.0, 5.0), rng.uniform(-5.0, 5.0), rng.uniform(-5.0, 5.0)),
                (0.0, 0.0, 0.0),
                rng.uniform(0.5, 1.0),
            )
        )
    return cubes


def _gen_shards(centre: Vec3, rng: Random) -> list[WorldCube]:
    """Steeply tilted plates hanging in the middle of the cell."""

    return [
        WorldCube(
            "mass",
            (
                centre[0] + rng.uniform(-11.0, 11.0),
                centre[1] + rng.uniform(-11.0, 11.0),
                centre[2] + rng.uniform(-11.0, 11.0),
            ),
            (rng.uniform(2.6, 7.0), rng.uniform(0.5, 1.4), rng.uniform(2.2, 5.5)),
            (rng.uniform(-46.0, 46.0), rng.uniform(0.0, 360.0), rng.uniform(-46.0, 46.0)),
            (0.0, 0.0, 0.0),
            rng.uniform(0.55, 1.0),
        )
        for _ in range(2)
    ]


def _gen_columns(centre: Vec3, rng: Random) -> list[WorldCube]:
    """Monoliths spanning the whole cell, floor to ceiling and on past it.

    Neighbouring cells of the same biome put theirs near the cell centre too, so
    a run of them reads as one shaft crossing many cells.
    """

    return [
        WorldCube(
            "mass",
            (
                centre[0] + rng.uniform(-9.0, 9.0),
                centre[1],
                centre[2] + rng.uniform(-9.0, 9.0),
            ),
            (rng.uniform(2.4, 5.5), VOLUME_CELL * 0.5, rng.uniform(2.4, 5.5)),
            (0.0, rng.uniform(0.0, 90.0), 0.0),
            (0.0, 0.0, 0.0),
            rng.uniform(0.5, 1.0),
        )
        for _ in range(1 if rng.random() < 0.7 else 2)
    ]


def _gen_overhangs(centre: Vec3, rng: Random) -> list[WorldCube]:
    """One giant flat slab, wider than the cell, jutting through it."""

    return [
        WorldCube(
            "mass",
            (
                centre[0] + rng.uniform(-8.0, 8.0),
                centre[1] + rng.uniform(-10.0, 10.0),
                centre[2] + rng.uniform(-8.0, 8.0),
            ),
            (rng.uniform(10.0, 22.0), rng.uniform(0.8, 2.4), rng.uniform(8.0, 18.0)),
            (rng.uniform(-12.0, 12.0), rng.uniform(0.0, 360.0), rng.uniform(-12.0, 12.0)),
            (0.0, 0.0, 0.0),
            rng.uniform(0.6, 1.0),
        )
    ]


_BIOME_GENERATOR: dict[str, Callable[[Vec3, Random], list[WorldCube]]] = {
    "reefs": _gen_reefs,
    "lattice": _gen_lattice,
    "shards": _gen_shards,
    "columns": _gen_columns,
    "overhangs": _gen_overhangs,
}


def _cell_interior(cell: Cell, world_seed: int) -> list[WorldCube]:
    """Fill one cell interior with the locally dominant biome's vocabulary.

    The biome is drawn from the blend weights rather than taken from the top
    one, so a border band is a real cell-by-cell mixture of two vocabularies.
    """

    rng = Random(_stable_seed(world_seed, _INTERIOR_SALT, cell[0], cell[1], cell[2]))
    if rng.random() < _EMPTY_CELL_SHARE:
        return []
    centre = cell_center(cell)
    weights = biome_weights(*centre, world_seed)
    draw = rng.random()
    total = 0.0
    name = max(weights, key=lambda key: (weights[key], key))
    for candidate in sorted(weights):
        total += weights[candidate]
        if draw <= total:
            name = candidate
            break
    return _BIOME_GENERATOR[name](centre, rng)


# --- chunk assembly --------------------------------------------------------


def chunk_cells(coord: ChunkCoord) -> list[Cell]:
    """The eight volume cells owned by a cubic chunk, at any altitude."""

    return [
        (coord[0] * CELLS_PER_CHUNK + dx, coord[1] * CELLS_PER_CHUNK + dy,
         coord[2] * CELLS_PER_CHUNK + dz)
        for dz in range(CELLS_PER_CHUNK)
        for dx in range(CELLS_PER_CHUNK)
        for dy in range(CELLS_PER_CHUNK)
    ]


def keep_interior(position: Vec3, world_seed: int) -> bool:
    """Approved C spacing: clear interior groups along the existing channels."""
    group = tuple(floor(value / 16.0) for value in position)
    removal = 0.05 + 0.60 * channel_weight(*position, world_seed)
    return _unit_float(world_seed, 0x5AACE, *group) >= removal


def _form_rotation(rotation: Vec3) -> np.ndarray:
    """Match the renderer's Rz @ Ry @ Rx transform for collision and placement."""
    x, y, z = map(radians, rotation)
    cx, cy, cz, sx, sy, sz = cos(x), cos(y), cos(z), sin(x), sin(y), sin(z)
    return np.array([
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ])


def _companion_forms(cubes: list[WorldCube], coord: ChunkCoord, seed: int) -> tuple[WorldForm, ...]:
    """Anchor B's forms beside existing structures, away from channel peaks.

    Each companion stays inside its owning cell with enough margin for the
    camera at every cell face. The original opening graph is unchanged.
    """
    forms = []
    for cube in cubes:
        parts = tuple(int(value * 8) for value in cube.position)
        pick = _unit_float(seed, 0xADD, *parts)
        share = .38 if cube.role == "mass" else .28
        if pick >= share:
            continue
        channel = channel_weight(*cube.position, seed)
        if channel >= .95 or pick >= share * (1.0 - .65 * channel):
            continue
        shape = _unit_float(seed, 0xC08BE, *parts)
        if cube.role == "panel":
            name = "weave" if shape < .36 else "rounded" if shape < .45 else "cube"
        else:
            name = ("cage" if shape < .25 else "ribs" if shape < .55 else
                    "weave" if shape < .80 else "rounded" if shape < .90 else "cube")
        if name == "cube":
            continue
        axis = int(np.argmin(cube.half_extents))
        half = np.asarray(cube.half_extents) * (.65 if name == "rounded" else .90)
        if name in ("weave", "ribs"):
            half[axis] = max(half[axis], min(v for k, v in enumerate(half) if k != axis) * .4)
            name = f"{name}{axis}"
        elif name == "cage":
            # A frame has a real opening wide enough for the camera.
            half = np.maximum(half, 5.6)
        rotation = _form_rotation(cube.rotation)
        extent = np.abs(rotation) @ half
        half *= min(1.0, (VOLUME_CELL * .5 - WALKER_RADIUS - 1.0) / float(extent.max()))
        extent = np.abs(rotation) @ half
        position = np.asarray(cube.position, dtype=float)
        direction = -1.0 if pick < share * .5 else 1.0
        position += rotation[:, axis] * direction * (cube.half_extents[axis] + half[axis] * .8)
        # Source panels lie on cell boundaries. Clamp ownership to this chunk
        # so streaming never leaves an addition stranded in its neighbour.
        owner = tuple(min(coord[k] * CELLS_PER_CHUNK + CELLS_PER_CHUNK - 1,
                          max(coord[k] * CELLS_PER_CHUNK, floor(cube.position[k] / VOLUME_CELL)))
                      for k in range(3))
        center = np.asarray(cell_center(owner))
        limit = VOLUME_CELL * .5 - WALKER_RADIUS - 1.0 - extent
        position = np.clip(position, center - limit, center + limit)
        if channel_weight(*position, seed) >= .95:
            continue
        color = object_color(seed, cube.position, cube.role, cube.tone)
        if name != "rounded":
            color = tuple(channel * .5 + tint * .5 for channel, tint in zip(color, (.68, .86, .92)))
        forms.append(WorldForm(name, tuple(position), tuple(half), cube.rotation, color))
        if len(forms) == MAX_FORMS_PER_CHUNK:
            break
    return tuple(forms)


@lru_cache(maxsize=128)
def generate_chunk(coord: ChunkCoord, world_seed: int) -> WorldChunk:
    """Generate one chunk independently of cache state or generation order.

    A chunk owns the three positive-side faces of each of its cells, so every
    face in the world is emitted exactly once. Panels come first because the
    per-chunk cap truncates the tail and the shell is what has to survive.

    A small cache shares recent spawn and collision queries. The renderer keeps
    packed arrays, so retaining entire windows of source objects here would
    add a large Python object graph to every full garbage collection.
    """

    panels: list[WorldCube] = []
    interiors: list[WorldCube] = []
    for cell in chunk_cells(coord):
        for axis in range(3):
            if not face_open(cell, axis, world_seed):
                panels.extend(_face_panels(cell, axis, world_seed))
        interiors.extend(_cell_interior(cell, world_seed))
    cubes = [cube for cube in (panels + interiors)[:MAX_OBJECTS_PER_CHUNK]
             if cube.role == "panel" or keep_interior(cube.position, world_seed)]

    weights = biome_weights(
        *((axis + 0.5) * CHUNK_SIZE for axis in coord), world_seed
    )
    present = tuple(
        sorted(
            (name for name, weight in weights.items() if weight >= MIN_BIOME_WEIGHT),
            key=lambda name: (-weights[name], name),
        )
    )
    return WorldChunk(
        coord=coord,
        biomes=present,
        objects=tuple(
            replace(cube, color=object_color(world_seed, cube.position, cube.role, cube.tone))
            for cube in cubes
        ),
        forms=_companion_forms(cubes, coord, world_seed),
    )


# --- collision -------------------------------------------------------------


def chunk_colliders(chunk: WorldChunk) -> np.ndarray:
    """Source cubes as yaw-aligned boxes, plus each added tube segment.

    Rows are (cx, cy, cz, cos_yaw, sin_yaw, half_x, half_y, half_z). Tilt is
    folded into the vertical half extent rather than modelled, so a tilted plate
    is a slightly taller box and never a surface the flier slips through.
    """

    rows = [
        (
            cube.position[0],
            cube.position[1],
            cube.position[2],
            cos(radians(cube.rotation[1])),
            sin(radians(cube.rotation[1])),
            cube.half_extents[0],
            cube.half_extents[1]
            + cube.half_extents[0] * abs(sin(radians(cube.rotation[2])))
            + cube.half_extents[2] * abs(sin(radians(cube.rotation[0]))),
            cube.half_extents[2],
        )
        for cube in chunk.objects
    ]
    cubes = np.asarray(rows, dtype=np.float64).reshape(-1, 8)
    if not chunk.forms:
        return cubes
    return np.concatenate([cubes, *(form_colliders(form) for form in chunk.forms)])


def form_colliders(form: WorldForm) -> np.ndarray:
    """Conservative bounds of individual bars, preserving the holes between them."""
    bounds = form_bounds(form.mesh)
    centers = (bounds[:, :3] + bounds[:, 3:]) * .5
    halves = (bounds[:, 3:] - bounds[:, :3]) * .5
    linear = _form_rotation(form.rotation) * np.asarray(form.half_extents)[None, :]
    rows = np.zeros((len(bounds), 8), dtype=np.float64)
    rows[:, :3] = centers @ linear.T + np.asarray(form.position)
    rows[:, 3] = 1.0
    rows[:, 5:] = halves @ np.abs(linear).T
    return rows


def _local_offsets(
    x: float, y: float, z: float, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Offsets from a point to each box centre, in each box's own frame."""

    dx, dz = x - rows[:, 0], z - rows[:, 2]
    cosine, sine = rows[:, 3], rows[:, 4]
    return dx * cosine - dz * sine, y - rows[:, 1], dx * sine + dz * cosine


def is_blocked(
    x: float, y: float, z: float, colliders: np.ndarray, radius: float = WALKER_RADIUS
) -> bool:
    """Whether a body of ``radius`` at (x, y, z) overlaps any form."""

    if colliders.shape[0] == 0:
        return False
    local_x, local_y, local_z = _local_offsets(x, y, z, colliders)
    return bool(
        np.any(
            (np.abs(local_x) < colliders[:, 5] + radius)
            & (np.abs(local_y) < colliders[:, 6] + radius)
            & (np.abs(local_z) < colliders[:, 7] + radius)
        )
    )


def resolve_collisions(
    x: float, y: float, z: float, colliders: np.ndarray, radius: float = WALKER_RADIUS
) -> Vec3:
    """Push (x, y, z) out of every box it overlaps, sliding along the surface.

    Each overlap is resolved along its shallowest of the three axes in the box's
    own frame, plus a nudge across the face toward the nearer edge, so head-on
    contact slides instead of pinning. The loop runs until nothing overlaps, so
    a body wedged between two forms settles rather than bouncing between them.
    """

    if colliders.shape[0] == 0:
        return x, y, z
    for _ in range(6):
        local_x, local_y, local_z = _local_offsets(x, y, z, colliders)
        pen = (
            colliders[:, 5] + radius - np.abs(local_x),
            colliders[:, 6] + radius - np.abs(local_y),
            colliders[:, 7] + radius - np.abs(local_z),
        )
        inside = (pen[0] > 0.0) & (pen[1] > 0.0) & (pen[2] > 0.0)
        if not inside.any():
            break
        shallow = np.minimum(np.minimum(pen[0], pen[1]), pen[2])
        index = int(np.argmax(np.where(inside, shallow, -np.inf)))
        offsets = (local_x[index], local_y[index], local_z[index])
        depths = (pen[0][index], pen[1][index], pen[2][index])
        halves = (colliders[index, 5], colliders[index, 6], colliders[index, 7])
        axis = int(np.argmin(depths))
        push = [
            SLIDE_NUDGE * depths[axis] * offsets[other] / (halves[other] + radius)
            for other in range(3)
        ]
        push[axis] = float(
            np.copysign(depths[axis], offsets[axis] if offsets[axis] else 1.0)
        )
        cosine, sine = colliders[index, 3], colliders[index, 4]
        x += push[0] * cosine + push[2] * sine
        y += push[1]
        z += -push[0] * sine + push[2] * cosine
    return x, y, z


@lru_cache(maxsize=512)
def _chunk_collider_rows(coord: ChunkCoord, world_seed: int) -> np.ndarray:
    return chunk_colliders(generate_chunk(coord, world_seed))


@lru_cache(maxsize=64)
def _neighbourhood_colliders(coord: ChunkCoord, world_seed: int) -> np.ndarray:
    """Boxes of one chunk and its 26 neighbours, for a local query."""

    return np.concatenate(
        [
            _chunk_collider_rows((coord[0] + dx, coord[1] + dy, coord[2] + dz), world_seed)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
        ]
    )


# --- heading search --------------------------------------------------------

# Directions the open-heading search tries: a ring of yaws at several pitches,
# plus straight up and straight down.
_HEADING_YAWS = 12
_HEADING_PITCHES = (-60.0, -30.0, 0.0, 30.0, 60.0, 88.0, -88.0)
SPAWN_PITCHES = (-30.0, -15.0, 0.0, 15.0, 30.0)
# Short enough that a line grazing a corner a metre ahead is rejected: the body
# is 2.2 wide, so a coarser march picks headings the flier immediately slides
# off.
_HEADING_STEP = 1.25
_HEADING_RANGE = 70.0


def direction_of(yaw: float, pitch: float) -> Vec3:
    """Unit look direction for a (yaw, pitch) pair, matching ``Camera.forward``."""

    yaw_r, pitch_r = radians(yaw), radians(pitch)
    return (sin(yaw_r) * cos(pitch_r), sin(pitch_r), -cos(yaw_r) * cos(pitch_r))


def _rows_near(origin: Vec3, colliders: np.ndarray, reach: float) -> np.ndarray:
    """Boxes whose centre is within ``reach`` of a point, on every axis.

    A ray march only ever touches this neighbourhood, and cutting a few thousand
    chunk boxes down to a couple of hundred is what makes the vectorised march
    cheap enough to run per spawn candidate.
    """

    if colliders.shape[0] == 0:
        return colliders
    keep = (
        (np.abs(colliders[:, 0] - origin[0]) < reach)
        & (np.abs(colliders[:, 1] - origin[1]) < reach)
        & (np.abs(colliders[:, 2] - origin[2]) < reach)
    )
    return colliders[keep]


def clear_distances(
    origin: Vec3,
    directions: tuple[Vec3, ...],
    colliders: np.ndarray,
    reach: float = _HEADING_RANGE,
    step: float = _HEADING_STEP,
) -> np.ndarray:
    """Distance to the first occupied march sample along each direction.

    Intersect rays with the expanded boxes, then snap each interval to the same
    sample grid as the flight probe. This avoids allocating one matrix for
    every sample along every ray when open frames add many small colliders.
    """
    count = len(directions)
    steps = max(1, int(reach / step))
    rows = _rows_near(origin, colliders, reach + 32.0)
    if rows.shape[0] == 0:
        return np.full(count, reach)
    directions_array = np.asarray(directions)
    local_origin = _local_offsets(*origin, rows)
    local_direction = (
        directions_array[:, 0, None] * rows[:, 3] - directions_array[:, 2, None] * rows[:, 4],
        np.broadcast_to(directions_array[:, 1, None], (count, len(rows))),
        directions_array[:, 0, None] * rows[:, 4] + directions_array[:, 2, None] * rows[:, 3],
    )
    entry = np.full((count, len(rows)), -np.inf)
    leave = np.full_like(entry, np.inf)
    for axis, velocity in enumerate(local_direction):
        half = rows[:, 5 + axis] + WALKER_RADIUS
        offset = local_origin[axis]
        moving = np.abs(velocity) > 1e-12
        low = np.full_like(entry, -np.inf)
        high = np.full_like(entry, np.inf)
        np.divide(-half - offset, velocity, out=low, where=moving)
        np.divide(half - offset, velocity, out=high, where=moving)
        axis_entry, axis_leave = np.minimum(low, high), np.maximum(low, high)
        parallel_outside = ~moving & (np.abs(offset) >= half)
        axis_entry[parallel_outside] = np.inf
        axis_leave[parallel_outside] = -np.inf
        np.maximum(entry, axis_entry, out=entry)
        np.minimum(leave, axis_leave, out=leave)
    first_sample = np.maximum(1., np.floor(entry / step) + 1.)
    hit = (first_sample <= steps) & (first_sample * step < leave)
    return np.where(hit, first_sample - 1., steps).min(axis=1) * step


def open_heading(
    x: float,
    y: float,
    z: float,
    colliders: np.ndarray,
    candidates: int = _HEADING_YAWS,
    pitches: tuple[float, ...] = _HEADING_PITCHES,
) -> tuple[float, float, float]:
    """The (yaw, pitch) with the longest clear line from a point, and its length.

    Used by the spawn and by the autopilot when it runs into something; a spawn
    facing a wall would otherwise pin the flier against it. ``pitches`` narrows
    the search: the spawn keeps its gaze near level, the autopilot may climb or
    dive straight out of a pocket.
    """

    x, y, z = resolve_collisions(x, y, z, colliders)
    poses = [
        (index * 360.0 / candidates, pitch)
        for index in range(candidates)
        for pitch in pitches
    ]
    distances = clear_distances(
        (x, y, z), tuple(direction_of(*pose) for pose in poses), colliders
    )
    best = int(np.argmax(distances))
    return poses[best][0], poses[best][1], float(distances[best])


def spawn_pose(world_seed: int) -> tuple[Vec3, float, float]:
    """Float in an open pocket of the volume, facing the longest clear line.

    Candidates are scored by how legible the pocket is - a long run to fly down
    plus something solid close by to read scale against. Empty pockets score
    badly on purpose: the first frame has to show walls, not blank fog.
    """

    scattered = [
        (
            (_unit_float(world_seed, _SPAWN_SALT, index, 1) - 0.5) * 512.0,
            (_unit_float(world_seed, _SPAWN_SALT, index, 2) - 0.5) * 2.0 * SPAWN_HEIGHT_RANGE,
            (_unit_float(world_seed, _SPAWN_SALT, index, 3) - 0.5) * 512.0,
        )
        for index in range(12)
    ]
    # Rank on the channel field first, which costs a few noise lookups, and only
    # build chunks for the best few: the full probe needs nine chunks per
    # candidate and a world reset already pays for one window fill.
    scattered.sort(key=lambda point: -channel_weight(*point, world_seed))
    best_score = -1e9
    best: tuple[Vec3, float, float] = (scattered[0], 0.0, 0.0)
    for x, y, z in scattered[:4]:
        colliders = _neighbourhood_colliders(world_to_chunk(x, y, z), world_seed)
        x, y, z = resolve_collisions(x, y, z, colliders)
        if is_blocked(x, y, z, colliders):
            continue
        yaw, pitch, distance = open_heading(x, y, z, colliders, pitches=SPAWN_PITCHES)
        if distance < 24.0:
            continue
        sideways = clear_distances(
            (x, y, z),
            tuple(direction_of(yaw + 90.0, angle) for angle in (-40.0, 0.0, 40.0)),
            colliders,
        )
        # Reward a long run ahead and a nearby surface off to the side.
        score = min(distance, 60.0) - 0.9 * min(float(sideways.min()), 40.0)
        if score > best_score:
            best_score = score
            best = ((x, y, z), yaw, pitch)
    return best


# --- volumetric probe ------------------------------------------------------

PROBE_CELL = 8.0
PROBE_RADIUS = 200.0
# The 26 compass directions the legibility statistic samples.
_COMPASS: tuple[Vec3, ...] = tuple(
    (
        dx / sqrt(dx * dx + dy * dy + dz * dz),
        dy / sqrt(dx * dx + dy * dy + dz * dz),
        dz / sqrt(dx * dx + dy * dy + dz * dz),
    )
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
)
# A direction counts as open when it runs this far clear, and as blocked when it
# meets a form this close. Two of each is what makes a pocket legible.
LEGIBLE_OPEN = 30.0
LEGIBLE_NEAR = 8.0
LEGIBLE_SAMPLES = 200


@dataclass(frozen=True, slots=True)
class Occupancy:
    """A boolean grid of the volume: True where a body would not fit."""

    cell: float
    origin: Vec3
    solid: np.ndarray

    def index_of(self, point: Vec3) -> tuple[int, int, int]:
        """Grid index nearest a world point; may fall outside the grid."""

        return (
            int(round((point[0] - self.origin[0]) / self.cell)),
            int(round((point[1] - self.origin[1]) / self.cell)),
            int(round((point[2] - self.origin[2]) / self.cell)),
        )

    def center_of(self, index: tuple[int, int, int]) -> Vec3:
        return (
            self.origin[0] + index[0] * self.cell,
            self.origin[1] + index[1] * self.cell,
            self.origin[2] + index[2] * self.cell,
        )

    def blocked(self, index: tuple[int, int, int]) -> bool:
        """Solid, or outside the grid, which the probes treat as unknown."""

        for axis in range(3):
            if not 0 <= index[axis] < self.solid.shape[axis]:
                return True
        return bool(self.solid[index])


def occupancy_grid(
    world_seed: int,
    origin: Vec3 | None = None,
    radius: float = PROBE_RADIUS,
    cell: float = PROBE_CELL,
) -> Occupancy:
    """Rasterise every form near ``origin`` into a cubic boolean grid.

    Each box is inflated by the body radius and marked through its yaw-aware
    bounding box, so a grid cell is solid roughly when a body at its centre
    would be inside something. Rasterising once and sharing the grid is what
    makes both the flood fill and the legibility march cheap enough per seed.
    """

    if origin is None:
        origin = spawn_pose(world_seed)[0]
    span = int(radius / cell)
    size = 2 * span + 1
    base = (origin[0] - span * cell, origin[1] - span * cell, origin[2] - span * cell)
    solid = np.zeros((size, size, size), dtype=bool)

    reach = int(radius / CHUNK_SIZE) + 1
    home = world_to_chunk(*origin)
    rows = np.concatenate(
        [
            _chunk_collider_rows((home[0] + dx, home[1] + dy, home[2] + dz), world_seed)
            for dx in range(-reach, reach + 1)
            for dy in range(-reach, reach + 1)
            for dz in range(-reach, reach + 1)
        ]
    )
    if rows.shape[0] == 0:
        return Occupancy(cell, base, solid)
    cosine, sine = np.abs(rows[:, 3]), np.abs(rows[:, 4])
    extent = np.stack(
        [
            rows[:, 5] * cosine + rows[:, 7] * sine + WALKER_RADIUS,
            rows[:, 6] + WALKER_RADIUS,
            rows[:, 5] * sine + rows[:, 7] * cosine + WALKER_RADIUS,
        ],
        axis=1,
    )
    anchor = np.asarray(base)
    low = np.ceil((rows[:, 0:3] - extent - anchor) / cell).astype(np.int64)
    high = np.floor((rows[:, 0:3] + extent - anchor) / cell).astype(np.int64)
    np.clip(low, 0, size, out=low)
    np.clip(high, -1, size - 1, out=high)
    for index in range(rows.shape[0]):
        solid[
            low[index, 0] : high[index, 0] + 1,
            low[index, 1] : high[index, 1] + 1,
            low[index, 2] : high[index, 2] + 1,
        ] = True
    return Occupancy(cell, base, solid)


@dataclass(frozen=True, slots=True)
class Walkability:
    """Flood-fill survey of the volume on an 8-unit grid.

    ``open_cells`` counts cells a body fits in and ``reachable`` the subset
    connected to the centre through face-adjacent open cells. ``reached`` is
    that subset as a mask over the grid, which is what the map slices draw.
    """

    cell: float
    open_cells: int
    reachable: int
    components: int
    reached: np.ndarray

    @property
    def reachable_fraction(self) -> float:
        """Share of the open volume connected to the spawn."""

        return self.reachable / max(1, self.open_cells)


def walkable_components(
    world_seed: int,
    origin: Vec3 | None = None,
    radius: float = PROBE_RADIUS,
    cell: float = PROBE_CELL,
    grid: Occupancy | None = None,
) -> Walkability:
    """Survey how much of the volume a flier can reach from the spawn.

    Six-connected flood fill over the open cells of ``occupancy_grid``, started
    at the open cell nearest the centre. Diagonal moves are ignored: a route
    that exists only on a diagonal needs the flier to thread a corner exactly,
    which is not a route.
    """

    if grid is None:
        grid = occupancy_grid(world_seed, origin, radius, cell)
    size = grid.solid.shape[0]
    flat = grid.solid.reshape(-1)
    strides = (size * size, size, 1)
    seen = np.zeros(flat.shape, dtype=bool)
    open_count = int((~flat).sum())
    if open_count == 0:
        return Walkability(cell, 0, 0, 0, np.zeros_like(grid.solid))

    def flood(start: int) -> int:
        queue = deque([start])
        seen[start] = True
        filled = 0
        while queue:
            current = queue.popleft()
            filled += 1
            first, remainder = divmod(current, strides[0])
            second, third = divmod(remainder, size)
            for axis, position in enumerate((first, second, third)):
                for step in (-1, 1):
                    if not 0 <= position + step < size:
                        continue
                    other = current + step * strides[axis]
                    if not flat[other] and not seen[other]:
                        seen[other] = True
                        queue.append(other)
        return filled

    centre = size // 2
    start = int(np.ravel_multi_index((centre, centre, centre), grid.solid.shape))
    if flat[start]:
        free = np.flatnonzero(~flat)
        start = int(free[np.argmin(np.abs(free - start))])
    reachable = flood(start)
    reached = seen.copy().reshape(grid.solid.shape)

    components = 1
    for index in np.flatnonzero(~flat):
        if not seen[index]:
            components += 1
            flood(int(index))
    return Walkability(cell, open_count, reachable, components, reached)


def legibility(
    world_seed: int,
    origin: Vec3 | None = None,
    radius: float = PROBE_RADIUS,
    cell: float = PROBE_CELL,
    grid: Occupancy | None = None,
    samples: int = LEGIBLE_SAMPLES,
) -> float:
    """Share of open pockets that read as a place with walls and a way out.

    From each sampled open cell the 26 compass directions are marched through
    the occupancy grid. The pocket is legible when at least two run clear for
    ``LEGIBLE_OPEN`` units - somewhere to go - and at least two meet a form
    within ``LEGIBLE_NEAR`` - something to judge that opening against. Scattered
    debris fails the second test; a sealed cellular foam fails the first.
    """

    if grid is None:
        grid = occupancy_grid(world_seed, origin, radius, cell)
    # Keep samples clear of the grid edge so a march never runs out of grid
    # before it runs out of range.
    margin = int(LEGIBLE_OPEN / cell) + 1
    interior = grid.solid[margin:-margin, margin:-margin, margin:-margin]
    free = np.argwhere(~interior) + margin
    if free.shape[0] == 0:
        return 0.0
    rng = Random(_stable_seed(world_seed, _LEGIBLE_SALT))
    step = cell * 0.5
    legible = 0
    for _ in range(samples):
        pick = free[rng.randrange(free.shape[0])]
        point = grid.center_of((int(pick[0]), int(pick[1]), int(pick[2])))
        far_open = 0
        near_blocked = 0
        for direction in _COMPASS:
            travelled = 0.0
            while travelled < LEGIBLE_OPEN:
                travelled += step
                probe = (
                    point[0] + direction[0] * travelled,
                    point[1] + direction[1] * travelled,
                    point[2] + direction[2] * travelled,
                )
                if grid.blocked(grid.index_of(probe)):
                    if travelled <= LEGIBLE_NEAR:
                        near_blocked += 1
                    break
            else:
                far_open += 1
        if far_open >= 2 and near_blocked >= 2:
            legible += 1
    return legible / samples


# --- autopilot -------------------------------------------------------------


class Autowalk:
    """Idle drift: fly at cruising pace, easing yaw and pitch, turning at walls.

    Heading and gaze both ease toward slowly re-rolled targets so the motion
    reads as someone looking around rather than a camera on rails. A blocked
    step - the flier gained under a third of it - retargets to the longest open
    direction in three dimensions, the same search the spawn uses.
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng = Random(seed)
        self._target_yaw: float | None = None
        self._target_pitch = 0.0
        self._retarget_at = 0.0
        self._turning_until = 0.0
        self._anchor: tuple[float, float, float, float] | None = None
        self._low_frames = 0

    def reset(self) -> None:
        self._target_yaw = None
        self._retarget_at = 0.0
        self._turning_until = 0.0
        self._anchor = None

    def turning(self, now: float) -> bool:
        """True while the flier hovers and turns after a blocked step."""

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
        """Advance the drift one frame; returns the new (yaw, pitch).

        ``gained`` is the distance the previous step actually moved in three
        dimensions after collision, against the ``step_length`` it asked for.
        """

        if self._target_yaw is None or now >= self._retarget_at:
            self._target_yaw = (yaw + self._rng.uniform(-70.0, 70.0)) % 360.0
            self._target_pitch = self._rng.uniform(-26.0, 26.0)
            self._retarget_at = now + self._rng.uniform(4.0, 9.0)
        # Progress is judged over a one-second window, not per frame: a flier
        # grinding along a panel still "gains" a sliver every frame.
        stuck = False
        if self.turning(now) or step_length <= 0.0:
            self._anchor = None
            self._low_frames = 0
            # Turning has an end: once the heading has swung round, fly on.
            if abs((self._target_yaw - yaw + 180.0) % 360.0 - 180.0) < 4.0:
                self._turning_until = 0.0
        else:
            self._low_frames = self._low_frames + 1 if gained < step_length * 0.35 else 0
            stuck = self._low_frames >= 10
            if self._anchor is None:
                self._anchor = (position[0], position[1], position[2], now)
            elif now - self._anchor[3] >= 1.0:
                moved = sqrt(
                    sum((position[axis] - self._anchor[axis]) ** 2 for axis in range(3))
                )
                stuck = stuck or moved < step_length / max(dt, 1e-6) * 0.35
                self._anchor = (position[0], position[1], position[2], now)
        if stuck:
            open_yaw, open_pitch, distance = open_heading(
                position[0], position[1], position[2], colliders
            )
            self._low_frames = 0
            if distance <= 4.0:
                # Boxed in on every heading: turn well away and try again.
                open_yaw = (yaw + self._rng.uniform(120.0, 240.0)) % 360.0
                open_pitch = self._rng.choice((-60.0, 60.0))
            elif distance < 25.0 and abs((open_yaw - yaw + 180.0) % 360.0 - 180.0) < 15.0:
                # A pocket: every heading is short and the best one is the one
                # just tried. Leave along a different edge instead.
                open_yaw = (open_yaw + self._rng.choice((-90.0, 90.0, 180.0))) % 360.0
            self._target_yaw = open_yaw
            self._target_pitch = max(-80.0, min(80.0, open_pitch))
            self._retarget_at = now + self._rng.uniform(3.0, 6.0)
            # Hover and turn for a moment: pushing into the wall while the
            # heading eases round is what made the drift jitter in place.
            self._turning_until = now + 1.2
        # Ease toward the targets along the shortest arc so a turn never spins.
        delta = (self._target_yaw - yaw + 180.0) % 360.0 - 180.0
        rate = 2.6 if self.turning(now) else 1.4
        new_yaw = (yaw + delta * min(1.0, rate * dt)) % 360.0
        new_pitch = pitch + (self._target_pitch - pitch) * min(1.0, rate * 0.7 * dt)
        return new_yaw, max(-88.0, min(88.0, new_pitch))
