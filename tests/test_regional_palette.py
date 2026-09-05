"""Travel changes local materials and atmosphere while leaving geometry cached."""
from colorsys import rgb_to_hsv

import numpy as np
import pytest

from app.renderer.world import (
    PALETTE_CELL, atmosphere_colors, object_color, world_label, world_palette,
)


@pytest.mark.parametrize("axis", range(3))
def test_local_palette_changes_along_every_flight_axis(axis):
    palettes, labels, horizons = set(), set(), set()
    for distance in range(-768, 769, 128):
        position = [64., 64., 64.]
        position[axis] += distance
        palettes.add(world_palette(934943880, *position))
        labels.add(world_label(934943880, *position).split(" / ")[1])
        horizons.add(atmosphere_colors(934943880, *position)[0])
    assert len(palettes) >= 4
    assert len(labels) >= 4
    assert len(horizons) >= 4


def test_local_surfaces_and_horizon_share_the_palette():
    for region in range(-6, 7):
        center = (region * PALETTE_CELL + 64., 64., 64.)
        hues = world_palette(934943880, *center)
        horizon = atmosphere_colors(934943880, *center)[0]
        assert rgb_to_hsv(*horizon)[0] == pytest.approx(hues[0])
        for x in (16., 48., 80., 112.):
            position = (region * PALETTE_CELL + x, 48., 48.)
            material_hue = rgb_to_hsv(*object_color(934943880, position, "mass", 1.))[0]
            assert min(abs(material_hue - hue) for hue in hues) < .01


@pytest.mark.parametrize("axis", range(3))
def test_atmosphere_is_continuous_through_positive_negative_and_remote_borders(axis):
    for border in (-1_000_000_000., -256., -128., 0., 128., 1_000_000_000.):
        position = np.array([64., 64., 64.])
        position[axis] = border - .001
        before = np.array(atmosphere_colors(934943880, *position))
        position[axis] = border + .001
        after = np.array(atmosphere_colors(934943880, *position))
        assert np.max(np.abs(after - before)) < .0001
        assert np.isfinite(after).all()
        assert after.min() >= 0 and after.max() <= 1
        assert after[2].max() < .2


def test_palette_is_repeatable_and_depends_on_world_seed():
    positions = [(x + 64., 64., 64.) for x in range(-768, 769, 128)]
    first = [world_palette(7, *point) for point in positions]
    assert first == [world_palette(7, *point) for point in positions]
    assert first != [world_palette(11, *point) for point in positions]


def test_palette_label_uses_distinct_hue_names_and_keeps_the_next_accent():
    for distance in range(-768, 769, 128):
        names = world_label(934943880, distance + 64., 64., 64.).split(" / ")[1].split("-")
        assert len(names) == len(set(names))
    # The pale secondary shares cobalt's hue name, so rose becomes the next cue.
    names = world_label(934943880, -448., 64., 64.).split(" / ")[1].split("-")
    assert names == ["cobalt", "rose"]
