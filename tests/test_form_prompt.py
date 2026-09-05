"""Keep the operator's subject and settings when composing local color cues."""

from dataclasses import asdict

import pytest

from app.config import AppConfig
from app.main import compose_prompt, hue_words


FORMS = "curved wire lattice sheets, open ribbed arches, cable nets"


@pytest.mark.parametrize("label", ["shards+voxels", "shards / amber", "voxels / cyan-lime"])
def test_palette_cues_preserve_original_prompt_and_settings(label):
    config = AppConfig()
    original = config.prompt
    settings = asdict(config)

    effective = compose_prompt(original, label)

    assert effective.startswith(original)
    assert effective.endswith(hue_words(label) or original)
    assert compose_prompt(effective, label) == effective
    assert asdict(config) == settings


def test_existing_form_suffix_is_not_duplicated():
    original = f"my existing cityscape, {FORMS.upper()}"
    effective = compose_prompt(original, "shards / amber")

    assert effective == f"{original}, amber"
    assert effective.casefold().count(FORMS) == 1
