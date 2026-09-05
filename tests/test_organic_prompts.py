"""Organic subjects participate in live rotation without losing their color cues."""

from itertools import combinations_with_replacement

import pytest

from app import main


ORGANIC_FAMILIES = ("Cellular Karst", "Root Networks", "Membrane Folds")


@pytest.mark.parametrize("family", ORGANIC_FAMILIES)
def test_organic_families_are_available_to_live_rotation(family, monkeypatch):
    entries = main.load_prompt_library(main.project_root() / "prompts.json")
    organic = [entry for entry in entries if entry.family == family]
    assert len(organic) == 4
    assert len({entry.prompt for entry in organic}) == 4
    assert all(entry.settings == {} for entry in organic)
    assert all("voxel" not in entry.prompt and "blocky" not in entry.prompt
               for entry in organic)

    # Exercise the same selector used by Space and the timer, using the full
    # installed library so additions cannot be stranded outside live rotation.
    monkeypatch.setattr(main.secrets, "choice", lambda options: next(
        entry for entry in options if entry.family == family))
    selected = main.advance_prompt(entries, entries[0])
    assert selected in organic
    assert selected.prompt.startswith("corrupted 3D render, ")


def test_library_and_palette_fit_local_clip_context():
    tokenizer_path = main.project_root() / "models" / "sd_turbo" / "tokenizer"
    if not tokenizer_path.is_dir():
        pytest.skip("Local SD Turbo tokenizer is not installed")
    from transformers import CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    entries = main.load_prompt_library(main.project_root() / "prompts.json")
    hues = ("red", "amber", "yellow", "lime", "green", "cyan", "cobalt", "violet", "magenta", "rose")
    for first, second in combinations_with_replacement(hues, 2):
        texts = [main.compose_prompt(entry.prompt, f"terrain / {first}-{second}")
                 for entry in entries]
        for entry, tokens in zip(entries, tokenizer(texts, truncation=False)["input_ids"]):
            assert len(tokens) <= tokenizer.model_max_length, (
                entry.family, first, second, len(tokens))
