"""Guards on the shipped config/prompt files and the settings plumbing between
app.config, the worker and the latent-walk backend."""

from __future__ import annotations

import re
from dataclasses import fields
from pathlib import Path

from app.config import BACKEND_SETTING_KEYS, RESOLUTION_MODES, AppConfig
from app.diffusion.latent_walk import SETTING_KEYS, normalise_settings
from app.main import compose_prompt, load_master_prefix, load_prompt_library

ROOT = Path(__file__).resolve().parents[1]


def test_live_settings_survive_the_whole_path_from_config_to_backend() -> None:
    field_names = {field.name for field in fields(AppConfig)}
    assert set(BACKEND_SETTING_KEYS) <= field_names
    assert set(BACKEND_SETTING_KEYS) == set(SETTING_KEYS)
    # Every configured value must survive the backend's own clamping unchanged.
    settings = AppConfig().backend_settings()
    assert normalise_settings(settings) == settings


def test_shipped_config_loads_and_meets_the_realtime_resolution_policy() -> None:
    config = AppConfig.load(ROOT / "config.json")
    assert config.backend == "latent_walk"
    assert config.diffusion_size in RESOLUTION_MODES
    # 512x512 measured 10.1 diffusion FPS at CFG 1 on the target GPU; 768x512
    # (7.8) and 1024x768 (3.5) fall under the 8 FPS acceptance bar.
    assert config.diffusion_size in ((512, 512), (640, 384), (512, 384), (384, 256))
    assert config.steps == 1


def test_shipped_prompt_library_uses_the_reference_vocabulary() -> None:
    prefix = load_master_prefix(ROOT / "prompts.json")
    entries = load_prompt_library(ROOT / "prompts.json")
    assert prefix == "corrupted 3D render"
    assert 12 <= len(entries) <= 16
    assert len({entry["prompt"] for entry in entries}) == len(entries)
    assert all(entry["prompt"].startswith(prefix) for entry in entries)
    joined = " ".join(entry["prompt"] for entry in entries).lower()
    assert all(
        term in joined
        for term in ("neural network highway", "blocky biophilia", "corrupted", "eye level")
    )
    # "glitch art" and "screenshot of a video game" are the two tokens that made
    # the 96-frame walk resolve into lettering, controllers and circuit boards
    # once the memory latent had drifted; they stay out of the library.
    assert all(term not in joined for term in ("glitch art", "screenshot"))
    # Run 2 rendered every seed icy blue: the palette lived in the prompt as one
    # fixed phrase instead of coming from the world. The hue now arrives at
    # runtime from world_label, so no entry may hard-code a base colour.
    assert "pale grey base" not in joined
    assert "saturated burst" not in joined
    # Pale/white vocabulary in at most a third of the library; the rest carry
    # dark or two-hue vocabulary so the walk has a contrast range to travel.
    pale = [
        entry
        for entry in entries
        if re.search(r"(pale|white|washed out)", entry["prompt"].lower())
    ]
    assert len(pale) <= len(entries) // 3
    dark = [
        entry
        for entry in entries
        if any(
            word in entry["prompt"].lower()
            for word in ("hard black shadows", "deep shadow", "high contrast", "hard shadows")
        )
    ]
    assert len(dark) >= len(entries) * 2 // 3


def test_negative_prompt_lists_only_observed_failure_modes() -> None:
    """Every clause must name something a 96-frame walk actually produced.

    A long speculative negative (pattern, tidy, isometric, tilemap, grass,
    furniture) was steering the sampler on every frame for failures that never
    happened, and cost contrast doing it.
    """
    negative = AppConfig.load(ROOT / "config.json").negative_prompt.lower()
    for term in (
        "text",
        "lettering",
        "logo",
        "ui",
        "hud",
        "game controller",
        "gamepad",
        "circuit board",
        "interior",
        "room",
        "person",
        "joystick",
        "cars",
        "road markings",
        "street lights",
    ):
        assert term in negative
    for term in (
        "pattern",
        "tidy",
        "isometric",
        "tilemap",
        "grass",
        "lawn",
        "furniture",
        "houseplant",
    ):
        assert term not in negative


def test_shipped_prompts_fit_the_clip_context() -> None:
    """CLIP truncates at 77 tokens and drops the tail of a longer prompt silently."""
    config = AppConfig.load(ROOT / "config.json")
    texts = [entry["prompt"] for entry in load_prompt_library(ROOT / "prompts.json")]
    texts += [config.prompt, config.default_prompt, config.negative_prompt]
    # The world's hue words are appended at runtime, so the budget has to hold
    # for the longest suffix compose_prompt can produce.
    texts = [compose_prompt(text, "shards+voxels / magenta-yellow-cyan") for text in texts]
    # One CLIP token is at most one word or punctuation mark, plus start and end.
    for text in texts:
        assert len(text.replace(",", " , ").split()) + 2 <= 77
