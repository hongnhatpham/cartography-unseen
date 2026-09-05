"""Guards on the shipped config/prompt files and the settings plumbing between
app.config, the worker and the latent-walk backend."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from app.config import BACKEND_SETTING_KEYS, RESOLUTION_MODES, AppConfig
from app.diffusion.latent_walk import SETTING_KEYS, normalise_settings
from app.main import compose_prompt, load_prompt_library

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


def test_shipped_library_is_abstract_and_varied() -> None:
    """Round 7: abstract vocabulary Hong asked for, nothing that can land concrete."""
    entries = load_prompt_library(ROOT / "prompts.json")
    families = {entry.family: entry for entry in entries}

    assert 6 <= len(families) <= 12
    assert len({entry.prompt for entry in entries}) == len(entries)
    assert all(len([e for e in entries if e.family == name]) >= 3 for name in families)
    joined = " ".join(entry.prompt for entry in entries).lower()
    # The vocabulary Hong named: wires, abstraction, corruption, data, mangled,
    # space and time, warped, lattice, biophilic cityscape.
    for term in ("wire", "abstraction", "corrupted", "data", "mangled", "space and time", "warped", "lattice", "biophilic"):
        assert term in joined, term
    # Every entry is anchored as an outdoor eye-level landscape so the sampler
    # never resolves a furnished interior, and material nouns that summon
    # product shots (chrome, glass, foil) stay out.
    assert all("eye level view inside a vast outdoor" in entry.prompt for entry in entries)
    for term in ("chrome", "glass", "foil", "honeycomb", "coral", "neon", "glitch art", "screenshot"):
        assert term not in joined, term


def test_every_family_ships_its_own_sampler_regime() -> None:
    """A family only changes the look if it changes the settings with it."""
    entries = load_prompt_library(ROOT / "prompts.json")
    families = {entry.family: entry.settings for entry in entries}
    required = {"timestep_min", "timestep_max", "guide_strength", "guidance_scale", "instability"}

    for name, settings in families.items():
        assert required <= set(settings), name
        assert set(settings) <= set(BACKEND_SETTING_KEYS), name
        AppConfig(**settings).validate()
        # The band Hong liked in round 5: below about 600 SD-Turbo redraws the
        # proxy as blocks, above about 780 the memory lets go and drifts.
        assert 600 <= settings["timestep_min"] < settings["timestep_max"] <= 780, name
        assert 0.6 <= settings["guide_strength"] <= 0.8, name


def test_negative_prompt_names_the_concrete_attractors() -> None:
    """Gamepads, desks, interiors and furniture are the residues Hong ruled out.

    People are deliberately absent: an occasional human residue is welcome.
    """
    negative = AppConfig.load(ROOT / "config.json").negative_prompt.lower()
    for term in ("gamepad", "game controller", "desk", "interior", "room", "furniture", "bed", "text", "ui"):
        assert term in negative, term
    for term in ("person", "figure"):
        assert term not in negative, term


def test_shipped_prompts_fit_the_clip_context() -> None:
    """CLIP truncates at 77 tokens and drops the tail of a longer prompt silently."""
    config = AppConfig.load(ROOT / "config.json")
    texts = [entry.prompt for entry in load_prompt_library(ROOT / "prompts.json")]
    texts += [config.prompt, config.default_prompt, config.negative_prompt]
    # The world's hue words are appended at runtime, so the budget has to hold
    # for the longest suffix compose_prompt can produce.
    texts = [compose_prompt(text, "shards+voxels / magenta-yellow-cyan") for text in texts]
    # One CLIP token is at most one word or punctuation mark, plus start and end.
    for text in texts:
        assert len(text.replace(",", " , ").split()) + 2 <= 77
