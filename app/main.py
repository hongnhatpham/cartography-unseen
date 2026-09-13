from __future__ import annotations

import argparse
import faulthandler
import json
import logging
import secrets
import sys
import textwrap
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, strftime

import numpy as np

from app.config import (
    BACKEND_SETTING_KEYS,
    MAX_FOG_DISTANCE,
    MIN_FOG_DISTANCE,
    RESOLUTION_MODES,
    AppConfig,
    configure_local_environment,
)
from app.utils.timing import ExponentialAverage, RateMeter
from app.idle_instructions import IdleInstructions
from app.screenshots import ScreenshotWriter
from app.monitoring import SnapshotPublisher

# Smoke mode also waits for the first generated frame; the model load alone is
# longer than the display-frame budget. This caps that extra wait.
SMOKE_AI_TIMEOUT_S = 150.0

CFG_LEVELS = (1.0, 1.25, 1.5, 2.0, 3.0)
NOISE_WALK_LEVELS = (0.0, 2.5, 5.0, 10.0, 20.0, 40.0)
PROMPT_WALK_LEVELS = (0.0, 3.0, 6.0, 12.0, 25.0, 50.0)
AUTO_ADVANCE_LEVELS = (0.0, 12.0, 24.0, 48.0, 96.0)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def configure_logging(root: Path, debug: bool) -> Path:
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"realtime_diffusion_{strftime('%Y%m%d_%H%M%S')}.log"
    handlers: list[logging.Handler] = [logging.FileHandler(log_path, encoding="utf-8")]
    if debug:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(threadName)s %(message)s",
        handlers=handlers,
        force=True,
    )
    # Keep native crash stacks even when the supervisor has no visible console.
    # Retain the handle: faulthandler writes directly to its file descriptor.
    global _fault_log
    _fault_log = log_path.with_suffix(".fault.log").open("a", encoding="utf-8")
    faulthandler.enable(file=_fault_log, all_threads=True)
    return log_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latent Space realtime diffusion renderer")
    parser.add_argument("--debug", action="store_true", help="windowed mode with diagnostics")
    parser.add_argument("--windowed", action="store_true", help="disable fullscreen")
    parser.add_argument("--no-map", action="store_true", help="run only the first-person window")
    parser.add_argument("--monitor", type=int, help="zero-based fullscreen monitor index")
    parser.add_argument("--list-monitors", action="store_true", help="print detected monitors and exit")
    parser.add_argument("--config", type=Path, help="alternate project-relative config file")
    parser.add_argument("--backend", choices=("latent_walk", "proxy_passthrough"))
    parser.add_argument(
        "--resolution",
        choices=tuple(f"{width}x{height}" for width, height in RESOLUTION_MODES),
        help="generation resolution/aspect mode",
    )
    parser.add_argument("--smoke-frames", type=int, default=0, help=argparse.SUPPRESS)
    return parser.parse_args()


def persist_config_values(config_path: Path, values: dict[str, object]) -> float:
    """Persist accepted runtime settings in one atomic replace."""
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw.update(values)
    temporary = config_path.with_suffix(config_path.suffix + ".tmp")
    temporary.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(config_path)
    return config_path.stat().st_mtime


def persist_config_value(config_path: Path, key: str, value: object) -> float:
    """Persist one accepted runtime setting without disturbing other config."""
    return persist_config_values(config_path, {key: value})


def persist_prompt(config_path: Path, prompt: str) -> float:
    return persist_config_value(config_path, "prompt", prompt)


def random_prompt_settings() -> dict[str, float | int]:
    """Independently combine visitor-selected ranges for each new subject."""
    return {
        "timestep_min": 80 + secrets.randbelow(121),
        "timestep_max": 280 + secrets.randbelow(121),
        "display_sharpen": (100 + secrets.randbelow(101)) / 100,
        "instability": secrets.randbelow(41) / 100,
        "guide_strength": (65 + secrets.randbelow(36)) / 100,
    }


@dataclass(frozen=True, slots=True)
class PromptEntry:
    """One variant of a style family with optional tuning metadata.

    Family settings remain available to offline comparison tools. Live prompt
    selection uses the independently sampled visitor ranges instead.
    """

    family: str
    prompt: str
    settings: dict[str, float | int]


def _family_entries(
    family: dict[str, object], index: int, master_prefix: str
) -> list[PromptEntry]:
    """Expand one family object into one entry per subject variant."""
    name = str(family.get("name", "")).strip()
    base = str(family.get("base", "")).strip()
    variants = family.get("variants")
    if not name or not base:
        raise RuntimeError(f"Prompt family {index} requires non-empty name and base")
    if not isinstance(variants, list) or not variants:
        raise RuntimeError(f"Prompt family {name!r} requires a non-empty 'variants' list")
    raw_settings = family.get("settings", {})
    if not isinstance(raw_settings, dict):
        raise RuntimeError(f"Prompt family {name!r} settings must be an object")
    unknown = sorted(set(raw_settings) - set(BACKEND_SETTING_KEYS))
    if unknown:
        raise RuntimeError(f"Prompt family {name!r} has unknown settings: {', '.join(unknown)}")
    settings = {key: value for key, value in raw_settings.items()}
    entries = []
    for variant in variants:
        subject = str(variant).strip()
        if not subject:
            raise RuntimeError(f"Prompt family {name!r} has an empty variant")
        prompt = apply_master_prefix(master_prefix, f"{subject}, {base}")
        entries.append(PromptEntry(family=name, prompt=prompt, settings=dict(settings)))
    return entries


def load_prompt_library(path: Path) -> list[PromptEntry]:
    """Read prompts.json into one entry per family variant.

    A "families" list is the current format. A legacy flat "prompts" list of
    name/prompt objects still loads, with each entry its own family and no
    settings override.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Prompt library not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid prompt library JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise RuntimeError("prompts.json must contain an object")
    master_prefix = str(raw.get("master_prefix", "")).strip()
    families = raw.get("families")
    if isinstance(families, list) and families:
        entries: list[PromptEntry] = []
        for index, family in enumerate(families, start=1):
            if not isinstance(family, dict):
                raise RuntimeError(f"Prompt family {index} must be an object")
            entries.extend(_family_entries(family, index, master_prefix))
        return entries
    flat = raw.get("prompts")
    if not isinstance(flat, list) or not flat:
        raise RuntimeError("prompts.json must contain a non-empty 'families' or 'prompts' list")
    legacy: list[PromptEntry] = []
    for index, entry in enumerate(flat, start=1):
        if not isinstance(entry, dict):
            raise RuntimeError(f"Prompt entry {index} must be an object")
        name = str(entry.get("name", "")).strip()
        prompt = str(entry.get("prompt", "")).strip()
        if not name or not prompt:
            raise RuntimeError(f"Prompt entry {index} requires non-empty name and prompt")
        legacy.append(
            PromptEntry(
                family=name, prompt=apply_master_prefix(master_prefix, prompt), settings={}
            )
        )
    return legacy


def load_master_prefix(path: Path) -> str:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Prompt library not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid prompt library JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise RuntimeError("prompts.json must contain an object")
    return str(raw.get("master_prefix", "")).strip()


def apply_master_prefix(prefix: str, prompt: str) -> str:
    """Prepend the library prefix unless the prompt already opens with it."""
    prefix = prefix.strip().rstrip(" ,")
    prompt = prompt.strip()
    if not prefix or prompt.casefold().startswith(prefix.casefold()):
        return prompt
    return f"{prefix}, {prompt}"


def world_hues(world_label: str) -> list[str]:
    """The hue names in a "biomes / hue-hue-hue" label, in order."""
    _, separator, palette = world_label.rpartition("/")
    hues = [name.strip() for name in palette.split("-")] if separator else []
    return [name for name in hues if name]


def hue_words(world_label: str, offset: int = 0) -> str:
    """A pair of the world's hues as prompt vocabulary, e.g. "violet and lime".

    ``ProxyRenderer.world_label`` reads "biomes / hue-hue-hue". Naming the local
    palette helps the model retain the proxy's colors. ``offset`` selects a
    different pair for offline comparisons.
    """
    hues = world_hues(world_label)
    if not hues:
        return ""
    if len(hues) == 1:
        return hues[0]
    start = offset % len(hues)
    return f"{hues[start]} and {hues[(start + 1) % len(hues)]}"


def compose_prompt(prompt: str, world_label: str, hue_offset: int = 0) -> str:
    """Append the local palette without changing the library's subject."""
    words = hue_words(world_label, hue_offset)
    if not words or words in prompt.casefold():
        return prompt
    return f"{prompt.rstrip().rstrip(',')}, {words}"


def choose_family_prompt(
    entries: list[PromptEntry], recent_families: str | Iterable[str]
) -> PromptEntry:
    """A variant from a family the last few presses did not use — what Space picks.

    Excluding only the current family still drew the same family three times in
    eight presses, and a repeat inside one sitting is what reads as "the reset
    did nothing".
    """
    if isinstance(recent_families, str):
        recent_families = (recent_families,)
    recent = list(recent_families)
    while recent:
        alternatives = [entry for entry in entries if entry.family not in recent]
        if alternatives:
            return secrets.choice(alternatives)
        # Small libraries can exhaust the history. Allow the oldest family
        # back first, keeping the current family excluded when possible.
        recent.pop(0)
    return secrets.choice(entries)


# How many past presses the Space family draw avoids repeating.
RECENT_FAMILY_MEMORY = 3


def advance_prompt(
    entries: list[PromptEntry], current: PromptEntry, recent_families: Iterable[str] = ()
) -> PromptEntry:
    """Choose a different subject family for both timed changes and Space."""
    alternatives = [entry for entry in entries if entry.prompt != current.prompt]
    if not alternatives:
        raise RuntimeError("Prompt library has no different prompt")
    return choose_family_prompt(alternatives, [*recent_families, current.family])


def entry_for_prompt(entries: list[PromptEntry], prompt: str) -> PromptEntry:
    """The library entry matching ``prompt``, or a settings-free custom entry."""
    for entry in entries:
        if entry.prompt == prompt:
            return entry
    return PromptEntry(family="Custom", prompt=prompt, settings={})


def next_level(levels: tuple[float, ...], current: float) -> float:
    """Step to the level after the one nearest to ``current`` (wrapping)."""
    nearest = min(range(len(levels)), key=lambda index: abs(levels[index] - current))
    return levels[(nearest + 1) % len(levels)]


def run() -> int:
    """Let UI and AI threads share Python without long kernel-submission gaps."""
    previous_interval = sys.getswitchinterval()
    sys.setswitchinterval(min(previous_interval, .001))
    try:
        return _run()
    finally:
        sys.setswitchinterval(previous_interval)


def _run() -> int:
    args = parse_args()
    root = project_root()
    configure_local_environment(root, offline=True)
    config_path = (root / args.config).resolve() if args.config else root / "config.json"
    prompt_library_path = root / "prompts.json"
    debug_requested = bool(args.debug)
    log_path = configure_logging(root, debug_requested)
    try:
        config = AppConfig.load(config_path)
        config.prompt = apply_master_prefix(
            load_master_prefix(prompt_library_path), config.prompt
        )
    except RuntimeError as exc:
        logging.critical("Configuration failed: %s", exc)
        print(exc, file=sys.stderr)
        return 2
    if args.backend:
        config.backend = args.backend
    if args.resolution:
        config.diffusion_resolution = args.resolution
    if args.monitor is not None:
        config.display_monitor = args.monitor
    if args.no_map:
        config.journey_map = False
    # An explicit map display enables unattended fullscreen startup. Legacy
    # configurations retain manual placement before F fills both projectors.
    fullscreen = config.fullscreen and not (args.debug or args.windowed or
        (config.journey_map and config.map_display_monitor is None))

    try:
        import pygame

        from app.diffusion.worker import DiffusionWorker
        from app.renderer.camera import Camera
        from app.renderer.player_trail import PlayerTrail
        from app.renderer.proxy_renderer import ProxyRenderer
        from app.renderer.world import Autowalk
    except Exception as exc:
        logging.exception("Startup dependency failure")
        print(str(exc), file=sys.stderr)
        return 3

    if args.list_monitors:
        pygame.display.init()
        try:
            for index, (width, height) in enumerate(pygame.display.get_desktop_sizes()):
                selected = " [configured]" if index == config.display_monitor else ""
                print(f"{index}: {width}x{height}{selected}")
        finally:
            pygame.display.quit()
        return 0

    renderer: ProxyRenderer | None = None
    worker: DiffusionWorker | None = None
    journey = None
    monitoring = SnapshotPublisher(root / "cache/monitoring/artwork.json", "artwork")
    monitoring.update(state="loading", displayed_frames=0)
    screenshots = ScreenshotWriter(root / "screenshot")
    try:
        renderer = ProxyRenderer(
            root,
            config.diffusion_size,
            fullscreen,
            window_size=(960, 640) if config.journey_map else config.diffusion_size,
            display_monitor=config.display_monitor,
            world_seed=config.world_seed,
            fog_distance=config.fog_distance,
            window_position=(60, 60) if config.journey_map and config.display_monitor == 0 and config.map_display_monitor is None else None,
        )
        renderer.loading_screen("INITIALIZING", "Starting renderer and diffusion worker")
        camera = Camera.create_default()
        renderer.spawn_camera(camera)
        trail = PlayerTrail()
        try:
            current_entry = entry_for_prompt(
                load_prompt_library(prompt_library_path), config.prompt
            )
        except RuntimeError as exc:
            logging.error("Prompt library ignored at startup: %s", exc)
            current_entry = PromptEntry(family="Custom", prompt=config.prompt, settings={})
        effective_prompt = compose_prompt(config.prompt, renderer.world_label())
        backend_config = config.backend_dict(root)
        backend_config["prompt"] = effective_prompt
        worker = DiffusionWorker(config.backend, backend_config)
        worker.start()
        logging.info("Effective prompt: %s", effective_prompt)

        clock = pygame.time.Clock()
        display_rate = RateMeter(120)
        proxy_rate = RateMeter(120)
        proxy_ms = ExponentialAverage(0.15)
        conditioning_ms = ExponentialAverage(0.15)
        frame_age_ms = ExponentialAverage(0.15)
        running = True
        session_started = perf_counter()
        overlay_enabled = bool(config.debug_overlay or debug_requested or
                               (config.journey_map and config.map_display_monitor is None))
        renderer.set_operator_mode(overlay_enabled)
        force_proxy = False
        diagnostic_mode = "none"
        frozen = False
        latest_ai = None
        latest_ai_version = 0
        last_config_check = 0.0
        last_prompt_advance = perf_counter()
        # Prompt selection preserves the local palette; travel changes it.
        hue_offset = 0
        sent_hues = hue_words(renderer.world_label(), hue_offset)
        last_palette_update = perf_counter()
        recent_families: list[str] = [current_entry.family]
        last_input = perf_counter()
        idle_instructions = IdleInstructions(last_input)
        autowalk = Autowalk(config.world_seed)
        autowalking = False
        config_mtime = config_path.stat().st_mtime
        displayed_frames = 0
        next_monitoring_update = 0.0
        conditioning = None
        last_conditioning_capture = 0.0
        prompt_editing = False
        prompt_caption_enabled = config.prompt_caption
        prompt_buffer = ""
        required_prompt_revision = 0
        minimum_ai_sequence = 0
        # Never expose the proxy as the startup fallback. The worker already
        # receives conditioning frames automatically, so hold the neutral
        # loading artwork until its first generated frame arrives.
        hide_proxy_until_ai = True
        notice = ""
        notice_until = 0.0
        ignore_prompt_hotkey_text = False
        seen_resolution_fallbacks = 0
        space_held = False
        if config.journey_map:
            from app.journey_session import JourneySession

            journey = JourneySession(root, config)
            journey.window.set_operator_mode(overlay_enabled)
            journey.prompt(config.prompt, 0, "initial", camera, {
                **config.backend_settings(), "submitted_prompt": effective_prompt,
                "negative_prompt": config.negative_prompt,
                "seed": config.seed, "backend": config.backend,
            }, timestamp=session_started)

        def toggle_projectors() -> None:
            """Use each window's current monitor and preserve its restore bounds."""
            nonlocal notice, notice_until
            try:
                enabled = renderer.set_fullscreen(not renderer.is_fullscreen)
                if journey is not None:
                    journey.window.set_fullscreen(enabled)
                notice = "BOTH WINDOWS FULLSCREEN - F restores placement" if enabled else "DRAG WINDOW TITLE BARS TO SCREENS - F fullscreen both"
            except RuntimeError as exc:
                notice = f"DISPLAY SETUP: {exc}"
                logging.error("Projector fullscreen failed: %s", exc)
            notice_until = perf_counter() + 5.0

        def request_quit() -> None:
            """Do not dismiss the only recoverable map when its final save fails."""
            nonlocal running, notice, notice_until, overlay_enabled
            try:
                if journey is not None:
                    journey.recorder.close()
                running = False
            except (RuntimeError, OSError) as exc:
                notice = f"MAP SAVE FAILED - still open. Free disk space, then press Escape to retry. {exc}"
                notice_until = perf_counter() + 60.0
                overlay_enabled = True
                renderer.set_operator_mode(True)
                if journey is not None:
                    journey.window.set_operator_mode(True)
                logging.exception("Exit deferred to preserve the journey")

        def apply_live(values: dict[str, object]) -> None:
            """Set settings on the config and forward the backend-tunable ones.

            Nothing is written to config.json, so ephemeral changes (auto-advance
            picking a new style family) cannot overwrite the configured startup
            values or move config_mtime past an external edit.
            """
            for key, value in values.items():
                setattr(config, key, value)
            forward = {
                key: value for key, value in values.items() if key in BACKEND_SETTING_KEYS
            }
            if forward:
                worker.request_settings(**forward)

        def commit_many(values: dict[str, object], live: bool = True) -> None:
            """Set, persist and (optionally) forward settings in one atomic write."""
            nonlocal config_mtime
            if live:
                apply_live(values)
            else:
                for key, value in values.items():
                    setattr(config, key, value)
            config_mtime = persist_config_values(config_path, values)

        def commit(key: str, value: object, live: bool = True) -> None:
            """Set, persist and (optionally) forward one runtime setting."""
            commit_many({key: value}, live)

        def send_prompt(
            prompt: str, negative: str | None = None, *,
            vary_settings: bool = False, persist: bool = False,
            trigger: str = "regional_palette",
        ) -> int:
            """Forward the original subject with the current region's colors."""
            nonlocal effective_prompt, sent_hues, last_palette_update
            values = random_prompt_settings() if vary_settings else {}
            if persist:
                commit_many({"prompt": prompt, **values}, live=False)
            else:
                for key, value in values.items():
                    setattr(config, key, value)
            effective_prompt = compose_prompt(prompt, renderer.world_label(), hue_offset)
            sent_hues = hue_words(renderer.world_label(), hue_offset)
            last_palette_update = perf_counter()
            if values:
                logging.info("Prompt variation: timestep %s-%s, sharpness %.2f, instability %.0f%%, guide %.0f%%",
                             config.timestep_min, config.timestep_max, config.display_sharpen,
                             config.instability * 100, config.guide_strength * 100)
                revision = worker.request_prompt(
                    effective_prompt, config.negative_prompt if negative is None else negative,
                    settings={key: value for key, value in values.items() if key in BACKEND_SETTING_KEYS},
                )
            else:
                revision = worker.request_prompt(
                    effective_prompt,
                    config.negative_prompt if negative is None else negative,
                )
            if journey is not None:
                journey.prompt(prompt, revision, trigger, camera, {
                    **config.backend_settings(), "submitted_prompt": effective_prompt,
                    "negative_prompt": config.negative_prompt if negative is None else negative,
                    "seed": config.seed, "backend": config.backend,
                })
            return revision

        while running:
            frame_started = perf_counter()
            quit_requested = False
            screenshot_requested = False
            screenshot_notice = screenshots.poll()
            if screenshot_notice:
                notice, notice_until = screenshot_notice, frame_started + 3.0
            input_event = False
            events = list(renderer.poll_events())
            if journey is not None:
                for message in journey.poll():
                    if message == '__map_display_ready__':
                        if not overlay_enabled:
                            renderer.focus()
                    elif message == "__toggle_fullscreen__":
                        events.append(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_f, mod=0))
                    elif message == "__toggle_overlay__":
                        events.append(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F1, mod=0))
                    else:
                        notice, notice_until = str(message), frame_started + 6.0
            for event in events:
                if event.type == pygame.KEYUP and event.key == pygame.K_SPACE:
                    space_held = False
                elif event.type == pygame.WINDOWFOCUSLOST:
                    space_held = False
                if event.type in (
                    pygame.KEYDOWN, pygame.KEYUP, pygame.TEXTINPUT, pygame.TEXTEDITING,
                    pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP,
                    pygame.MOUSEWHEEL,
                ):
                    input_event = True
                if event.type == pygame.QUIT:
                    logging.info('Exit requested by SDL window-close event')
                    quit_requested = True
                elif prompt_editing and event.type == pygame.TEXTINPUT:
                    if ignore_prompt_hotkey_text and event.text.lower() == "p":
                        ignore_prompt_hotkey_text = False
                    else:
                        ignore_prompt_hotkey_text = False
                        prompt_buffer = (prompt_buffer + event.text)[:600]
                elif event.type == pygame.KEYDOWN:
                    if prompt_editing:
                        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                            accepted = prompt_buffer.strip()
                            if accepted:
                                accepted = apply_master_prefix(
                                    load_master_prefix(prompt_library_path), accepted
                                )
                                prompt_changed = accepted != config.prompt
                                current_entry = PromptEntry(
                                    family="Custom", prompt=accepted, settings={}
                                )
                                required_prompt_revision = send_prompt(
                                    accepted, vary_settings=prompt_changed, persist=True,
                                    trigger="editor",
                                )
                                # Restart the auto-advance countdown so a manual
                                # prompt is not overwritten moments after entry.
                                last_prompt_advance = frame_started
                                latest_ai = None
                                latest_ai_version, _ = worker.generated.get()
                                notice = "PROMPT APPLIED — walking to the new prompt"
                                notice_until = frame_started + 4.0
                                logging.info("Prompt updated from in-app editor")
                            else:
                                notice = "EMPTY PROMPT IGNORED"
                                notice_until = frame_started + 3.0
                            prompt_editing = False
                            renderer.set_prompt_editing(False)
                        elif event.key == pygame.K_ESCAPE:
                            prompt_editing = False
                            renderer.set_prompt_editing(False)
                            notice = "PROMPT EDIT CANCELLED"
                            notice_until = frame_started + 2.0
                        elif event.key == pygame.K_BACKSPACE:
                            prompt_buffer = prompt_buffer[:-1]
                        elif event.key == pygame.K_d and event.mod & pygame.KMOD_CTRL:
                            prompt_buffer = config.default_prompt
                        continue
                    # Temporary status/notice panels do not unlock settings.
                    # Movement is polled separately below, so flight stays live.
                    if not overlay_enabled and event.key not in (
                        pygame.K_ESCAPE, pygame.K_F1, pygame.K_SPACE,
                        pygame.K_RETURN, pygame.K_KP_ENTER,
                    ):
                        continue
                    if event.key == pygame.K_ESCAPE:
                        logging.info('Exit requested by Escape key')
                        quit_requested = True
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        if not screenshots.busy:
                            screenshot_requested = True
                    elif event.key == pygame.K_f:
                        if not getattr(event, "repeat", False):
                            toggle_projectors()
                    elif event.key == pygame.K_F11:
                        try:
                            is_fullscreen = renderer.toggle_fullscreen()
                            commit("fullscreen", is_fullscreen, live=False)
                            notice = (
                                "FULLSCREEN — F11 returns to windowed mode"
                                if is_fullscreen
                                else "WINDOWED — F11 returns to fullscreen"
                            )
                        except RuntimeError as exc:
                            notice = str(exc)
                            logging.error("Fullscreen toggle failed: %s", exc)
                        notice_until = frame_started + 3.0
                    elif event.key == pygame.K_F10:
                        current_size = config.diffusion_size
                        try:
                            current_index = RESOLUTION_MODES.index(current_size)
                        except ValueError:
                            current_index = -1
                        next_size = RESOLUTION_MODES[(current_index + 1) % len(RESOLUTION_MODES)]
                        renderer.set_resolution(next_size)
                        worker.request_resolution(next_size)
                        commit(
                            "diffusion_resolution",
                            f"{next_size[0]}x{next_size[1]}",
                            live=False,
                        )
                        latest_ai = None
                        latest_ai_version, _ = worker.generated.get()
                        minimum_ai_sequence = renderer.sequence + 1
                        conditioning = None
                        last_conditioning_capture = 0.0
                        hide_proxy_until_ai = True
                        force_proxy = False
                        diagnostic_mode = "none"
                        notice = f"RESOLUTION: {config.resolution_label} — generating"
                        notice_until = frame_started + 4.0
                    elif event.key == pygame.K_F7:
                        prompt_caption_enabled = not prompt_caption_enabled
                        commit("prompt_caption", prompt_caption_enabled, live=False)
                    elif event.key == pygame.K_F1:
                        overlay_enabled = not overlay_enabled
                        renderer.set_operator_mode(overlay_enabled)
                        if journey is not None:
                            journey.window.set_operator_mode(overlay_enabled)
                            if not overlay_enabled:
                                renderer.focus()
                        commit("debug_overlay", overlay_enabled, live=False)
                    elif event.key == pygame.K_F2:
                        # From a diagnostic view, F2 is a one-press return to AI.
                        # From AI/proxy it retains the familiar toggle behavior.
                        if diagnostic_mode != "none":
                            force_proxy = False
                            diagnostic_mode = "none"
                        else:
                            force_proxy = not force_proxy
                        notice = "PROXY VIEW: WASD + MOUSE, Q/E DOWN/UP" if force_proxy else "AI VIEW"
                        notice_until = frame_started + 3.0
                    elif event.key == pygame.K_F3:
                        modes = ("none", "depth", "edges")
                        diagnostic_mode = modes[(modes.index(diagnostic_mode) + 1) % len(modes)]
                        force_proxy = False
                        notice = "AI VIEW" if diagnostic_mode == "none" else f"{diagnostic_mode.upper()} DIAGNOSTIC"
                        notice_until = frame_started + 2.0
                    elif event.key == pygame.K_F6:
                        force_proxy = False
                        diagnostic_mode = "none"
                        notice = "AI VIEW"
                        notice_until = frame_started + 3.0
                    elif event.key == pygame.K_F4:
                        frozen = not frozen
                        worker.set_frozen(frozen)
                    elif event.key == pygame.K_F5:
                        commit("reprojection", not config.reprojection, live=False)
                        notice = "DEPTH REPROJECTION ON" if config.reprojection else "DEPTH REPROJECTION OFF"
                        notice_until = frame_started + 3.0
                    elif event.key == pygame.K_F8:
                        commit("feedback_reprojection", not config.feedback_reprojection)
                        notice = (
                            "FEEDBACK REPROJECTION ON — slower, stickier forms"
                            if config.feedback_reprojection
                            else "FEEDBACK REPROJECTION OFF"
                        )
                        notice_until = frame_started + 3.0
                    elif event.key == pygame.K_F12:
                        commit(
                            "autowalk_idle_seconds",
                            0.0 if config.autowalk_idle_seconds > 0.0 else 60.0,
                            live=False,
                        )
                        autowalking = False
                        notice = (
                            "AUTOWALK OFF"
                            if config.autowalk_idle_seconds == 0.0
                            else f"AUTOWALK after {config.autowalk_idle_seconds:.0f}s idle"
                        )
                        notice_until = frame_started + 3.0
                    elif event.key == pygame.K_F9:
                        commit(
                            "prompt_auto_advance_seconds",
                            next_level(AUTO_ADVANCE_LEVELS, config.prompt_auto_advance_seconds),
                            live=False,
                        )
                        last_prompt_advance = frame_started
                        notice = (
                            "PROMPT AUTO ADVANCE OFF"
                            if config.prompt_auto_advance_seconds == 0.0
                            else f"PROMPT AUTO ADVANCE: {config.prompt_auto_advance_seconds:.0f}s"
                        )
                        notice_until = frame_started + 3.0
                    elif event.key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
                        direction = -1.0 if event.key == pygame.K_LEFTBRACKET else 1.0
                        commit(
                            "reprojection_strength",
                            float(max(0.0, min(1.0, round(config.reprojection_strength + direction * 0.1, 2)))),
                            live=False,
                        )
                        notice = f"REPROJECTION STRENGTH: {config.reprojection_strength:.0%}"
                        notice_until = frame_started + 3.0
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        direction = -1 if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS) else 1
                        commit("steps", max(1, min(4, config.steps + direction)))
                        notice = f"DIFFUSION STEPS: {config.steps} — {'cleaner / slower' if config.steps > 1 else 'fastest'}"
                        notice_until = frame_started + 3.0
                    elif event.key in (pygame.K_COMMA, pygame.K_PERIOD):
                        direction = -1.0 if event.key == pygame.K_COMMA else 1.0
                        commit(
                            "display_sharpen",
                            float(max(0.0, min(2.0, round(config.display_sharpen + direction * 0.1, 2)))),
                            live=False,
                        )
                        notice = f"DISPLAY SHARPNESS: {config.display_sharpen:.1f}"
                        notice_until = frame_started + 3.0
                    elif event.key in (pygame.K_i, pygame.K_o):
                        direction = -0.1 if event.key == pygame.K_i else 0.1
                        commit(
                            "instability",
                            float(max(0.0, min(1.0, round(config.instability + direction, 2)))),
                        )
                        notice = f"INSTABILITY: {config.instability:.0%}"
                        notice_until = frame_started + 3.0
                    elif event.key in (pygame.K_h, pygame.K_j):
                        direction = -10.0 if event.key == pygame.K_h else 10.0
                        commit(
                            "fog_distance",
                            max(MIN_FOG_DISTANCE, min(MAX_FOG_DISTANCE, config.fog_distance + direction)),
                            live=False,
                        )
                        notice = f"FOG DISTANCE: {config.fog_distance:g}   H nearer / J farther"
                        notice_until = frame_started + 3.0
                    elif event.key == pygame.K_v:
                        commit("player_trail", not config.player_trail, live=False)
                        notice = "PLAYER TRAIL ON: fades in 10s" if config.player_trail else "PLAYER TRAIL OFF"
                        notice_until = frame_started + 3.0
                    elif event.key in (pygame.K_k, pygame.K_l):
                        direction = -0.05 if event.key == pygame.K_k else 0.05
                        commit(
                            "guide_strength",
                            float(max(0.0, min(1.0, round(config.guide_strength + direction, 2)))),
                        )
                        notice = f"GUIDE STRENGTH: {config.guide_strength:.0%} of the proxy"
                        notice_until = frame_started + 3.0
                    elif event.key in (pygame.K_t, pygame.K_y):
                        direction = -25 if event.key == pygame.K_t else 25
                        span = config.timestep_max - config.timestep_min
                        low = max(1, min(999 - span, config.timestep_min + direction))
                        # One write: a half-applied pair fails AppConfig.validate.
                        commit_many({"timestep_min": low, "timestep_max": low + span})
                        notice = f"TIMESTEP RANGE: {config.timestep_min}–{config.timestep_max}"
                        notice_until = frame_started + 3.0
                    elif event.key == pygame.K_n:
                        commit(
                            "noise_walk_seconds",
                            next_level(NOISE_WALK_LEVELS, config.noise_walk_seconds),
                        )
                        notice = (
                            "NOISE WALK OFF — fresh noise every frame"
                            if config.noise_walk_seconds == 0.0
                            else f"NOISE WALK: {config.noise_walk_seconds:.0f}s per keyframe"
                        )
                        notice_until = frame_started + 3.0
                    elif event.key == pygame.K_m:
                        commit(
                            "prompt_walk_seconds",
                            next_level(PROMPT_WALK_LEVELS, config.prompt_walk_seconds),
                        )
                        notice = (
                            "PROMPT WALK OFF — prompts cut instantly"
                            if config.prompt_walk_seconds == 0.0
                            else f"PROMPT WALK: {config.prompt_walk_seconds:.0f}s"
                        )
                        notice_until = frame_started + 3.0
                    elif event.key == pygame.K_c:
                        commit("guidance_scale", next_level(CFG_LEVELS, config.guidance_scale))
                        notice = f"CFG: {config.guidance_scale:.2g}"
                        notice_until = frame_started + 3.0
                    elif event.key == pygame.K_r and event.mod & pygame.KMOD_SHIFT:
                        new_seed = worker.request_reseed()
                        commit("seed", new_seed, live=False)
                        notice = f"DIFFUSION RESEEDED: {new_seed}"
                        notice_until = frame_started + 3.0
                        logging.info("Manual reseed: %s", new_seed)
                    elif event.key == pygame.K_SPACE:
                        if space_held or getattr(event, "repeat", False):
                            continue
                        space_held = True
                        # Keep the current world and live tuning while the
                        # backend walks toward another library prompt.
                        try:
                            next_entry = advance_prompt(
                                load_prompt_library(prompt_library_path), current_entry, recent_families
                            )
                            if journey is not None:
                                journey.reset(frame_started)
                            current_entry = next_entry
                            recent_families.append(current_entry.family)
                            del recent_families[:-RECENT_FAMILY_MEMORY]
                            required_prompt_revision = send_prompt(
                                current_entry.prompt, vary_settings=True, persist=True,
                                trigger="space",
                            )
                            last_prompt_advance = frame_started
                            notice = f"NEW PROMPT: {current_entry.family}"
                            notice_until = frame_started + 4.0
                            logging.info("Prompt selected: %s", current_entry.family)
                        except (RuntimeError, OSError) as exc:
                            notice = f"NEW JOURNEY NOT STARTED: {exc}"
                            notice_until = frame_started + 8.0
                            logging.error("Prompt selection skipped: %s", exc)
                    elif event.key == pygame.K_p:
                        prompt_editing = True
                        prompt_buffer = config.prompt
                        ignore_prompt_hotkey_text = True
                        renderer.set_prompt_editing(True)
                        notice = ""

            (mouse_x, mouse_y), keys, mouse_buttons = renderer.read_input()
            if not prompt_editing and not overlay_enabled:
                camera.rotate(mouse_x, mouse_y, config.mouse_sensitivity)
            speed = config.movement_speed
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                speed *= config.sprint_multiplier
            dt = min(clock.get_time() / 1000.0, 0.1)
            strafe = float(keys[pygame.K_d]) - float(keys[pygame.K_a])
            ahead = float(keys[pygame.K_w]) - float(keys[pygame.K_s])
            rise = float(keys[pygame.K_e]) - float(keys[pygame.K_q])
            physical_input = bool(
                input_event or mouse_x or mouse_y or strafe or ahead or rise
                or any(keys) or any(mouse_buttons)
            )
            if physical_input:
                last_input = frame_started
                if autowalking:
                    autowalking = False
                    logging.info("Autowalk stopped by input")
            idle_limit = config.autowalk_idle_seconds
            if not autowalking and idle_limit > 0.0 and frame_started - last_input >= idle_limit:
                autowalking = True
                autowalk.reset()
                logging.info("Autowalk started after %.0f s idle", idle_limit)
            if autowalking and not prompt_editing:
                step_length = 0.0 if autowalk.turning(frame_started) else config.movement_speed * dt
                before = camera.position.copy()
                camera.fly(0.0, 0.0, 1.0, step_length)
                renderer.constrain_camera(camera, dt)
                gained = float(np.linalg.norm(camera.position - before))
                camera.yaw, camera.pitch = autowalk.step(
                    camera.yaw,
                    camera.pitch,
                    tuple(float(value) for value in camera.position),
                    frame_started,
                    dt,
                    gained,
                    step_length,
                    renderer.nearby_colliders(camera),
                )
            else:
                if not prompt_editing:
                    camera.fly(strafe, rise, ahead, speed * dt)
                renderer.constrain_camera(camera, dt)

            trail.update(camera.position, frame_started, config.player_trail)
            if journey is not None:
                journey.observe(
                    camera.position, [camera.pitch, camera.yaw, 0.0], frame_started,
                    interacting=bool(strafe or ahead or rise or (
                        not overlay_enabled and (mouse_x or mouse_y)
                    )), autowalking=autowalking, suppressed=prompt_editing,
                )
            capture_due = (
                conditioning is None
                or force_proxy
                or diagnostic_mode != "none"
                or frame_started - last_conditioning_capture >= 1.0 / config.conditioning_fps
            )
            renderer.fog_distance = config.fog_distance
            if capture_due:
                proxy_started = perf_counter()
                snapshot = renderer.render_scene(camera)
                proxy_ms.update((perf_counter() - proxy_started) * 1000.0)
                proxy_rate.tick()
            else:
                # Reprojection draws its own interpolated camera later. Between
                # captures, neither view consumes a render from this live camera.
                # Streaming and the location palette still follow every tick.
                renderer.update_world(camera.position)
            # Let the existing prompt interpolation finish before retargeting
            # colors. Crossing a region changes no subject or sampler setting.
            if (
                not prompt_editing
                and frame_started - last_palette_update >= max(1.0, config.prompt_walk_seconds)
                and hue_words(renderer.world_label(), hue_offset) != sent_hues
            ):
                required_prompt_revision = send_prompt(config.prompt)
                logging.info("Landscape palette: %s", sent_hues)
            if capture_due:
                capture_started = perf_counter()
                conditioning = renderer.capture_conditioning(
                    snapshot,
                    frame_started,
                    include_edges=config.backend != "latent_walk" or diagnostic_mode == "edges",
                )
                conditioning_ms.update((perf_counter() - capture_started) * 1000.0)
                last_conditioning_capture = frame_started
                worker.publish(conditioning)
            assert conditioning is not None

            ai_version, published_ai = worker.generated.get()
            if journey is not None:
                journey.offer_frame(published_ai)
            if (
                ai_version > latest_ai_version
                and published_ai is not None
                and int(published_ai.stats.get("prompt_revision", 0)) >= required_prompt_revision
                and published_ai.sequence >= minimum_ai_sequence
            ):
                first_ai_arrival = latest_ai is None
                silent_ai_arrival = hide_proxy_until_ai
                latest_ai_version = ai_version
                latest_ai = published_ai
                hide_proxy_until_ai = False
                if first_ai_arrival:
                    # Diagnostics selected while the model was loading must not
                    # hide the first generated frame and make startup appear
                    # stuck. The operator can re-enter them after AI is visible.
                    force_proxy = False
                    diagnostic_mode = "none"
                    if not silent_ai_arrival:
                        notice = "AI READY — GENERATED VIEW"
                        notice_until = perf_counter() + 4.0

            reproject_frame = None
            if diagnostic_mode != "none":
                display_image = renderer.diagnostic_image(conditioning, diagnostic_mode)
                view_name = diagnostic_mode.upper()
            elif force_proxy:
                display_image = conditioning.rgb
                view_name = "PROXY"
            elif latest_ai is None:
                display_image = renderer.loading_image if hide_proxy_until_ai else conditioning.rgb
                view_name = "AI LOADING" if hide_proxy_until_ai else "LIVE PROXY — AI LOADING"
            else:
                display_image = latest_ai.image
                if config.reprojection:
                    reproject_frame = latest_ai
                    view_name = "AI REPROJECTED"
                else:
                    view_name = "AI RAW"

            now = perf_counter()
            if latest_ai is not None:
                frame_age_ms.update((now - latest_ai.generation_timestamp) * 1000.0)
            status = worker.status()
            if status.resolution_fallbacks != seen_resolution_fallbacks:
                # The backend dropped a mode after a CUDA OOM; follow it so the
                # proxy, the display and the next launch all agree.
                seen_resolution_fallbacks = status.resolution_fallbacks
                width_text, height_text = status.active_resolution.split("x", 1)
                renderer.set_resolution((int(width_text), int(height_text)))
                commit("diffusion_resolution", status.active_resolution, live=False)
                latest_ai = None
                latest_ai_version, _ = worker.generated.get()
                minimum_ai_sequence = renderer.sequence + 1
                conditioning = None
                last_conditioning_capture = 0.0
                hide_proxy_until_ai = True
                notice = f"CUDA OOM: RESOLUTION {status.active_resolution}"
                notice_until = now + 4.0
            stats = worker.stats()
            overlay: list[str] | None = None
            if overlay_enabled or prompt_editing or now < notice_until or status.state != "ready":
                overlay = build_overlay(
                    display_rate.fps,
                    proxy_rate.fps,
                    proxy_ms.value,
                    conditioning_ms.value,
                    renderer.reproject_ms,
                    frame_age_ms.value if latest_ai is not None else 0.0,
                    view_name,
                    frozen,
                    config,
                    status,
                    stats,
                    renderer.world_label(),
                    current_entry.family,
                    log_path,
                    prompt_editing,
                    prompt_buffer,
                    notice if now < notice_until else "",
                    effective_prompt=effective_prompt,
                    hue_offset=hue_offset,
                )
            idle_opacity = idle_instructions.update(
                now, active=physical_input,
                suppressed=overlay is not None or hide_proxy_until_ai,
            )
            screenshot = None
            if screenshot_requested:
                if latest_ai is not None and not force_proxy and diagnostic_mode == "none":
                    screenshot = screenshots
                else:
                    notice = "SCREENSHOT: wait for the generated AI view"
                    notice_until = now + 3.0
            if reproject_frame is not None:
                renderer.display_reprojected(
                    reproject_frame,
                    camera,
                    overlay,
                    strength=config.reprojection_strength,
                    max_translation=config.reprojection_max_translation,
                    max_rotation=config.reprojection_max_rotation,
                    sharpen=config.display_sharpen,
                    prompt_caption=config.prompt if prompt_caption_enabled else None,
                    trail=trail if config.player_trail else None,
                    trail_time=now,
                    idle_opacity=idle_opacity,
                    screenshot=screenshot,
                )
            else:
                # Match the camera and depth of the image actually on screen,
                # including raw AI frames held between generation updates.
                trail_frame = None
                if diagnostic_mode == "none" and (force_proxy or latest_ai is not None):
                    trail_frame = conditioning if force_proxy else latest_ai
                renderer.display(
                    display_image,
                    overlay,
                    sharpen=config.display_sharpen,
                    prompt_caption=config.prompt if prompt_caption_enabled else None,
                    trail=trail if config.player_trail else None,
                    trail_frame=trail_frame,
                    trail_time=now,
                    idle_opacity=idle_opacity,
                    screenshot=screenshot,
                )
            display_rate.tick()
            displayed_frames += 1
            # Publish only scalar counters after the display call succeeded.
            if now >= next_monitoring_update:
                monitoring.update(state="error" if status.state == "error" else
                                  "loading" if latest_ai is None else "running",
                                  displayed_frames=displayed_frames, display_fps=display_rate.fps,
                                  generation_fps=stats.get("diffusion_fps"),
                                  frame_age_ms=frame_age_ms.value if latest_ai is not None else None)
                next_monitoring_update = now + 1.0

            # Automatic prompt walking keeps the world it is walking through
            # intact while sampling a fresh prompt and tuning combination.
            if (
                config.prompt_auto_advance_seconds > 0.0
                and not prompt_editing
                and now - last_prompt_advance >= config.prompt_auto_advance_seconds
            ):
                last_prompt_advance = now
                try:
                    current_entry = advance_prompt(
                        load_prompt_library(prompt_library_path), current_entry, recent_families
                    )
                    recent_families.append(current_entry.family)
                    del recent_families[:-RECENT_FAMILY_MEMORY]
                    # Ephemeral: persisting would move config_mtime past an
                    # external edit and overwrite the configured startup prompt.
                    apply_live({"prompt": current_entry.prompt})
                    required_prompt_revision = send_prompt(current_entry.prompt, vary_settings=True, trigger="automatic")
                    logging.info("Prompt auto-advanced to: %s", current_entry.family)
                except RuntimeError as exc:
                    logging.error("Prompt auto-advance skipped: %s", exc)

            if now - last_config_check > 0.5:
                last_config_check = now
                try:
                    current_mtime = config_path.stat().st_mtime
                    if current_mtime != config_mtime:
                        updated = AppConfig.load(config_path)
                        if config.diffusion_size != updated.diffusion_size:
                            renderer.set_resolution(updated.diffusion_size)
                            worker.request_resolution(updated.diffusion_size)
                            config.diffusion_resolution = updated.diffusion_resolution
                            latest_ai = None
                            latest_ai_version, _ = worker.generated.get()
                            minimum_ai_sequence = renderer.sequence + 1
                            conditioning = None
                            last_conditioning_capture = 0.0
                            hide_proxy_until_ai = True
                        updated.prompt = apply_master_prefix(
                            load_master_prefix(prompt_library_path), updated.prompt
                        )
                        prompt_changed = updated.prompt != config.prompt
                        prompt_request_changed = (updated.prompt, updated.negative_prompt) != (
                            config.prompt,
                            config.negative_prompt,
                        )
                        if prompt_request_changed:
                            current_entry = PromptEntry(
                                family=current_entry.family
                                if updated.prompt == current_entry.prompt
                                else "Custom",
                                prompt=updated.prompt,
                                settings={},
                            )
                            last_prompt_advance = frame_started
                        changed = {
                            key: getattr(updated, key)
                            for key in BACKEND_SETTING_KEYS
                            if getattr(config, key) != getattr(updated, key)
                        }
                        if changed:
                            worker.request_settings(**changed)
                        for key in (
                            *BACKEND_SETTING_KEYS,
                            "prompt",
                            "negative_prompt",
                            "movement_speed",
                            "sprint_multiplier",
                            "mouse_sensitivity",
                            "reprojection",
                            "reprojection_strength",
                            "reprojection_max_translation",
                            "reprojection_max_rotation",
                            "display_sharpen",
                            "fog_distance",
                            "player_trail",
                            "debug_overlay",
                            "prompt_caption",
                            "prompt_auto_advance_seconds",
                        ):
                            setattr(config, key, getattr(updated, key))
                        if prompt_request_changed:
                            required_prompt_revision = send_prompt(
                                config.prompt, vary_settings=prompt_changed,
                                trigger="configuration",
                            )
                        overlay_enabled = bool(config.debug_overlay or debug_requested)
                        renderer.set_operator_mode(overlay_enabled)
                        if journey is not None:
                            journey.window.set_operator_mode(overlay_enabled)
                        prompt_caption_enabled = config.prompt_caption
                        config_mtime = current_mtime
                        logging.info("Hot-reloaded config: %s", ", ".join(sorted(changed)) or "prompt/navigation")
                except Exception:
                    logging.exception("Config hot reload failed; retaining last valid values")

            if args.smoke_frames and displayed_frames >= args.smoke_frames:
                # Wait for proof that the backend produced a frame, but never
                # hang a headless smoke run on a stuck or failed worker.
                if (
                    latest_ai is not None
                    or status.state == "error"
                    or now - session_started > SMOKE_AI_TIMEOUT_S
                ):
                    running = False
            clock.tick(config.target_display_fps)
            if quit_requested:
                request_quit()
        logging.info(
            "Session ended: %s displayed frames, %.1f display FPS, %.1f proxy FPS",
            displayed_frames,
            display_rate.fps,
            proxy_rate.fps,
        )
        monitoring.update(state="stopped")
        return 0
    except Exception as exc:
        monitoring.update(state="error")
        logging.exception("Fatal application error")
        print(f"Fatal application error: {exc}", file=sys.stderr)
        return 4
    finally:
        monitoring.close()
        screenshots.close()
        if worker is not None:
            worker.stop()
        if journey is not None:
            try:
                journey.close()
            except Exception as exc:
                logging.exception("Journey save failed on exit; checkpoint retained")
                print(f"Journey save failed: {exc}. Check journeys/ for the checkpoint.", file=sys.stderr)
        if renderer is not None:
            renderer.close()


def build_overlay(
    display_fps: float,
    proxy_fps: float,
    proxy_ms: float,
    conditioning_ms: float,
    reproject_ms: float,
    ai_age_ms: float,
    view_name: str,
    frozen: bool,
    config: AppConfig,
    status: object,
    stats: dict[str, float | int | str],
    world_label: str,
    family: str,
    log_path: Path,
    prompt_editing: bool = False,
    prompt_buffer: str = "",
    notice: str = "",
    effective_prompt: str = "",
    hue_offset: int = 0,
) -> list[str]:
    """Diagnostic lines for the F1 overlay: rates, walk state and hotkeys."""
    state = getattr(status, "state", "unknown")
    error = getattr(status, "error", "")
    active_resolution = getattr(status, "active_resolution", config.resolution_label)
    lines = [
        "FLIGHT        WASD move  Mouse look  Q/E down/up  Shift faster",
        "FUNCTION KEYS  F1 overlay  F2 proxy/AI  F3 diagnostics  F4 freeze",
        "               F5 reprojection  F6 AI  F7 caption  F8 feedback",
        "               F9 auto advance  F10 resolution  F11 fullscreen  F12 autowalk",
        "PROJECTORS    F fullscreen both  F1 releases mouse for window placement",
        "",
        f"DISPLAY       {display_fps:6.1f} FPS     VIEW  {view_name}{' (FROZEN)' if frozen else ''}",
        f"PROXY         {proxy_fps:6.1f} FPS     RENDER {proxy_ms:6.2f} ms",
        f"CONDITIONING  {config.conditioning_fps:6d} FPS     CAPTURE {conditioning_ms:5.2f} ms",
        f"DIFFUSION     {float(stats.get('diffusion_fps', 0.0)):6.1f} FPS     INFER  {float(stats.get('inference_ms', 0.0)):6.2f} ms",
        f"REPROJECT     {reproject_ms:6.2f} ms      MODE    {'ON' if config.reprojection else 'OFF'}",
        f"WARP AMOUNT   [{'#' * round(config.reprojection_strength * 10)}{'-' * (10 - round(config.reprojection_strength * 10))}] {config.reprojection_strength:4.0%}    [ / ] adjust",
        f"AI AGE        {ai_age_ms:6.1f} ms      VRAM   {float(stats.get('vram_allocated_gb', 0.0)):5.2f} GB",
        f"RES           {active_resolution}       F10 cycle       STEPS  {stats.get('steps', config.steps)}    - / = adjust",
        f"CFG           {float(stats.get('guidance_scale', config.guidance_scale)):4.2g}                  C cycle",
        f"SHARPNESS     {config.display_sharpen:4.1f}                  , / . adjust",
        f"FOG DISTANCE  {config.fog_distance:4.0f}                   H nearer / J farther",
        f"PLAYER TRAIL  {'ON' if config.player_trail else 'OFF'}                     V toggle   fades in 10s",
        f"TIMESTEP      {config.timestep_min}-{config.timestep_max} now {float(stats.get('timestep_now', 0.0)):5.0f}    T / Y shift",
        f"INSTABILITY   {config.instability:4.0%}                  I less / O more",
        f"GUIDE         {config.guide_strength:4.0%} now {float(stats.get('guide_strength_now', 0.0)):4.0%}         K less / L more",
        f"NOISE WALK    {float(stats.get('noise_walk_t', 0.0)):4.0%} of {config.noise_walk_seconds:5.1f}s    N cycle",
        f"PROMPT WALK   {float(stats.get('prompt_walk_t', 1.0)):4.0%} of {config.prompt_walk_seconds:5.1f}s    M cycle",
        f"FEEDBACK      {'ON' if config.feedback_reprojection else 'OFF'}                   F8 toggle",
        f"BACKEND       {config.backend}       STATE  {state.upper()}",
        f"SEED          {stats.get('active_seed', config.seed)}        Shift+R reseed",
        f"WORLD         {config.world_seed}  {world_label}",
        f"FAMILY        {family}",
        "              SPACE new prompt   P edit prompt",
        f"HUE           {hue_words(world_label, hue_offset) or 'none'}",
        f"PROMPT        {(effective_prompt or config.prompt)[:58]}",
        f"AUTO ADVANCE  {config.prompt_auto_advance_seconds:5.1f}s               F9 cycle",
    ]
    if notice:
        lines.extend(["", notice])
    if prompt_editing:
        lines.extend(
            [
                "",
                "PROMPT EDITOR — type normally; Enter applies; Escape cancels; Ctrl+D restores default",
                *[f"> {line}" for line in textwrap.wrap(prompt_buffer + "|", width=60) or ["|"]],
            ]
        )
    if error:
        lines.extend(["", f"ERROR: {error}", f"LOG: {log_path}"])
    elif state != "ready":
        lines.append(f"STATUS: {getattr(status, 'message', '')}")
    return lines


if __name__ == "__main__":
    raise SystemExit(run())
