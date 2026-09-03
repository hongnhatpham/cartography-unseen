from __future__ import annotations

import argparse
import json
import logging
import secrets
import sys
import textwrap
from pathlib import Path
from time import perf_counter, strftime

from app.config import (
    RESOLUTION_MODES,
    AppConfig,
    configure_local_environment,
    structure_lock_percent,
)
from app.utils.timing import ExponentialAverage, RateMeter


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
    return log_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime Diffusion WASD Art Renderer")
    parser.add_argument("--debug", action="store_true", help="windowed mode with diagnostics")
    parser.add_argument("--windowed", action="store_true", help="disable fullscreen")
    parser.add_argument("--monitor", type=int, help="zero-based fullscreen monitor index")
    parser.add_argument("--list-monitors", action="store_true", help="print detected monitors and exit")
    parser.add_argument("--config", type=Path, help="alternate project-relative config file")
    parser.add_argument("--backend", choices=("sd_turbo_stream", "proxy_passthrough"))
    parser.add_argument(
        "--resolution",
        choices=tuple(f"{width}x{height}" for width, height in RESOLUTION_MODES),
        help="generation resolution/aspect mode",
    )
    parser.add_argument("--smoke-frames", type=int, default=0, help=argparse.SUPPRESS)
    return parser.parse_args()


def persist_config_value(config_path: Path, key: str, value: object) -> float:
    """Persist one accepted runtime setting without disturbing other config."""
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw[key] = value
    temporary = config_path.with_suffix(config_path.suffix + ".tmp")
    temporary.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(config_path)
    return config_path.stat().st_mtime


def persist_prompt(config_path: Path, prompt: str) -> float:
    return persist_config_value(config_path, "prompt", prompt)


def load_prompt_library(path: Path) -> list[dict[str, str]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Prompt library not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid prompt library JSON: {exc}") from exc
    entries = raw.get("prompts") if isinstance(raw, dict) else None
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("prompts.json must contain a non-empty 'prompts' list")
    master_prefix = str(raw.get("master_prefix", "")).strip()
    cleaned: list[dict[str, str]] = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise RuntimeError(f"Prompt entry {index} must be an object")
        name = str(entry.get("name", "")).strip()
        prompt = str(entry.get("prompt", "")).strip()
        if not name or not prompt:
            raise RuntimeError(f"Prompt entry {index} requires non-empty name and prompt")
        cleaned.append(
            {"name": name, "prompt": apply_master_prefix(master_prefix, prompt)}
        )
    return cleaned


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
    prefix = prefix.strip().rstrip(" ,")
    prompt = prompt.strip()
    if not prefix or prompt.casefold().startswith(prefix.casefold()):
        return prompt
    return f"{prefix}, {prompt}"


def choose_different_prompt(
    entries: list[dict[str, str]], current_prompt: str
) -> dict[str, str]:
    alternatives = [entry for entry in entries if entry["prompt"] != current_prompt]
    return secrets.choice(alternatives or entries)


def run() -> int:
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
    fullscreen = config.fullscreen and not (args.debug or args.windowed)

    try:
        import pygame

        from app.diffusion.worker import DiffusionWorker
        from app.renderer.camera import EYE_HEIGHT, Camera
        from app.renderer.proxy_renderer import ProxyRenderer
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
    try:
        renderer = ProxyRenderer(
            root,
            config.diffusion_size,
            fullscreen,
            window_size=config.diffusion_size,
            display_monitor=config.display_monitor,
            world_seed=config.world_seed,
        )
        renderer.loading_screen("INITIALIZING", "Starting renderer and diffusion worker")
        camera = Camera.create_default()
        road_contact = renderer.spawn_road_contact()
        camera.yaw = road_contact.heading
        camera.position[:] = (
            road_contact.x,
            road_contact.surface_y + EYE_HEIGHT,
            road_contact.z,
        )
        worker = DiffusionWorker(config.backend, config.backend_dict(root))
        worker.start()

        clock = pygame.time.Clock()
        display_rate = RateMeter(120)
        proxy_rate = RateMeter(120)
        proxy_ms = ExponentialAverage(0.15)
        conditioning_ms = ExponentialAverage(0.15)
        frame_age_ms = ExponentialAverage(0.15)
        running = True
        overlay_enabled = bool(config.debug_overlay or debug_requested)
        force_proxy = False
        diagnostic_mode = "none"
        frozen = False
        latest_ai = None
        latest_ai_version = 0
        last_config_check = 0.0
        config_mtime = config_path.stat().st_mtime
        displayed_frames = 0
        conditioning = None
        last_conditioning_capture = 0.0
        prompt_editing = False
        prompt_caption_enabled = config.prompt_caption
        prompt_buffer = ""
        required_prompt_revision = 0
        minimum_ai_sequence = 0
        hold_previous_ai = False
        # Never expose the proxy as the startup fallback. The worker already
        # receives conditioning frames automatically, so hold the neutral
        # loading artwork until its first generated frame arrives.
        hide_proxy_until_ai = True
        notice = ""
        notice_until = 0.0
        ignore_prompt_hotkey_text = False

        while running:
            frame_started = perf_counter()
            for event in renderer.poll_events():
                if event.type == pygame.QUIT:
                    running = False
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
                                config.prompt = accepted
                                required_prompt_revision = worker.request_prompt(
                                    accepted, config.negative_prompt
                                )
                                config_mtime = persist_prompt(config_path, accepted)
                                latest_ai = None
                                latest_ai_version, _ = worker.generated.get()
                                notice = "PROMPT APPLIED — generating a new AI frame"
                                notice_until = frame_started + 4.0
                                logging.info("Prompt updated from in-app editor")
                            else:
                                notice = "EMPTY PROMPT IGNORED"
                                notice_until = frame_started + 3.0
                            prompt_editing = False
                            pygame.key.stop_text_input()
                            pygame.event.set_grab(True)
                            pygame.mouse.set_visible(False)
                            pygame.mouse.get_rel()
                        elif event.key == pygame.K_ESCAPE:
                            prompt_editing = False
                            pygame.key.stop_text_input()
                            pygame.event.set_grab(True)
                            pygame.mouse.set_visible(False)
                            pygame.mouse.get_rel()
                            notice = "PROMPT EDIT CANCELLED"
                            notice_until = frame_started + 2.0
                        elif event.key == pygame.K_BACKSPACE:
                            prompt_buffer = prompt_buffer[:-1]
                        elif event.key == pygame.K_d and event.mod & pygame.KMOD_CTRL:
                            prompt_buffer = config.default_prompt
                        continue
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_F11:
                        try:
                            is_fullscreen = renderer.toggle_fullscreen()
                            config.fullscreen = is_fullscreen
                            config_mtime = persist_config_value(
                                config_path, "fullscreen", config.fullscreen
                            )
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
                        config.diffusion_resolution = f"{next_size[0]}x{next_size[1]}"
                        renderer.set_resolution(next_size)
                        worker.request_resolution(next_size)
                        config_mtime = persist_config_value(
                            config_path, "diffusion_resolution", config.diffusion_resolution
                        )
                        latest_ai = None
                        latest_ai_version, _ = worker.generated.get()
                        minimum_ai_sequence = renderer.sequence + 1
                        conditioning = None
                        last_conditioning_capture = 0.0
                        hold_previous_ai = False
                        hide_proxy_until_ai = True
                        force_proxy = False
                        diagnostic_mode = "none"
                        notice = f"RESOLUTION: {config.resolution_label} — generating"
                        notice_until = frame_started + 4.0
                    elif event.key == pygame.K_F7:
                        prompt_caption_enabled = not prompt_caption_enabled
                        config.prompt_caption = prompt_caption_enabled
                        config_mtime = persist_config_value(
                            config_path, "prompt_caption", config.prompt_caption
                        )
                    elif event.key == pygame.K_F1:
                        overlay_enabled = not overlay_enabled
                        config.debug_overlay = overlay_enabled
                        config_mtime = persist_config_value(
                            config_path, "debug_overlay", config.debug_overlay
                        )
                    elif event.key == pygame.K_F2:
                        # From a diagnostic view, F2 is a one-press return to AI.
                        # From AI/proxy it retains the familiar toggle behavior.
                        if diagnostic_mode != "none":
                            force_proxy = False
                            diagnostic_mode = "none"
                        else:
                            force_proxy = not force_proxy
                        notice = "PROXY VIEW — WASD + MOUSE" if force_proxy else "AI VIEW"
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
                        config.reprojection = not config.reprojection
                        config_mtime = persist_config_value(
                            config_path, "reprojection", config.reprojection
                        )
                        notice = "DEPTH REPROJECTION ON" if config.reprojection else "DEPTH REPROJECTION OFF"
                        notice_until = frame_started + 3.0
                    elif event.key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
                        direction = -1.0 if event.key == pygame.K_LEFTBRACKET else 1.0
                        config.reprojection_strength = float(
                            max(0.0, min(1.0, round(config.reprojection_strength + direction * 0.1, 2)))
                        )
                        config_mtime = persist_config_value(
                            config_path, "reprojection_strength", config.reprojection_strength
                        )
                        notice = f"REPROJECTION STRENGTH: {config.reprojection_strength:.0%}"
                        notice_until = frame_started + 3.0
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        direction = -1 if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS) else 1
                        config.steps = max(1, min(4, config.steps + direction))
                        worker.request_steps(config.steps)
                        config_mtime = persist_config_value(config_path, "steps", config.steps)
                        notice = f"DIFFUSION STEPS: {config.steps} — {'cleaner / slower' if config.steps > 1 else 'fastest'}"
                        notice_until = frame_started + 3.0
                    elif event.key in (pygame.K_COMMA, pygame.K_PERIOD):
                        direction = -1.0 if event.key == pygame.K_COMMA else 1.0
                        config.display_sharpen = float(
                            max(0.0, min(2.0, round(config.display_sharpen + direction * 0.1, 2)))
                        )
                        config_mtime = persist_config_value(
                            config_path, "display_sharpen", config.display_sharpen
                        )
                        notice = f"DISPLAY SHARPNESS: {config.display_sharpen:.1f}"
                        notice_until = frame_started + 3.0
                    elif event.key in (pygame.K_k, pygame.K_l):
                        if config.steps == 1:
                            direction = 50 if event.key == pygame.K_k else -50
                            config.one_step_timestep = max(
                                250, min(900, config.one_step_timestep + direction)
                            )
                            worker.request_one_step_timestep(config.one_step_timestep)
                            config_mtime = persist_config_value(
                                config_path, "one_step_timestep", config.one_step_timestep
                            )
                        else:
                            direction = 0.05 if event.key == pygame.K_k else -0.05
                            config.img2img_strength = float(
                                max(0.25, min(1.0, round(config.img2img_strength + direction, 2)))
                            )
                            worker.request_img2img_strength(config.img2img_strength)
                            config_mtime = persist_config_value(
                                config_path, "img2img_strength", config.img2img_strength
                            )
                        notice = f"STRUCTURE LOCK: {structure_lock_percent(config):.0f}%"
                        notice_until = frame_started + 3.0
                    elif event.key == pygame.K_b:
                        levels = (0.0, 0.75, 1.5, 2.5, 4.0)
                        nearest = min(range(len(levels)), key=lambda index: abs(levels[index] - config.edge_softness))
                        config.edge_softness = levels[(nearest + 1) % len(levels)]
                        worker.request_edge_softness(config.edge_softness)
                        config_mtime = persist_config_value(
                            config_path, "edge_softness", config.edge_softness
                        )
                        notice = f"GEOMETRY EDGE SOFTNESS: {config.edge_softness:.2g} px"
                        notice_until = frame_started + 3.0
                    elif event.key == pygame.K_g:
                        levels = (0.0, 0.15, 0.3, 0.5, 0.7)
                        nearest = min(
                            range(len(levels)),
                            key=lambda index: abs(levels[index] - config.edge_strength),
                        )
                        config.edge_strength = levels[(nearest + 1) % len(levels)]
                        worker.request_edge_strength(config.edge_strength)
                        config_mtime = persist_config_value(
                            config_path, "edge_strength", config.edge_strength
                        )
                        notice = f"GEOMETRY GUIDE: {config.edge_strength:.0%}"
                        notice_until = frame_started + 3.0
                    elif event.key == pygame.K_c:
                        levels = (0.0, 1.25, 1.5, 2.0, 3.0)
                        nearest = min(
                            range(len(levels)),
                            key=lambda index: abs(levels[index] - config.guidance_scale),
                        )
                        config.guidance_scale = levels[(nearest + 1) % len(levels)]
                        worker.request_guidance_scale(config.guidance_scale)
                        config_mtime = persist_config_value(
                            config_path, "guidance_scale", config.guidance_scale
                        )
                        notice = f"CFG: {config.guidance_scale:.2g}"
                        notice_until = frame_started + 3.0
                    elif event.key == pygame.K_n:
                        modes = ("fixed", "drift", "random_each_frame")
                        config.seed_mode = modes[(modes.index(config.seed_mode) + 1) % len(modes)]
                        worker.request_seed_mode(config.seed_mode)
                        config_mtime = persist_config_value(
                            config_path, "seed_mode", config.seed_mode
                        )
                        notice = (
                            "SEED MODE: RANDOM EVERY AI FRAME"
                            if config.seed_mode == "random_each_frame"
                            else f"SEED MODE: {config.seed_mode.upper()}"
                        )
                        notice_until = frame_started + 4.0
                    elif event.key == pygame.K_r and event.mod & pygame.KMOD_SHIFT:
                        new_seed = worker.request_reseed()
                        config.seed = new_seed
                        config_mtime = persist_config_value(
                            config_path, "seed", config.seed
                        )
                        notice = f"DIFFUSION RESEEDED: {new_seed}"
                        notice_until = frame_started + 3.0
                        logging.info("Manual reseed: %s", new_seed)
                    elif event.key == pygame.K_SPACE:
                        camera.reset()
                        config.world_seed = renderer.randomize_world()
                        road_contact = renderer.spawn_road_contact()
                        camera.yaw = road_contact.heading
                        camera.position[:] = (
                            road_contact.x,
                            road_contact.surface_y + EYE_HEIGHT,
                            road_contact.z,
                        )
                        selected_prompt_name = "current prompt"
                        try:
                            selected_prompt = choose_different_prompt(
                                load_prompt_library(prompt_library_path), config.prompt
                            )
                            selected_prompt_name = selected_prompt["name"]
                            config.prompt = selected_prompt["prompt"]
                            required_prompt_revision = worker.request_prompt(
                                config.prompt, config.negative_prompt
                            )
                            config_mtime = persist_prompt(config_path, config.prompt)
                        except RuntimeError as exc:
                            logging.error("Prompt library ignored: %s", exc)
                            selected_prompt_name = f"library error: {exc}"
                        config_mtime = persist_config_value(
                            config_path, "world_seed", config.world_seed
                        )
                        # Preserve the last finished artwork while the new
                        # world and prompt generate. Reprojecting that old
                        # image with the new world's depth would expose the
                        # proxy and break the transition, so it is displayed
                        # raw until its replacement arrives.
                        hold_previous_ai = latest_ai is not None
                        hide_proxy_until_ai = latest_ai is None
                        force_proxy = False
                        diagnostic_mode = "none"
                        latest_ai_version, _ = worker.generated.get()
                        minimum_ai_sequence = renderer.sequence + 1
                        conditioning = None
                        last_conditioning_capture = 0.0
                        notice = ""
                        notice_until = 0.0
                        logging.info(
                            "World randomized: %s; prompt: %s",
                            config.world_seed,
                            selected_prompt_name,
                        )
                    elif event.key == pygame.K_p:
                        prompt_editing = True
                        prompt_buffer = config.prompt
                        ignore_prompt_hotkey_text = True
                        pygame.key.start_text_input()
                        pygame.event.set_grab(False)
                        pygame.mouse.set_visible(True)
                        notice = ""

            mouse_x, mouse_y = pygame.mouse.get_rel()
            if not prompt_editing:
                camera.rotate(mouse_x, mouse_y, config.mouse_sensitivity)
            keys = pygame.key.get_pressed()
            speed = config.movement_speed
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                speed *= config.sprint_multiplier
            dt = min(clock.get_time() / 1000.0, 0.1)
            if not prompt_editing:
                camera.move(
                    float(keys[pygame.K_d]) - float(keys[pygame.K_a]),
                    float(keys[pygame.K_w]) - float(keys[pygame.K_s]),
                    speed * dt,
                )
                road_contact = renderer.move_road_contact(
                    road_contact,
                    float(camera.position[0]),
                    float(camera.position[2]),
                )
                camera.position[:] = (
                    road_contact.x,
                    road_contact.surface_y + EYE_HEIGHT,
                    road_contact.z,
                )

            proxy_started = perf_counter()
            snapshot = renderer.render_scene(camera)
            proxy_ms.update((perf_counter() - proxy_started) * 1000.0)
            proxy_rate.tick()
            capture_due = (
                conditioning is None
                or force_proxy
                or diagnostic_mode != "none"
                or frame_started - last_conditioning_capture >= 1.0 / config.conditioning_fps
            )
            if capture_due:
                capture_started = perf_counter()
                conditioning = renderer.capture_conditioning(snapshot, frame_started)
                conditioning_ms.update((perf_counter() - capture_started) * 1000.0)
                last_conditioning_capture = frame_started
                worker.publish(conditioning)
            assert conditioning is not None

            ai_version, published_ai = worker.generated.get()
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
                hold_previous_ai = False
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
                if config.reprojection and not hold_previous_ai:
                    reproject_frame = latest_ai
                    view_name = "AI REPROJECTED"
                else:
                    view_name = "AI HOLD" if hold_previous_ai else "AI RAW"

            now = perf_counter()
            if latest_ai is not None:
                frame_age_ms.update((now - latest_ai.generation_timestamp) * 1000.0)
            status = worker.status()
            stats = worker.stats()
            overlay: list[str] | None = None
            if overlay_enabled or prompt_editing or now < notice_until or status.state == "error" or status.state != "ready":
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
                    log_path,
                    prompt_editing,
                    prompt_buffer,
                    notice if now < notice_until else "",
                )
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
                )
            else:
                renderer.display(
                    display_image,
                    overlay,
                    sharpen=config.display_sharpen,
                    prompt_caption=config.prompt if prompt_caption_enabled else None,
                )
            display_rate.tick()
            displayed_frames += 1

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
                            hold_previous_ai = False
                            hide_proxy_until_ai = True
                        updated.prompt = apply_master_prefix(
                            load_master_prefix(prompt_library_path), updated.prompt
                        )
                        required_prompt_revision = worker.request_prompt(
                            updated.prompt, updated.negative_prompt
                        )
                        config.prompt = updated.prompt
                        config.negative_prompt = updated.negative_prompt
                        config.movement_speed = updated.movement_speed
                        config.sprint_multiplier = updated.sprint_multiplier
                        config.mouse_sensitivity = updated.mouse_sensitivity
                        config.reprojection = updated.reprojection
                        config.debug_overlay = updated.debug_overlay
                        overlay_enabled = bool(config.debug_overlay or debug_requested)
                        config.prompt_caption = updated.prompt_caption
                        prompt_caption_enabled = config.prompt_caption
                        config.reprojection_strength = updated.reprojection_strength
                        config.reprojection_max_translation = updated.reprojection_max_translation
                        config.reprojection_max_rotation = updated.reprojection_max_rotation
                        config.display_sharpen = updated.display_sharpen
                        if config.one_step_timestep != updated.one_step_timestep:
                            config.one_step_timestep = updated.one_step_timestep
                            worker.request_one_step_timestep(config.one_step_timestep)
                        if config.edge_softness != updated.edge_softness:
                            config.edge_softness = updated.edge_softness
                            worker.request_edge_softness(config.edge_softness)
                        if config.img2img_strength != updated.img2img_strength:
                            config.img2img_strength = updated.img2img_strength
                            worker.request_img2img_strength(config.img2img_strength)
                        if config.edge_strength != updated.edge_strength:
                            config.edge_strength = updated.edge_strength
                            worker.request_edge_strength(config.edge_strength)
                        if config.steps != updated.steps:
                            config.steps = updated.steps
                            worker.request_steps(config.steps)
                        if config.guidance_scale != updated.guidance_scale:
                            config.guidance_scale = updated.guidance_scale
                            worker.request_guidance_scale(config.guidance_scale)
                        if config.seed_mode != updated.seed_mode:
                            config.seed_mode = updated.seed_mode
                            worker.request_seed_mode(config.seed_mode)
                        if config.noise_persistence != updated.noise_persistence:
                            config.noise_persistence = updated.noise_persistence
                            worker.request_noise_persistence(config.noise_persistence)
                        config_mtime = current_mtime
                        logging.info("Hot-reloaded prompt and navigation settings")
                except Exception:
                    logging.exception("Config hot reload failed; retaining last valid values")

            if args.smoke_frames and displayed_frames >= args.smoke_frames:
                running = False
            clock.tick(config.target_display_fps)
        logging.info(
            "Session ended: %s displayed frames, %.1f display FPS, %.1f proxy FPS",
            displayed_frames,
            display_rate.fps,
            proxy_rate.fps,
        )
        return 0
    except Exception as exc:
        logging.exception("Fatal application error")
        print(f"Fatal application error: {exc}", file=sys.stderr)
        return 4
    finally:
        if worker is not None:
            worker.stop()
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
    log_path: Path,
    prompt_editing: bool = False,
    prompt_buffer: str = "",
    notice: str = "",
) -> list[str]:
    state = getattr(status, "state", "unknown")
    error = getattr(status, "error", "")
    active_resolution = getattr(status, "active_resolution", config.resolution_label)
    inference_ms = float(stats.get("inference_ms", 0.0))
    lines = [
        "FUNCTION KEYS  F1 overlay  F2 proxy/AI  F3 diagnostics",
        "               F4 freeze  F5 reprojection  F6 AI",
        "               F7 prompt  F10 resolution  F11 fullscreen",
        "",
        f"DISPLAY       {display_fps:6.1f} FPS     VIEW  {view_name}{' (FROZEN)' if frozen else ''}",
        f"PROXY         {proxy_fps:6.1f} FPS     RENDER {proxy_ms:6.2f} ms",
        f"CONDITIONING  {config.conditioning_fps:6d} FPS     CAPTURE {conditioning_ms:5.2f} ms",
        f"DIFFUSION     {float(stats.get('diffusion_fps', 0.0)):6.1f} FPS     INFER  {inference_ms:6.2f} ms",
        f"REPROJECT     {reproject_ms:6.2f} ms      MODE    {'ON' if config.reprojection else 'OFF'}",
        f"WARP AMOUNT   [{'#' * round(config.reprojection_strength * 10)}{'-' * (10 - round(config.reprojection_strength * 10))}] {config.reprojection_strength:4.0%}    [ / ] adjust",
        f"AI AGE        {ai_age_ms:6.1f} ms      VRAM   {float(stats.get('vram_allocated_gb', 0.0)):5.2f} GB",
        f"RES           {active_resolution}       F10 cycle       STEPS  {stats.get('steps', config.steps)}    - / = adjust",
        f"CFG           {float(stats.get('guidance_scale', config.guidance_scale)):4.2g}                  C cycle",
        f"SHARPNESS     {config.display_sharpen:4.1f}                  , / . adjust",
        f"STRUCTURE     {structure_lock_percent(config):4.0f}%                  K less / L more",
        f"EDGE SOFT     {config.edge_softness:4.2g} px                B cycle",
        f"GEO GUIDE     {config.edge_strength:4.0%}                  G cycle",
        f"BACKEND       {config.backend}       STATE  {state.upper()}",
        f"SEED          {stats.get('active_seed', config.seed)}        MODE {config.seed_mode.upper()}    N cycle mode",
        f"WORLD         {config.world_seed}        SPACE new world  Shift+R reseed  P edit prompt",
        f"PROMPT        {config.prompt[:58]}",
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
