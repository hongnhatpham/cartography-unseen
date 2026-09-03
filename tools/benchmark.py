from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import AppConfig, RESOLUTION_MODES, VALID_RESOLUTIONS, configure_local_environment
from app.diffusion.factory import create_backend
from app.renderer.camera import Camera
from app.types import ConditioningFrame


def make_conditioning(
    size: int | tuple[int, int], sequence: int = 0
) -> ConditioningFrame:
    width, height = (size, size) if isinstance(size, int) else size
    yy, xx = np.mgrid[0:height, 0:width]
    checker = ((xx // 48 + yy // 48) % 2).astype(np.float32)
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.clip(48 + xx * 145 / width + checker * 28, 0, 255)
    rgb[:, :, 1] = np.clip(55 + yy * 135 / height, 0, 255)
    rgb[:, :, 2] = np.clip(155 - checker * 52, 0, 255)
    depth = np.clip(0.15 + yy / height * 0.8, 0, 1).astype(np.float32)
    edges = np.zeros((height, width), dtype=np.uint8)
    edges[:, 47::48] = 255
    edges[47::48, :] = 255
    return ConditioningFrame(rgb, depth, edges, Camera.create_default().snapshot(), perf_counter(), sequence)


def benchmark_one(
    config: AppConfig,
    resolution: int | tuple[int, int],
    warmup: int,
    frames: int,
) -> dict[str, object]:
    width, height = (resolution, resolution) if isinstance(resolution, int) else resolution
    config.diffusion_resolution = width if width == height else f"{width}x{height}"
    config.warmup_passes = warmup
    backend = create_backend(config.backend)
    result: dict[str, object] = {
        "backend": config.backend,
        "resolution": f"{width}x{height}",
        "steps": config.steps,
        "guidance_scale": config.guidance_scale,
        "status": "failed",
    }
    try:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            torch = None  # type: ignore[assignment]
        load_started = perf_counter()
        backend.load(config.backend_dict(ROOT))
        result["backend_load_ms"] = (perf_counter() - load_started) * 1000.0
        warm_started = perf_counter()
        backend.warmup()
        result["warmup_total_ms"] = (perf_counter() - warm_started) * 1000.0

        timings: list[float] = []
        vram_samples: list[float] = []
        previous = None
        for index in range(frames):
            conditioning = make_conditioning((width, height), index)
            started = perf_counter()
            previous = backend.generate(conditioning, previous_frame=previous)
            if torch is not None and torch.cuda.is_available():
                torch.cuda.synchronize()
            timings.append((perf_counter() - started) * 1000.0)
            vram_samples.append(float(backend.stats().get("vram_allocated_gb", 0.0)))
        backend_stats = backend.stats()
        average_ms = statistics.fmean(timings)
        result.update(
            {
                "status": "ok",
                "warm_inference_ms": average_ms,
                "median_inference_ms": statistics.median(timings),
                "min_inference_ms": min(timings),
                "max_inference_ms": max(timings),
                "fps": 1000.0 / average_ms,
                "first_frame_ms": backend_stats.get("first_frame_ms", timings[0]),
                "peak_vram_gb": backend_stats.get("peak_vram_gb", 0.0),
                "average_vram_gb": statistics.fmean(vram_samples),
                "samples": frames,
                "backend_stats": backend_stats,
            }
        )
    except Exception as exc:
        result["error"] = str(exc)
    finally:
        backend.unload()
    return result


def write_text(results: dict[str, object], path: Path) -> None:
    lines = [
        "REALTIME DIFFUSION ART - BENCHMARK RESULTS",
        f"GPU: {results.get('gpu', 'unknown')}",
        f"Backend: {results['backend']}",
        "",
    ]
    for item in results["results"]:  # type: ignore[index]
        if item["status"] == "ok":
            lines.append(
                f"{item['resolution']}  CFG {item['guidance_scale']:.2g}  "
                f"{item['warm_inference_ms']:.2f} ms  "
                f"{item['fps']:.2f} FPS  peak VRAM {item['peak_vram_gb']:.2f} GB"
            )
        else:
            lines.append(
                f"{item['resolution']}  CFG {item['guidance_scale']:.2g}  "
                f"FAILED  {item.get('error', '')}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("sd_turbo_stream", "proxy_passthrough"))
    parser.add_argument(
        "--resolutions",
        nargs="+",
        default=[str(value) for value in VALID_RESOLUTIONS],
        help="square sizes or WIDTHxHEIGHT modes",
    )
    parser.add_argument("--guidance-scales", nargs="+", type=float)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    resolutions: list[int | tuple[int, int]] = []
    for value in args.resolutions:
        try:
            if "x" in value.lower():
                width_text, height_text = value.lower().split("x", 1)
                size = (int(width_text), int(height_text))
                if size not in RESOLUTION_MODES:
                    raise ValueError
                resolutions.append(size)
            else:
                size = int(value)
                if size not in VALID_RESOLUTIONS:
                    raise ValueError
                resolutions.append(size)
        except ValueError:
            parser.error(f"unsupported resolution: {value}")
    configure_local_environment(ROOT, offline=True)
    config = AppConfig.load(ROOT / "config.json")
    if args.backend:
        config.backend = args.backend
    guidance_scales = args.guidance_scales or [config.guidance_scale]
    if any(not 0.0 <= value <= 4.0 for value in guidance_scales):
        parser.error("guidance scales must be in [0, 4]")
    gpu = "unavailable"
    try:
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    payload: dict[str, object] = {
        "backend": config.backend,
        "gpu": gpu,
        "config": asdict(config),
        "results": [],
    }
    for guidance_scale in guidance_scales:
        config.guidance_scale = guidance_scale
        for resolution in resolutions:
            payload["results"].append(
                benchmark_one(config, resolution, args.warmup, args.frames)
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "benchmark_results.json"
    text_path = args.output_dir / "benchmark_results.txt"
    json_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    write_text(payload, text_path)
    print(text_path.read_text(encoding="utf-8"))
    return 0 if all(item["status"] == "ok" for item in payload["results"]) else 1  # type: ignore[index]


if __name__ == "__main__":
    raise SystemExit(main())
