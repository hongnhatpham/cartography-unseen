from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from time import perf_counter

ROOT_DEFAULT = Path(__file__).resolve().parents[1]

# Files the realtime path opens. The full VAE is deliberately absent: TAESD
# does both the encode and the decode.
REQUIRED_FILES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "model_path",
        "models/sd_turbo",
        (
            "unet/config.json",
            "text_encoder/config.json",
            "tokenizer/vocab.json",
            "scheduler/scheduler_config.json",
        ),
    ),
    (
        "taesd_path",
        "models/taesd",
        ("config.json", "diffusion_pytorch_model.safetensors"),
    ),
)


def missing_model_files(root: Path, settings: dict[str, object]) -> list[str]:
    """List the pinned model files a first run must have downloaded."""
    missing: list[str] = []
    for key, default, names in REQUIRED_FILES:
        folder = Path(str(settings.get(key, default)))
        if not folder.is_absolute():
            folder = root / folder
        missing.extend(str(folder / name) for name in names if not (folder / name).exists())
    return missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--backend")
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    from app.config import AppConfig, configure_local_environment

    configure_local_environment(root, offline=True)
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    config = AppConfig.load(root / "config.json")
    if args.backend:
        config.backend = args.backend
    from app.diffusion.factory import create_backend
    from tools.benchmark import make_conditioning

    settings = config.backend_dict(root)
    missing = missing_model_files(root, settings)
    if missing:
        print(
            "OFFLINE VERIFY FAILED: missing model files: " + ", ".join(missing),
            file=sys.stderr,
        )
        return 1

    backend = create_backend(config.backend)
    try:
        started = perf_counter()
        backend.load(settings)
        backend.warmup()
        width, height = config.diffusion_size
        output = backend.generate(make_conditioning((width, height)))
        if output.shape[:2] != (height, width):
            raise RuntimeError(f"Unexpected generated shape: {output.shape}")
        elapsed = perf_counter() - started
        print(f"OFFLINE VERIFY OK: {config.backend}, {output.shape}, {elapsed:.2f}s")
        return 0
    except Exception as exc:
        print(f"OFFLINE VERIFY FAILED: {exc}", file=sys.stderr)
        return 1
    finally:
        backend.unload()


if __name__ == "__main__":
    raise SystemExit(main())
