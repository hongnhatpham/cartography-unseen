from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from time import perf_counter

ROOT_DEFAULT = Path(__file__).resolve().parents[1]


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

    backend = create_backend(config.backend)
    try:
        started = perf_counter()
        backend.load(config.backend_dict(root))
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
