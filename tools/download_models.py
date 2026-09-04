"""Fetch every pinned model the offline app needs into ``models/``."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# One entry per model folder. Revisions are pinned so an exhibition machine
# rebuilt months later gets byte-identical weights.
MODELS: tuple[dict[str, Any], ...] = (
    {
        "name": "sd_turbo",
        "repo": "stabilityai/sd-turbo",
        "revision": "b261bac6fd2cf515557d5d0707481eafa0485ec2",
        "target": "models/sd_turbo",
        "allow_patterns": [
            "model_index.json",
            "scheduler/*",
            "tokenizer/*",
            "text_encoder/config.json",
            "text_encoder/model.fp16.safetensors",
            "unet/config.json",
            "unet/diffusion_pytorch_model.fp16.safetensors",
        ],
    },
    {
        # TAESD is the realtime encoder/decoder; the full VAE is never loaded.
        "name": "taesd",
        "repo": "madebyollin/taesd",
        "revision": "614f76814bbe30edbe2e627ace1c2234c81a2c0e",
        "target": "models/taesd",
        "allow_patterns": ["config.json", "diffusion_pytorch_model.safetensors"],
    },
)


def download(root: Path, model: dict[str, Any]) -> Path:
    """Snapshot one pinned model and write its provenance manifest."""
    from huggingface_hub import snapshot_download

    target = root / model["target"]
    target.mkdir(parents=True, exist_ok=True)
    resolved = snapshot_download(
        repo_id=model["repo"],
        revision=model["revision"],
        local_dir=target,
        local_dir_use_symlinks=False,
        allow_patterns=model["allow_patterns"],
    )
    manifest = {
        "repository": model["repo"],
        "revision": model["revision"],
        "resolved_path": str(Path(resolved).resolve()),
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
    }
    (target / "LOCAL_MODEL_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--only",
        action="append",
        choices=[model["name"] for model in MODELS],
        help="Download a subset by name; repeatable. Defaults to every model.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    os.environ.update(
        {
            "HF_HOME": str(root / "cache" / "huggingface"),
            "HUGGINGFACE_HUB_CACHE": str(root / "cache" / "huggingface" / "hub"),
            "HF_HUB_OFFLINE": "0",
            "TRANSFORMERS_OFFLINE": "0",
            "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
            "HF_HUB_DISABLE_XET": "1",
        }
    )
    wanted = set(args.only) if args.only else {model["name"] for model in MODELS}
    for model in MODELS:
        if model["name"] not in wanted:
            continue
        target = download(root, model)
        print(f"Prepared {model['repo']}@{model['revision']} at {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
