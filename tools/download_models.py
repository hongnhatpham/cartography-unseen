from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", default="stabilityai/sd-turbo")
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    target = root / "models" / "sd_turbo"
    target.mkdir(parents=True, exist_ok=True)
    cache = root / "cache" / "huggingface"
    os.environ.update(
        {
            "HF_HOME": str(cache),
            "HUGGINGFACE_HUB_CACHE": str(cache / "hub"),
            "HF_HUB_OFFLINE": "0",
            "TRANSFORMERS_OFFLINE": "0",
            "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
            "HF_HUB_DISABLE_XET": "1",
        }
    )
    from huggingface_hub import snapshot_download

    resolved = snapshot_download(
        repo_id=args.repo,
        revision=args.revision,
        local_dir=target,
        local_dir_use_symlinks=False,
        allow_patterns=[
            "model_index.json",
            "scheduler/*",
            "tokenizer/*",
            "text_encoder/config.json",
            "text_encoder/model.fp16.safetensors",
            "unet/config.json",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.fp16.safetensors",
        ],
    )
    manifest = {
        "repository": args.repo,
        "revision": args.revision,
        "variant": "fp16",
        "resolved_path": str(Path(resolved).resolve()),
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
    }
    (target / "LOCAL_MODEL_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared {args.repo}@{args.revision} at {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
