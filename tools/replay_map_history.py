"""Probe accumulated-map scaling with synthetic history and real AI traversal.

Pass --image an existing generated PNG. The same bitmap is reused for all 1,200
historical planes, so this measures history scaling, not diverse disk assets.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import shutil
import sys
from time import perf_counter, strftime
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.journey import JourneyRecorder
from tools.replay_performance import run


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--recorder-only", action="store_true")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error("--image must be an existing generated PNG")
    from PIL import Image
    with Image.open(args.image) as image:
        width, height = image.size
    original_init = JourneyRecorder.__init__

    def seeded(self, *values, **kwargs):
        original_init(self, *values, **kwargs)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.image, self.archive_dir / "prefill.png")
        now = perf_counter()
        positions = [[120 * math.sin(i * .013), 20 + 20 * math.sin(i * .007), -i * .1]
                     for i in range(12000)]
        self._data["segments"] = [dict(id="prefill", started=now-2000, ended=now-1,
            points=[dict(position=p, timestamp=now-2000+i/6) for i, p in enumerate(positions)])]
        self._data["images"] = [dict(id=f"prefill-{i}", path="prefill.png", segment_id="prefill",
            position=positions[i*10], rotation=[15, (i*.7) % 360, 0], timestamp=now-2000+i*1.6,
            generation_timestamp=now-2000+i*1.6, sequence=-i-1, width=width, height=height,
            prompt_revision=0, plane_width=8) for i in range(1200)]

    output = args.output or ROOT / "logs/performance" / f"history-{strftime('%Y%m%d-%H%M%S')}"
    with patch.object(JourneyRecorder, "__init__", seeded):
        return run(90, output.resolve(), ROOT / "config.json", 100,
                   map_check="recorder" if args.recorder_only else "active")


if __name__ == "__main__":
    raise SystemExit(cli())
