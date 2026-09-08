"""Create an embedded-image SVG from a saved journey without running the installation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.journey import _export_svg


def export_journey_svg(archive: Path, output: Path | None = None) -> Path:
    """Read a completed PNG or WebP archive and write its optional SVG export."""
    archive = archive.resolve()
    data = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
    if not data.get("completion_reason"):
        raise ValueError("The journey is still being recorded; finish it before exporting.")
    for sample in data["images"]:
        image = (archive / sample["path"]).resolve()
        if not image.is_relative_to(archive) or image.suffix.lower() not in {".png", ".webp"}:
            raise ValueError("Journey images must be PNG or WebP files inside the archive.")
    output = output.resolve() if output else archive / "map.svg"
    if output.suffix.lower() != ".svg":
        raise ValueError("The output path must end in .svg.")
    output.parent.mkdir(parents=True, exist_ok=True)
    _export_svg(output, data, image_directory=archive)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Completed journey directory")
    parser.add_argument("--output", type=Path, help="Output SVG path, default: ARCHIVE/map.svg")
    args = parser.parse_args()
    try:
        print(export_journey_svg(args.archive, args.output))
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
