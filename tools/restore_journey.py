"""Download a Wrangler journey backup, verify its parts, and restore its viewer."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.journey_storage import sync_environment
from app.journey_sync import ArchiveError, _no_links, _relative_file, _sha256, inspect_archive
from app.journey_bundle_sync import WranglerTransport


def restore_bundle(transport, key: str, output_root: Path) -> Path:
    """Restore a new directory only; existing archives are never overwritten."""
    _relative_file(key)
    fields = key.split("/")
    if len(fields) < 3 or not re.fullmatch(r"[0-9a-f]{64}\.zip\.json", fields[-1]):
        raise ArchiveError("Use the backup descriptor key ending in <sha256>.zip.json")
    archive_id, prefix = fields[-2], "/".join(fields[:-2])
    output_root = output_root.absolute()
    target = output_root / archive_id
    _no_links(target)
    if target.exists():
        raise ArchiveError("Restore destination already exists; choose a different output root")
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".journey-restore-", dir=output_root) as temporary:
        work = Path(temporary)
        descriptor = work / "bundle.json"
        transport.get(key, descriptor)
        data = json.loads(descriptor.read_text(encoding="utf-8"))
        expected_destination = {**transport.identity, "prefix": prefix}
        if (data.get("schema_version") != 1 or data.get("key") != key or
                data.get("sha256") != fields[-1].removesuffix(".zip.json") or
                data.get("destination") != expected_destination):
            raise ArchiveError("Backup descriptor does not match this destination")
        inventory = data["files"]
        if not isinstance(inventory, dict) or not {"manifest.json", "index.html", "complete.json"} <= inventory.keys():
            raise ArchiveError("Backup has no complete viewer inventory")
        for name, item in inventory.items():
            _relative_file(name)
            if (name not in {"manifest.json", "index.html", "complete.json", "map.svg"} and
                    not name.startswith("images/")):
                raise ArchiveError("Unrecognized backup file")
            if (type(item.get("size")) is not int or item["size"] < 0 or
                    not re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", ""))):
                raise ArchiveError("Invalid backup file checksum")
        if not isinstance(data["parts"], list) or not data["parts"]:
            raise ArchiveError("Backup has no ZIP parts")
        archive_path = work / "archive.zip"
        whole = hashlib.sha256()
        with archive_path.open("wb") as combined:
            for part in data["parts"]:
                checksum = part["sha256"]
                if not re.fullmatch(r"[0-9a-f]{64}", checksum):
                    raise ArchiveError("Invalid ZIP part checksum")
                allowed = {f"{prefix}/{archive_id}/parts/{checksum}.part",
                           f"{prefix}/{archive_id}/{checksum}.zip"}
                if part["key"] not in allowed or type(part["size"]) is not int or part["size"] <= 0:
                    raise ArchiveError("Invalid ZIP part path or size")
                download = work / "part"
                transport.get(part["key"], download)
                if download.stat().st_size != part["size"] or _sha256(download) != checksum:
                    raise ArchiveError("Downloaded ZIP part checksum mismatch")
                with download.open("rb") as source:
                    while chunk := source.read(1024 * 1024):
                        combined.write(chunk)
                        whole.update(chunk)
                download.unlink()
        if whole.hexdigest() != data["sha256"]:
            raise ArchiveError("Reassembled ZIP checksum mismatch")
        staged = work / archive_id
        staged.mkdir()
        with zipfile.ZipFile(archive_path) as bundle:
            names = bundle.namelist()
            if len(names) != len(set(names)) or set(names) != inventory.keys():
                raise ArchiveError("ZIP entries do not match the backup inventory")
            for name in names:
                entry = bundle.getinfo(name)
                if entry.file_size != inventory[name]["size"]:
                    raise ArchiveError("ZIP entry size does not match the backup inventory")
                path = staged / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(name) as incoming, path.open("xb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
                if _sha256(path) != inventory[name]["sha256"]:
                    raise ArchiveError("Restored file checksum mismatch")
        inspect_archive(work, staged)
        _no_links(target)
        if target.exists():
            raise ArchiveError("Restore destination appeared during download; refusing to overwrite it")
        staged.rename(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("key", help="R2 descriptor key, journeys/<id>/<sha256>.zip.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "restored-journeys")
    args = parser.parse_args()
    os.environ.update(sync_environment(ROOT))
    try:
        transport = WranglerTransport(os.environ.get("JOURNEY_S3_BUCKET", ""))
        print(restore_bundle(transport, args.key, args.output_root))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f"Restore failed: {error}\n")


if __name__ == "__main__":
    main()
