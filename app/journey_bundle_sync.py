"""Back up complete viewer archives through an authenticated Wrangler session."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Protocol
import zipfile
from app.monitoring import UploadProgress

from app.journey_sync import (
    Archive, ArchiveError, SyncResult, _files, _no_links, _relative_file,
    _remaining_files, _sha256, _write_receipt, inspect_archive,
)

PART_BYTES = 128 * 1024 ** 2
TEMP_RESERVE_BYTES = 64 * 1024 ** 2


class BundleTransport(Protocol):
    identity: dict[str, str]

    def put(self, key: str, path: Path) -> None: ...
    def get(self, key: str, path: Path) -> None: ...


class WranglerTransport:
    """Run the installed CLI without a shell; never print its credential diagnostics."""

    def __init__(self, bucket: str):
        if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket):
            raise ValueError("A valid R2 bucket name is required")
        self.identity = {"transport": "wrangler-zip", "bucket": bucket,
                         "account_id": os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")}
        executable = shutil.which("wrangler.cmd" if os.name == "nt" else "wrangler")
        if not executable:
            raise RuntimeError("Install Wrangler and run wrangler login before enabling uploads")
        if os.name == "nt":
            script = Path(executable).parent / "node_modules/wrangler/bin/wrangler.js"
            node = shutil.which("node")
            if not node or not script.is_file():
                raise RuntimeError("The installed Node.js or Wrangler entry point is missing")
            self.command = [node, str(script)]
        else:
            self.command = [executable]

    def _run(self, arguments: list[str]) -> None:
        environment = {**os.environ, "CI": "true", "WRANGLER_SEND_METRICS": "false"}
        if self.identity["account_id"]:
            environment["CLOUDFLARE_ACCOUNT_ID"] = self.identity["account_id"]
        try:
            result = subprocess.run(self.command + arguments, capture_output=True,
                                    stdin=subprocess.DEVNULL, env=environment, timeout=1200,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError("Wrangler could not finish; local originals were retained") from error
        if result.returncode:
            raise RuntimeError("Wrangler failed; check network access and run wrangler login if needed")

    def check_bucket(self) -> None:
        self._run(["r2", "bucket", "info", self.identity["bucket"]])

    def put(self, key: str, path: Path) -> None:
        # Wrangler's object upload endpoint supports at most 300 MiB per object.
        if path.stat().st_size > 300 * 1024 ** 2:
            raise ArchiveError("Bundle exceeds Wrangler's 300 MiB limit; retain it locally and use S3 sync")
        content_type = ("application/json" if key.endswith(".json") else
                        "application/zip" if key.endswith(".zip") else "application/octet-stream")
        self._run(["r2", "object", "put", f"{self.identity['bucket']}/{_relative_file(key)}",
                   "--remote", "--file", str(path), "--content-type", content_type])

    def get(self, key: str, path: Path) -> None:
        self._run(["r2", "object", "get", f"{self.identity['bucket']}/{_relative_file(key)}",
                   "--remote", "--file", str(path)])


class _PartWriter:
    """Stream an unseekable ZIP into one temporary part, uploading and verifying each."""

    def __init__(self, path: Path, transport: BundleTransport, prefix: str, progress=None):
        self.progress = progress or UploadProgress()
        self.path, self.transport, self.prefix = path, transport, prefix
        self.handle = path.open("wb")
        self.digest = hashlib.sha256()
        self.position = self.part_size = 0
        self.parts = []
        self.failed = False

    def tell(self):
        return self.position

    def seek(self, *args):
        raise OSError("ZIP output is a stream")

    def flush(self):
        if not self.handle.closed:
            self.handle.flush()

    def write(self, data):
        if self.failed:
            raise ArchiveError("Bundle upload was interrupted")
        try:
            view = memoryview(data)
            offset = 0
            while offset < len(view):
                chunk = view[offset:offset + PART_BYTES - self.part_size]
                self.handle.write(chunk)
                self.digest.update(chunk)
                self.part_size += len(chunk)
                self.position += len(chunk)
                offset += len(chunk)
                if self.part_size == PART_BYTES:
                    self._send()
            return len(data)
        except Exception:
            self.failed = True
            raise

    def _send(self, *, final=False):
        if not self.part_size:
            return
        self.handle.close()
        checksum = _sha256(self.path)
        key = (f"{self.prefix}/{checksum}.zip" if final and not self.parts else
               f"{self.prefix}/parts/{checksum}.part")
        self.progress.update(phase="uploading")
        self.transport.put(key, self.path)
        self.progress.add("completed_bytes", self.part_size)
        # Reuse the same pathname, so verification needs no second part on disk.
        self.progress.update(phase="verifying")
        self.transport.get(key, self.path)
        if self.path.stat().st_size != self.part_size or _sha256(self.path) != checksum:
            raise ArchiveError("Remote bundle part checksum mismatch; local originals were retained")
        self.parts.append({"key": key, "sha256": checksum, "size": self.part_size})
        self.progress.verified(self.part_size)
        self.progress.update(phase="packing")
        self.part_size = 0
        self.handle = self.path.open("wb")

    def finish(self):
        self._send(final=True)

    def close(self):
        self.handle.close()


def _bundle(archive: Archive, output: _PartWriter) -> None:
    """Stable ordering and metadata give identical archives the same object key."""
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in sorted(archive.inventory):
            source = archive.directory / name
            _no_links(source)
            entry = zipfile.ZipInfo(_relative_file(name), date_time=(1980, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = 0o100600 << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            digest, size = hashlib.sha256(), 0
            with source.open("rb") as incoming, bundle.open(entry, "w", force_zip64=True) as outgoing:
                while chunk := incoming.read(1024 * 1024):
                    outgoing.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            if {"sha256": digest.hexdigest(), "size": size} != archive.inventory[name]:
                raise ArchiveError("Local archive changed while creating its bundle")


def _receipt_path(root: Path, name: str) -> Path:
    path = root / ".sync" / f"bundle-{name}.json"
    _no_links(path)
    return path


def _read_receipt(root: Path, directory: Path, destination: dict) -> dict | None:
    path = _receipt_path(root, directory.name)
    if not path.exists():
        return None
    receipt = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version", "destination", "files", "key", "sha256", "parts"}
    if (not isinstance(receipt, dict) or set(receipt) not in (required, required | {"pruning"})
            or receipt.get("schema_version") != 1 or receipt.get("destination") != destination
            or ("pruning" in receipt and receipt["pruning"] is not True)):
        raise ArchiveError("Bundle receipt differs from the destination or has an invalid format")
    checksum = receipt["sha256"]
    if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise ArchiveError("Invalid bundle checksum")
    object_prefix = f"{destination['prefix']}/{directory.name}"
    if receipt["key"] != f"{object_prefix}/{checksum}.zip.json":
        raise ArchiveError("Bundle receipt key differs from this archive")
    parts = receipt["parts"]
    if not isinstance(parts, list) or not parts:
        raise ArchiveError("Invalid bundle parts")
    for part in parts:
        if (not isinstance(part, dict) or set(part) != {"key", "sha256", "size"}
                or type(part["size"]) is not int or not 0 < part["size"] <= PART_BYTES
                or not isinstance(part["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", part["sha256"])):
            raise ArchiveError("Invalid bundle part checksum")
        expected = {f"{object_prefix}/parts/{part['sha256']}.part"}
        if len(parts) == 1:
            expected.add(f"{object_prefix}/{checksum}.zip")
        if part["key"] not in expected:
            raise ArchiveError("Invalid bundle part key")
    if len(parts) == 1 and parts[0]["sha256"] != checksum:
        raise ArchiveError("Bundle and single part checksum differ")
    inventory = receipt["files"]
    if not isinstance(inventory, dict) or not {"manifest.json", "complete.json", "index.html"} <= inventory.keys():
        raise ArchiveError("Invalid bundle inventory")
    for name, item in inventory.items():
        _relative_file(name)
        if name not in ("manifest.json", "complete.json", "index.html", "map.svg") and not name.startswith("images/"):
            raise ArchiveError("Unknown file in bundle inventory")
        if (not isinstance(item, dict) or set(item) != {"size", "sha256"}
                or type(item["size"]) is not int or item["size"] < 0
                or not isinstance(item["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])):
            raise ArchiveError("Invalid bundle file checksum")
    return receipt


def _verify(transport: BundleTransport, receipt: dict, temporary: Path, progress=None) -> None:
    progress = progress or UploadProgress()
    digest = hashlib.sha256()
    for part in receipt["parts"]:
        progress.update(phase="verifying")
        transport.get(part["key"], temporary)
        if temporary.stat().st_size != part["size"] or _sha256(temporary) != part["sha256"]:
            raise ArchiveError("Remote bundle part checksum mismatch; local originals were retained")
        progress.verified(part["size"])
        with temporary.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    if digest.hexdigest() != receipt["sha256"]:
        raise ArchiveError("Remote bundle checksum mismatch; local originals were retained")


def _descriptor(receipt: dict) -> dict:
    return {key: value for key, value in receipt.items() if key != "pruning"}


def _verify_descriptor(transport: BundleTransport, receipt: dict, temporary: Path) -> int:
    transport.get(receipt["key"], temporary)
    if json.loads(temporary.read_text(encoding="utf-8")) != _descriptor(receipt):
        raise ArchiveError("Remote bundle descriptor differs from its receipt")
    return temporary.stat().st_size


def _upload(root: Path, archive: Archive, transport: BundleTransport, destination: dict,
            receipt: dict | None, progress=None) -> dict:
    progress = progress or UploadProgress()
    if receipt is not None:
        if receipt["files"] != archive.inventory:
            raise ArchiveError("Local archive differs from its verified bundle receipt")
        return receipt
    with tempfile.TemporaryDirectory(prefix="journey-bundle-") as temporary:
        part_path = Path(temporary) / "archive.part"
        if part_path.is_relative_to(root):
            raise ArchiveError("Temporary bundle directory must be outside the journey root")
        if shutil.disk_usage(temporary).free < PART_BYTES + TEMP_RESERVE_BYTES:
            raise ArchiveError("Insufficient temporary space for one bundle part and 64 MiB reserve; local originals were retained")
        object_prefix = f"{destination['prefix']}/{archive.directory.name}"
        progress.update(phase="packing")
        output = _PartWriter(part_path, transport, object_prefix, progress)
        try:
            _bundle(archive, output)
            output.finish()
        finally:
            output.close()
        checksum = output.digest.hexdigest()
        receipt = {"schema_version": 1, "destination": destination, "files": archive.inventory,
                   "key": f"{object_prefix}/{checksum}.zip.json",
                   "sha256": checksum, "parts": output.parts}
        if inspect_archive(root, archive.directory).inventory != archive.inventory:
            raise ArchiveError("Local archive changed during upload")
        descriptor = Path(temporary) / "descriptor.json"
        descriptor.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
        # Publish discovery metadata only after every part verifies.
        descriptor_size = descriptor.stat().st_size
        progress.update(phase="uploading")
        transport.put(receipt["key"], descriptor)
        progress.add("completed_bytes", descriptor_size)
        progress.update(phase="verifying")
        verified_size = _verify_descriptor(transport, receipt, descriptor)
        progress.verified(verified_size)
    if inspect_archive(root, archive.directory).inventory != archive.inventory:
        raise ArchiveError("Local archive changed during upload")
    _write_receipt(_receipt_path(root, archive.directory.name), receipt)
    progress.committed()
    return receipt


def _prune(root: Path, archive: Archive, transport: BundleTransport, receipt: dict, progress=None) -> None:
    """Verify the complete remote bundle again before deleting a known local subset."""
    progress = progress or UploadProgress()
    partial = receipt.get("pruning", False)
    _remaining_files(root, archive, partial=partial)
    with tempfile.TemporaryDirectory(prefix="journey-verify-") as temporary:
        _verify(transport, receipt, Path(temporary) / "download.part", progress)
        verified_size = _verify_descriptor(transport, receipt, Path(temporary) / "descriptor.json")
        progress.verified(verified_size)
    progress.committed()
    remaining = _remaining_files(root, archive, partial=partial)
    _write_receipt(_receipt_path(root, archive.directory.name), {**receipt, "pruning": True})
    progress.update(phase="pruning")
    for path in remaining:
        _no_links(path)
        if not path.resolve().is_relative_to(archive.directory.resolve()):
            raise ArchiveError("File escaped its archive")
        if _sha256(path) != archive.inventory[path.relative_to(archive.directory).as_posix()]["sha256"]:
            raise ArchiveError("Local file changed before removal")
        path.unlink()
    # Never recursively delete: unexpected files prevent directory removal.
    directories = sorted((path for path in archive.directory.rglob("*") if path.is_dir()),
                         key=lambda path: len(path.parts), reverse=True)
    for path in directories:
        _no_links(path)
        path.rmdir()
    archive.directory.rmdir()


def sync_bundles_once(root: Path, transport: BundleTransport, *, prefix: str = "journeys",
                      cache_bytes: int = 5 * 1024 ** 3, prune: bool = False, progress=None) -> SyncResult:
    """Upload completed journeys and bound verified local storage. Caller holds sync_lock."""
    if not math.isfinite(cache_bytes) or cache_bytes < 0:
        raise ValueError("Cache size must be finite and nonnegative")
    destination = {**transport.identity, "prefix": _relative_file(prefix)}
    root = Path(root).absolute()
    _no_links(root)
    root.mkdir(parents=True, exist_ok=True)
    progress = progress or UploadProgress()
    progress.begin(root)
    result, available = SyncResult(), []
    for directory in sorted(root.iterdir()):
        if directory.name == ".sync":
            continue
        validated = completed = False
        try:
            _no_links(directory)
            if not directory.is_dir():
                result.local_bytes += directory.stat().st_size
                continue
            completed = (directory / "complete.json").exists()
            result.local_bytes += sum(path.stat().st_size for path in _files(directory))
            receipt = _read_receipt(root, directory, destination)
            if receipt and receipt.get("pruning"):
                validated = True
                if not prune:
                    raise ArchiveError("Interrupted pruning is pending; rerun with --prune to finish")
                archive = Archive(directory, 0, receipt["files"])
                remaining_size = sum(path.stat().st_size for path in _remaining_files(root, archive, partial=True))
                _prune(root, archive, transport, receipt, progress)
                result.local_bytes -= remaining_size
                result.pruned.append(directory.name)
                progress.resumed_prune_done(completed)
                continue
            if not (directory / "complete.json").exists():
                continue
            archive = inspect_archive(root, directory)
            validated = True
            existing = receipt is not None
            receipt = _upload(root, archive, transport, destination, receipt, progress)
            progress.archive_done()
            if not existing:
                result.uploaded.append(directory.name)
            available.append((archive, receipt))
        except Exception as error:
            progress.failed(invalid=not validated, completed=completed)
            result.errors.append(f"{directory.name}: {error}")
    if prune:
        for archive, receipt in sorted(available, key=lambda item: (item[0].ended, item[0].directory.name)):
            if result.local_bytes <= cache_bytes:
                break
            try:
                _prune(root, archive, transport, receipt, progress)
                result.local_bytes -= archive.size
                result.pruned.append(archive.directory.name)
                progress.add("pruned_archives", 1)
                progress.add("verified_archives", -1)
            except Exception as error:
                progress.failed()
                result.errors.append(f"{archive.directory.name}: {error}")
    progress.update(phase="error" if result.errors else "idle", local_bytes=result.local_bytes)
    return result
