"""Upload immutable completed journeys and optionally prune verified local copies."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import math
import mimetypes
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Protocol
from urllib.parse import urlsplit
from app.monitoring import UploadProgress


class ArchiveError(ValueError):
    """An archive or receipt is unsafe to synchronize."""


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return _stream_sha256(source)


def _stream_sha256(source) -> str:
    digest = hashlib.sha256()
    while chunk := source.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def _no_links(path: Path) -> None:
    """Reject symlinks and Windows reparse points, including junctions."""
    for component in (path, *path.parents):
        if not component.exists() and not component.is_symlink():
            continue
        info = component.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ArchiveError(f"Links and junctions are not supported: {component}")


def _relative_file(name: str) -> str:
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ArchiveError(f"Invalid archive path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in name.split("/")):
        raise ArchiveError(f"Invalid archive path: {name!r}")
    return name


def _files(directory: Path, *, allow_missing: bool = False) -> list[Path]:
    _no_links(directory)
    found = []
    for path in directory.iterdir():
        try:
            _no_links(path)
            if path.is_dir():
                found.extend(_files(path, allow_missing=allow_missing))
            elif path.is_file():
                found.append(path)
            elif not (allow_missing and not path.exists()):
                raise ArchiveError(f"Unsupported archive entry: {path}")
        except FileNotFoundError:
            if not allow_missing:
                raise
    return found


def _directory_size(directory: Path, *, mutable: bool = False) -> int:
    """Estimate active storage despite atomic temp-file renames; finalized data is strict."""
    total = 0
    for path in _files(directory, allow_missing=mutable):
        try:
            total += path.stat().st_size
        except FileNotFoundError:
            if not mutable:
                raise
    return total


@dataclass(frozen=True)
class Archive:
    directory: Path
    ended: float
    inventory: dict[str, dict]

    @property
    def size(self) -> int:
        return sum(item["size"] for item in self.inventory.values())


def inspect_archive(root: Path, directory: Path) -> Archive:
    """Only completed, self-contained archives with a matching manifest qualify."""
    _no_links(directory)
    if directory.resolve().parent != root.resolve():
        raise ArchiveError("Archive must be an immediate child of the journey root")
    actual = {path.relative_to(directory).as_posix(): path for path in _files(directory)}
    marker = json.loads(actual["complete.json"].read_text(encoding="utf-8"))
    manifest_path = actual["manifest.json"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if marker.get("schema_version") != 1 or marker.get("manifest_sha256") != _sha256(manifest_path):
        raise ArchiveError("Completion marker does not match the manifest")
    if manifest.get("id") != directory.name or not manifest.get("ended_at"):
        raise ArchiveError("Manifest must name this completed archive")
    ended = datetime.fromisoformat(manifest["ended_at"].replace("Z", "+00:00"))
    if ended.tzinfo is None:
        raise ArchiveError("Archive end time must include a timezone")
    expected = {"manifest.json", "index.html", "complete.json"}
    for descriptor in manifest["images"]:
        name = _relative_file(descriptor["path"])
        if not name.startswith("images/"):
            raise ArchiveError("Captured images must be inside images/")
        expected.add(name)
    if "map.svg" in actual:
        expected.add("map.svg")
    if expected != actual.keys():
        raise ArchiveError("Archive has missing or unrecognized files; preserve it locally")
    inventory = {name: {"sha256": _sha256(path), "size": path.stat().st_size}
                 for name, path in sorted(actual.items())}
    return Archive(directory, ended.timestamp(), inventory)


class ObjectStore(Protocol):
    identity: dict[str, str]

    def head_sha256(self, key: str) -> str | None: ...
    def upload(self, key: str, path: Path, sha256: str) -> None: ...
    def verify(self, key: str, sha256: str) -> None: ...


class S3Store:
    """S3-compatible transport. Credentials stay in boto3's normal provider chain."""

    def __init__(self, client, endpoint: str, bucket: str, prefix: str = "journeys"):
        parsed = urlsplit(endpoint)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("JOURNEY_S3_ENDPOINT must be an HTTPS URL without credentials")
        if parsed.query or parsed.fragment or parsed.path not in ("", "/"):
            raise ValueError("JOURNEY_S3_ENDPOINT must be an HTTPS service origin")
        if not bucket or "/" in bucket or "\\" in bucket or ":" in bucket:
            raise ValueError("JOURNEY_S3_BUCKET must be a bucket name")
        prefix = _relative_file(prefix)
        self.client = client
        self.identity = {"endpoint": endpoint.rstrip("/"), "bucket": bucket, "prefix": prefix}

    @classmethod
    def from_env(cls) -> S3Store:
        endpoint = os.environ.get("JOURNEY_S3_ENDPOINT", "")
        bucket = os.environ.get("JOURNEY_S3_BUCKET", "")
        prefix = os.environ.get("JOURNEY_S3_PREFIX", "journeys")
        # Validate before constructing a client or searching for credentials.
        store = cls(None, endpoint, bucket, prefix)
        try:
            import boto3
            from botocore.config import Config
        except ImportError as error:
            raise RuntimeError("Install requirements-storage.txt to enable journey uploads") from error
        store.client = boto3.client("s3", endpoint_url=endpoint, region_name="auto",
                                   config=Config(connect_timeout=10, read_timeout=60,
                                                 request_checksum_calculation="when_required",
                                                 response_checksum_validation="when_required",
                                                 retries={"max_attempts": 3, "mode": "standard"}))
        return store

    def _args(self, key: str) -> dict:
        return {"Bucket": self.identity["bucket"], "Key": key}

    def head_sha256(self, key: str) -> str | None:
        try:
            response = self.client.head_object(**self._args(key))
        except Exception as error:
            code = getattr(error, "response", {}).get("Error", {}).get("Code")
            if code in ("404", "NoSuchKey", "NotFound"):
                return None
            raise
        # An existing object without our metadata is a conflict, never a missing file.
        return response.get("Metadata", {}).get("sha256", "")

    def upload(self, key: str, path: Path, sha256: str) -> None:
        if path.stat().st_size > 5 * 1024 ** 3:
            raise ArchiveError("Individual files over 5 GiB require multipart upload")
        content_type = {".webp": "image/webp", ".png": "image/png", ".svg": "image/svg+xml",
                        ".html": "text/html; charset=utf-8", ".json": "application/json"}.get(
                            path.suffix.lower(), mimetypes.guess_type(path.name)[0] or
                            "application/octet-stream")
        with path.open("rb") as source:
            self.client.put_object(**self._args(key), Body=source, Metadata={"sha256": sha256},
                                   ContentType=content_type, IfNoneMatch="*")

    def verify(self, key: str, sha256: str) -> None:
        response = self.client.get_object(**self._args(key))
        source = response["Body"]
        try:
            actual = _stream_sha256(source)
        finally:
            source.close()
        if actual != sha256:
            raise ArchiveError(f"Remote checksum mismatch: {key}")


def _key(store: ObjectStore, archive: Archive, name: str) -> str:
    return f"{store.identity['prefix']}/{archive.directory.name}/{name}"


def _receipt_path(root: Path, archive: Archive) -> Path:
    path = root / ".sync" / f"{archive.directory.name}.json"
    _no_links(path)
    return path


def _receipt(store: ObjectStore, archive: Archive) -> dict:
    return {"schema_version": 1, "destination": store.identity, "files": archive.inventory}


def _write_receipt(path: Path, data: dict) -> None:
    _no_links(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    _no_links(temporary)
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(data, output, sort_keys=True)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def upload_archive(root: Path, archive: Archive, store: ObjectStore, progress=None) -> None:
    progress = progress or UploadProgress()
    path = _receipt_path(root, archive)
    expected = _receipt(store, archive)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise ArchiveError("Existing receipt differs from the destination or local files")
        return
    # Publish the manifest and completion marker only after all viewer assets verify.
    names = sorted(archive.inventory, key=lambda name: (name in ("manifest.json", "complete.json"),
                                                        name == "complete.json", name))
    for name in names:
        key, checksum = _key(store, archive, name), archive.inventory[name]["sha256"]
        existing = store.head_sha256(key)
        if existing is None:
            progress.update(phase="uploading")
            store.upload(key, archive.directory / name, checksum)
            progress.add("completed_bytes", archive.inventory[name]["size"])
        elif existing != checksum:
            raise ArchiveError(f"Remote object conflicts with this archive: {key}")
        progress.update(phase="verifying")
        store.verify(key, checksum)
        progress.verified(archive.inventory[name]["size"])
    if inspect_archive(root, archive.directory).inventory != archive.inventory:
        raise ArchiveError("Local archive changed during upload")
    _write_receipt(path, expected)
    progress.committed()


def _remaining_files(root: Path, archive: Archive, *, partial: bool) -> list[Path]:
    if archive.directory.resolve().parent != root.resolve():
        raise ArchiveError("Archive escaped the journey root")
    actual = {path.relative_to(archive.directory).as_posix(): path
              for path in _files(archive.directory)}
    if not actual.keys() <= archive.inventory.keys() or (not partial and actual.keys() != archive.inventory.keys()):
        raise ArchiveError("Archive has missing or unrecognized files; refusing to remove it")
    for name, path in actual.items():
        if path.stat().st_size != archive.inventory[name]["size"] or _sha256(path) != archive.inventory[name]["sha256"]:
            raise ArchiveError("Local file changed; refusing to remove it")
    return [actual[name] for name in sorted(actual)]


def _pending_prune(root: Path, directory: Path, store: ObjectStore) -> Archive | None:
    path = root / ".sync" / f"{directory.name}.json"
    _no_links(path)
    if not path.exists():
        return None
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if not receipt.get("pruning"):
        return None
    if (receipt.get("schema_version") != 1 or receipt.get("destination") != store.identity
            or receipt.get("pruning") is not True):
        raise ArchiveError("Pending prune receipt differs from this destination")
    inventory = receipt["files"]
    if not isinstance(inventory, dict) or not {"manifest.json", "complete.json", "index.html"} <= inventory.keys():
        raise ArchiveError("Invalid pending prune inventory")
    for name, item in inventory.items():
        _relative_file(name)
        if name not in ("manifest.json", "complete.json", "index.html", "map.svg") and not name.startswith("images/"):
            raise ArchiveError("Invalid pending prune path")
        if (not isinstance(item, dict) or set(item) != {"size", "sha256"}
                or not isinstance(item["size"], int) or item["size"] < 0
                or not isinstance(item["sha256"], str) or len(item["sha256"]) != 64
                or any(char not in "0123456789abcdef" for char in item["sha256"])):
            raise ArchiveError("Invalid pending prune checksum inventory")
    archive = Archive(directory, 0, inventory)
    if receipt != {**_receipt(store, archive), "pruning": True}:
        raise ArchiveError("Invalid pending prune receipt")
    return archive


def prune_archive(root: Path, archive: Archive, store: ObjectStore, progress=None) -> None:
    """Recheck every local and remote byte before removing only the known files."""
    progress = progress or UploadProgress()
    receipt_path = _receipt_path(root, archive)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = _receipt(store, archive)
    pruning = {**expected, "pruning": True}
    partial = receipt == pruning
    if receipt != expected and not partial:
        raise ArchiveError("Receipt differs from the archive or destination")
    _remaining_files(root, archive, partial=partial)
    for name, item in archive.inventory.items():
        progress.update(phase="verifying")
        store.verify(_key(store, archive, name), item["sha256"])
        progress.verified(item["size"])
    progress.committed()
    remaining = _remaining_files(root, archive, partial=partial)
    # Persist intent before the first unlink, so crashes can resume a known subset.
    _write_receipt(receipt_path, pruning)
    # No recursive deletion. Unknown additions make rmdir fail and remain untouched.
    progress.update(phase="pruning")
    for path in remaining:
        name = path.relative_to(archive.directory).as_posix()
        _no_links(path)
        if not path.resolve().is_relative_to(archive.directory.resolve()):
            raise ArchiveError("File escaped its archive")
        if _sha256(path) != archive.inventory[name]["sha256"]:
            raise ArchiveError("Local file changed before removal")
        path.unlink()
    directories = sorted((path for path in archive.directory.rglob("*") if path.is_dir()),
                         key=lambda path: len(path.parts), reverse=True)
    for path in directories:
        _no_links(path)
        path.rmdir()
    archive.directory.rmdir()


@dataclass
class SyncResult:
    uploaded: list[str] = field(default_factory=list)
    pruned: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    local_bytes: int = 0


def sync_once(root: Path, store: ObjectStore, *, cache_bytes: int = 5 * 1024 ** 3,
              prune: bool = False, progress=None) -> SyncResult:
    """Synchronize completed archives. Caller must hold sync_lock for this root."""
    if not math.isfinite(cache_bytes) or cache_bytes < 0:
        raise ValueError("Cache size must be finite and nonnegative")
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
            result.local_bytes += _directory_size(directory, mutable=not completed)
            pending = _pending_prune(root, directory, store)
            if pending is not None:
                validated = True
                if not prune:
                    raise ArchiveError("Interrupted pruning is pending; rerun with --prune to finish")
                remaining_size = sum(path.stat().st_size for path in _remaining_files(root, pending, partial=True))
                prune_archive(root, pending, store, progress)
                result.local_bytes -= remaining_size
                result.pruned.append(directory.name)
                progress.resumed_prune_done(completed)
                continue
            if not (directory / "complete.json").exists():
                continue
            archive = inspect_archive(root, directory)
            validated = True
            upload_archive(root, archive, store, progress)
            progress.archive_done()
            result.uploaded.append(directory.name)
            available.append(archive)
        except Exception as error:
            progress.failed(invalid=not validated, completed=completed)
            result.errors.append(f"{directory.name}: {error}")
    if prune:
        for archive in sorted(available, key=lambda item: (item.ended, item.directory.name)):
            if result.local_bytes <= cache_bytes:
                break
            try:
                prune_archive(root, archive, store, progress)
                result.local_bytes -= archive.size
                result.pruned.append(archive.directory.name)
                progress.add("pruned_archives", 1)
                progress.add("verified_archives", -1)
            except Exception as error:
                progress.failed()
                result.errors.append(f"{archive.directory.name}: {error}")
    progress.update(phase="error" if result.errors else "idle", local_bytes=result.local_bytes)
    return result


@contextmanager
def sync_lock(root: Path):
    """Hold a nonblocking OS lock that releases when the process exits."""
    root = Path(root).absolute()
    _no_links(root / ".sync" / "lock")
    (root / ".sync").mkdir(parents=True, exist_ok=True)
    with (root / ".sync" / "lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Another journey sync process holds this root") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
