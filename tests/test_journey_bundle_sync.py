"""Bundle backups retain originals until a fresh remote download verifies."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import zipfile

import pytest

from app.journey_bundle_sync import WranglerTransport, sync_bundles_once


def make_archive(root, name="first", ended="2026-09-01T10:00:00Z"):
    directory = root / name
    (directory / "images").mkdir(parents=True)
    (directory / "images/000001.webp").write_bytes(b"original captured pixels")
    (directory / "index.html").write_text("interactive viewer", encoding="utf-8")
    manifest = json.dumps({"id": name, "ended_at": ended,
                           "images": [{"path": "images/000001.webp"}]}).encode()
    (directory / "manifest.json").write_bytes(manifest)
    (directory / "complete.json").write_text(json.dumps({"schema_version": 1,
        "manifest_sha256": hashlib.sha256(manifest).hexdigest()}), encoding="utf-8")
    return directory


class FakeTransport:
    def __init__(self):
        self.identity = {"transport": "wrangler-zip", "bucket": "private", "account_id": "test"}
        self.objects = {}
        self.uploads = []
        self.downloads = []
        self.fail_upload = False
        self.corrupt = False
        self.on_get = None

    def put(self, key, path):
        if self.fail_upload:
            raise OSError("Upload interrupted")
        self.objects[key] = b"corrupt" if self.corrupt else path.read_bytes()
        self.uploads.append(key)

    def get(self, key, path):
        self.downloads.append(key)
        path.write_bytes(self.objects[key])
        if self.on_get:
            self.on_get()


def test_bundle_preserves_viewer_files_and_is_deterministic(tmp_path):
    first = make_archive(tmp_path / "one")
    second = make_archive(tmp_path / "two")
    os.utime(second / "index.html", (10, 10))
    store = FakeTransport()
    assert not sync_bundles_once(first.parent, store).errors
    first_key = store.uploads[0]
    first_data = store.objects[first_key]
    assert not sync_bundles_once(second.parent, store).errors
    assert store.uploads[::2] == [first_key, first_key]
    assert store.objects[first_key] == first_data
    assert first_key.endswith(hashlib.sha256(first_data).hexdigest() + ".zip")
    with zipfile.ZipFile(io.BytesIO(first_data)) as bundle:
        assert bundle.namelist() == sorted(bundle.namelist())
        assert set(bundle.namelist()) == {"complete.json", "index.html", "manifest.json", "images/000001.webp"}
        assert all(bundle.read(name) == (first / name).read_bytes() for name in bundle.namelist())


@pytest.mark.parametrize("failure", ["upload", "corrupt"])
def test_failed_roundtrip_does_not_receipt_or_prune(tmp_path, failure):
    directory = make_archive(tmp_path)
    store = FakeTransport()
    store.fail_upload = failure == "upload"
    store.corrupt = failure == "corrupt"
    result = sync_bundles_once(tmp_path, store, prune=True, cache_bytes=0)
    assert result.errors and directory.exists() and not result.pruned
    assert not (tmp_path / ".sync/bundle-first.json").exists()
    store.fail_upload = store.corrupt = False
    assert not sync_bundles_once(tmp_path, store).errors


def test_cached_archive_skips_transfers_but_prune_rechecks_remote(tmp_path):
    directory = make_archive(tmp_path)
    store = FakeTransport()
    assert sync_bundles_once(tmp_path, store).uploaded == ["first"]
    assert not sync_bundles_once(tmp_path, store).uploaded
    assert len(store.uploads) == len(store.downloads) == 2
    store.objects[store.uploads[0]] = b"modified remote"
    result = sync_bundles_once(tmp_path, store, prune=True, cache_bytes=0)
    assert result.errors and directory.exists() and not result.pruned
    assert len(store.downloads) == 3


def test_active_and_unknown_files_are_preserved(tmp_path):
    active = make_archive(tmp_path, "active")
    (active / "complete.json").unlink()
    invalid = make_archive(tmp_path, "invalid")
    (invalid / "notes.txt").write_text("keep")
    store = FakeTransport()
    result = sync_bundles_once(tmp_path, store, prune=True, cache_bytes=0)
    assert len(result.errors) == 1 and not store.uploads
    assert active.exists() and invalid.exists()


@pytest.mark.parametrize("change", ["local", "bucket", "account", "prefix"])
def test_receipt_rejects_changed_archive_or_destination(tmp_path, change):
    directory = make_archive(tmp_path)
    store = FakeTransport()
    sync_bundles_once(tmp_path, store)
    if change == "local":
        (directory / "images/000001.webp").write_bytes(b"new pixels")
    elif change in ("bucket", "account"):
        store.identity["bucket" if change == "bucket" else "account_id"] = "changed"
    result = sync_bundles_once(tmp_path, store, prefix="other" if change == "prefix" else "journeys",
                               prune=True, cache_bytes=0)
    assert result.errors and directory.exists() and len(store.uploads) == 2


def test_local_change_during_fresh_remote_check_prevents_deletion(tmp_path):
    directory = make_archive(tmp_path)
    store = FakeTransport()
    sync_bundles_once(tmp_path, store)
    before = {path: path.read_bytes() for path in directory.rglob("*") if path.is_file()}
    store.on_get = lambda: (directory / "notes.txt").write_text("keep")
    result = sync_bundles_once(tmp_path, store, prune=True, cache_bytes=0)
    assert result.errors and all(path.read_bytes() == data for path, data in before.items())


def test_cache_evicts_oldest_and_protects_active_bytes(tmp_path):
    old = make_archive(tmp_path, "old")
    new = make_archive(tmp_path, "new", "2026-09-02T10:00:00Z")
    active = tmp_path / "active"
    active.mkdir()
    (active / "recording").write_bytes(b"active")
    budget = sum(path.stat().st_size for path in new.rglob("*") if path.is_file()) + 6
    result = sync_bundles_once(tmp_path, FakeTransport(), prune=True, cache_bytes=budget)
    assert not result.errors and result.pruned == ["old"]
    assert not old.exists() and new.exists() and active.exists()
    assert result.local_bytes == budget


def interrupt_prune(root, store, monkeypatch):
    unlink = Path.unlink
    calls = 0

    def fail_second(path, *args, **kwargs):
        nonlocal calls
        if path.parent == root / "first" or path.parent == root / "first/images":
            calls += 1
            if calls == 2:
                raise OSError("Disk interrupted")
        return unlink(path, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "unlink", fail_second)
        result = sync_bundles_once(root, store, prune=True, cache_bytes=0)
    assert result.errors
    assert json.loads((root / ".sync/bundle-first.json").read_text())["pruning"] is True
    assert not (root / "first/complete.json").exists()


def test_interrupted_prune_requires_flag_and_rechecks_bundle(tmp_path, monkeypatch):
    directory = make_archive(tmp_path)
    store = FakeTransport()
    interrupt_prune(tmp_path, store, monkeypatch)
    assert sync_bundles_once(tmp_path, store).errors
    transfers = len(store.downloads)
    result = sync_bundles_once(tmp_path, store, prune=True, cache_bytes=0)
    assert not result.errors and result.pruned == ["first"] and result.local_bytes == 0
    assert len(store.downloads) == transfers + 2 and not directory.exists()


@pytest.mark.parametrize("change", ["local", "remote", "unknown", "destination", "traversal"])
def test_interrupted_prune_rejects_changes(tmp_path, monkeypatch, change):
    directory = make_archive(tmp_path)
    store = FakeTransport()
    interrupt_prune(tmp_path, store, monkeypatch)
    if change == "local":
        (directory / "images/000001.webp").write_bytes(b"new pixels")
    elif change == "remote":
        store.objects[store.uploads[0]] = b"changed"
    elif change == "unknown":
        (directory / "notes.txt").write_text("keep")
    elif change == "destination":
        store.identity["bucket"] = "other"
    else:
        path = tmp_path / ".sync/bundle-first.json"
        receipt = json.loads(path.read_text())
        receipt["files"]["../outside"] = receipt["files"]["index.html"]
        path.write_text(json.dumps(receipt))
    before = {path: path.read_bytes() for path in directory.rglob("*") if path.is_file()}
    result = sync_bundles_once(tmp_path, store, prune=True, cache_bytes=0)
    assert result.errors and not result.pruned
    assert all(path.read_bytes() == data for path, data in before.items())


@pytest.mark.skipif(os.name != "nt", reason="Windows junction protection")
def test_junction_is_rejected(tmp_path):
    import _winapi

    real = make_archive(tmp_path / "source")
    root = tmp_path / "root"
    root.mkdir()
    _winapi.CreateJunction(str(real), str(root / "first"))
    store = FakeTransport()
    result = sync_bundles_once(root, store, prune=True, cache_bytes=0)
    assert result.errors and real.exists() and not store.uploads


def test_wrangler_uses_remote_private_objects_and_hides_output(tmp_path, monkeypatch):
    import app.journey_bundle_sync as module

    command = tmp_path / ("wrangler.cmd" if os.name == "nt" else "wrangler")
    script = tmp_path / "node_modules/wrangler/bin/wrangler.js"
    script.parent.mkdir(parents=True)
    script.touch()
    monkeypatch.setattr(module.shutil, "which", lambda name: "node" if name == "node" else str(command))
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", run)
    store = WranglerTransport("private-bucket")
    path = tmp_path / "archive.zip"
    path.write_bytes(b"bundle")
    store.check_bucket()
    store.put("journeys/first/hash.zip", path)
    store.get("journeys/first/hash.zip", path)
    assert all("--remote" in command for command, _ in calls[1:])
    assert "application/zip" in calls[1][0]
    assert all(kwargs["capture_output"] and not kwargs.get("shell") for _, kwargs in calls)
    monkeypatch.setattr(module.subprocess, "run", lambda command, **kwargs:
                        subprocess.CompletedProcess(command, 1, stdout=b"secret", stderr=b"secret"))
    with pytest.raises(RuntimeError, match="Wrangler failed") as error:
        store.check_bucket()
    assert "secret" not in str(error.value)


def test_large_bundle_streams_bounded_parts_and_reassembles(tmp_path, monkeypatch):
    import app.journey_bundle_sync as module

    monkeypatch.setattr(module, "PART_BYTES", 128)
    directory = make_archive(tmp_path)
    store = FakeTransport()
    result = sync_bundles_once(tmp_path, store)
    assert not result.errors
    receipt = json.loads((tmp_path / ".sync/bundle-first.json").read_text())
    assert len(receipt["parts"]) > 1
    assert all(part["size"] <= 128 for part in receipt["parts"])
    assert store.uploads[-1] == receipt["key"]
    assert json.loads(store.objects[receipt["key"]]) == receipt
    combined = b"".join(store.objects[part["key"]] for part in receipt["parts"])
    assert hashlib.sha256(combined).hexdigest() == receipt["sha256"]
    with zipfile.ZipFile(io.BytesIO(combined)) as bundle:
        assert all(bundle.read(name) == (directory / name).read_bytes() for name in bundle.namelist())
    result = sync_bundles_once(tmp_path, store, prune=True, cache_bytes=0)
    assert not result.errors and result.pruned == ["first"]


@pytest.mark.skipif(os.name != "nt", reason="Windows npm installation")
def test_wrangler_found_after_install_with_stale_desktop_path(tmp_path, monkeypatch):
    import app.journey_bundle_sync as module
    npm = tmp_path / "npm"
    script = npm / "node_modules/wrangler/bin/wrangler.js"
    script.parent.mkdir(parents=True)
    script.touch()
    (npm / "wrangler.cmd").touch()
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setattr(module.shutil, "which", lambda name: "node" if name == "node" else None)
    assert WranglerTransport("private-bucket").command == ["node", str(script)]


def test_changed_source_during_stream_never_publishes_descriptor(tmp_path, monkeypatch):
    import app.journey_bundle_sync as module

    monkeypatch.setattr(module, "PART_BYTES", 64)
    directory = make_archive(tmp_path)
    store = FakeTransport()
    store.on_get = lambda: (directory / "images/000001.webp").write_bytes(b"changed source")
    result = sync_bundles_once(tmp_path, store, prune=True, cache_bytes=0)
    assert result.errors and directory.exists()
    assert not (tmp_path / ".sync/bundle-first.json").exists()
    assert not any(key.endswith(".json") for key in store.objects)


def test_missing_remote_descriptor_prevents_pruning(tmp_path):
    directory = make_archive(tmp_path)
    store = FakeTransport()
    sync_bundles_once(tmp_path, store)
    del store.objects[store.uploads[-1]]
    result = sync_bundles_once(tmp_path, store, prune=True, cache_bytes=0)
    assert result.errors and directory.exists() and not result.pruned
