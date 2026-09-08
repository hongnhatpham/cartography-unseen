"""Protect the only local originals across interrupted uploads and cache eviction."""
import hashlib
import io
import json
import os
from pathlib import Path

import pytest

from app.journey_sync import (
    ArchiveError, S3Store, inspect_archive, prune_archive, sync_lock, sync_once,
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def make_archive(root, name="first", ended="2026-09-01T10:00:00+00:00"):
    directory = root / name
    (directory / "images").mkdir(parents=True)
    (directory / "images" / "000001.webp").write_bytes(b"original captured pixels")
    (directory / "index.html").write_text("interactive viewer", encoding="utf-8")
    manifest = {"schema_version": 1, "id": name, "ended_at": ended,
                "images": [{"path": "images/000001.webp"}]}
    write_manifest(directory, manifest)
    return directory


def write_manifest(directory, manifest):
    data = json.dumps(manifest).encode()
    (directory / "manifest.json").write_bytes(data)
    (directory / "complete.json").write_text(json.dumps(
        {"schema_version": 1, "manifest_sha256": digest(data)}), encoding="utf-8")


class FakeStore:
    def __init__(self):
        self.identity = {"endpoint": "https://account.r2.cloudflarestorage.com",
                         "bucket": "private", "prefix": "journeys"}
        self.objects = {}
        self.uploads = []
        self.fail_upload = None
        self.corrupt_upload = False
        self.on_verify = None
        self.verifications = []

    def head_sha256(self, key):
        return self.objects[key][1] if key in self.objects else None

    def upload(self, key, path, sha256):
        if key == self.fail_upload:
            raise OSError("Upload interrupted")
        self.objects[key] = (b"corrupt" if self.corrupt_upload else path.read_bytes(), sha256)
        self.uploads.append(key)

    def verify(self, key, sha256):
        self.verifications.append(key)
        if self.on_verify:
            self.on_verify()
        if key not in self.objects or digest(self.objects[key][0]) != sha256:
            raise ArchiveError("Remote checksum mismatch")


def test_active_and_invalid_completed_archives_are_preserved(tmp_path):
    active = make_archive(tmp_path, "active")
    (active / "complete.json").unlink()
    invalid = make_archive(tmp_path, "invalid")
    (invalid / "manifest.json").write_text("{}")
    store = FakeStore()
    result = sync_once(tmp_path, store, prune=True, cache_bytes=0)
    assert active.exists() and invalid.exists()
    assert not store.uploads
    assert len(result.errors) == 1


def test_interrupted_upload_resumes_without_reuploading_verified_assets(tmp_path):
    directory = make_archive(tmp_path)
    store = FakeStore()
    store.fail_upload = "journeys/first/manifest.json"
    result = sync_once(tmp_path, store, prune=True, cache_bytes=0)
    assert result.errors and directory.exists()
    assert not (tmp_path / ".sync" / "first.json").exists()
    assert "journeys/first/complete.json" not in store.objects
    uploaded = list(store.uploads)
    store.fail_upload = None
    result = sync_once(tmp_path, store)
    assert not result.errors
    assert all(store.uploads.count(key) == 1 for key in uploaded)
    assert store.uploads[-2:] == ["journeys/first/manifest.json", "journeys/first/complete.json"]


def test_metadata_is_not_enough_to_issue_receipt_or_remove_originals(tmp_path):
    directory = make_archive(tmp_path)
    store = FakeStore()
    store.corrupt_upload = True
    result = sync_once(tmp_path, store, prune=True, cache_bytes=0)
    assert result.errors and directory.exists()
    assert not (tmp_path / ".sync" / "first.json").exists()


def test_receipt_avoids_repeated_downloads_but_prune_rechecks_remote_bytes(tmp_path):
    directory = make_archive(tmp_path)
    store = FakeStore()
    assert not sync_once(tmp_path, store).errors
    verified = len(store.verifications)
    assert not sync_once(tmp_path, store).errors
    assert len(store.verifications) == verified
    key = "journeys/first/images/000001.webp"
    store.objects[key] = (b"changed remote body", store.objects[key][1])
    result = sync_once(tmp_path, store, prune=True, cache_bytes=0)
    assert result.errors and directory.exists()
    assert (directory / "images/000001.webp").read_bytes() == b"original captured pixels"


def test_changed_local_files_or_destination_cannot_use_receipt(tmp_path):
    directory = make_archive(tmp_path)
    store = FakeStore()
    sync_once(tmp_path, store)
    original_uploads = list(store.uploads)
    store.identity = {**store.identity, "bucket": "another-bucket"}
    result = sync_once(tmp_path, store, prune=True, cache_bytes=0)
    assert result.errors and directory.exists()
    assert store.uploads == original_uploads
    store.identity = {**store.identity, "bucket": "private"}
    (directory / "images/000001.webp").write_bytes(b"new local original")
    result = sync_once(tmp_path, store, prune=True, cache_bytes=0)
    assert result.errors and directory.exists()
    assert store.uploads == original_uploads


def test_cache_evicts_oldest_and_keeps_newest(tmp_path):
    old = make_archive(tmp_path, "old", "2026-09-01T10:00:00Z")
    new = make_archive(tmp_path, "new", "2026-09-02T10:00:00Z")
    size = inspect_archive(tmp_path, new).size
    store = FakeStore()
    result = sync_once(tmp_path, store, prune=True, cache_bytes=size)
    assert not result.errors
    assert result.pruned == ["old"]
    assert not old.exists() and new.exists()
    assert result.local_bytes == size
    assert (tmp_path / ".sync" / "old.json").exists()


def test_default_never_prunes_and_active_bytes_count_toward_budget(tmp_path):
    directory = make_archive(tmp_path)
    active = tmp_path / "active"
    active.mkdir()
    (active / "recording").write_bytes(b"still recording")
    store = FakeStore()
    result = sync_once(tmp_path, store, cache_bytes=0)
    assert not result.pruned and directory.exists()
    result = sync_once(tmp_path, store, cache_bytes=0, prune=True)
    assert result.pruned == ["first"]
    assert active.exists() and result.local_bytes == len(b"still recording")


@pytest.mark.parametrize("name", ["../outside.png", "/outside.png", "images/../../outside.png",
                                  "images\\outside.png", "images/C:outside.png", "images//x.png"])
def test_manifest_path_traversal_never_uploads_or_prunes(tmp_path, name):
    directory = make_archive(tmp_path)
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["images"][0]["path"] = name
    write_manifest(directory, manifest)
    store = FakeStore()
    result = sync_once(tmp_path, store, cache_bytes=0, prune=True)
    assert result.errors and directory.exists() and not store.uploads


def test_unknown_files_block_upload_and_eviction(tmp_path):
    directory = make_archive(tmp_path)
    (directory / "notes.txt").write_text("keep my notes")
    store = FakeStore()
    result = sync_once(tmp_path, store, cache_bytes=0, prune=True)
    assert result.errors and directory.exists() and not store.uploads


def test_local_change_during_remote_verification_prevents_any_deletion(tmp_path):
    directory = make_archive(tmp_path)
    store = FakeStore()
    sync_once(tmp_path, store)
    archive = inspect_archive(tmp_path, directory)
    store.on_verify = lambda: (directory / "notes.txt").write_text("new file")
    with pytest.raises(ArchiveError):
        prune_archive(tmp_path, archive, store)
    assert all((directory / name).exists() for name in archive.inventory)


def test_symlink_archive_is_rejected(tmp_path):
    real = make_archive(tmp_path / "source")
    root = tmp_path / "root"
    root.mkdir()
    try:
        (root / "first").symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("Creating symlinks is unavailable on this Windows account")
    store = FakeStore()
    result = sync_once(root, store, cache_bytes=0, prune=True)
    assert result.errors and real.exists() and not store.uploads


def test_sync_lock_rejects_concurrent_process_and_releases(tmp_path):
    with sync_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another journey sync"):
            with sync_lock(tmp_path):
                pytest.fail("Acquired the same lock twice")
    with sync_lock(tmp_path):
        pass


@pytest.mark.skipif(os.name != "nt", reason="Windows junction protection")
def test_junction_archive_is_rejected_without_symlink_privileges(tmp_path):
    import _winapi

    real = make_archive(tmp_path / "source")
    root = tmp_path / "root"
    root.mkdir()
    _winapi.CreateJunction(str(real), str(root / "first"))
    store = FakeStore()
    result = sync_once(root, store, cache_bytes=0, prune=True)
    assert result.errors and real.exists() and not store.uploads


def test_remote_object_conflict_is_not_overwritten(tmp_path):
    directory = make_archive(tmp_path)
    store = FakeStore()
    key = "journeys/first/images/000001.webp"
    store.objects[key] = (b"another archive", digest(b"another archive"))
    result = sync_once(tmp_path, store, cache_bytes=0, prune=True)
    assert result.errors and directory.exists()
    assert store.objects[key][0] == b"another archive"


def test_missing_remote_after_receipt_preserves_every_local_file(tmp_path):
    directory = make_archive(tmp_path)
    store = FakeStore()
    sync_once(tmp_path, store)
    inventory = inspect_archive(tmp_path, directory).inventory
    del store.objects["journeys/first/manifest.json"]
    result = sync_once(tmp_path, store, cache_bytes=0, prune=True)
    assert result.errors
    assert all((directory / name).exists() for name in inventory)


def interrupt_prune(root, store, monkeypatch):
    unlink = Path.unlink
    calls = 0

    def fail_second_unlink(path, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("Injected disk interruption")
        return unlink(path, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "unlink", fail_second_unlink)
        result = sync_once(root, store, cache_bytes=0, prune=True)
    assert result.errors
    receipt = json.loads((root / ".sync/first.json").read_text())
    assert receipt["pruning"] is True
    assert not (root / "first/complete.json").exists()
    assert (root / "first/images/000001.webp").exists()


def test_interrupted_prune_resumes_only_with_explicit_flag_and_verifies_full_remote(tmp_path, monkeypatch):
    directory = make_archive(tmp_path)
    store = FakeStore()
    interrupt_prune(tmp_path, store, monkeypatch)
    result = sync_once(tmp_path, store, cache_bytes=0)
    assert result.errors and directory.exists()
    store.verifications.clear()
    result = sync_once(tmp_path, store, cache_bytes=0, prune=True)
    assert not result.errors and result.pruned == ["first"]
    assert not directory.exists()
    assert set(store.verifications) == set(store.objects)
    assert result.local_bytes == 0


@pytest.mark.parametrize("change", ["local", "unknown", "remote", "destination"])
def test_interrupted_prune_preserves_remaining_files_if_any_check_changes(tmp_path, monkeypatch, change):
    directory = make_archive(tmp_path)
    store = FakeStore()
    interrupt_prune(tmp_path, store, monkeypatch)
    if change == "local":
        (directory / "images/000001.webp").write_bytes(b"new captured pixels")
    elif change == "unknown":
        (directory / "notes.txt").write_text("new notes")
    elif change == "remote":
        key = "journeys/first/complete.json"
        store.objects[key] = (b"changed", store.objects[key][1])
    else:
        store.identity = {**store.identity, "bucket": "different"}
    remaining = {path: path.read_bytes() for path in directory.rglob("*") if path.is_file()}
    result = sync_once(tmp_path, store, cache_bytes=0, prune=True)
    assert result.errors and not result.pruned
    assert all(path.read_bytes() == contents for path, contents in remaining.items())


def test_s3_adapter_checks_actual_body_and_uses_private_conditional_put(tmp_path):
    class Client:
        def put_object(self, **kwargs):
            self.put = {**kwargs, "Body": kwargs["Body"].read()}

        def head_object(self, **kwargs):
            return {"Metadata": {"sha256": "claimed"}}

        def get_object(self, **kwargs):
            return {"Body": io.BytesIO(b"actual")}

    client = Client()
    store = S3Store(client, "https://account.r2.cloudflarestorage.com", "private")
    path = tmp_path / "image.webp"
    path.write_bytes(b"pixels")
    store.upload("journeys/a/image.webp", path, digest(b"pixels"))
    assert client.put["IfNoneMatch"] == "*" and "ACL" not in client.put
    assert client.put["ContentType"] == "image/webp"
    assert client.put["Body"] == b"pixels"
    assert store.head_sha256("x") == "claimed"
    store.verify("x", digest(b"actual"))
    with pytest.raises(ArchiveError, match="checksum"):
        store.verify("x", "claimed")


@pytest.mark.parametrize("endpoint", ["http://host", "https://user:password@host", "https://host/path",
                                      "https://host?token=secret", ""])
def test_s3_requires_https_without_embedded_secrets(endpoint):
    with pytest.raises(ValueError):
        S3Store(None, endpoint, "private")
