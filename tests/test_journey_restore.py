import hashlib
import json

import pytest

from app.journey_bundle_sync import sync_bundles_once
from app.journey_sync import ArchiveError
from tools.restore_journey import restore_bundle


class Store:
    identity = {"transport": "wrangler-zip", "bucket": "private", "account_id": "test"}

    def __init__(self):
        self.objects = {}

    def put(self, key, path):
        self.objects[key] = path.read_bytes()

    def get(self, key, path):
        path.write_bytes(self.objects[key])


@pytest.fixture
def backup(tmp_path):
    source = tmp_path / "original" / "first"
    (source / "images").mkdir(parents=True)
    (source / "images/000001.webp").write_bytes(b"original captured pixels")
    (source / "index.html").write_text("viewer")
    manifest = json.dumps({"id": "first", "ended_at": "2026-09-07T00:00:00Z",
                           "images": [{"path": "images/000001.webp"}]}).encode()
    (source / "manifest.json").write_bytes(manifest)
    (source / "complete.json").write_text(json.dumps({"schema_version": 1,
        "manifest_sha256": hashlib.sha256(manifest).hexdigest()}))
    store = Store()
    assert not sync_bundles_once(source.parent, store).errors
    key = next(key for key in store.objects if key.endswith(".zip.json"))
    return source, store, key


def test_restore_rebuilds_every_file_without_overwriting_existing_archive(tmp_path, backup):
    source, store, key = backup
    restored = restore_bundle(store, key, tmp_path / "restored")
    for path in source.rglob("*"):
        if path.is_file():
            assert (restored / path.relative_to(source)).read_bytes() == path.read_bytes()
    with pytest.raises(ArchiveError, match="already exists"):
        restore_bundle(store, key, restored.parent)


@pytest.mark.parametrize("failure", ["part", "path", "destination"])
def test_restore_rejects_corruption_or_invalid_destination(tmp_path, backup, failure):
    _, store, key = backup
    descriptor = json.loads(store.objects[key])
    if failure == "part":
        store.objects[descriptor["parts"][0]["key"]] = b"corrupt"
    elif failure == "path":
        descriptor["files"]["../outside.txt"] = {"size": 0, "sha256": "0" * 64}
        store.objects[key] = json.dumps(descriptor).encode()
    else:
        descriptor["destination"]["bucket"] = "different-bucket"
        store.objects[key] = json.dumps(descriptor).encode()
    with pytest.raises(ArchiveError):
        restore_bundle(store, key, tmp_path / "restored")
    assert not (tmp_path / "restored/first").exists()
    assert not (tmp_path / "outside.txt").exists()
