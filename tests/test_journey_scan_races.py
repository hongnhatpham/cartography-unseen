"""Live recorder temp files may disappear while the uploader counts disk use."""
from pathlib import Path
from types import SimpleNamespace
import pytest
from app import journey_sync
from app.journey_sync import sync_once
from app.journey_bundle_sync import sync_bundles_once


@pytest.mark.parametrize('sync', [sync_once, sync_bundles_once])
@pytest.mark.parametrize('completed', [False, True])
@pytest.mark.parametrize('stage', ['listing', 'size'])
def test_atomic_temp_rename_is_tolerated_only_for_active_size_estimate(tmp_path, monkeypatch, sync, completed, stage):
    directory = tmp_path / 'visitor'
    directory.mkdir()
    (directory / 'manifest.json').write_bytes(b'active manifest')
    temporary = directory / 'manifest.json.tmp'
    temporary.write_bytes(b'new manifest')
    if completed:
        (directory / 'complete.json').write_text('{}')
    if stage == 'listing':
        original = Path.iterdir
        def listing(path):
            entries = list(original(path))
            if path == directory:
                temporary.unlink(missing_ok=True)
            return iter(entries)
        monkeypatch.setattr(Path, 'iterdir', listing)
    else:
        original = journey_sync._files
        def listing(path, **kwargs):
            entries = original(path, **kwargs)
            if path == directory:
                temporary.unlink(missing_ok=True)
            return entries
        monkeypatch.setattr(journey_sync, '_files', listing)
    # Any attempt to upload or prune this active/invalid fixture would fail.
    store = SimpleNamespace(identity={'prefix': 'journeys'})
    result = sync(tmp_path, store, prune=True, cache_bytes=0)
    assert bool(result.errors) is completed
    assert directory.exists() and (directory / 'manifest.json').exists()
    assert not result.uploaded and not result.pruned
    if not completed:
        assert result.local_bytes == len(b'active manifest')
