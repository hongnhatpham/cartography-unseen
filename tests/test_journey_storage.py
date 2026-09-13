import json
from types import SimpleNamespace

import pytest

from app.config import AppConfig
from app.journey_storage import DiskSpaceMonitor, start_sync, sync_environment


@pytest.mark.parametrize("settings", [
    {"map_image_format": "jpeg"}, {"map_export_svg": 1}, {"map_sync_enabled": "yes"},
    {"map_cache_gib": 0}, {"map_min_free_gib": float("nan")},
    {"map_cache_gib": float("inf")},
])
def test_invalid_archive_settings_are_rejected(settings):
    with pytest.raises(RuntimeError):
        AppConfig(**settings).validate()


def test_disk_monitor_stops_capture_on_low_space_or_disk_error(tmp_path, monkeypatch):
    monkeypatch.setattr("app.journey_storage.shutil.disk_usage", lambda path: SimpleNamespace(free=100))
    monitor = DiskSpaceMonitor(tmp_path, 100)
    try:
        assert monitor.available
        monkeypatch.setattr("app.journey_storage.shutil.disk_usage", lambda path: SimpleNamespace(free=99))
        monitor._sample()
        assert not monitor.available
        def unavailable(path):
            raise OSError("disk unavailable")
        monkeypatch.setattr("app.journey_storage.shutil.disk_usage", unavailable)
        monitor._sample()
        assert not monitor.available
    finally:
        monitor.close()


def test_saved_setup_is_scoped_to_child_environment(tmp_path, monkeypatch):
    for key in ("AWS_PROFILE", "AWS_ACCESS_KEY_ID", "JOURNEY_S3_ENDPOINT", "JOURNEY_S3_BUCKET"):
        monkeypatch.delenv(key, raising=False)
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache/journey-storage.json").write_text(json.dumps({
        "endpoint": "https://example.r2.cloudflarestorage.com", "bucket": "journeys"}))
    environment = sync_environment(tmp_path)
    assert environment["AWS_PROFILE"] == "journey"
    assert environment["AWS_SHARED_CREDENTIALS_FILE"] == str(tmp_path / "cache/journey-credentials.ini")
    monkeypatch.setenv("JOURNEY_S3_BUCKET", "explicit-bucket")
    monkeypatch.setenv("AWS_PROFILE", "explicit-profile")
    environment = sync_environment(tmp_path)
    assert environment["JOURNEY_S3_BUCKET"] == "explicit-bucket"
    assert environment["AWS_PROFILE"] == "explicit-profile"


def test_uploader_requires_a_destination_before_launch(tmp_path, monkeypatch):
    monkeypatch.delenv("JOURNEY_S3_ENDPOINT", raising=False)
    monkeypatch.delenv("JOURNEY_S3_BUCKET", raising=False)
    with pytest.raises(RuntimeError, match="JOURNEY_S3_ENDPOINT"):
        start_sync(tmp_path, tmp_path / "journeys", 5)


def test_wrangler_setup_uses_existing_login_without_aws_credentials(tmp_path, monkeypatch):
    for key in ("JOURNEY_STORAGE_TRANSPORT", "JOURNEY_S3_BUCKET", "AWS_PROFILE",
                "AWS_SHARED_CREDENTIALS_FILE", "CLOUDFLARE_ACCOUNT_ID"):
        monkeypatch.delenv(key, raising=False)
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache/journey-storage.json").write_text(json.dumps({
        "transport": "wrangler", "bucket": "private-journeys", "account_id": "test-account"}))
    environment = sync_environment(tmp_path)
    assert environment["JOURNEY_STORAGE_TRANSPORT"] == "wrangler"
    assert environment["CLOUDFLARE_ACCOUNT_ID"] == "test-account"
    assert environment["JOURNEY_S3_BUCKET"] == "private-journeys"
    assert "AWS_SHARED_CREDENTIALS_FILE" not in environment
    assert "AWS_PROFILE" not in environment


def test_background_uploader_uses_physical_windows_login_location(tmp_path, monkeypatch):
    import os
    monkeypatch.setenv("XDG_CONFIG_HOME", "logical-desktop-location")
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache/journey-storage.json").write_text(json.dumps({
        "transport": "wrangler", "bucket": "private-journeys",
        "wrangler_config_home": "physical-shared-location"}))
    assert sync_environment(tmp_path)["XDG_CONFIG_HOME"] == "physical-shared-location"
    assert os.environ["XDG_CONFIG_HOME"] == "logical-desktop-location"
