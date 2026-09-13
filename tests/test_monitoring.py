"""Collector privacy, atomic state, failure isolation and bounded replay."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import uuid

import pytest

from app import monitoring as m


class EmptySampler:
    def metrics(self, now):
        return {}

    def process_alive(self, pid):
        return None


def sample(directory, sequence=0, now=100):
    return m.heartbeat("12345678-1234-4234-8234-123456789abc", sequence, directory,
                       EmptySampler(), now=now)


def test_atomic_snapshots_allowlist_and_failure_retains_previous(tmp_path, monkeypatch):
    path = tmp_path / "artwork.json"
    assert m.atomic_snapshot(path, "artwork", {"state": "running", "updated_at": 100,
        "display_fps": 60, "prompt": "private", "routes": [1, 2], "hostname": "private"})
    before = path.read_bytes()
    assert b"private" not in before and b"routes" not in before
    def fail_replace(*args):
        raise PermissionError("private path and token")
    monkeypatch.setattr(m.os, "replace", fail_replace)
    assert not m.atomic_snapshot(path, "artwork", {"state": "error"})
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))
    assert m.read_snapshot(path, "artwork", now=110)["stale"] is False
    assert m.read_snapshot(path, "artwork", now=121)["stale"] is True


@pytest.mark.parametrize("raw", [b"{", b"[]", b"null", b"{}", b"x" * 9000])
def test_partial_malformed_and_missing_status_are_unknown(tmp_path, raw):
    path = tmp_path / "artwork.json"
    assert m.read_snapshot(path, "artwork")["stale"] is True
    path.write_bytes(raw)
    result = m.read_snapshot(path, "artwork")
    assert result["stale"] is True and result["state"] is None


def test_snapshot_published_during_system_probe_is_not_stalled(tmp_path, monkeypatch):
    tick = [1000.0]
    monkeypatch.setattr(m.time, 'monotonic', lambda: tick[0])
    class PublishingSampler(EmptySampler):
        def metrics(self, now):
            m.atomic_snapshot(tmp_path / 'artwork.json', 'artwork',
                              {'state': 'running', 'updated_at': now + .5, 'pid': 123})
            tick[0] += .75
            return {}
        def process_alive(self, pid):
            return True
    value = m.heartbeat('test', 0, tmp_path, PublishingSampler(), now=100)
    assert value['app']['state'] == 'running'


def test_snapshot_published_during_read_is_not_from_the_future(tmp_path, monkeypatch):
    path = tmp_path / 'artwork.json'
    m.atomic_snapshot(path, 'artwork', {'state': 'running', 'updated_at': 100.25})
    tick = [1000.0]
    monkeypatch.setattr(m.time, 'monotonic', lambda: tick[0])
    original = Path.open
    def delayed_open(self, *args, **kwargs):
        tick[0] += .5
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', delayed_open)
    assert not m.read_snapshot(path, 'artwork', now=100)['stale']
    assert m.read_snapshot(path, 'artwork', now=80)['stale']
    assert m.read_snapshot(path, 'artwork', now=121)['stale']


def test_oversized_json_integer_cannot_stop_uploader_startup(tmp_path):
    path = tmp_path / "uploader.json"
    path.write_text(json.dumps({"schema_version": 1, "phase": "idle", "last_verified_at": 10 ** 400,
                               "updated_at": 10 ** 400, "pid": 10 ** 400}))
    previous = m.read_snapshot(path, "uploader")
    progress = m.UploadProgress()
    progress.update(phase="scan", last_verified_at=previous["last_verified_at"])
    assert previous["last_verified_at"] is None
    assert previous["pid"] is None
    assert previous["stale"] is True
    assert progress.values["phase"] == "scan"


def test_deeply_nested_ignored_json_cannot_stop_uploader_startup(tmp_path):
    path = tmp_path / "uploader.json"
    path.write_text('{"schema_version":1,"phase":"idle","ignored":' + '[' * 3000 + '0' + ']' * 3000 + '}')
    assert path.stat().st_size < m.MAX_SNAPSHOT_BYTES
    previous = m.read_snapshot(path, "uploader")
    progress = m.UploadProgress()
    progress.update(phase="scan", last_verified_at=previous["last_verified_at"])
    assert previous["phase"] is None
    assert previous["last_verified_at"] is None
    assert previous["stale"] is True
    assert progress.values["phase"] == "scan"


def test_publisher_never_refreshes_timestamp_without_producer(tmp_path, monkeypatch):
    # Avoid scheduling races while exercising the actual publisher's bounded slot.
    monkeypatch.setattr(m.threading.Thread, "start", lambda self: None)
    monkeypatch.setattr(m.threading.Thread, "join", lambda *args, **kwargs: None)
    publisher = m.SnapshotPublisher(tmp_path / "artwork.json", "artwork")
    monkeypatch.setattr(m.time, "time", lambda: 100)
    publisher.update(state="running", displayed_frames=12, prompt="secret")
    assert publisher.latest["updated_at"] == 100
    monkeypatch.setattr(m.time, "time", lambda: 500)
    assert publisher.latest["updated_at"] == 100
    assert "prompt" not in publisher.latest
    publisher.close()


def test_counter_rates_reset_missing_and_time_reversal():
    rates = m.CounterRates()
    assert rates.rate("sent", 100, 1) is None
    assert rates.rate("sent", 120, 3) == 10
    assert rates.rate("sent", 3, 5) is None
    assert rates.rate("sent", 13, 5) is None
    assert rates.rate("sent", None, 7) is None
    assert rates.rate("sent", 23, 9) is None
    assert rates.rate("sent", 33, 11) == 5


@pytest.mark.parametrize("result", [None, "broken", "[N/A], 10, 20, 30", "nan, inf, -1, unknown"])
def test_gpu_missing_or_malformed_values_stay_null(result):
    def run(command, **kwargs):
        assert kwargs["timeout"] == 2 and "--id=0" in command
        if result is None:
            raise FileNotFoundError("private path")
        return SimpleNamespace(returncode=0, stdout=result)
    values = m.gpu_metrics(run)
    assert values["gpu_util_percent"] is None
    if result == "[N/A], 10, 20, 30":
        assert values["gpu_memory_used_bytes"] == 10 * 1024 ** 2
    else:
        assert all(value is None for value in values.values())


def test_missing_system_metrics_remain_null(tmp_path, monkeypatch):
    class Unavailable:
        def __getattr__(self, name):
            def fail(*args, **kwargs):
                raise OSError("secret exception")
            return fail
    monkeypatch.setattr(m, "gpu_metrics", lambda: dict.fromkeys(m.METRICS[-4:]))
    sampler = m.SystemSampler(tmp_path, Unavailable())
    assert all(value is None for value in sampler.metrics(100).values())
    assert sampler.process_alive(123) is None


def test_wire_allowlist_does_not_include_device_pid_paths_or_error_text(tmp_path):
    m.atomic_snapshot(tmp_path / "artwork.json", "artwork", {"state": "error", "pid": 555,
        "updated_at": 100, "error": "secret prompt", "display_fps": float("nan")})
    m.atomic_snapshot(tmp_path / "uploader.json", "uploader", {"phase": "error", "pid": 777,
        "updated_at": 100, "error_code": "secret token", "last_verified_at": 2 ** 53 - 1})
    body = sample(tmp_path)
    encoded = json.dumps(body)
    for forbidden in ("device_id", "pid", "secret", "hostname", "prompt", str(tmp_path)):
        assert forbidden not in encoded
    assert body["schemaVersion"] == 1
    assert body["app"]["processRunning"] is None
    assert body["app"]["displayFps"] is None
    assert body["app"]["errorCode"] == "application_error"
    assert body["archiveSync"]["lastVerifiedAt"] is None
    assert body["archiveSync"]["errorCode"] is None
    assert len(encoded) < 16384


def test_stale_live_application_is_stalled_and_dead_process_stopped(tmp_path):
    m.atomic_snapshot(tmp_path / "artwork.json", "artwork", {"state": "running", "pid": 42, "updated_at": 1})
    assert sample(tmp_path)["app"]["state"] == "stalled"
    sampler = EmptySampler()
    sampler.process_alive = lambda pid: False
    assert m.heartbeat(str(uuid.uuid4()), 0, tmp_path, sampler, now=100)["app"]["state"] == "stopped"


@pytest.mark.parametrize("phase,alive,age,expected", [
    ("uploading", True, 1801, "unknown"), ("verifying", False, 1, "error"),
    ("idle", False, 1, "idle"), ("stopped", False, 1, "idle"),
    ("packing", None, 1, "packing"), ("pruning", None, 1801, "unknown"),
])
def test_uploader_stale_or_dead_never_reports_live_work(tmp_path, phase, alive, age, expected):
    m.atomic_snapshot(tmp_path / "uploader.json", "uploader", {"phase": phase, "pid": 42, "updated_at": 2000 - age})
    sampler = EmptySampler()
    sampler.process_alive = lambda pid: alive
    sync = m.heartbeat(str(uuid.uuid4()), 0, tmp_path, sampler, now=2000)["archiveSync"]
    assert sync["state"] == expected
    assert sync["errorCode"] == ("sync_failed" if expected == "error" else None)


def test_exact_wire_contract_and_utc_timestamps(tmp_path):
    m.atomic_snapshot(tmp_path / "uploader.json", "uploader", {"phase": "idle", "updated_at": 100, "last_verified_at": 99})
    body = sample(tmp_path)
    assert set(body) == {"schemaVersion", "bootId", "sequence", "sampledAt", "agentVersion", "system", "app", "archiveSync"}
    assert set(body["system"]) == {"cpuPercent", "ramUsedBytes", "ramTotalBytes", "diskFreeBytes", "diskTotalBytes",
        "diskReadBytesPerSecond", "diskWriteBytesPerSecond", "networkRxBytesPerSecond", "networkTxBytesPerSecond",
        "gpuPercent", "vramUsedBytes", "vramTotalBytes", "gpuTemperatureC"}
    assert set(body["app"]) == {"state", "processRunning", "statusAgeSeconds", "displayFps", "generationFps", "lastFrameAgeSeconds", "errorCode"}
    assert set(body["archiveSync"]) == {"enabled", "processRunning", "state", "statusAgeSeconds", "pendingArchives", "pendingBytes",
        "verifiedLocalArchives", "incompleteArchives", "invalidArchives", "localBytes", "currentArchiveId", "completedPayloadBytes",
        "verifiedBytes", "lastVerifiedAt", "errorCode"}
    assert body["sampledAt"].endswith("Z") and body["archiveSync"]["lastVerifiedAt"].endswith("Z")
    assert body["agentVersion"] == "1.0.0"


def test_outbox_dedupes_expires_and_replays_freshest_before_history(tmp_path):
    outbox = m.Outbox(tmp_path / "outbox.db", max_age=10)
    for index in range(20):
        outbox.add(sample(tmp_path, index, index))
    outbox.add(sample(tmp_path, 19, 19))
    assert len(outbox.pending(100)) == 11
    sent = []
    def post(endpoint, token, payload):
        sent.append((endpoint, payload))
        return True
    assert m.deliver(outbox, {"endpoint": "https://example.test/api/v1/heartbeat", "token": "secret"}, post=post)
    assert sent[0][0].endswith("/heartbeat") and sent[0][1]["sequence"] == 19
    assert sent[1][0].endswith("/history")
    assert [s["sequence"] for s in sent[1][1]["samples"]] == list(range(18, 8, -1))
    assert outbox.pending() == []
    outbox.close()


def test_failed_delivery_keeps_unacknowledged_samples_and_hides_errors(tmp_path, capsys):
    outbox = m.Outbox(tmp_path / "outbox.db")
    for index in range(3):
        outbox.add(sample(tmp_path, index))
    def failed(*args):
        raise RuntimeError("token secret host path")
    assert not m.deliver(outbox, {"endpoint": "https://example.test/api/v1/heartbeat", "token": "secret"}, post=failed)
    assert len(outbox.pending()) == 3
    def partial(endpoint, *args):
        return endpoint.endswith("/heartbeat")
    assert not m.deliver(outbox, {"endpoint": "https://example.test/api/v1/heartbeat", "token": "secret"}, post=partial)
    assert len(outbox.pending()) == 2
    assert capsys.readouterr().out == "" and capsys.readouterr().err == ""
    outbox.close()


def test_outbox_physical_size_and_replay_limits(tmp_path):
    path = tmp_path / "outbox.db"
    outbox = m.Outbox(path, max_bytes=150000)
    observed_sizes = []
    def observe(statement):
        if statement == "COMMIT":
            observed_sizes.append(sum(p.stat().st_size for p in tmp_path.glob("outbox.db*")))
    outbox.db.set_trace_callback(observe)
    for index in range(250):
        outbox.add(sample(tmp_path, index, index))
    assert path.stat().st_size <= 150000
    assert max(observed_sizes) <= 150000
    assert outbox.pending(1)[0]["sequence"] == 249
    history = outbox.history()
    assert len(history) <= 120
    assert len(json.dumps({"schemaVersion": 1, "samples": history}, separators=(",", ":")).encode()) <= 256 * 1024
    outbox.close()


@pytest.mark.parametrize("flag,value", [("--spool-mib", "21"), ("--spool-hours", "25"),
    ("--spool-mib", "0"), ("--spool-hours", "nan"), ("--spool-hours", "0")])
def test_cli_rejects_excessive_or_invalid_spool_limits(monkeypatch, flag, value):
    from tools import monitor_agent
    monkeypatch.setattr("sys.argv", ["monitor_agent", flag, value])
    with pytest.raises(SystemExit) as result:
        monitor_agent.main()
    assert result.value.code == 2


@pytest.mark.parametrize("once", [True, False])
def test_collector_separates_writable_state_from_read_only_snapshots(tmp_path, monkeypatch, once):
    from tools import monitor_agent
    state = tmp_path / "protected/state"
    snapshots = tmp_path / "deployment/cache/monitoring"
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"token": "x" * 32, "endpoint": "https://example.test/api/v1/heartbeat"}))
    monkeypatch.setattr("sys.argv", ["monitor_agent", "--config", str(config),
        "--state-directory", str(state), "--snapshot-directory", str(snapshots)] + (["--once"] if once else []))
    monkeypatch.setattr(monitor_agent, "SystemSampler", lambda directory: EmptySampler())
    observed = []
    def collect(boot_id, sequence, directory, sampler):
        observed.append(directory)
        return m.heartbeat(boot_id, sequence, directory, sampler, now=100)
    monkeypatch.setattr(monitor_agent, "heartbeat", collect)
    monkeypatch.setattr(monitor_agent, "deliver", lambda *args: False)
    thread_args = []
    class SenderThread:
        def __init__(self, *, target, daemon, args):
            assert target is monitor_agent.send_loop
            thread_args.append(args)

        def start(self):
            pass

        def is_alive(self):
            return False
    monkeypatch.setattr(monitor_agent.threading, "Thread", SenderThread)
    assert monitor_agent.main() == 1
    assert observed == [snapshots]
    assert (state / "outbox.sqlite3").is_file()
    assert not snapshots.exists()
    assert not snapshots.parent.exists()
    if not once:
        assert thread_args[0][0] == state / "outbox.sqlite3"


def test_collector_default_directories_remain_compatible(tmp_path, monkeypatch):
    from tools import monitor_agent
    monkeypatch.setattr(monitor_agent, "PROJECT_ROOT", tmp_path)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"token": "x" * 32, "endpoint": "https://example.test/api/v1/heartbeat"}))
    monkeypatch.setattr("sys.argv", ["monitor_agent", "--once", "--config", str(config)])
    monkeypatch.setattr(monitor_agent, "SystemSampler", lambda directory: EmptySampler())
    monkeypatch.setattr(monitor_agent, "deliver", lambda *args: True)
    assert monitor_agent.main() == 0
    assert (tmp_path / "cache/monitoring/outbox.sqlite3").is_file()


def test_collector_directory_validation_rejects_relative_network_and_files(tmp_path):
    from tools.monitor_agent import local_directory
    with pytest.raises(ValueError):
        local_directory(Path("relative/cache"))
    with pytest.raises(ValueError):
        local_directory(Path("//server/share/cache"))
    regular_file = tmp_path / "file"
    regular_file.write_text("test")
    with pytest.raises(ValueError):
        local_directory(regular_file)
    with pytest.raises(ValueError):
        local_directory(regular_file / "child")
    missing = tmp_path / "missing/subdirectory"
    assert local_directory(missing) == missing
    assert not missing.exists()


def test_collector_directory_validation_rejects_reparse_ancestors(tmp_path, monkeypatch):
    from tools.monitor_agent import local_directory
    actual_lstat = Path.lstat
    def lstat(path, *args, **kwargs):
        if path == tmp_path:
            return SimpleNamespace(st_mode=0o40755, st_file_attributes=0x400)
        return actual_lstat(path, *args, **kwargs)
    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(ValueError, match="reparse"):
        local_directory(tmp_path / "missing")


def test_https_auth_no_redirect_and_network_failure_isolation():
    class Offline:
        def open(self, request, timeout):
            assert timeout == 5
            assert request.headers["Authorization"] == "Bearer secret"
            assert request.get_header("User-agent") == "Cartography-Monitor/1.0"
            raise OSError("secret token in raw error")
    assert not m.post_heartbeat("https://example.test/api/v1/heartbeat", "secret", {}, opener=Offline())
    assert not m.post_heartbeat("http://example.test/api/v1/heartbeat", "secret", {})
    assert m.NoRedirect().redirect_request(None, None, 302, "", {}, "https://other.test") is None


def test_config_contains_only_monitor_token_and_endpoint(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"token": "a" * 32, "endpoint": "https://example.test/api/v1/heartbeat",
                               "deviceId": "local-label", "AWS_SECRET_ACCESS_KEY": "must not load"}))
    assert set(m.load_config(path)) == {"token", "endpoint"}


class Recorder:
    def __init__(self):
        self.events = []

    def update(self, **values):
        self.events.append(dict(values))


def test_s3_progress_only_counts_completed_uploads_and_verified_downloads(tmp_path):
    from test_journey_sync import FakeStore, make_archive
    from app.journey_sync import sync_once
    make_archive(tmp_path)
    recorder = Recorder()
    progress = m.UploadProgress(recorder)
    store = FakeStore()
    store.corrupt_upload = True
    result = sync_once(tmp_path, store, progress=progress, prune=True, cache_bytes=0)
    assert result.errors
    assert progress.values["completed_bytes"] > 0
    assert progress.values["verified_bytes"] == 0
    assert progress.values["last_verified_at"] is None
    assert progress.values["pending_archives"] == 1
    assert progress.values["invalid_archives"] == 0
    assert "pruning" not in [event["phase"] for event in recorder.events]
    assert progress.values["phase"] == "error"


def test_bundle_progress_includes_packing_upload_verify_prune_and_receipt_reuse(tmp_path):
    from test_journey_bundle_sync import FakeTransport, make_archive
    from app.journey_bundle_sync import sync_bundles_once
    make_archive(tmp_path)
    recorder = Recorder()
    progress = m.UploadProgress(recorder)
    store = FakeTransport()
    assert not sync_bundles_once(tmp_path, store, progress=progress).errors
    completed, verified = progress.values["completed_bytes"], progress.values["verified_bytes"]
    assert completed == verified > 0
    assert progress.values["last_verified_at"] is not None
    assert {"scan", "packing", "uploading", "verifying", "idle"} <= {event["phase"] for event in recorder.events}
    assert not sync_bundles_once(tmp_path, store, progress=progress).errors
    assert progress.values["completed_bytes"] == completed
    assert progress.values["verified_bytes"] == verified
    assert not sync_bundles_once(tmp_path, store, progress=progress, prune=True, cache_bytes=0).errors
    assert progress.values["verified_bytes"] > verified
    assert progress.values["verified_archives"] == 0
    assert "pruning" in {event["phase"] for event in recorder.events}


def test_failed_publisher_cannot_fail_archive_sync(tmp_path):
    from test_journey_sync import FakeStore, make_archive
    from app.journey_sync import sync_once
    class BrokenPublisher:
        def update(self, **values):
            raise OSError("disk offline")
    make_archive(tmp_path)
    result = sync_once(tmp_path, FakeStore(), progress=m.UploadProgress(BrokenPublisher()))
    assert result.uploaded == ["first"] and not result.errors


@pytest.mark.parametrize("transport", ["s3", "bundle"])
def test_resumed_prune_removes_archive_from_status_queues(tmp_path, monkeypatch, transport):
    if transport == "s3":
        from test_journey_sync import FakeStore, make_archive, interrupt_prune
        from app.journey_sync import sync_once as synchronize
        store = FakeStore()
    else:
        from test_journey_bundle_sync import FakeTransport, make_archive, interrupt_prune
        from app.journey_bundle_sync import sync_bundles_once as synchronize
        store = FakeTransport()
    make_archive(tmp_path)
    interrupt_prune(tmp_path, store, monkeypatch)
    progress = m.UploadProgress()
    assert not synchronize(tmp_path, store, progress=progress, prune=True, cache_bytes=0).errors
    assert progress.values["pending_archives"] == 0
    assert progress.values["active_archives"] == 0
    assert progress.values["verified_archives"] == 0
    assert progress.values["pruned_archives"] == 1
