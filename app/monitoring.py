"""Private, bounded telemetry. Rendering only publishes scalar state in memory.

No archive, credential, environment, log, prompt or image readers belong here.
Network and system probes run exclusively in tools/monitor_agent.py.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

STATES = {"loading", "running", "error", "stopped"}
PHASES = {"idle", "scan", "packing", "uploading", "verifying", "pruning", "retry", "error", "stopped"}
ARTWORK_NUMBERS = {"pid", "displayed_frames", "display_fps", "generation_fps", "frame_age_ms"}
UPLOADER_NUMBERS = {"pid", "completed_bytes", "verified_bytes", "local_bytes", "pending_archives",
                    "verified_archives", "active_archives", "error_archives", "pruned_archives",
                    "invalid_archives", "last_verified_at"}
METRICS = ("cpu_percent", "ram_used_bytes", "ram_total_bytes", "disk_used_bytes", "disk_total_bytes",
           "disk_read_bytes_per_sec", "disk_write_bytes_per_sec", "network_sent_bytes_per_sec",
           "network_recv_bytes_per_sec", "gpu_util_percent", "gpu_memory_used_bytes",
           "gpu_memory_total_bytes", "gpu_temperature_c")
MAX_SNAPSHOT_BYTES = 8192


def number(value):
    # Bound integers before math.isfinite converts them to a C double. JSON can
    # contain integers far larger than a float without being malformed syntax.
    return value if type(value) in (int, float) and 0 <= value <= 2 ** 53 - 1 and math.isfinite(value) else None


def sanitize_snapshot(kind: str, data: dict) -> dict:
    fields = ARTWORK_NUMBERS if kind == "artwork" else UPLOADER_NUMBERS
    result = {key: number(data.get(key)) for key in fields}
    key, choices = ("state", STATES) if kind == "artwork" else ("phase", PHASES)
    value = data.get(key)
    result[key] = value if isinstance(value, str) and value in choices else None
    if kind == "uploader":
        result["error_code"] = "sync_failed" if data.get("error_code") == "sync_failed" else None
    result["updated_at"] = number(data.get("updated_at"))
    return result


def atomic_snapshot(path: Path, kind: str, data: dict) -> bool:
    """Best effort atomic replacement; failures never reach the producer."""
    temporary = None
    try:
        payload = json.dumps({"schema_version": 1, **sanitize_snapshot(kind, data)}, allow_nan=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name, suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            output.write(payload)
        os.replace(temporary, path)
        return True
    except Exception:
        return False
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def read_snapshot(path: Path, kind: str, *, now=None, stale_after=20) -> dict:
    now = time.time() if now is None else now
    read_started = time.monotonic()
    data = {}
    try:
        with path.open("rb") as source:
            raw = source.read(MAX_SNAPSHOT_BYTES + 1)
        if len(raw) <= MAX_SNAPSHOT_BYTES:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and parsed.get("schema_version") == 1:
                data = parsed
    except (OSError, ValueError, UnicodeError, RecursionError):
        pass
    result = sanitize_snapshot(kind, data)
    updated = result["updated_at"]
    # A producer may publish after this read started. Judge freshness at the
    # end of the read, preserving the supplied clock's epoch for replay/tests.
    now += max(0, time.monotonic() - read_started)
    result["stale"] = updated is None or not 0 <= now - updated <= stale_after
    return result


class SnapshotPublisher:
    """One bounded latest-value slot. Only the daemon thread touches the disk.

    The timestamp advances when the producer publishes, never on a timer. A
    stalled renderer therefore cannot look healthy because this thread is alive.
    """

    def __init__(self, path: Path, kind: str, interval=5):
        self.path, self.kind, self.interval = path, kind, interval
        self.latest = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread = None
        try:
            self._thread = threading.Thread(target=self._run, name=f"{kind}-status", daemon=True)
            self._thread.start()
        except Exception:
            self._thread = None

    def update(self, **values):
        try:
            old = self.latest or {}
            self.latest = sanitize_snapshot(self.kind, {**old, **values, "pid": os.getpid(),
                                                       "updated_at": time.time()})
            self._wake.set()
        except Exception:
            pass

    def _run(self):
        while not self._stop.is_set():
            self._wake.wait(self.interval)
            self._wake.clear()
            if self.latest is not None:
                atomic_snapshot(self.path, self.kind, self.latest)
            self._stop.wait(self.interval)
        if self.latest is not None:
            atomic_snapshot(self.path, self.kind, self.latest)

    def close(self):
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=0.1)


class UploadProgress:
    """Scalar session counters; upload completion and checksum verification differ."""

    def __init__(self, publisher=None):
        self.publisher = publisher
        self.values = dict(completed_bytes=0, verified_bytes=0, last_verified_at=None)

    def update(self, **values):
        self.values.update(values)
        try:
            if self.publisher is not None:
                self.publisher.update(**self.values)
        except Exception:
            pass

    def add(self, key, amount):
        self.update(**{key: self.values.get(key, 0) + amount})

    def begin(self, root):
        # Counts are provisional until the normal safety checks inspect each archive.
        pending = active = 0
        try:
            for path in root.iterdir():
                if path.name != ".sync" and path.is_dir():
                    if (path / "complete.json").is_file():
                        pending += 1
                    else:
                        active += 1
        except Exception:
            pending = active = None
        self.update(phase="scan", pending_archives=pending, active_archives=active,
                    verified_archives=0, error_archives=0, invalid_archives=0, pruned_archives=0,
                    local_bytes=None, error_code=None)

    def archive_done(self):
        pending = self.values.get("pending_archives")
        self.update(pending_archives=max(0, pending - 1) if pending is not None else None,
                    verified_archives=self.values.get("verified_archives", 0) + 1)

    def resumed_prune_done(self, completed):
        key = "pending_archives" if completed else "active_archives"
        count = self.values.get(key)
        self.update(**{key: max(0, count - 1) if count is not None else None})
        self.add("pruned_archives", 1)

    def verified(self, size):
        self.add("verified_bytes", size)

    def committed(self):
        self.update(last_verified_at=time.time())

    def failed(self, *, invalid=False, completed=False):
        self.add("error_archives", 1)
        if invalid:
            self.add("invalid_archives", 1)
            pending = self.values.get("pending_archives")
            if completed and pending is not None:
                self.update(pending_archives=max(0, pending - 1))
        self.update(phase="error", error_code="sync_failed")


class CounterRates:
    def __init__(self):
        self.previous = {}

    def rate(self, key, value, now):
        value = number(value)
        prior = self.previous.get(key)
        self.previous[key] = (value, now)
        if value is None or prior is None or prior[0] is None or now <= prior[1] or value < prior[0]:
            return None
        return (value - prior[0]) / (now - prior[1])


def gpu_metrics(run=subprocess.run):
    keys = METRICS[-4:]
    result = dict.fromkeys(keys)
    try:
        response = run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                        "--format=csv,noheader,nounits", "--id=0"], capture_output=True,
                       text=True, timeout=2, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if response.returncode == 0 and len(response.stdout) < 1024:
            cells = response.stdout.strip().split(",")
            if len(cells) == 4:
                for key, cell, multiplier in zip(keys, cells, (1, 1024 ** 2, 1024 ** 2, 1)):
                    try:
                        result[key] = number(float(cell.strip()) * multiplier)
                    except ValueError:
                        pass
    except Exception:
        pass
    return result


class SystemSampler:
    def __init__(self, disk: Path, psutil_module=None):
        self.disk, self.rates = disk, CounterRates()
        self.cpu_primed = False
        self.psutil = psutil_module
        if self.psutil is None:
            try:
                import psutil
                self.psutil = psutil
            except ImportError:
                pass

    def metrics(self, now):
        result = dict.fromkeys(METRICS)
        p = self.psutil
        if p is not None:
            probes = [
                (lambda: {"cpu_percent": p.cpu_percent(interval=None)}),
                (lambda: {"ram_used_bytes": p.virtual_memory().used, "ram_total_bytes": p.virtual_memory().total}),
                (lambda: {"disk_used_bytes": p.disk_usage(str(self.disk)).used,
                          "disk_total_bytes": p.disk_usage(str(self.disk)).total}),
            ]
            for probe in probes:
                try:
                    result.update({k: number(v) for k, v in probe().items()})
                except Exception:
                    pass
            if not self.cpu_primed:
                result["cpu_percent"] = None
                self.cpu_primed = True
            for fn, fields in ((p.disk_io_counters, {"disk_read_bytes_per_sec": "read_bytes", "disk_write_bytes_per_sec": "write_bytes"}),
                               (p.net_io_counters, {"network_sent_bytes_per_sec": "bytes_sent", "network_recv_bytes_per_sec": "bytes_recv"})):
                try:
                    counters = fn()
                except Exception:
                    counters = None
                for key, attribute in fields.items():
                    result[key] = self.rates.rate(key, getattr(counters, attribute, None), now)
        result.update(gpu_metrics())
        return result

    def process_alive(self, pid):
        if self.psutil is None or type(pid) is not int or pid <= 0:
            return None
        try:
            return self.psutil.pid_exists(pid)
        except Exception:
            return None


def iso_time(value):
    try:
        return datetime.fromtimestamp(value, timezone.utc).isoformat().replace("+00:00", "Z") if value is not None else None
    except (OSError, ValueError, OverflowError):
        return None


def heartbeat(boot_id, sequence, directory: Path, sampler, *, now=None):
    now = time.time() if now is None else now
    started = time.monotonic()
    metrics = sampler.metrics(now)
    mapping = dict(zip(METRICS, ("cpuPercent", "ramUsedBytes", "ramTotalBytes", "diskUsedBytes", "diskTotalBytes",
        "diskReadBytesPerSecond", "diskWriteBytesPerSecond", "networkTxBytesPerSecond", "networkRxBytesPerSecond",
        "gpuPercent", "vramUsedBytes", "vramTotalBytes", "gpuTemperatureC")))
    system = {mapping[key]: number(metrics.get(key)) for key in METRICS}
    used = system.pop("diskUsedBytes")
    system["diskFreeBytes"] = system["diskTotalBytes"] - used if used is not None and system["diskTotalBytes"] is not None else None
    app = read_snapshot(directory / "artwork.json", "artwork", now=now + max(0, time.monotonic() - started))
    sync = read_snapshot(directory / "uploader.json", "uploader", now=now + max(0, time.monotonic() - started), stale_after=1800)
    app_alive, sync_alive = sampler.process_alive(app["pid"]), sampler.process_alive(sync["pid"])
    app_state = {"loading": "starting", "running": "running", "stopped": "stopped", "error": "error"}.get(app["state"], "unknown")
    if app_alive is False:
        app_state = "stopped"
    elif app["stale"] and app_state in ("starting", "running"):
        app_state = "stalled"
    phase = {"scan": "scanning", "retry": "retrying", "stopped": "idle"}.get(sync["phase"], sync["phase"] or "unknown")
    sync_error = sync["error_code"]
    if sync_alive is False and phase not in ("idle", "unknown"):
        phase, sync_error = "error", "sync_failed"
    elif sync["stale"]:
        phase, sync_error = "unknown", None
    age = lambda snapshot: max(0, now - snapshot["updated_at"]) if snapshot["updated_at"] is not None else None
    return {"schemaVersion": 1, "bootId": boot_id, "sequence": sequence, "sampledAt": iso_time(now),
        "agentVersion": "1.0.0", "system": system,
        "app": {"state": app_state, "processRunning": app_alive, "statusAgeSeconds": age(app),
            "displayFps": app["display_fps"], "generationFps": app["generation_fps"],
            "lastFrameAgeSeconds": app["frame_age_ms"] / 1000 if app["frame_age_ms"] is not None else None,
            "errorCode": "application_error" if app_state == "error" else None},
        "archiveSync": {"enabled": True if sync["pid"] is not None else None,
            "processRunning": sync_alive, "state": phase, "statusAgeSeconds": age(sync),
            "pendingArchives": sync["pending_archives"], "pendingBytes": None,
            "verifiedLocalArchives": sync["verified_archives"], "incompleteArchives": sync["active_archives"],
            "invalidArchives": sync["invalid_archives"], "localBytes": sync["local_bytes"], "currentArchiveId": None,
            "completedPayloadBytes": sync["completed_bytes"], "verifiedBytes": sync["verified_bytes"],
            "lastVerifiedAt": iso_time(sync["last_verified_at"]), "errorCode": sync_error}}


class Outbox:
    """A deduplicated, size- and age-bounded queue. SQLite reuses freed pages."""

    def __init__(self, path, max_bytes=20 * 1024 ** 2, max_age=86400):
        self.max_bytes, self.max_age = max_bytes, max_age
        # A rollback journal can briefly approach the database's size. Reserve
        # space for it and for one maximum-sized sample before trimming.
        self.database_limit = max(16384, max_bytes // 2 - 32768)
        self.db = sqlite3.connect(path, timeout=1)
        self.db.execute("PRAGMA journal_mode=DELETE")
        self.db.execute("PRAGMA auto_vacuum=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS outbox (id TEXT PRIMARY KEY, sampled REAL NOT NULL, payload TEXT NOT NULL, size INTEGER NOT NULL)")
        self.db.commit()

    def trim(self, now):
        with self.db:
            self.db.execute("DELETE FROM outbox WHERE sampled < ?", (now - self.max_age,))
            total = self.db.execute("SELECT coalesce(sum(size),0) FROM outbox").fetchone()[0]
            while total > max(0, self.database_limit - 16384):
                row = self.db.execute("SELECT id,size FROM outbox ORDER BY sampled,id LIMIT 1").fetchone()
                if row is None:
                    break
                self.db.execute("DELETE FROM outbox WHERE id=?", (row[0],))
                total -= row[1]
        # Include SQLite indexes and page overhead in the disk bound, not just JSON.
        while self.db.execute("PRAGMA page_count").fetchone()[0] * self.db.execute("PRAGMA page_size").fetchone()[0] > self.database_limit:
            with self.db:
                cursor = self.db.execute("DELETE FROM outbox WHERE id IN (SELECT id FROM outbox ORDER BY sampled,id LIMIT 16)")
            if cursor.rowcount == 0:
                break

    def add(self, payload):
        encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False)
        if len(encoded.encode("utf-8")) > 16384:
            raise ValueError("Heartbeat exceeds the size limit")
        sample_id = f"{payload['bootId']}:{payload['sequence']}"
        sampled = datetime.fromisoformat(payload["sampledAt"].replace("Z", "+00:00")).timestamp()
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO outbox VALUES (?,?,?,?)",
                            (sample_id, sampled, encoded, len(encoded.encode("utf-8"))))
        self.trim(sampled)

    def pending(self, limit=5):
        # The freshest snapshot always wins. Replay is bounded per POST interval.
        return [json.loads(row[0]) for row in self.db.execute(
            "SELECT payload FROM outbox ORDER BY sampled DESC,id DESC LIMIT ?", (limit,))]

    def history(self, limit=120):
        samples, size = [], 64
        for sample in self.pending(limit):
            encoded_size = len(json.dumps(sample, separators=(",", ":")).encode("utf-8")) + 1
            if size + encoded_size > 256 * 1024:
                break
            samples.append(sample)
            size += encoded_size
        return samples

    def ack(self, sample_id):
        with self.db:
            self.db.execute("DELETE FROM outbox WHERE id=?", (sample_id,))

    def close(self):
        self.db.close()


def validate_endpoint(endpoint):
    parsed = urlsplit(endpoint)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment):
        raise ValueError("Monitoring endpoint must be an HTTPS URL without credentials, query or fragment")
    return endpoint


def load_config(path: Path, endpoint=None):
    with path.open("rb") as source:
        raw = source.read(8193)
    if len(raw) > 8192:
        raise ValueError("Monitoring configuration is too large")
    settings = json.loads(raw)
    token = settings.get("token", "")
    if not isinstance(token, str) or not 16 <= len(token) <= 512 or any(ord(c) < 33 or ord(c) > 126 for c in token):
        raise ValueError("Monitoring token must be 16-512 printable characters")
    return {"token": token, "endpoint": validate_endpoint(endpoint or settings.get("endpoint", ""))}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def post_heartbeat(endpoint, token, payload, *, opener=None):
    """Do not follow redirects: a device bearer token must stay on its endpoint."""
    try:
        validate_endpoint(endpoint)
        request = Request(endpoint, data=json.dumps(payload, separators=(",", ":"), allow_nan=False).encode(),
                          headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}",
                                   "User-Agent": "Cartography-Monitor/1.0"},
                          method="POST")
        with (opener or build_opener(NoRedirect)).open(request, timeout=5) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def deliver(outbox, settings, *, post=post_heartbeat):
    """Freshest goes to live ingestion; bounded replay only goes to history."""
    try:
        pending = outbox.pending(1)
        if not pending:
            return True
        payload = pending[0]
        if not post(settings["endpoint"], settings["token"], payload):
            return False
        outbox.ack(f"{payload['bootId']}:{payload['sequence']}")
        samples = outbox.history()
        if samples:
            endpoint = settings["endpoint"].rsplit("/", 1)[0] + "/history"
            if not post(endpoint, settings["token"], {"schemaVersion": 1, "samples": samples}):
                return False
            for sample in samples:
                outbox.ack(f"{sample['bootId']}:{sample['sequence']}")
        return True
    except Exception:
        return False
