"""Opt-in system telemetry process. Reads only its config and scalar snapshots."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import random
import stat
import sys
import threading
import time
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.monitoring import Outbox, SystemSampler, deliver, heartbeat, load_config


def local_directory(path: Path) -> Path:
    """Validate local absolute directories without creating or following links."""
    if not path.is_absolute() or str(path).startswith(("\\\\", "//")):
        raise ValueError("Monitoring directories must be absolute local paths")
    for component in (path, *path.parents):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Monitoring directories must not contain links or reparse points")
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("Monitoring directory paths must not contain files")
    resolved = path.resolve()
    if os.name == "nt":
        import ctypes
        drive_type = ctypes.windll.kernel32.GetDriveTypeW
        drive_type.argtypes = [ctypes.c_wchar_p]
        drive_type.restype = ctypes.c_uint
        if drive_type(resolved.anchor) in (0, 1, 4):
            raise ValueError("Monitoring directories must be on a local drive")
    return resolved


def send_loop(path, settings, stop, ready, interval, max_bytes, max_age):
    """Network waits never delay the five-second sampling loop."""
    outbox = None
    try:
        outbox = Outbox(path, max_bytes, max_age)
        ready.wait()
        failures = 0
        while not stop.is_set():
            outbox.trim(time.time())
            ok = deliver(outbox, settings)
            failures = 0 if ok else min(failures + 1, 6)
            delay = interval if ok else min(300, interval * 2 ** failures) * random.uniform(0.8, 1.2)
            stop.wait(delay)
    except Exception:
        # The process can restart this optional collector; never reveal raw errors.
        print("Monitoring delivery unavailable; retained samples stay in the local outbox.", file=sys.stderr)
    finally:
        if outbox is not None:
            outbox.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "cache/monitoring/config.json")
    parser.add_argument("--endpoint", help="HTTPS heartbeat ingestion URL, overrides config")
    parser.add_argument("--state-directory", type=Path, default=PROJECT_ROOT / "cache/monitoring",
                        help="Absolute local directory for writable collector state and outbox")
    parser.add_argument("--snapshot-directory", type=Path, default=PROJECT_ROOT / "cache/monitoring",
                        help="Absolute local directory to read artwork and uploader snapshots; never created")
    parser.add_argument("--sample-seconds", type=float, default=5)
    parser.add_argument("--post-seconds", type=float, default=30)
    parser.add_argument("--spool-mib", type=float, default=20)
    parser.add_argument("--spool-hours", type=float, default=24)
    parser.add_argument("--once", action="store_true", help="Collect and attempt delivery once")
    args = parser.parse_args()
    for key in ("sample_seconds", "post_seconds", "spool_mib", "spool_hours"):
        value = getattr(args, key)
        if not math.isfinite(value) or value <= 0:
            parser.error("Intervals and spool limits must be finite and positive")
    if not 1 <= args.spool_mib <= 20:
        parser.error("--spool-mib must be between 1 and 20")
    if args.spool_hours > 24:
        parser.error("--spool-hours must not exceed 24")
    outbox = None
    stop, ready = threading.Event(), threading.Event()
    sender = None
    try:
        settings = load_config(args.config, args.endpoint)
        state_directory = local_directory(args.state_directory)
        snapshot_directory = local_directory(args.snapshot_directory)
        state_directory.mkdir(parents=True, exist_ok=True)
        state_directory = local_directory(state_directory)
        if os.name == 'nt' and not args.once:
            # The scheduled collector must survive closing unrelated console
            # windows. Keep bounded, redacted diagnostics before detaching.
            error_log = state_directory / 'agent-errors.log'
            if error_log.exists() and error_log.stat().st_size >= 65536:
                os.replace(error_log, state_directory / 'agent-errors.previous.log')
            log_stream = error_log.open('a', encoding='utf-8', buffering=1)
            sys.stdout = sys.stderr = log_stream
            import ctypes
            ctypes.windll.kernel32.FreeConsole()
        outbox_path = state_directory / "outbox.sqlite3"
        outbox = Outbox(outbox_path, int(args.spool_mib * 1024 ** 2), args.spool_hours * 3600)
        sampler = SystemSampler(PROJECT_ROOT)
        sequence = 0
        boot_id = str(uuid.uuid4())
        if not args.once:
            sender = threading.Thread(target=send_loop, daemon=True, args=(outbox_path,
                settings, stop, ready, args.post_seconds, int(args.spool_mib * 1024 ** 2), args.spool_hours * 3600))
            sender.start()
        while True:
            started = time.monotonic()
            outbox.add(heartbeat(boot_id, sequence, snapshot_directory, sampler))
            sequence += 1
            ready.set()
            if args.once:
                return 0 if deliver(outbox, settings) else 1
            if sender is not None and not sender.is_alive():
                return 1
            time.sleep(max(0, args.sample_seconds - (time.monotonic() - started)))
    except KeyboardInterrupt:
        return 0
    except Exception:
        # Raw exceptions can contain URLs, paths, hostnames or credentials.
        print("Monitoring agent stopped; check its configuration and local cache access.", file=sys.stderr)
        return 1
    finally:
        stop.set()
        ready.set()
        if outbox is not None:
            outbox.close()


if __name__ == "__main__":
    raise SystemExit(main())
