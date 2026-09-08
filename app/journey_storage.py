"""Disk monitoring and optional uploader startup outside the render loop."""
from __future__ import annotations

import logging
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading


class DiskSpaceMonitor:
    """Sample free space in a thread; an unavailable disk also pauses capture."""

    def __init__(self, path: Path, minimum_bytes: int):
        self.path = path
        self.minimum_bytes = minimum_bytes
        self.available = False
        self._stop = threading.Event()
        self._sample()
        self._thread = threading.Thread(target=self._run, name="journey-disk", daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        try:
            self.available = shutil.disk_usage(self.path).free >= self.minimum_bytes
        except OSError:
            self.available = False

    def _run(self) -> None:
        while not self._stop.wait(5):
            self._sample()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1)


def sync_environment(root: Path) -> dict[str, str]:
    """Load this installation's private setup without changing other AWS tools."""
    environment = os.environ.copy()
    setup = root / "cache" / "journey-storage.json"
    if setup.exists():
        settings = json.loads(setup.read_text(encoding="utf-8"))
        environment.setdefault("JOURNEY_STORAGE_TRANSPORT", settings.get("transport", "s3"))
        if settings.get("endpoint"):
            environment.setdefault("JOURNEY_S3_ENDPOINT", settings["endpoint"])
        environment.setdefault("JOURNEY_S3_BUCKET", settings["bucket"])
        if settings.get("account_id"):
            environment.setdefault("CLOUDFLARE_ACCOUNT_ID", settings["account_id"])
        # A saved setup uses its dedicated profile unless explicit credentials exist.
        if (environment["JOURNEY_STORAGE_TRANSPORT"] == "s3" and
                not environment.get("AWS_ACCESS_KEY_ID") and not environment.get("AWS_PROFILE")):
            environment["AWS_SHARED_CREDENTIALS_FILE"] = str(root / "cache" / "journey-credentials.ini")
            environment["AWS_PROFILE"] = "journey"
    return environment


def start_sync(root: Path, archive_root: Path, cache_gib: float) -> None:
    """Start an opt-in uploader that can finish the final archive after app exit."""
    environment = sync_environment(root)
    transport = environment.get("JOURNEY_STORAGE_TRANSPORT", "s3")
    if transport == "wrangler":
        if not environment.get("JOURNEY_S3_BUCKET"):
            raise RuntimeError("Configure the R2 bucket before enabling map sync")
    elif transport == "s3":
        if not environment.get("JOURNEY_S3_ENDPOINT") or not environment.get("JOURNEY_S3_BUCKET"):
            raise RuntimeError("Set JOURNEY_S3_ENDPOINT and JOURNEY_S3_BUCKET before enabling map sync")
        import importlib.util
        if importlib.util.find_spec("boto3") is None:
            raise RuntimeError("Install requirements-storage.txt before enabling map sync")
    else:
        raise RuntimeError("Journey storage transport must be s3 or wrangler")
    executable = Path(sys.executable)
    options = {}
    if os.name == "nt":
        windowless = executable.with_name("pythonw.exe")
        if windowless.exists():
            executable = windowless
        options["creationflags"] = subprocess.CREATE_NO_WINDOW
    else:
        options["start_new_session"] = True
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / "journey-sync.log").open("ab") as output:
        subprocess.Popen(
            [str(executable), "-u", str(root / "tools" / "sync_journeys.py"),
             "--root", str(archive_root), "--watch", "--prune", "--cache-gib", str(cache_gib)],
            cwd=root, env=environment, stdin=subprocess.DEVNULL, stdout=output, stderr=output, **options,
        )
    logging.info("Journey uploader started; status is in logs/journey-sync.log")
