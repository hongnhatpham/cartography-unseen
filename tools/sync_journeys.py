"""Run optional private journey uploads independently of the viewer and renderer."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.journey_sync import S3Store, sync_lock, sync_once
from app.journey_storage import sync_environment
from app.monitoring import SnapshotPublisher, UploadProgress, read_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "journeys")
    parser.add_argument("--transport", choices=("s3", "wrangler"),
                        help="Override the saved storage transport")
    parser.add_argument("--watch", nargs="?", const=60.0, type=float, metavar="SECONDS",
                        help="Repeat uploads every SECONDS, default 60")
    parser.add_argument("--cache-gib", type=float, default=5.0,
                        help="Local cache target when --prune is enabled, default 5 GiB")
    parser.add_argument("--prune", action="store_true",
                        help="Remove oldest uploaded local archives after verifying remote bytes")
    args = parser.parse_args()
    if not math.isfinite(args.cache_gib) or args.cache_gib < 0:
        parser.error("--cache-gib must be finite and nonnegative")
    if args.watch is not None and (not math.isfinite(args.watch) or args.watch <= 0):
        parser.error("--watch must be finite and positive")
    publisher = None
    progress = UploadProgress()
    try:
        os.environ.update(sync_environment(PROJECT_ROOT))
        transport = args.transport or os.environ.get("JOURNEY_STORAGE_TRANSPORT", "s3")
        if transport == "wrangler":
            from app.journey_bundle_sync import WranglerTransport, sync_bundles_once
            store = WranglerTransport(os.environ.get("JOURNEY_S3_BUCKET", ""))
            def synchronize():
                return sync_bundles_once(args.root, store,
                    prefix=os.environ.get("JOURNEY_S3_PREFIX", "journeys"),
                    cache_bytes=int(args.cache_gib * 1024 ** 3), prune=args.prune, progress=progress)
        elif transport == "s3":
            store = S3Store.from_env()
            def synchronize():
                return sync_once(args.root, store, cache_bytes=int(args.cache_gib * 1024 ** 3),
                                 prune=args.prune, progress=progress)
        else:
            raise ValueError("Journey storage transport must be s3 or wrangler")
        with sync_lock(args.root):
            previous = read_snapshot(PROJECT_ROOT / "cache/monitoring/uploader.json", "uploader")
            publisher = SnapshotPublisher(PROJECT_ROOT / "cache/monitoring/uploader.json", "uploader")
            progress.publisher = publisher
            progress.update(phase="scan", last_verified_at=previous["last_verified_at"])
            while True:
                result = synchronize()
                print(f"Verified receipts: {len(result.uploaded)}; removed: {len(result.pruned)}; "
                      f"local data: {result.local_bytes / 1024 ** 3:.2f} GiB", flush=True)
                for error in result.errors:
                    print(error, file=sys.stderr, flush=True)
                if args.prune and result.local_bytes > args.cache_gib * 1024 ** 3:
                    print("Cache remains over target; active or unverified data is retained.",
                          file=sys.stderr, flush=True)
                if args.watch is None:
                    return 1 if result.errors else 0
                if result.errors:
                    progress.update(phase="retry")
                time.sleep(args.watch)
    except KeyboardInterrupt:
        progress.update(phase="stopped")
        return 0
    except Exception as error:
        progress.failed()
        print(f"Journey sync stopped: {error}", file=sys.stderr)
        return 1
    finally:
        if publisher is not None:
            publisher.close()


if __name__ == "__main__":
    raise SystemExit(main())
