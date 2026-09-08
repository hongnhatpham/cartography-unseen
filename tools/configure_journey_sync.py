"""Enable private R2 backup with Wrangler's existing login or scoped S3 keys."""
from __future__ import annotations

import configparser
import argparse
from getpass import getpass
from io import StringIO
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = "https://9591985981c37caaf2ed403cb28f52e5.r2.cloudflarestorage.com"
BUCKET = "cartography-unseen-journeys"
ACCOUNT_ID = "9591985981c37caaf2ed403cb28f52e5"


def enable_sync() -> None:
    config_path = ROOT / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["map_sync_enabled"] = True
    atomic_text(config_path, json.dumps(config, ensure_ascii=False, indent=2) + "\n")


def configure_wrangler() -> None:
    sys.path.insert(0, str(ROOT))
    from app.journey_bundle_sync import WranglerTransport
    os.environ["CLOUDFLARE_ACCOUNT_ID"] = ACCOUNT_ID
    WranglerTransport(BUCKET).check_bucket()
    atomic_text(ROOT / "cache" / "journey-storage.json", json.dumps({
        "transport": "wrangler", "bucket": BUCKET, "account_id": ACCOUNT_ID}))
    enable_sync()
    print("Wrangler backup configured with the existing login. No new credentials were created.")
    print("Restart the installation to start automatic uploads and verified cache cleanup.")


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main() -> None:
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        raise SystemExit("Install first: runtime\\python\\python.exe -m pip install -r requirements-storage.txt")
    print(f"Private R2 bucket: {BUCKET}")
    print("Enter an Object Read & Write token restricted to this bucket. Input is hidden.")
    key = getpass("Access key ID: ").strip()
    secret = getpass("Secret access key: ").strip()
    if not key or not secret or any(c in key + secret for c in "\r\n"):
        raise SystemExit("Both keys are required and must each be one line.")
    client = boto3.client("s3", endpoint_url=ENDPOINT, region_name="auto",
                          aws_access_key_id=key, aws_secret_access_key=secret,
                          config=Config(connect_timeout=10, read_timeout=30,
                                        retries={"max_attempts": 2}))
    try:
        client.head_bucket(Bucket=BUCKET)
    except Exception:
        raise SystemExit("Bucket access failed. Check the keys, bucket permission and connection. Setup was not saved.")
    finally:
        client.close()
    credentials = configparser.RawConfigParser()
    credentials["journey"] = {"aws_access_key_id": key, "aws_secret_access_key": secret}
    encoded = StringIO()
    credentials.write(encoded)
    atomic_text(ROOT / "cache" / "journey-credentials.ini", encoded.getvalue())
    atomic_text(ROOT / "cache" / "journey-storage.json", json.dumps({"endpoint": ENDPOINT, "bucket": BUCKET}))
    enable_sync()
    print("Setup saved locally under ignored cache/. No map was uploaded during setup.")
    print("Restart the installation to start uploads and verified local cache cleanup.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrangler", action="store_true", help="Use the existing Wrangler login without new keys")
    args = parser.parse_args()
    if args.wrangler:
        configure_wrangler()
    else:
        main()
