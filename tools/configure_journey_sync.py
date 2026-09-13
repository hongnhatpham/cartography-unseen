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
    config_home = None
    if os.name == "nt":
        import ctypes
        import msvcrt
        configured = os.environ.get("XDG_CONFIG_HOME")
        logical_home = Path(configured) if configured else Path(os.environ["APPDATA"]) / "xdg.config"
        login_path = logical_home / ".wrangler/config/default.toml"
        if login_path.is_file():
            # MSIX AppData redirection applies inside the desktop but not to
            # Task Scheduler. Resolve the actual existing login; do not copy it
            # and create two independently refreshed OAuth credential files.
            function = ctypes.windll.kernel32.GetFinalPathNameByHandleW
            function.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32]
            function.restype = ctypes.c_uint32
            buffer = ctypes.create_unicode_buffer(32768)
            with login_path.open("rb") as stream:
                length = function(msvcrt.get_osfhandle(stream.fileno()), buffer, len(buffer), 0)
            if not 0 < length < len(buffer):
                raise RuntimeError("Could not resolve the Windows Wrangler login location")
            physical = Path(buffer.value.removeprefix("\\\\?\\"))
            config_home = str(physical.parents[2])
            os.environ["XDG_CONFIG_HOME"] = config_home
    WranglerTransport(BUCKET).check_bucket()
    settings = {"transport": "wrangler", "bucket": BUCKET, "account_id": ACCOUNT_ID}
    if config_home:
        settings["wrangler_config_home"] = config_home
    atomic_text(ROOT / "cache" / "journey-storage.json", json.dumps(settings))
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
