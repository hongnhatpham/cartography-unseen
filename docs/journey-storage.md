# Journey storage

New archives save lossless WebP images, the complete route and prompt manifest,
an interactive HTML viewer, and `complete.json`. The completion marker includes
the manifest's SHA-256 checksum and is written only after all requested exports
finish. A checkpoint without that marker is never uploaded or removed.

SVG export is optional. It embeds another copy of every captured image and is
not needed for the interactive viewer. Generate it later with:

```powershell
runtime/python/python.exe tools/export_journey_svg.py journeys/<id>
```

Use `--output path/to/map.svg` to keep the export outside the archive. Existing
PNG archives remain readable and exportable. This change does not migrate or
delete old archives, screenshots or performance logs. Older archives without
completion markers remain local and require a separate validated migration
before automatic backup can include them.

## Private R2 setup

The private bucket `cartography-unseen-journeys` was created on 7 September 2026
in APAC, using Standard storage. Public r2.dev access is disabled and no custom
domain is attached. Uploading here does not publish an archive to Emergent Play.

### Wrangler login

This machine can back up through its existing Wrangler login without creating
new S3 credentials. To configure that mode:

```powershell
runtime/python/python.exe tools/configure_journey_sync.py --wrangler
```

Wrangler mode stores a portable ZIP per completed journey. Each ZIP contains
the manifest, captures, completion marker and interactive viewer with their
relative paths intact. Small archives upload as a single ZIP; larger archives
stream into parts of at most 128 MiB to stay below Wrangler's 300 MiB object
limit. Upload uses only one temporary part on disk. Each object key includes its
SHA-256 checksum, so different contents have different keys. A final
`<zip-sha256>.zip.json` descriptor records the ordered parts and file inventory.
The uploader downloads and verifies the parts before recording success and again
before local cleanup. At least 192 MiB free temporary space is required.

The receipt in `journeys/.sync/bundle-<id>.json` contains the descriptor's `key`.
Restore with:

```powershell
runtime/python/python.exe tools/restore_journey.py "journeys/<id>/<sha256>.zip.json"
```

The command verifies the descriptor, every part, the complete ZIP and every
extracted file, then creates `restored-journeys/<id>/`. Existing directories are
never overwritten. Use `--output-root path/to/restore` for another destination.
Restoring needs room for the downloaded ZIP and extracted archive. Open the
restored `index.html` or serve the directory on the website. Backing up does not
publish that directory.

This mode requires Node, Wrangler and a valid Wrangler login for the pinned
Cloudflare account. If the login expires and cannot refresh, local archives stay
on disk and uploads report an error. Run `wrangler login` interactively to restore
access. Setup stores only the transport, bucket and account ID under `cache/`;
Wrangler manages its own credentials.

### Scoped S3 credentials

The S3 mode below stores individual files directly in R2. It remains available
for installations using a bucket-scoped access key instead of a Wrangler login.

1. In Cloudflare, open R2, then manage R2 API tokens. Create an Object Read & Write
   token restricted to `cartography-unseen-journeys`.
2. Install the optional uploader dependency into the portable runtime:

   ```powershell
   runtime/python/python.exe -m pip install -r requirements-storage.txt
   ```

3. With the installation closed, run:

   ```powershell
   runtime/python/python.exe tools/configure_journey_sync.py
   ```

   Enter the access key ID and secret in the hidden local prompts. Do not paste
   them into chat or commit them. The helper checks bucket access, stores a
   dedicated AWS credential profile in ignored `cache/journey-credentials.ini`,
   and enables `map_sync_enabled` in `config.json`. It stores the endpoint and
   bucket in `cache/journey-storage.json`. It uploads no map during setup.
4. Restart the installation. A separate uploader checks for completed maps every
   60 seconds and continues after the artwork exits, so it can upload the final
   map. Later artwork launches can start it again after a reboot. An OS lock
   prevents two uploaders from processing the same folder simultaneously.

Keep the local credential file private when copying the installation to another
machine. The helper uses the bucket endpoint
`https://9591985981c37caaf2ed403cb28f52e5.r2.cloudflarestorage.com`.

For a different destination or externally managed credentials, set
`JOURNEY_S3_ENDPOINT`, `JOURNEY_S3_BUCKET`, and optionally `JOURNEY_S3_PREFIX`.
The prefix defaults to `journeys`. The uploader uses boto3's AWS credentials
chain in S3 mode; explicit environment settings take precedence over saved setup.
`JOURNEY_STORAGE_TRANSPORT` selects `s3` or `wrangler`. Wrangler setup pins
`CLOUDFLARE_ACCOUNT_ID` to the configured account in the uploader process.
Do not change destinations while an uploader is running.

## Settings and operation

| Config key | Default | Behavior |
|---|---|---|
| `map_image_format` | `webp` | Lossless WebP or PNG for new captures |
| `map_export_svg` | `false` | Include SVG at each completed save |
| `map_sync_enabled` | `false` | Start private upload and verified local cleanup |
| `map_cache_gib` | `5` | Target total archive size on this machine |
| `map_min_free_gib` | `1` | Pause path and image capture below this free-space reserve |

These settings apply at launch. Free space is sampled every five seconds outside
the render loop. Prompt metadata and checkpoints continue during a low-disk pause.
The reserve is not a filesystem quota. External writers or a sudden disk failure
can still cause a save error; the app retains pending data for retry.

Upload runs independently of recording. Status is in `logs/journey-sync.log`.
To run one upload pass manually without deleting local copies:

```powershell
runtime/python/python.exe tools/sync_journeys.py
```

Add `--watch` for repeated passes. Add `--prune --cache-gib 5` to allow verified
local cleanup. Close the existing uploader first when switching to a manual run.
To stop automatic backup, disable `map_sync_enabled` and end the Python process
whose command line contains `tools/sync_journeys.py`. Disabling the setting alone
does not stop an already running uploader.

## Verification and recovery

In S3 mode, files keep their relative paths under `journeys/<id>/` in R2. Uploads refuse to
overwrite conflicting objects, publish the manifest and completion marker last,
and verify the uploaded bytes by downloading and hashing them. Successful
receipts live in `journeys/.sync/` and include the destination and file checksums.
Retries reuse already uploaded files; they do not re-upload a whole archive.

Before local removal, the uploader checks every remote checksum again and
confirms that the local files still match. It removes the oldest verified
archives first. Unknown files, checksum mismatches, interrupted uploads and
destination changes preserve local data. Explicit pruning state allows an
interrupted cleanup to resume after remote verification.

The cache can exceed its target during an outage or a long active journey.
Active and unsynced captures are retained even when the cache is full. At the
free-space reserve, recording pauses. Space finalizes the current map so it can
be uploaded; capturing resumes when disk space returns. No unsynced archive is
deleted to force compliance with the target.

To restore an S3 archive, download its entire `journeys/<id>/` object prefix,
preserving the manifest and image paths, and open `index.html`. Use the viewer's
Open archive action if the browser blocks direct local image loading. The website
can consume these same files using one shared viewer; public selection, delivery
and website deployment remain separate work. For Wrangler backups, use
`tools/restore_journey.py` with the descriptor key as described above.
