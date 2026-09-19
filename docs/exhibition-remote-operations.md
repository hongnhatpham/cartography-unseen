# Exhibition remote operations

The exhibition PC is the Windows host reached through the SSH alias
`hp-zbook`. The deployment is `C:\Exhibition\Cartography`. The `hong-ssh`
account has key-only, high-integrity administrator access from the approved
Tailscale fleet peers.

## Start with telemetry, not session commands

OpenSSH runs outside the interactive desktop. `query user` and administrative
CIM/WMI now work, but UI APIs still cannot inspect or control the console window
station reliably. Use telemetry for application health and the console for
visual confirmation.

The authoritative cross-session state is in:

- `cache\monitoring\supervisor.json`: supervisor heartbeat, app PID, restarts,
  and fault/maintenance state.
- `cache\monitoring\artwork.json`: application heartbeat, display FPS,
  generation FPS, frame age, rendered-frame count, and runtime state.
- `cache\monitoring\uploader.json`: uploader heartbeat, pending/error/invalid
  archive counts, verified archives, local bytes, and upload phase.
- `logs\supervisor\supervisor.log`: app starts, exits, and restart history.
- `logs\journey-sync.log`: detailed archive upload activity.

The app is launched by the existing interactive scheduled task
`CartographyExhibition-exhibition`, running interactively as `exhibition`. Its supervisor is
`tools\run_exhibition.ps1`; it validates the offline runtime and CUDA, writes a
heartbeat, restarts the app with bounded backoff, and latches repeated failures
with `cache\monitoring\maintenance.stop`. Do not add an HKCU startup entry or
start `run.bat` over SSH: either can create a duplicate or invisible app.

## Live watch

Copy the current watcher when it changes. For a watch that must survive SSH
disconnects, register or start it as a temporary SYSTEM scheduled task. An
attached diagnostic run is also useful:

```powershell
scp tools/exhibition_watchdog.ps1 hp-zbook:C:/Exhibition/Cartography/tools/exhibition_watchdog.ps1
ssh hp-zbook "powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\Exhibition\Cartography\tools\exhibition_watchdog.ps1 -Until 2026-09-19T20:00:00+07:00 -IntervalSeconds 30"
```

Adjust `-Until` for the requested local cutoff. The remote loop performs cheap
checks every 30 seconds and emits only startup, state changes, alerts, recovery,
and completion. Waiting in the attached command requires no model polling or
token use, but the process ends if its SSH session is closed.

The watch alerts when the supervisor, artwork, or uploader heartbeat is stale;
the app or supervisor is not running; the app restarts; display FPS falls below
45; generation FPS falls below 2; AI frame age exceeds 500 ms; uploader errors
or invalid archives appear; uploads are disabled; journey maps/fullscreen are
disabled in the exhibition config; or free space falls below 5 GiB.

## Power and storage baseline

The active Windows power plan should have these AC and DC values:

- display timeout: never (`VIDEOIDLE = 0`)
- sleep timeout: never (`STANDBYIDLE = 0`)
- hibernate timeout: never (`HIBERNATEIDLE = 0`)
- hybrid sleep: off
- adaptive brightness: off

Verify with `powercfg /query SCHEME_CURRENT SUB_VIDEO` and
`powercfg /query SCHEME_CURRENT SUB_SLEEP`.

The exhibition config is `cache\exhibition\config.json`, not the root
`config.json`. Its expected storage settings are `map_sync_enabled: true`,
`map_cache_gib: 5`, and `map_min_free_gib: 1`. The uploader verifies remote
objects before pruning local archives. Never delete an unverified archive to
recover space.

## Two-projector state

Expected config is `journey_map: true`, `fullscreen: true`, with distinct
`display_monitor` and `map_display_monitor` values. Assigned exhibition startup
restores each window to its saved monitor, fullscreens both, and starts with the
F1 overlay closed. Moving either window and toggling fullscreen updates its
saved monitor for the next launch.

The current telemetry proves rendering health but does not expose each window's
live monitor, rectangle, or fullscreen flag. SSH cannot inspect the interactive
window station. Do not claim current two-screen fullscreen from
config or an SSH screen count. Verify it visually at the machine, or add these
fields to the in-app `artwork.json` snapshot in a planned deployment. The
existing `tools\check_fullscreen.py` is a commissioning test that launches its
own app; do not run it beside the live exhibition.

## Fast triage

1. Read the three monitoring JSON files and compare their timestamps with the
   PC clock.
2. If the supervisor is stale or faulted, inspect `supervisor.log` and the
   latest `realtime_diffusion_*.log` before restarting anything.
3. If artwork telemetry is fresh, trust its FPS/frame-age values over SSH
   process enumeration.
4. If uploader telemetry is fresh and has zero pending/error/invalid archives,
   upload and cleanup are healthy even when the phase is `idle`.
5. Treat fullscreen as unverified unless confirmed in the interactive session.
6. Preserve `journeys` and `.sync` receipts during all recovery work.

## Fleet administrative SSH

The intended administrative boundary is the two Tailscale fleet peers only:

- `aria`: `100.87.222.71` and `fd7a:115c:a1e0::b434:de48`
- `asus-rog`: `100.96.33.72` and `fd7a:115c:a1e0::9f34:2149`

Run `tools\unlock_exhibition_admin_ssh.ps1` once from an elevated PowerShell on
the exhibition PC. It adds `hong-ssh` to the built-in Administrators group,
adds Aria's dedicated public key, removes forwarding/tunneling restrictions
from that user's managed SSH block, retains public-key-only authentication,
keeps the firewall restricted to the four fleet addresses, validates the SSH
candidate before publishing it, and creates timestamped rollback copies.

Afterward, open fresh connections from both peers and verify administrator group
membership. Existing SSH sessions do not receive the new group token; reconnect.
Administrative SSH still does not bypass Windows interactive-session isolation,
so fullscreen/window placement needs in-app telemetry or console inspection.

Verified on 19 September 2026: a fresh `ssh hp-zbook` session contains the
built-in Administrators SID, has a High mandatory integrity level, and can run
`net session`. Do not rerun the unlock helper during routine maintenance.

## Automatic console sign-in

`tools\enable_exhibition_autologon.ps1` configures the standard local
`exhibition` account for console autologon. This machine's RMIT Acceptable Use
notice blocked unattended sign-in; its removal was explicitly approved. The
helper requires `-BlankPasswordConfirmed -ClearLegalNotice`, backs up the notice
and previous Winlogon values under `C:\ProgramData\CartographyExhibition`, and
rolls changes back independently on failure.

Autologon was commissioned with two consecutive reboots on 19 September 2026.
Both boots created an active `exhibition` console session and started
`CartographyExhibition-exhibition` with zero supervisor restarts. Boot-to-SSH
can take about three minutes; do not diagnose failure from an early timeout.

## Upload and low-storage acceptance

The live transport is private Cloudflare R2 through the `exhibition` account's
Wrangler installation. On 19 September 2026, automatic discovery uploaded the
real archive `20260919T071904-e0311e1346db`. Restoring its descriptor produced
four files / 5,118,993 bytes with every SHA-256 matching the local source.

For cleanup testing, restore a verified archive into an isolated root and call
the production `sync_bundles_once` API with `prune=True, cache_bytes=0`. Never
use a zero cache target on the live `journeys` root. The acceptance test
reverified and pruned the completed copy, retained an incomplete active archive,
and returned no errors. Relevant unit coverage is 65 passed / 1 skipped across
journey storage, S3 sync, Wrangler bundle sync, and session pause/resume.

Storage guarantees are deliberately conservative: only remotely verified
completed archives are pruned. Low free space pauses map path/image capture but
keeps the artwork rendering and uploader alive. An active journey is never
deleted or auto-finalized; an exceptionally long active journey may remain
paused until Space finalizes it or an operator frees disk space.
