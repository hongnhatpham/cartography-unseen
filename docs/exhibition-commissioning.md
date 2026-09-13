# Exhibition commissioning and persistence

The September 13, 2026 installation uses the game on the left display (index 0)
and the journey map on the right (index 1), both borderless fullscreen. Windows
must remain in **Extend these displays**. The idle Cartography Unseen title has
no F1 guide, including while operator controls are open.

## Reproduce the working installation

1. Prepare the runtime and models with `setup_first_run.bat`, then verify offline
   generation on the target GPU as described in [Windows setup](exhibition-windows.md).
2. Apply `tools/setup_exhibition_windows.ps1` using the intended account and fleet
   public keys. Fresh installations copy the versioned `config.exhibition.json`
   into `cache/exhibition/config.json`. Reruns preserve existing operator settings.
   Display indices belong to the target Windows layout; verify left/right placement
   at the venue before opening to visitors.
3. Sign into the exhibition account. The registered logon task starts the supervisor,
   which requests the Windows high-performance GPU for both bundled Python
   executables before creating the artwork windows. It reapplies the preference
   using the deployment's current path, while preserving other graphics preferences.
   On this host the working OpenGL renderer is the NVIDIA RTX 4070 Laptop GPU.
   Verify the reported renderer after a driver, display or hardware change.
4. Configure monitoring and journey upload credentials privately using the Windows
   and [storage](journey-storage.md) runbooks. Credentials, OAuth sessions, visitor
   archives, models and runtime caches are intentionally excluded from Git and the
   lightweight package. They persist on this host; another host needs its own setup.

For an existing installation, the active settings are still
`cache/exhibition/config.json`. To intentionally adopt the preset, stop the artwork
with the documented maintenance flag, back up the active settings, copy
`config.exhibition.json` over them, and restart the task. Ordinary setup never
overwrites a tuned active configuration.

The supervisor restarts an exited artwork and keeps bounded logs. Monitoring runs
in its own logon task, and the journey uploader resumes verified pending uploads.
Both artwork and monitoring require the exhibition user to be signed in. Follow
the stop/restart procedure in the Windows runbook before editing a live deployment.

## Repeat the operational test

Run from PowerShell in the exhibition account, with both displays connected:

```powershell
Set-Location C:\Exhibition\Cartography
& .\tools\start_soak.ps1
```

The launcher snapshots the active configuration, creates a new timestamped results
directory under the deployment parent's `Commissioning` folder, and starts the
instrumented artwork and independent sampler. `-ResultsDirectory` selects another
results base directory. Temporary supervisor source changes are restored by the
launcher. The test simulates visitor sessions lasting 2–15 minutes without pauses
or operator shortcuts. Inspect `status.json` and the evidence in that run directory;
elapsed wall time alone is not a pass. After commissioning, use the standard task
restart procedure to load the normal artwork without synthetic input.

The original successful run is documented in
[the September operational result](performance/exhibition-soak-20260913.md).
The fullscreen repair report distinguishes preliminary crash tests from the final
dual-display validation. Repeat commissioning after material hardware or graphics
changes, and test a complete Windows restart and exhibition-account sign-in before
the show.

## Remote monitor and billing

`monitor.bynhat.com` is protected by Cloudflare Access. Dashboard history loads an
initial 24-hour window, then indexed receipt-cursor deltas every five minutes;
live status refreshes every 15 seconds and hidden tabs suspend polling. The worker
migration and regression coverage are versioned in `dashboard`; see
[incremental history](../dashboard/INCREMENTAL-HISTORY.md) for deployment details.
This prevents repeatedly scanning the complete telemetry history.

Workers Paid was activated for one billing period and renewal was cancelled at the
user's request. The dashboard showed paid access ending **October 6, 2026**. R2
storage is a separate subscription. Before the planned November 19 exhibition,
review actual D1 usage and the available plan: the paid period does not cover that
show, and this configuration does not authorize another subscription purchase.

Windows SSH and Tailscale are configured on this host. Authenticated recovery from
the agents on the other machines and their Discord workflow still require a test
from those machines; local service health does not establish that remote workflow.
