# Windows exhibition host

Use a dedicated Windows installation and deployment directory. The exhibition runs as a new standard local user. Approved fleet machines connect inbound through Tailscale to Windows OpenSSH using public keys. The host receives public keys only. It does not need fleet private keys, an outbound SSH configuration, or a Tailscale enrollment key in the project.

The setup script prints a read-only plan by default. `-Apply` creates the account, changes the dedicated deployment ACLs, configures OpenSSH and its firewall rules, and registers an interactive logon task. Run it from the local console so a failed SSH change cannot strand you. Keep a separate administrator login available.

Privileged apply and commissioning require the actual target Windows machine.
Building or reviewing a package on another computer does not provision the host.
The normal `tools/pack_exhibition.ps1 -Zip` package downloads its runtime and
models on first launch. For offline deployment, complete `setup_first_run.bat`
in a trusted source checkout, install any optional monitoring/storage dependencies,
then build with `tools/pack_exhibition.ps1 -PreparedOffline -Zip`. That mode copies
the prepared runtime and pinned model files and writes a fresh installation marker
only after the copied runtime passes CUDA imports and a real offline generation.
It never copies source caches, operator credentials or personal archives. Repeat
verification on the target GPU as described below.

## Before setup

1. Install Windows updates, the NVIDIA driver, and Tailscale. Enroll Tailscale manually with the appropriate exhibition device identity. The script neither enrolls the host nor changes tailnet policy. Arrange tailnet ACLs/grants and device/key expiry with the fleet administrator before the exhibition. Use Windows OpenSSH, not Tailscale SSH.
2. Copy the application from a trusted release to a dedicated administrator-controlled directory such as `C:\Exhibition\Cartography`. Include the runtime, models, `tools`, and installation marker after completing `setup_first_run.bat`. Do not deploy a personal checkout with credentials, private keys, or unrelated files. Setup assigns file ownership to Administrators, resets child ACLs and grants the exhibition user read+execute on the deployment, including code, tools, runtime, models and the root configuration template. Only `cache`, `logs`, `journeys` and `screenshot` receive Modify access. Keep the deployment's parent directory administrator-controlled too, so the standard user cannot replace the entire deployment. Other ordinary users lose project access.
3. Run the existing offline verification on this machine. Disconnect internet access for the final graphics/input test. The supervisor will never run an installer or download missing models. Model cache flags do not prohibit optional journey uploads when credentials and application settings enable them.
4. Obtain one OpenSSH `.pub` file for each approved fleet operator key, and the exact Tailscale IPv4 and/or IPv6 addresses of approved source machines. Include both families if clients may use either. Subnets, DNS names, wildcards, and addresses outside Tailscale ranges are rejected. Keys and source machines form independent allowlists: any approved key can authenticate from any approved address.
5. Use elevated, 64-bit Windows PowerShell 5.1. All Windows firewall profiles must be enabled with inbound blocking. OpenSSH capability installation can require Windows Update or an administrator-provided Features on Demand source. If policy owns an inbound rule that permits SSH, narrow it through that policy before running setup.

For a new deployment parent, create it at the administrator console before copying the release. This example refuses to reuse an existing directory. If the parent already exists, inspect and secure its ACLs deliberately instead. Setup checks ancestor ownership and permissions that could allow another user to replace the deployment.

```powershell
$parent = 'C:\Exhibition'
if (Test-Path -LiteralPath $parent) { throw 'Choose a new dedicated parent or review the existing directory ACLs.' }
New-Item -ItemType Directory -Path $parent
icacls $parent /inheritance:r /grant:r '*S-1-5-32-544:(OI)(CI)F' '*S-1-5-18:(OI)(CI)F' '*S-1-5-32-545:(OI)(CI)RX'
# Copy the trusted release into C:\Exhibition\Cartography, then complete commissioning.
```

Use actual fleet addresses and public key paths in place of the examples:

```powershell
$setup = @{
    UserName = 'exhibition'
    DeploymentPath = 'C:\Exhibition\Cartography'
    PublicKeyFiles = @('C:\Commissioning\operator-a.pub', 'C:\Commissioning\operator-b.pub')
    ApprovedPeerAddresses = @('100.101.102.103', 'fd7a:115c:a1e0::1234')
    ExhibitionEndDate = [datetime]'2026-12-01T00:00:00'
    TailscaleInterfaceAlias = 'Tailscale'
}
& C:\Exhibition\Cartography\tools\setup_exhibition_windows.ps1 @setup
# Review the account, directory, ACL change, and firewall rules printed above.
& C:\Exhibition\Cartography\tools\setup_exhibition_windows.ps1 @setup -Apply
```

The expiry is local Windows time and prevents subsequent account logons after that instant. The password has no separate age expiry, so password aging does not interrupt an approved show period. Account expiry does not terminate an existing exhibition session or disable the SSH listener. Schedule teardown separately. Enter the local password only at the secure prompt. The script does not store it or enable autologon. Existing unmanaged accounts and task-name collisions are rejected. Repeating setup for its own recorded account updates the peer/key allowlists, expiry and task without creating another user. A failed apply stops `sshd`; inspect the error, repair the prerequisite, and rerun locally.

The supervisor uses `cache\exhibition\config.json`. Setup copies the administrator-managed root `config.json` there only if the mutable copy is absent. UI tuning and remote configuration edits target that mutable copy; its writable directory supports atomic temporary-file replacement. Repeating setup preserves it. Root `config.json` and `prompts.json` remain release templates that require administrator access to change. Do not run code placed in writable cache/output directories with elevation. Sign the exhibition user out before applying or repairing the deployment, and run setup from a trusted administrator-controlled copy. If an earlier setup granted the account write access to code, restore those files from a trusted release before running anything elevated.

The script keeps `%ProgramData%\ssh\sshd_config` intact and points the `sshd` service to an administrator-owned configuration under `%ProgramData%\CartographyExhibition\exhibition`. It permits only this local user, requires public-key authentication, and disables SSH forwarding and tunneling. The user can read the authorized public keys but cannot replace them. A separate Windows firewall rule allows TCP 22 only on the selected Tailscale adapter from the exact approved peer addresses. Other enabled local inbound rules capable of admitting this SSH service are disabled, including the default OpenSSH rule. This can affect other services if they depended on a broad inbound allow rule, so read the plan.

This is an inbound access policy, not an egress sandbox. Shell access permits ordinary programs under the standard account. No outbound SSH is configured or used by this workflow. Locally generated SSH host private keys belong to this server and are distinct from fleet client credentials.

## Console logon and unattended startup

Sign in locally as `exhibition`. Task Scheduler runs `CartographyExhibition-exhibition` with that interactive user's token and limited privileges. The task does not store a password and cannot create a visible desktop before console logon. Close the first-login privacy/setup prompts, verify the correct display arrangement and controller, and confirm the artwork appears. Sign out of other graphical sessions to avoid an unattended display landing on the wrong session.

For unattended reboot recovery, manually configure [Microsoft Sysinternals Autologon](https://learn.microsoft.com/en-us/sysinternals/downloads/autologon) at the physical console after commissioning. Use the local account, this computer's name, and its password. Do not pass the password on a command line or put it in a script, task XML, transcript, or registry `DefaultPassword` value. Autologon uses an LSA secret; an administrator can recover it, so it is not protection against someone who already controls the machine. Test a complete reboot. Autologon is optional and is not installed or enabled by setup.

Decide power and update settings with the venue. Setup does not change sleep, display timeout, lock policy, Windows Update, Defender, or reboot policy. If approved for a mains-powered exhibition, an administrator can explicitly choose:

```powershell
# Optional. Record the current values first and restore after the exhibition.
powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE
powercfg /query SCHEME_CURRENT SUB_VIDEO VIDEOIDLE
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 0
```

Set an update/maintenance window through supported Windows Settings or the venue's management policy. Do not disable security updates indefinitely. Check BIOS power-after-loss behavior separately if the venue expects recovery from a power cut.

## Optional monitoring before console logon

Monitoring is opt-in. Omitting `-MonitorCredentialPath` leaves any existing monitor task and its protected configuration unchanged. The artwork remains independent of monitoring and network availability.

On a trusted administrator computer with the dashboard source checkout, use the dashboard credential generator described in [the dashboard runbook](../dashboard/README.md). The exhibition package includes that README for reference, but does not contain dashboard server code or credentials. The generator writes a JSON file containing `machineId` and `token` and a separate SQL file containing only the token hash. Register that hash with the intended deployed dashboard during an authorized commissioning session. A machine token permits telemetry ingestion only. It cannot read the dashboard, open SSH, or execute commands.

Copy the generated credential JSON through your approved private channel into an administrator-only directory on the exhibition host, outside the project and journey archive tree. Install `requirements-monitor.txt` into the deployed runtime during commissioning. Pass the exact dashboard heartbeat URL separately; the generator's credential JSON does not contain an endpoint. From the local administrator console:

```powershell
# Reuse the account/deployment/peer parameters in $setup from the setup example.
$monitor = @{
    MonitorCredentialPath = 'C:\Commissioning\gallery-a.json'
    MonitorEndpoint = 'https://monitor.example.org/api/v1/heartbeat'
}
& C:\Exhibition\Cartography\tools\setup_exhibition_windows.ps1 @setup @monitor
# The plan reports the credential source path and endpoint host, without reading the token.
& C:\Exhibition\Cartography\tools\setup_exhibition_windows.ps1 @setup @monitor -Apply
Get-ScheduledTask -TaskName 'CartographyExhibition-Monitor-exhibition'
Get-ScheduledTaskInfo -TaskName 'CartographyExhibition-Monitor-exhibition'
```

Apply validates the credential as a regular JSON file with no reparse points in its path, then writes `{endpoint,token}` to `%ProgramData%\CartographyExhibition\exhibition\monitor\config.json`. Only SYSTEM and Administrators can read that file or directory. The rollback record contains the config path and task name, never the token. Setup does not print the token or put it in task arguments. Remove the administrator's extra credential copy after verifying installation, according to your credential-handling policy.

`CartographyExhibition-Monitor-exhibition` starts at boot under SYSTEM, uses no stored password, and restarts a failed process after one minute. Apply also starts it immediately. It runs the protected deployment's Python in isolated mode and calls `tools\monitor_agent.py` with a protected state directory and the project snapshot directory. Its SQLite outbox stays under the protected ProgramData monitor directory. It reads artwork and uploader snapshots from `cache\monitoring`; it does not write privileged state into that user-writable directory. The task continues reporting machine health when nobody is logged on or the artwork task is stopped.

Verify this behavior with a reboot before console logon. Check the machine's dashboard receipt time and confirm the artwork is reported as stopped or unknown until its interactive session starts. Stop and restart the artwork and confirm the collector task remains Running. Then disconnect networking and verify the local artwork continues; reconnect and confirm collection resumes without starting a second collector.

For rotation, generate a replacement credential in a separate private directory and update the existing dashboard machine row's token hash as described in the dashboard runbook. Preserve that row and its history. Rerun setup with the new `-MonitorCredentialPath` and `-MonitorEndpoint`. This explicitly replaces the protected config and restarts the collector. Reruns without a credential path do not rotate, create, or restart the monitor task. `-MonitorEndpoint` alone is rejected.

To remove monitoring while keeping the exhibition running, use the administrator console:

```powershell
Stop-ScheduledTask -TaskName 'CartographyExhibition-Monitor-exhibition'
Unregister-ScheduledTask -TaskName 'CartographyExhibition-Monitor-exhibition' -Confirm:$false
Remove-Item -LiteralPath 'C:\ProgramData\CartographyExhibition\exhibition\monitor\config.json'
```

Revoke the machine credential in the dashboard as a separate authorized operation. The local deletion removes the token copy; the bounded SQLite outbox may be retained for investigation or removed later during teardown. Omitting monitoring parameters does not remove an existing installation.

## Observe, stop, and restart

From an approved fleet machine, verify the server host-key fingerprint using the physical console before trusting the first connection. Then connect to the host's Tailscale address:

```text
ssh -i <fleet-private-key-on-this-fleet-machine> exhibition@<exhibition-tailscale-address>
```

Windows OpenSSH normally opens `cmd.exe`. Enter `powershell -NoProfile` in the remote shell before these commands:

```powershell
$project = 'C:\Exhibition\Cartography'
& "$project\tools\exhibition_status.ps1" -DeploymentPath $project -UserName exhibition -Json
# Exit codes: 0 process healthy; 1 SSH/disk warning; 2 app/task/heartbeat unhealthy;
# 3 maintenance flag present; 4 status inspection failed.

# Gracefully stop the owned app. The supervisor checks this flag within five seconds.
New-Item -ItemType File -Path "$project\cache\monitoring\maintenance.stop" -Force
# Wait until childPid is null and the task is no longer Running before changing files.
Get-Content "$project\cache\monitoring\supervisor.json"
Get-ScheduledTask -TaskName 'CartographyExhibition-exhibition'

# Restart in the already logged-on exhibition desktop, not in the SSH session.
Remove-Item -LiteralPath "$project\cache\monitoring\maintenance.stop"
Start-ScheduledTask -TaskName 'CartographyExhibition-exhibition'
```

Do not launch `run.bat` or `app.main` from SSH for the public display. The scheduled task is bound to the existing interactive session. The setup grants the exhibition user read/run rights on that task. If the supervisor is stuck, use `Stop-ScheduledTask -TaskName 'CartographyExhibition-exhibition'` before restarting; verify this permission during commissioning. An administrator can always stop it locally. A Windows Job Object kills the owned renderer when the supervisor process closes, including forced task stops.

The supervisor checks the install marker, mutable configuration, configured model files, Python imports and CUDA availability before it starts `python.exe -m app.main --config cache/exhibition/config.json`. It remains alive as the foreground task owner. App exits trigger exponential backoff capped at 60 seconds. Eight consecutive runs shorter than five minutes latch a maintenance stop, requiring an operator to inspect and restart. A run lasting five minutes resets that short-run count. Task Scheduler can restart a failed supervisor up to three times at one-minute intervals.

`cache\monitoring\supervisor.json` is replaced atomically and contains bounded status fields, child PID, restart count and timestamp. `logs\supervisor\supervisor.log` rotates at 1 MiB with five retained files. These logs describe supervision; application logs retain the application's own retention policy. A fresh heartbeat and live child prove process liveness only. They do not prove that the display is rendering, input works, or image generation is progressing. Inspect the display and application telemetry during commissioning and after repairs. Status itself makes no network calls.

## Commissioning checklist

- Complete a real offline generation with `runtime\python\python.exe tools\verify_offline.py --root C:\Exhibition\Cartography`, then run the interactive task with the network disconnected. Confirm fullscreen display, controller input, model output, journey saving and the expected optional upload behavior.
- Test a connection from each approved fleet address, a denied key from an approved address, and a connection from an unapproved tailnet machine. Verify ordinary LAN/public-interface access to TCP 22 fails. Test IPv4 and IPv6 explicitly where both are deployed.
- Confirm password authentication fails, other usernames fail, and forwarding is refused. Inspect `Get-NetFirewallRule -Name 'CartographyExhibition-SSH-exhibition'` and its address/interface filters as administrator. The script validates SSH syntax before startup; these live tests verify the whole access path.
- Stop/start through the maintenance flag and scheduled task. Kill the child once and confirm bounded restart. Stop the task and confirm its child disappears. Verify `exhibition_status.ps1` returns the documented codes for running, maintenance and stopped states.
- As the standard user, confirm edits to `tools\setup_exhibition_windows.ps1`, app code and root `config.json` fail. Change one UI setting and confirm atomic persistence in `cache\exhibition\config.json`; check journey and screenshot writes still succeed. Never run a user-supplied permissions test script elevated.
- Reboot from the console. Confirm manual logon or the approved Autologon configuration reaches the artwork, status becomes fresh, and SSH recovers. Repeat a network disconnect/reconnect without interrupting the artwork. Check that the end date and Tailscale key expiry cover the required show period.
- Keep a local administrator credential and this runbook with the venue's authorized operator. Keep it outside the standard user's readable project. Record the host-key fingerprint, deployment version, peer/key approvals, and commissioning results.

## Recovery and teardown

Before repair, create the maintenance flag and wait for the task to stop. Preserve `journeys`, relevant logs and application settings. Repair runtime/models through the commissioning workflow with a local administrator, rerun offline verification, and only then clear the flag and start the task. Do not let startup attempt downloads while the exhibition is open.

Every apply writes an administrator-controlled rollback record at `%ProgramData%\CartographyExhibition\exhibition\rollback.json`. It records the account SID, original OpenSSH capability/service state, disabled firewall rule names, original deployment ACL backup, and last completed phase. It contains no password or fleet private key. The original default SSH configuration is untouched; the previous managed configuration is retained as `sshd_config.previous`. Setup intentionally does not perform automatic broad rollback after a failure because restoring permissive firewall rules could reopen SSH.

At the end, perform teardown from the local administrator console. Use the recorded names and paths, not an unrelated account or deployment:

```powershell
$state = 'C:\ProgramData\CartographyExhibition\exhibition'
$record = Get-Content "$state\rollback.json" -Raw | ConvertFrom-Json
if ($record.PSObject.Properties.Name -contains 'monitorTaskName') {
    Stop-ScheduledTask -TaskName $record.monitorTaskName -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $record.monitorTaskName -Confirm:$false -ErrorAction SilentlyContinue
}
Stop-ScheduledTask -TaskName $record.taskName -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName $record.taskName -Confirm:$false
Stop-Service sshd
Remove-NetFirewallRule -Name $record.firewallRuleName
Disable-LocalUser -Name $record.userName
# Restore the saved deployment ACLs only if the same deployment still occupies this path.
icacls (Split-Path -Parent $record.deploymentPath) /restore $record.originalAclFile
```

Disable Sysinternals Autologon manually before removing the account. Archive exhibition data before deleting any project/profile files. Account deletion and Tailscale device removal are separate administrator decisions. Remove the host from the tailnet and revoke its fleet approvals through the normal fleet process. Restore any approved power/update changes from your commissioning record.

If OpenSSH existed before setup, restore its recorded service executable/configuration path, start mode, and recovery settings using Services/`sc.exe`, validate that configuration with `sshd.exe -t`, and review the old firewall rules before re-enabling them. If setup installed OpenSSH solely for this exhibition, leave it stopped/disabled or remove the `OpenSSH.Server~~~~0.0.1.0` capability. Do not blindly re-enable a broad SSH rule. Retain the rollback record until recovery is complete, then remove the managed state directory after confirming no service refers to it.

Windows behavior references: [OpenSSH server configuration](https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh-server-configuration), [interactive scheduled-task principals](https://learn.microsoft.com/en-us/powershell/module/scheduledtasks/new-scheduledtaskprincipal), and [Sysinternals Autologon](https://learn.microsoft.com/en-us/sysinternals/downloads/autologon).
