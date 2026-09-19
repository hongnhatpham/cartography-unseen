# Windows exhibition host

Create a clean, standard local Windows account for the one-week exhibition starting November 19, 2026. The exhibition account has no local password or expiry date. Setup leaves other accounts and administrator membership alone and gives the exhibition user Modify access throughout the project, so ordinary editing and troubleshooting work normally. There is no kiosk mode or date-based shutdown. For unattended power-on, run `tools/enable_exhibition_autologon.ps1` once from an elevated console after setup; it refuses administrator accounts and logon-notice policies.

`tools/setup_exhibition_windows.ps1` prints a read-only plan by default. `-Apply` creates the account, grants project access, configures inbound OpenSSH, and registers tasks that run at the exhibition user's logon. Run setup from an elevated, 64-bit Windows PowerShell console on the target machine. It does not enable autologon or change Windows Update, sleep, display timeout, lock policy, or Defender.

## Prepare the project and keys

1. Install the NVIDIA driver and Tailscale. Enroll the machine in the fleet manually and confirm its Tailscale device/key lifetime covers the show. Use Windows OpenSSH for remote access. Setup does not enroll devices or change tailnet grants.
2. Put a clean project copy in `C:\Exhibition\Cartography`, outside anyone's personal profile. Include no personal files, credentials, or fleet private keys. This short, shared path is easy to open in Explorer or a terminal. Setup adds the exhibition user's access without replacing existing project permissions or file ownership. Choose a parent directory that the new user can browse.
3. Complete `setup_first_run.bat` before the show and run `runtime\python\python.exe tools\verify_offline.py --root C:\Exhibition\Cartography` on the target GPU. The supervisor never installs missing dependencies or downloads models. Test the artwork offline, including the display, controller, generation and journey saving.
4. Obtain the intended OpenSSH public key from aria and asus-rog as `aria.pub` and `asus-rog.pub`. Transfer only `.pub` files to the exhibition host. Each private key stays on its original fleet machine. Confirm these are the keys used by those machines before applying setup.

The source addresses default to this fixed fleet allowlist. An explicit `ApprovedPeerAddresses` parameter may narrow it, but cannot add other machines.

| Machine | Tailscale IPv4 | Tailscale IPv6 |
| --- | --- | --- |
| aria | `100.87.222.71` | `fd7a:115c:a1e0::b434:de48` |
| asus-rog | `100.96.33.72` | `fd7a:115c:a1e0::9f34:2149` |

Public keys and addresses are independent allowlists. Either approved key can authenticate from either approved machine. SSH requires a public key, with password and keyboard-interactive authentication disabled. Staff sign in at the local console without a password. Setup does not change Windows restrictions on remote use of blank passwords.

The normal `tools/pack_exhibition.ps1 -Zip` package downloads runtime and models on first launch. For an offline package, complete setup in a trusted source checkout, install optional monitoring/storage dependencies, and build with `tools/pack_exhibition.ps1 -PreparedOffline -Zip`. This copies the prepared runtime and pinned models and verifies the copied runtime with a real offline generation. It excludes source caches, operator credentials and personal archives. Verify again on the target GPU.

## Preview and apply

```powershell
$setup = @{
    UserName = 'exhibition'
    DeploymentPath = 'C:\Exhibition\Cartography'
    PublicKeyFiles = @('C:\Commissioning\aria.pub', 'C:\Commissioning\asus-rog.pub')
    TailscaleInterfaceAlias = 'Tailscale'
}
& C:\Exhibition\Cartography\tools\setup_exhibition_windows.ps1 @setup
# Review the account, project grant, added SSH user block and Tailscale firewall rule.
& C:\Exhibition\Cartography\tools\setup_exhibition_windows.ps1 @setup -Apply
```

Setup does not prompt for a password. It creates a new account disabled and passwordless, installs and activates its SSH restrictions, then enables it. A failure leaves that new account disabled. The rollback record tracks pending activation, so a successful rerun can finish enabling an account created by an interrupted setup and clear a password entered during an older setup. Every apply clears the managed account's password after SSH activation. Existing managed accounts otherwise keep their enabled or disabled state. Setup rejects unmanaged accounts or task names. Repeating setup updates SSH access and tasks, preserves application settings, and clears any expiry left by an older setup. Group checks target only the exhibition account, so unrelated stale domain members do not interrupt setup.

Keep the exhibition user signed out while applying setup. Run a trusted copy of the setup script when elevated, because the exhibition user can edit project code. Existing protected deployments receive an added Modify grant on all files. The active application configuration remains `cache\exhibition\config.json`; setup copies the versioned `config.exhibition.json` preset there only when it is absent (falling back to `config.json` for older packages). Edit the active copy when tuning the show.

If an earlier run stopped after granting project permissions, keep its rollback record and rerun with the same account and deployment path. Setup records the `deployment-acl` phase before SSH staging. The additive grant can be applied again; the original ACL backup and existing active config are preserved. The pending account stays disabled until SSH activation succeeds.

Setup keeps the existing SSH service executable, config path, start mode and recovery settings. It inserts a marked `Match User exhibition` block into the active config before existing Match blocks. Existing global settings and administrator authentication remain intact. Only the new exhibition user requires the approved public keys and has SSH forwarding disabled. Its administrator-owned key file and config backup live beside the rollback record at `%ProgramData%\CartographyExhibition\exhibition`.

An installed OpenSSH capability does not guarantee that the `sshd` service is registered. When it is missing, setup registers `OpenSSH SSH Server` from the canonical `%SystemRoot%\System32\OpenSSH\sshd.exe`, requiring a valid Microsoft signature and no reparse points in its path. It runs as LocalSystem with no service dependencies and initially uses Manual startup. After publishing the validated config and keys, setup starts it, verifies that it is Running, then sets Automatic startup. The installed capability is not reinstalled. If the service registry key and Windows service manager disagree, setup stops for a reboot or registration repair.

The plan reports `registerSshService` separately from `installOpenSsh`. It also reports `sshOperationalLogAvailable`; an absent `OpenSSH/Operational` event log limits diagnostics but does not block registration or working SSH. Setup does not install an event-log provider as part of this repair.

Setup adds a firewall allow rule on the current SSH port, normally TCP 22, scoped to the Tailscale adapter and the two approved machines. All other firewall rules and profiles remain unchanged. Existing rules may still admit connections for existing users. The exhibition user's public keys carry `from=` source-address restrictions, so a broader existing firewall rule does not let that account authenticate from other machines. Certificates are excluded for that account to keep authorization tied to the supplied public keys. This follows OpenSSH's [per-user configuration](https://man.openbsd.org/sshd_config) and [authorized-key restrictions](https://man.openbsd.org/sshd#AUTHORIZED_KEYS_FILE_FORMAT).

Apply stages both the config and public keys, validates them, and completes task setup before activating either file. Each live file is replaced atomically after its original is backed up. A failure during activation or service restart restores both originals. After attempting to clear an already-enabled account's password, a later failure retains the new SSH restrictions because the old password cannot be restored. The config uses `PubkeyAcceptedKeyTypes` for compatibility with inbox Windows OpenSSH, and `KbdInteractiveAuthentication no` to disable keyboard-interactive authentication inside the exhibition user's Match block. Apply restarts an already-running service to load the addition. Run at the local console because this brief restart can interrupt SSH sessions. A stopped existing service stays stopped and its startup policy stays unchanged; start it manually when ready. A newly installed service is set to Automatic and started. Configurations with Include directives, multiple listener ports or custom service command-line options need manual integration. Existing deny and group-access policies still apply to the new account; resolve any conflict explicitly during commissioning.

See [commissioning and persistence](exhibition-commissioning.md) for the dual-display preset, startup GPU preference, operational test and monitoring billing handover.

## Log on and run

Select `exhibition` at the Windows sign-in screen and sign in with the password field empty. `CartographyExhibition-exhibition` starts the artwork supervisor in that user's desktop with limited privileges. Open `C:\Exhibition\Cartography` in Explorer and create a desktop shortcut there if useful. Confirm the user can open, edit and save project files. No task stores a password or runs project code as SYSTEM.

Without the optional autologon helper, staff restart Windows, select `exhibition`, and sign in without a password to start the artwork again. For unattended recovery, run the helper from an elevated PowerShell console:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File C:\Exhibition\Cartography\tools\enable_exhibition_autologon.ps1
```

It verifies that an empty password can perform an interactive logon, records the previous Winlogon values under `%ProgramData%\CartographyExhibition`, then enables automatic sign-in for the non-administrator account. This makes anyone with physical access able to enter the exhibition desktop; use it only on the dedicated installation machine. Reboot twice and verify automatic sign-in and both fullscreen windows after each boot; the registry method is not considered commissioned until both reboots pass. Also check BIOS power-after-loss behavior for power-cut recovery. The project task can be stopped and started from the exhibition account using the commands below.

Choose the venue's power and update settings in Windows Settings. Record manual changes so they can be restored afterward. Check BIOS power-after-loss behavior if unattended recovery from a power cut matters.

## Optional monitoring

Monitoring uses a second task, `CartographyExhibition-Monitor-exhibition`, under the same exhibition user at logon. It has its own process and remains active when the artwork task stops. Both tasks require the exhibition user to be logged on. The dashboard becomes stale while the machine is off or signed out; a successful SSH connection is a separate check that Windows is reachable.

Generate a machine credential using [the dashboard runbook](../dashboard/README.md), register its token hash with the dashboard, and transfer the credential JSON through an approved private channel. Install `requirements-monitor.txt` into the deployed runtime before setup. The token can submit telemetry only and does not grant SSH or remote command access.

```powershell
$monitor = @{
    MonitorCredentialPath = 'C:\Commissioning\gallery-a.json'
    MonitorEndpoint = 'https://monitor.example.org/api/v1/heartbeat'
}
& C:\Exhibition\Cartography\tools\setup_exhibition_windows.ps1 @setup @monitor
& C:\Exhibition\Cartography\tools\setup_exhibition_windows.ps1 @setup @monitor -Apply
# Sign in as exhibition to start both tasks.
```

The plan reports only the credential source path and endpoint host without reading the token. Apply writes `{endpoint,token}` to `cache\exhibition-monitor\config.json`. The exhibition user, SYSTEM and Administrators can access this directory; other users receive no access grant. Its SQLite outbox lives beside the config. Keep that directory out of shared archives. Credentials never appear in task arguments or the rollback record. Remove the extra commissioning copy according to your credential-handling policy after verifying installation.

The monitor reads snapshots from `cache\monitoring` and restarts after process failures. Sign in and check the dashboard receipt time, stop the artwork and confirm machine telemetry continues, then disconnect/reconnect networking and confirm reporting resumes with one collector. To rotate a token, update the dashboard machine's token hash, then rerun setup with the replacement credential and endpoint. The new collector starts at the next exhibition logon, or use `Start-ScheduledTask` while that user is already logged on.

Omitting the monitor parameters leaves an existing interactive monitor task unchanged. To migrate an older SYSTEM monitor task, supply the credential and endpoint again. Setup removes the old task before granting project write access and registers the user task. Any old protected credential/outbox under ProgramData can be removed manually after verifying migration; setup preserves it for recovery.

To remove monitoring, stop and unregister its named task and remove `cache\exhibition-monitor\config.json`. Revoke the machine credential in the dashboard separately. The outbox may be retained for investigation. Removing monitoring does not stop the artwork or SSH.

## Observe, stop and restart

Check the server host-key fingerprint at the physical console before trusting the first SSH connection. From aria or asus-rog:

```text
ssh -i <private-key-already-on-this-fleet-machine> exhibition@<exhibition-tailscale-address>
```

Windows OpenSSH normally opens `cmd.exe`; enter `powershell -NoProfile` before these commands.

```powershell
$project = 'C:\Exhibition\Cartography'
Set-Location $project
& "$project\tools\exhibition_status.ps1" -DeploymentPath $project -UserName exhibition -Json
# Exit codes: 0 process healthy; 1 SSH/disk warning; 2 app/task/heartbeat unhealthy;
# 3 maintenance flag present; 4 status inspection failed.

New-Item -ItemType File -Path "$project\cache\monitoring\maintenance.stop" -Force
# The supervisor checks the flag within five seconds. Wait for childPid to become null.
Get-Content "$project\cache\monitoring\supervisor.json"
Get-ScheduledTask -TaskName 'CartographyExhibition-exhibition'
# Edit files after the child exits and the task is no longer Running.

Remove-Item -LiteralPath "$project\cache\monitoring\maintenance.stop"
Start-ScheduledTask -TaskName 'CartographyExhibition-exhibition'
```

Restart through the task so the artwork appears in the existing local desktop. Launching `run.bat` or `app.main` from SSH does not target that desktop. The exhibition user receives read/run/stop rights on its tasks. If needed, stop the supervisor with `Stop-ScheduledTask`; its Windows Job Object closes the owned renderer too. Verify task permissions from SSH during commissioning.

App exits trigger restart delays capped at 60 seconds. Eight consecutive runs shorter than five minutes latch a maintenance stop. Inspect the error, clear the flag, and start the task after repair. `cache\monitoring\supervisor.json` holds current status and `logs\supervisor\supervisor.log` rotates at 1 MiB with five retained files. Status makes no network calls. A live process and heartbeat do not prove rendering or input works; inspect the display too.

## Commission and recover

- Verify existing administrator SSH access still works. For the exhibition account, test a successful connection from aria and asus-rog, a rejected key from an approved address, and rejected authentication from another tailnet machine or the ordinary LAN. Check IPv4 and IPv6, password and keyboard-interactive rejection, and forwarding. Other accounts retain their existing SSH policy.
- Verify offline artwork operation, file editing under the exhibition account, journey/screenshot writes, maintenance stop/restart, child recovery and the optional monitor. Restart Windows from the exhibition desktop, sign back in without a password, and confirm the artwork starts.
- Keep the existing administrator login available to the venue operator. Record the deployment version, server host-key fingerprint and approved public keys.

Before repairs, stop the artwork and preserve journeys, settings and relevant logs. After dependency/model changes, run offline verification before restarting the show. There is no automatic end-of-show action. Keep using the account afterward or remove the setup when you choose.

Each apply records its phases in `%ProgramData%\CartographyExhibition\exhibition\rollback.json`, including the account SID, task names, the added firewall rule, active SSH config and backup paths, previous service state, `createdService` ownership and original project ACL backup. A failed apply stops and removes only an SSH service created by that same run. A later run preserves an already-existing service even when an earlier run recorded `createdService: true`. The record has no passwords or tokens. If cleanup is wanted, stop and unregister the named project and monitor tasks, disable Autologon if enabled, then archive data. Account deletion is a separate decision.

For SSH cleanup, remove the marked exhibition Match User block and the named exhibition firewall rule, then validate the config with `sshd.exe -t` before restarting the service. Preserve any unrelated changes made since setup. The backup is available for immediate recovery; restoring it later could overwrite newer SSH edits. If SSH was installed solely for the show, it may be stopped/disabled or uninstalled. Keep the rollback record until recovery is complete. Restore the original project ACL backup only if the same deployment still occupies the path and restoring its earlier permissions is intended.

Windows references: [OpenSSH server configuration](https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh-server-configuration) and [interactive scheduled-task principals](https://learn.microsoft.com/en-us/powershell/module/scheduledtasks/new-scheduledtaskprincipal).
