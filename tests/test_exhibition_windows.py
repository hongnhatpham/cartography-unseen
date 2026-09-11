"""Exercise Windows supervisor failure/maintenance paths without launching the artwork."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = shutil.which("powershell.exe")
pytestmark = pytest.mark.skipif(
    sys.platform != "win32" or not POWERSHELL,
    reason="Requires Windows PowerShell",
)


def run_supervisor(deployment: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(POWERSHELL),
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "tools" / "run_exhibition.ps1"),
            "-DeploymentPath",
            str(deployment),
        ],
        text=True,
        capture_output=True,
        timeout=30,
    )


def test_missing_runtime_reports_fault_without_running_installer(tmp_path: Path) -> None:
    result = run_supervisor(tmp_path)
    assert result.returncode == 2, result.stdout + result.stderr
    status = json.loads((tmp_path / "cache/monitoring/supervisor.json").read_text())
    assert status["state"] == "faulted"
    assert "runtime" in status["detail"]
    assert status["childPid"] is None
    assert not (tmp_path / "runtime").exists()
    assert not (tmp_path / "cache/monitoring/supervisor.json.tmp").exists()


def test_maintenance_prevents_python_launch(tmp_path: Path) -> None:
    # The invalid executable would fail if the maintenance branch tried to start it.
    for relative in (
        "runtime/python/python.exe",
        "cache/INSTALL_COMPLETE.txt",
        "tools/verify_offline.py",
        "cache/exhibition/config.json",
        "app/main.py",
        "cache/monitoring/maintenance.stop",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder", encoding="utf-8")
    result = run_supervisor(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    status = json.loads((tmp_path / "cache/monitoring/supervisor.json").read_text())
    assert status["state"] == "maintenance"
    assert status["maintenance"] is True
    assert status["childPid"] is None
    assert status["restarts"] == 0
    # A second launch must acquire the released lock and replace existing status atomically.
    assert run_supervisor(tmp_path).returncode == 0
    assert json.loads((tmp_path / "cache/monitoring/supervisor.json").read_text())["state"] == "maintenance"


def test_job_object_closes_owned_child() -> None:
    script_path = str(ROOT / "tools/run_exhibition.ps1").replace("'", "''")
    command = f"""
$ErrorActionPreference = 'Stop'
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile('{script_path}', [ref]$tokens, [ref]$errors)
$literal = $ast.Find({{ param($node)
    $node -is [System.Management.Automation.Language.StringConstantExpressionAst] -and $node.Value.Contains('public static class ExhibitionJob')
}}, $true)
Add-Type -TypeDefinition $literal.Value
$job = [ExhibitionJob]::Create()
$info = New-Object Diagnostics.ProcessStartInfo
$info.FileName = (Get-Command powershell.exe).Source
$info.Arguments = '-NoProfile -Command "Start-Sleep -Seconds 60"'
$info.UseShellExecute = $false
$info.CreateNoWindow = $true
$child = [Diagnostics.Process]::Start($info)
try {{
    [ExhibitionJob]::Assign($job, $child.Handle)
    $null = [ExhibitionJob]::CloseHandle($job)
    $job = [IntPtr]::Zero
    if (-not $child.WaitForExit(5000)) {{ throw 'Owned child survived job close' }}
}} finally {{
    if (-not $child.HasExited) {{ $child.Kill() }}
    $child.Dispose()
    if ($job -ne [IntPtr]::Zero) {{ $null = [ExhibitionJob]::CloseHandle($job) }}
}}
"""
    result = subprocess.run(
        [str(POWERSHELL), "-NoProfile", "-Command", command],
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_project_access_preserves_permissions_and_allows_edits(tmp_path: Path) -> None:
    """Grant project access using the real ACL command, confined to a temporary directory."""
    script_path = str(ROOT / "tools/setup_exhibition_windows.ps1").replace("'", "''")
    deployment = str(tmp_path).replace("'", "''")
    command = rf"""
$ErrorActionPreference = 'Stop'
$testSid = [Security.Principal.WindowsIdentity]::GetCurrent().User
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile('{script_path}', [ref]$tokens, [ref]$errors)
foreach ($name in @('Invoke-CheckedNative', 'Grant-ProjectAccess')) {{
    $function = $ast.Find({{ param($node)
        $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name
    }}, $true)
    Invoke-Expression $function.Extent.Text
}}
$code = Join-Path '{deployment}' 'setup.ps1'
$cache = Join-Path '{deployment}' 'cache'
$null = New-Item -ItemType Directory -Path $cache
[IO.File]::WriteAllText($code, 'original code')
$saved = Get-Acl -LiteralPath '{deployment}'
try {{
    Grant-ProjectAccess -Path '{deployment}' -UserSid $testSid.Value
    $after = Get-Acl -LiteralPath '{deployment}'
    if ($after.Owner -ne $saved.Owner) {{ throw 'Owner changed' }}
    foreach ($rule in $saved.Access) {{
        if (-not @($after.Access | Where-Object {{ $_.IdentityReference -eq $rule.IdentityReference -and
            $_.AccessControlType -eq $rule.AccessControlType -and
            ($_.FileSystemRights -band $rule.FileSystemRights) -eq $rule.FileSystemRights }}).Count) {{
            throw 'Existing access was removed'
        }}
    }}
    foreach ($path in @('{deployment}', $cache, $code)) {{
        $rules = (Get-Acl -LiteralPath $path).GetAccessRules($true, $true, [Security.Principal.SecurityIdentifier])
        if (-not @($rules | Where-Object {{ $_.IdentityReference -eq $testSid -and
            ($_.FileSystemRights -band [Security.AccessControl.FileSystemRights]::Modify) -eq [Security.AccessControl.FileSystemRights]::Modify }}).Count) {{
            throw "Project Modify access missing: $path"
        }}
    }}
    [IO.File]::WriteAllText($code, 'replacement')
    [IO.File]::WriteAllText((Join-Path '{deployment}' 'new-code.ps1'), 'replacement')
    $config = Join-Path $cache 'config.json'
    [IO.File]::WriteAllText($config, '{{"version":1}}')
    [IO.File]::WriteAllText("$config.tmp", '{{"version":2}}')
    [IO.File]::Replace("$config.tmp", $config, [NullString]::Value)
    if ((Get-Content -LiteralPath $config -Raw | ConvertFrom-Json).version -ne 2) {{ throw 'Atomic mutable config replacement failed' }}
}} finally {{
    $restore = New-Object Security.AccessControl.DirectorySecurity
    $restore.SetSecurityDescriptorSddlForm($saved.GetSecurityDescriptorSddlForm('Access'), 'Access')
    [IO.Directory]::SetAccessControl('{deployment}', $restore)
}}
"""
    result = subprocess.run(
        [str(POWERSHELL), "-NoProfile", "-Command", command],
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("right", ["DeleteSubdirectoriesAndFiles", "ChangePermissions", "TakeOwnership"])
def test_privileged_state_rejects_unsafe_ancestor(tmp_path: Path, right: str) -> None:
    """A protected child DACL cannot compensate for replacement rights on its parent."""
    script_path = str(ROOT / "tools/setup_exhibition_windows.ps1").replace("'", "''")
    protected_path = str(tmp_path / "CartographyExhibition").replace("'", "''")
    ancestor = str(tmp_path).replace("'", "''")
    command = rf"""
$ErrorActionPreference = 'Stop'
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile('{script_path}', [ref]$tokens, [ref]$errors)
$function = $ast.Find({{ param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Assert-ProtectedParents'
}}, $true)
Invoke-Expression $function.Extent.Text
$safe = New-Object Security.AccessControl.DirectorySecurity
$safe.SetOwner([Security.Principal.SecurityIdentifier]'S-1-5-32-544')
$safe.AddAccessRule((New-Object Security.AccessControl.FileSystemAccessRule(
    [Security.Principal.SecurityIdentifier]'S-1-5-32-544', 'FullControl', 'Allow')))
$unsafe = New-Object Security.AccessControl.DirectorySecurity
$unsafe.SetSecurityDescriptorSddlForm($safe.GetSecurityDescriptorSddlForm('All'))
$unsafe.AddAccessRule((New-Object Security.AccessControl.FileSystemAccessRule(
    [Security.Principal.SecurityIdentifier]'S-1-5-32-545', '{right}', 'Allow')))
$useUnsafe = $false
function Get-Acl {{
    param($LiteralPath)
    if ($useUnsafe -and $LiteralPath -eq '{ancestor}') {{ return $unsafe }}
    return $safe
}}
# Both cases run before the managed state directory exists, as on first setup.
Assert-ProtectedParents '{protected_path}'
$useUnsafe = $true
$rejected = $false
try {{ Assert-ProtectedParents '{protected_path}' }} catch {{
    if ($_.Exception.Message -notlike '*replace protected content*') {{ throw }}
    $rejected = $true
}}
if (-not $rejected) {{ throw 'Unsafe state ancestor was accepted' }}
if (Test-Path -LiteralPath '{protected_path}') {{ throw 'Ancestor validation created state' }}
"""
    result = subprocess.run(
        [str(POWERSHELL), "-NoProfile", "-Command", command],
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def setup_function_command(body: str) -> subprocess.CompletedProcess[str]:
    script_path = str(ROOT / "tools/setup_exhibition_windows.ps1").replace("'", "''")
    command = rf"""
$ErrorActionPreference = 'Stop'
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile('{script_path}', [ref]$tokens, [ref]$errors)
foreach ($name in @('Get-MonitorPlan', 'Read-MonitorCredential', 'New-ExhibitionMonitorTask', 'Add-ExhibitionSshConfiguration', 'Get-SshServicePaths', 'Publish-SshFile', 'Restore-SshFile')) {{
    $function = $ast.Find({{ param($node)
        $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name
    }}, $true)
    Invoke-Expression $function.Extent.Text
}}
{body}
"""
    return subprocess.run(
        [str(POWERSHELL), "-NoProfile", "-Command", command],
        text=True,
        capture_output=True,
        timeout=20,
    )


def test_monitor_plan_does_not_read_credentials() -> None:
    result = setup_function_command(r"""
function Read-MonitorCredential { throw 'Plan attempted to read token' }
$omitted = Get-MonitorPlan
if ($omitted.action -ne 'unchanged') { throw 'Omitted monitor configuration must remain unchanged' }
$plan = Get-MonitorPlan -CredentialPath 'C:\nonexistent\credential.json' -Endpoint 'https://monitor.example/api/v1/heartbeat'
if ($plan.action -ne 'install-or-update' -or $plan.endpointHost -ne 'monitor.example') { throw 'Unexpected plan' }
$plan | ConvertTo-Json -Compress
""")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "token" not in result.stdout.lower()


@pytest.mark.parametrize("endpoint", ["http://monitor.example/api/v1/heartbeat", "https://a:b@monitor.example/api/v1/heartbeat", "https://monitor.example/api/v1/heartbeat?token=secret", "https://monitor.example/api/v1/other"])
def test_monitor_rejects_unsafe_endpoint(endpoint: str) -> None:
    result = setup_function_command(f"Get-MonitorPlan -CredentialPath 'credential.json' -Endpoint '{endpoint}'")
    assert result.returncode != 0
    # Endpoint validation does not include the supplied URL in the error message.
    assert endpoint not in result.stdout


@pytest.mark.parametrize("invalid", [False, True])
def test_monitor_credential_validation_never_prints_token(tmp_path: Path, invalid: bool) -> None:
    token = "private-device-token-not-for-output-123456789"
    credential = tmp_path / "credential.json"
    credential.write_text(
        '{"token":"' + token + '","machineId":' if invalid else json.dumps({"token": token, "machineId": "gallery-a"}),
        encoding="utf-8",
    )
    path = str(credential).replace("'", "''")
    result = setup_function_command(f"$credentialToken = Read-MonitorCredential '{path}'; Write-Output 'validated'")
    assert (result.returncode != 0) == invalid, result.stdout + result.stderr
    assert token not in result.stdout + result.stderr


def test_monitor_task_uses_exhibition_logon_and_project_paths() -> None:
    result = setup_function_command(r"""
$testUser = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$task = New-ExhibitionMonitorTask -Root 'C:\Exhibition\Cartography' -ConfigPath 'C:\Exhibition\Cartography\cache\exhibition-monitor\config.json' -StateDirectory 'C:\Exhibition\Cartography\cache\exhibition-monitor' -UserId $testUser
$principalSid = ([Security.Principal.NTAccount]$task.Principal.UserId).Translate([Security.Principal.SecurityIdentifier]).Value
if ($principalSid -ne $testUser -or $task.Principal.LogonType -ne 'Interactive' -or $task.Principal.RunLevel -ne 'Limited') { throw 'Collector must run as the interactive exhibition user' }
if ($task.Triggers[0].CimClass.CimClassName -ne 'MSFT_TaskLogonTrigger' -or $task.Triggers[0].UserId -ne $testUser) { throw 'Collector must start at exhibition user logon' }
if ($task.Triggers[0].EndBoundary) { throw 'Collector logon trigger must not expire' }
if ($task.Settings.RestartCount -lt 1) { throw 'Collector must recover from process failure' }
$task.Actions[0] | Select-Object Execute, Arguments, WorkingDirectory | ConvertTo-Json -Compress
""")
    assert result.returncode == 0, result.stdout + result.stderr
    action = json.loads(result.stdout)
    assert action["Execute"].endswith(r"runtime\python\python.exe")
    assert "-I -B" in action["Arguments"]
    assert '--state-directory "C:\\Exhibition\\Cartography\\cache\\exhibition-monitor"' in action["Arguments"]
    assert '--snapshot-directory "C:\\Exhibition\\Cartography\\cache\\monitoring"' in action["Arguments"]
    assert "--config" in action["Arguments"]


def test_setup_account_creation_and_update_have_no_expiry() -> None:
    script_path = str(ROOT / "tools/setup_exhibition_windows.ps1").replace("'", "''")
    command = rf"""
$ErrorActionPreference = 'Stop'
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile('{script_path}', [ref]$tokens, [ref]$errors)
if ($errors.Count) {{ throw ($errors | Out-String) }}
if ($ast.ParamBlock.Parameters.Name.VariablePath.UserPath -contains 'ExhibitionEndDate') {{ throw 'Setup still requires a show date' }}
$UserName = 'exhibition'
$password = ConvertTo-SecureString 'temporary-test-password' -AsPlainText -Force
function New-LocalUser {{
    param($Name, $Password, [switch]$Disabled, [switch]$PasswordNeverExpires, [switch]$AccountNeverExpires, $Description, $AccountExpires)
    if (-not $AccountNeverExpires -or -not $PasswordNeverExpires -or $AccountExpires) {{ throw 'New account would expire' }}
    if (-not $Disabled) {{ throw 'New account could log in before SSH restrictions activate' }}
    Write-Output 'created'
}}
function Set-LocalUser {{
    param($Name, [bool]$PasswordNeverExpires, [switch]$AccountNeverExpires, $AccountExpires)
    if (-not $AccountNeverExpires -or -not $PasswordNeverExpires -or $AccountExpires) {{ throw 'Existing account expiry would not be cleared' }}
    Write-Output 'updated'
}}
try {{
    $commands = $ast.FindAll({{ param($node)
        $node -is [System.Management.Automation.Language.CommandAst] -and
        $node.GetCommandName() -in @('New-LocalUser', 'Set-LocalUser')
    }}, $true)
    foreach ($command in $commands) {{ Invoke-Expression $command.Extent.Text }}
}} finally {{ $password.Dispose() }}
"""
    result = subprocess.run(
        [str(POWERSHELL), "-NoProfile", "-Command", command],
        text=True, capture_output=True, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.split() == ["created", "updated"]


@pytest.mark.parametrize("existing", [
    "Port 22\nPasswordAuthentication yes\nSubsystem sftp sftp-server.exe\n",
    "Port 2222\nAllowUsers admin\nPasswordAuthentication yes\n"
    "Match Group administrators\n    AuthorizedKeysFile __PROGRAMDATA__/ssh/administrators_authorized_keys\n"
    "Match all\n    AllowTcpForwarding yes\n",
])
def test_ssh_addition_preserves_existing_global_and_admin_settings(existing: str) -> None:
    escaped = existing.replace("'", "''")
    result = setup_function_command(rf"""
$existing = '{escaped}'
$candidate = Add-ExhibitionSshConfiguration -Existing $existing -Account exhibition -KeyPath 'C:\ProgramData\CartographyExhibition\exhibition\authorized_keys'
$again = Add-ExhibitionSshConfiguration -Existing $candidate -Account exhibition -KeyPath 'C:\ProgramData\CartographyExhibition\exhibition\authorized_keys'
if ($again -ne $candidate) {{ throw 'Rerun changed or duplicated the user block' }}
$candidate
""")
    assert result.returncode == 0, result.stdout + result.stderr
    candidate = result.stdout
    begin = candidate.index("# BEGIN Cartography exhibition exhibition")
    end = candidate.index("# END Cartography exhibition exhibition")
    block = candidate[begin:end]
    assert block.splitlines()[1] == "Match User exhibition"
    assert "    AllowUsers exhibition\n" in block
    assert "    PasswordAuthentication no\n" in block
    assert "    AuthenticationMethods publickey\n" in block
    assert "    PubkeyAcceptedKeyTypes ssh-ed25519," in block
    assert "PubkeyAcceptedAlgorithms" not in block
    assert '    AuthorizedKeysFile "C:/ProgramData/CartographyExhibition/exhibition/authorized_keys"' in block
    assert "-cert-" not in block
    restored = candidate[:begin] + candidate[end:].split("\n", 1)[1]
    assert restored.strip() == existing.strip()
    if "Match Group administrators" in existing:
        assert begin < candidate.index("Match Group administrators")


@pytest.mark.parametrize("custom", [False, True])
def test_existing_ssh_service_keeps_executable_and_config_paths(custom: bool) -> None:
    command_line = (
        '"C:\\Tools\\OpenSSH\\sshd.exe" -f "C:\\SSH config\\sshd_config"'
        if custom else r"C:\Windows\System32\OpenSSH\sshd.exe"
    )
    result = setup_function_command(f"Get-SshServicePaths -CommandLine '{command_line}' | ConvertTo-Json -Compress")
    assert result.returncode == 0, result.stdout + result.stderr
    paths = json.loads(result.stdout)
    assert paths["executable"] == (r"C:\Tools\OpenSSH\sshd.exe" if custom else r"C:\Windows\System32\OpenSSH\sshd.exe")
    assert paths["config"].lower() == (r"C:\SSH config\sshd_config" if custom else r"C:\ProgramData\ssh\sshd_config").lower()


@pytest.mark.parametrize("failure", ["staging", "activation", "after_activation"])
def test_ssh_failure_preserves_existing_config_and_keys(tmp_path: Path, failure: str) -> None:
    deployment = str(tmp_path).replace("'", "''")
    result = setup_function_command(rf"""
$root = '{deployment}'
$config = Join-Path $root 'sshd_config'
$keys = Join-Path $root 'authorized_keys'
$configCandidate = "$config.candidate"
$keyCandidate = "$keys.candidate"
[IO.File]::WriteAllText($config, 'original admin configuration')
[IO.File]::WriteAllText($keys, 'original exhibition public keys')
$changes = @()
$failed = $false
$heldFile = $null
try {{
    [IO.File]::WriteAllText($configCandidate, 'new configuration')
    [IO.File]::WriteAllText($keyCandidate, 'new exhibition public keys')
    if ('{failure}' -eq 'staging') {{
        $heldFile = [IO.File]::Open($configCandidate, 'Open', 'ReadWrite', 'None')
        [IO.File]::WriteAllText($configCandidate, 'staging write must fail')
    }}
    $changes += Publish-SshFile -CandidatePath $keyCandidate -Path $keys -BackupPath "$keys.previous"
    if ('{failure}' -eq 'activation') {{
        # A missing candidate makes atomic activation fail after the first file was published.
        Remove-Item -LiteralPath $configCandidate
    }}
    $changes += Publish-SshFile -CandidatePath $configCandidate -Path $config -BackupPath "$config.previous"
    throw 'simulated service restart failure'
}} catch {{
    $failed = $true
    [array]::Reverse($changes)
    foreach ($change in $changes) {{ Restore-SshFile $change }}
}} finally {{ if ($heldFile) {{ $heldFile.Dispose() }} }}
if (-not $failed) {{ throw 'Failure injection did not run' }}
if ([IO.File]::ReadAllText($config) -ne 'original admin configuration') {{ throw 'Existing SSH configuration changed after failure' }}
if ([IO.File]::ReadAllText($keys) -ne 'original exhibition public keys') {{ throw 'Existing authorized keys changed after failure' }}
if (@(Get-ChildItem -LiteralPath $root -Filter '*.restore').Count) {{ throw 'Recovery left a temporary file' }}
Write-Output 'restored'
""")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "restored"


@pytest.mark.parametrize("account_state,failure", [
    ("new", "none"), ("new", "keys"), ("new", "config"),
    ("new", "restart"), ("new", "save"),
    ("pending", "none"), ("pending", "restart"),
    ("enabled", "none"), ("enabled", "restart"), ("disabled", "none"),
])
def test_account_activation_waits_for_ssh_and_preserves_existing_state(account_state: str, failure: str) -> None:
    result = setup_function_command(rf"""
$pending = '{account_state}' -in @('new', 'pending')
$script:accountEnabled = '{account_state}' -eq 'enabled'
$user = if ('{account_state}' -eq 'new') {{ $null }} else {{ [pscustomobject]@{{ Enabled = $script:accountEnabled }} }}
$record = [pscustomobject]@{{ accountActivationPending = $pending; completedPhase = 'ssh-staged' }}
$prior = if ('{account_state}' -eq 'new') {{ $null }} else {{ $record }}
foreach ($variable in @('$activationPending', '$activateAccount')) {{
    $assignment = $ast.Find({{ param($node)
        $node -is [System.Management.Automation.Language.AssignmentStatementAst] -and $node.Left.Extent.Text -eq $variable
    }}, $true)
    Invoke-Expression $assignment.Extent.Text
}}
$UserName = 'exhibition'; $keyPath = 'keys'; $configPath = 'config'
$keyCandidate = 'keys.candidate'; $candidatePath = 'config.candidate'
$stateRoot = 'C:\state'; $configBackup = 'config.previous'; $recordPath = 'unused'
$service = [pscustomobject]@{{ State = 'Running' }}
$sshChanges = @(); $accountActivationAttempted = $false
$script:published = 0; $script:restarted = $false; $script:saveFailed = $false
$script:disableCount = 0; $script:restoreCount = 0
function Publish-SshFile {{
    param($CandidatePath, $Path, $BackupPath)
    if ('{failure}' -eq $Path) {{ throw 'Injected file activation failure' }}
    $script:published++
    return [pscustomobject]@{{ path = $Path; backup = $BackupPath }}
}}
function Restart-Service {{
    param($Name)
    if ('{failure}' -eq 'restart') {{ throw 'Injected service failure' }}
    $script:restarted = $true
}}
function Enable-LocalUser {{
    param($Name)
    if ($script:published -ne 2 -or -not $script:restarted) {{ throw 'Account enabled before SSH was ready' }}
    $script:accountEnabled = $true
}}
function Disable-LocalUser {{ param($Name); $script:accountEnabled = $false; $script:disableCount++ }}
function Get-Service {{ param($Name); return [pscustomobject]@{{ Status = 'Running' }} }}
function Restore-SshFile {{
    param($Change)
    if ($pending -and $script:accountEnabled) {{ throw 'SSH policy restored while new account was enabled' }}
    $script:restoreCount++
}}
function Save-Record {{
    if ('{failure}' -eq 'save' -and -not $script:saveFailed) {{ $script:saveFailed = $true; throw 'Injected record failure' }}
}}
$outerTry = $ast.Find({{ param($node)
    $node -is [System.Management.Automation.Language.TryStatementAst] -and
    $node.Body.Extent.Text.Contains('$sshChanges += Publish-SshFile')
}}, $true)
$tail = @(); $capture = $false
foreach ($statement in $outerTry.Body.Statements) {{
    if ($statement.Extent.Text.StartsWith('$sshChanges += Publish-SshFile')) {{ $capture = $true }}
    if ($capture) {{ $tail += $statement.Extent.Text }}
}}
$caught = $false
try {{ Invoke-Expression ($tail -join "`n") }} catch {{
    try {{ Invoke-Expression ($outerTry.CatchClauses[0].Body.Statements.Extent.Text -join "`n") }} catch {{ $caught = $true }}
}}
if ($caught -ne ('{failure}' -ne 'none')) {{ throw 'Unexpected transaction result' }}
$expectedEnabled = '{account_state}' -eq 'enabled' -or ($pending -and '{failure}' -eq 'none')
if ($script:accountEnabled -ne $expectedEnabled) {{ throw 'Wrong final account enabled state' }}
if (-not $pending -and $script:disableCount) {{ throw 'Pre-existing account was disabled' }}
if ($pending -and '{failure}' -ne 'none' -and -not $record.accountActivationPending) {{ throw 'Failed setup lost its pending activation record' }}
if ('{failure}' -eq 'save' -and ($script:disableCount -ne 1 -or $script:restoreCount -ne 2)) {{ throw 'Late failure did not disable before restoring both files' }}
Write-Output 'state verified'
""")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().endswith("state verified")
