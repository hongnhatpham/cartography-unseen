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


def test_readonly_code_and_atomic_mutable_config(tmp_path: Path) -> None:
    """Exercise production DACLs as an ordinary token without changing file ownership."""
    script_path = str(ROOT / "tools/setup_exhibition_windows.ps1").replace("'", "''")
    deployment = str(tmp_path).replace("'", "''")
    command = rf"""
$ErrorActionPreference = 'Stop'
$testSid = [Security.Principal.WindowsIdentity]::GetCurrent().User
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile('{script_path}', [ref]$tokens, [ref]$errors)
$function = $ast.Find({{ param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Set-PrivateAcl'
}}, $true)
Invoke-Expression $function.Extent.Text
function Set-Acl {{
    param($LiteralPath, $AclObject)
    if ($AclObject.GetOwner([Security.Principal.SecurityIdentifier]).Value -ne 'S-1-5-32-544') {{ throw 'Production owner must be Administrators' }}
    # Ownership assignment needs elevation. Assert the requested owner above and apply
    # only the DACL so ordinary-token writes exercise the real file access rules.
    $accessOnly = New-Object Security.AccessControl.DirectorySecurity
    $accessOnly.SetSecurityDescriptorSddlForm($AclObject.GetSecurityDescriptorSddlForm('Access'), 'Access')
    [IO.Directory]::SetAccessControl($LiteralPath, $accessOnly)
}}
$code = Join-Path '{deployment}' 'setup.ps1'
$cache = Join-Path '{deployment}' 'cache'
$null = New-Item -ItemType Directory -Path $cache
[IO.File]::WriteAllText($code, 'trusted code')
$saved = Get-Acl -LiteralPath '{deployment}'
try {{
    Set-PrivateAcl -Path '{deployment}' -ReaderSid $testSid.Value -Directory
    Set-PrivateAcl -Path $cache -ReaderSid $testSid.Value -Directory -Modify
    $denied = $false
    try {{ [IO.File]::WriteAllText($code, 'replacement') }} catch [UnauthorizedAccessException] {{ $denied = $true }}
    if (-not $denied) {{ throw 'Code replacement was allowed' }}
    $denied = $false
    try {{ [IO.File]::WriteAllText((Join-Path '{deployment}' 'new-code.ps1'), 'replacement') }} catch [UnauthorizedAccessException] {{ $denied = $true }}
    if (-not $denied) {{ throw 'Root file creation was allowed' }}
    $config = Join-Path $cache 'config.json'
    [IO.File]::WriteAllText($config, '{{"version":1}}')
    [IO.File]::WriteAllText("$config.tmp", '{{"version":2}}')
    [IO.File]::Replace("$config.tmp", $config, [NullString]::Value)
    if ((Get-Content -LiteralPath $config -Raw | ConvertFrom-Json).version -ne 2) {{ throw 'Atomic mutable config replacement failed' }}
}} finally {{
    $restore = New-Object Security.AccessControl.DirectorySecurity
    $restore.SetSecurityDescriptorSddlForm($saved.GetSecurityDescriptorSddlForm('Access'), 'Access')
    [IO.Directory]::SetAccessControl('{deployment}', $restore)
    # Remove the protected test cache ACL so the temporary-directory fixture can clean up.
    $cacheAcl = Get-Acl -LiteralPath $cache
    $cacheAcl.SetAccessRuleProtection($false, $false)
    [IO.Directory]::SetAccessControl($cache, $cacheAcl)
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


def monitor_function_command(body: str) -> subprocess.CompletedProcess[str]:
    script_path = str(ROOT / "tools/setup_exhibition_windows.ps1").replace("'", "''")
    command = rf"""
$ErrorActionPreference = 'Stop'
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile('{script_path}', [ref]$tokens, [ref]$errors)
foreach ($name in @('Get-MonitorPlan', 'Read-MonitorCredential', 'New-ExhibitionMonitorTask')) {{
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
    result = monitor_function_command(r"""
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
    result = monitor_function_command(f"Get-MonitorPlan -CredentialPath 'credential.json' -Endpoint '{endpoint}'")
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
    result = monitor_function_command(f"$credentialToken = Read-MonitorCredential '{path}'; Write-Output 'validated'")
    assert (result.returncode != 0) == invalid, result.stdout + result.stderr
    assert token not in result.stdout + result.stderr


def test_monitor_task_is_system_startup_with_protected_paths() -> None:
    result = monitor_function_command(r"""
$task = New-ExhibitionMonitorTask -Root 'C:\Exhibition\Cartography' -ConfigPath 'C:\ProgramData\CartographyExhibition\exhibition\monitor\config.json' -StateDirectory 'C:\ProgramData\CartographyExhibition\exhibition\monitor'
if ($task.Principal.UserId -notin @('SYSTEM', 'S-1-5-18', 'NT AUTHORITY\SYSTEM')) { throw 'Collector must run as SYSTEM' }
if ($task.Triggers[0].CimClass.CimClassName -ne 'MSFT_TaskBootTrigger') { throw 'Collector must start at boot' }
if ($task.Settings.RestartCount -lt 1) { throw 'Collector must recover from process failure' }
$task.Actions[0] | Select-Object Execute, Arguments, WorkingDirectory | ConvertTo-Json -Compress
""")
    assert result.returncode == 0, result.stdout + result.stderr
    action = json.loads(result.stdout)
    assert action["Execute"].endswith(r"runtime\python\python.exe")
    assert "-I -B" in action["Arguments"]
    assert '--state-directory "C:\\ProgramData\\CartographyExhibition\\exhibition\\monitor"' in action["Arguments"]
    assert '--snapshot-directory "C:\\Exhibition\\Cartography\\cache\\monitoring"' in action["Arguments"]
    assert "--config" in action["Arguments"]
