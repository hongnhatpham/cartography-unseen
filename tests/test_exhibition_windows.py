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


def test_gpu_preference_is_repeatable_and_preserves_other_settings(tmp_path: Path) -> None:
    import uuid
    import winreg

    for name in ("python.exe", "pythonw.exe"):
        executable = tmp_path / "runtime/python" / name
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.touch()
    key_name = "Software\\CartographyGpuTest-" + uuid.uuid4().hex
    python_path = str(tmp_path / "runtime/python/python.exe")
    try:
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_name) as key:
            winreg.SetValueEx(key, python_path, 0, winreg.REG_SZ, "GpuPreference=1;AutoHDREnable=1;")
            winreg.SetValueEx(key, "unrelated.exe", 0, winreg.REG_SZ, "GpuPreference=1;")
        for _ in range(2):
            result = subprocess.run([
                str(POWERSHELL), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                str(ROOT / "tools/configure_exhibition_gpu.ps1"),
                "-DeploymentPath", str(tmp_path), "-RegistryPath", "HKCU:\\" + key_name,
            ], text=True, capture_output=True, timeout=30)
            assert result.returncode == 0, result.stdout + result.stderr
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_name) as key:
                assert winreg.QueryValueEx(key, python_path)[0] == "GpuPreference=2;AutoHDREnable=1;"
                assert winreg.QueryValueEx(key, str(tmp_path / "runtime/python/pythonw.exe"))[0] == "GpuPreference=2;"
                assert winreg.QueryValueEx(key, "unrelated.exe")[0] == "GpuPreference=1;"
    finally:
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, key_name)


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
foreach ($name in @('Get-MonitorPlan', 'Read-MonitorCredential', 'New-ExhibitionMonitorTask', 'Add-ExhibitionSshConfiguration', 'Get-SshServicePaths', 'Get-SshServiceRegistration', 'Resolve-TrustedSshd', 'Publish-SshFile', 'Restore-SshFile')) {{
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
$user = $null
$record = [pscustomobject]@{{ userSid = $null; createdUser = $false; completedPhase = 'begin' }}
function Save-Record {{ }}
function Read-Host {{ throw 'Passwordless setup must not prompt' }}
function New-LocalUser {{
    param($Name, $Password, [switch]$NoPassword, [switch]$Disabled, [switch]$AccountNeverExpires, $Description, $AccountExpires)
    if (-not $AccountNeverExpires -or $AccountExpires) {{ throw 'New account would expire' }}
    if (-not $NoPassword -or $Password) {{ throw 'New account must be passwordless' }}
    if (-not $Disabled) {{ throw 'New account could log in before SSH restrictions activate' }}
    Write-Host 'created'
    return [pscustomobject]@{{ SID = [Security.Principal.SecurityIdentifier]'S-1-5-21-1-2-3-1001' }}
}}
function Set-LocalUser {{
    param($Name, [bool]$PasswordNeverExpires, [switch]$AccountNeverExpires, $AccountExpires)
    if (-not $AccountNeverExpires -or -not $PasswordNeverExpires -or $AccountExpires) {{ throw 'Existing account expiry would not be cleared' }}
    Write-Output 'updated'
}}
$creation = $ast.Find({{ param($node)
    $node -is [System.Management.Automation.Language.IfStatementAst] -and
    $node.Extent.Text.StartsWith('if (-not $user)') -and $node.Extent.Text.Contains('New-LocalUser')
}}, $true)
if (-not $creation) {{ throw 'Missing production account creation seam' }}
Invoke-Expression $creation.Extent.Text
if (-not $record.createdUser -or -not $record.accountActivationPending) {{ throw 'New account was not recorded for pending activation' }}
$commands = $ast.FindAll({{ param($node)
        $node -is [System.Management.Automation.Language.CommandAst] -and
        $node.GetCommandName() -eq 'Set-LocalUser' -and $node.Extent.Text.Contains('-AccountNeverExpires')
    }}, $true)
foreach ($command in $commands) {{ Invoke-Expression $command.Extent.Text }}
"""
    result = subprocess.run(
        [str(POWERSHELL), "-NoProfile", "-Command", command],
        text=True, capture_output=True, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.split() == ["created", "updated"]


@pytest.mark.parametrize("outcome", [
    "added", "already", "already-record", "already-wrapped", "denied", "trust",
    "denied-wrapped", "trust-wrapped", "ambiguous-message", "unrelated-command",
])
def test_setup_users_membership_never_enumerates_unrelated_sids(outcome: str) -> None:
    """Run the real account-update statements with the reported enumeration failure injected."""
    result = setup_function_command(rf"""
Import-Module Microsoft.PowerShell.LocalAccounts
$UserName = 'exhibition'
$user = [pscustomobject]@{{ SID = [Security.Principal.SecurityIdentifier]'S-1-5-21-1-2-3-1001' }}
$script:addCalls = 0
function Set-LocalUser {{ param($Name, [switch]$AccountNeverExpires, $PasswordNeverExpires) }}
function Get-LocalGroup {{ param($SID); return [pscustomobject]@{{ SID = $SID }} }}
function Get-LocalGroupMember {{ throw [ComponentModel.Win32Exception]::new(1789) }}
function Add-LocalGroupMember {{
    [CmdletBinding()]param($SID, $Member)
    if ($SID -ne 'S-1-5-32-545' -or $Member -ne $user.SID.Value) {{ throw 'Membership target must use exact group and local account SIDs' }}
    if ($ErrorActionPreference -ne 'Stop') {{ throw 'Membership errors must terminate' }}
    $script:addCalls++
    if ('{outcome}' -eq 'already') {{ throw [Microsoft.PowerShell.Commands.MemberExistsException]::new('already a member') }}
    if ('{outcome}' -eq 'denied') {{ throw [ComponentModel.Win32Exception]::new(5) }}
    if ('{outcome}' -eq 'trust') {{ throw [ComponentModel.Win32Exception]::new(1789) }}
    if ('{outcome}' -eq 'already-record') {{
        $errorRecord = [Management.Automation.ErrorRecord]::new(
            [Microsoft.PowerShell.Commands.MemberExistsException]::new('already a member'),
            'MemberExists', [Management.Automation.ErrorCategory]::ResourceExists, $Member)
        $PSCmdlet.WriteError($errorRecord)
    }}
    if ('{outcome}' -in @('already-wrapped', 'denied-wrapped', 'trust-wrapped', 'ambiguous-message', 'unrelated-command')) {{
        $errorId = switch ('{outcome}') {{
            'already-wrapped' {{ 'MemberExists,Microsoft.PowerShell.Commands.AddLocalGroupMemberCommand' }}
            'denied-wrapped' {{ 'AccessDenied,Microsoft.PowerShell.Commands.AddLocalGroupMemberCommand' }}
            'trust-wrapped' {{ '1789,Microsoft.PowerShell.Commands.AddLocalGroupMemberCommand' }}
            'unrelated-command' {{ 'MemberExists,OtherCommand' }}
            default {{ 'UnidentifiedError,Microsoft.PowerShell.Commands.AddLocalGroupMemberCommand' }}
        }}
        # Preserve the cmdlet's stable ID while reproducing ErrorAction Stop's wrapper.
        # All messages deliberately look like duplicates: text alone must never authorize swallowing.
        $wrapped = [Management.Automation.ActionPreferenceStopException]::new('already a member')
        $errorRecord = [Management.Automation.ErrorRecord]::new(
            $wrapped, $errorId, [Management.Automation.ErrorCategory]::NotSpecified, $Member)
        $script:injectedError = $errorRecord
        throw $errorRecord
    }}
}}
$outerTry = $ast.Find({{ param($node)
    $node -is [System.Management.Automation.Language.TryStatementAst] -and
    $node.Body.Extent.Text.Contains('$sshChanges += Publish-SshFile')
}}, $true)
$capture = $false; $statements = @()
foreach ($statement in $outerTry.Body.Statements) {{
    if ($statement.Extent.Text.StartsWith('Set-LocalUser')) {{ $capture = $true }}
    if ($statement.Extent.Text.StartsWith('if ($MonitorCredentialPath')) {{ break }}
    if ($capture) {{ $statements += $statement.Extent.Text }}
}}
if (-not $statements.Count) {{ throw 'Missing production membership seam' }}
$caughtCode = 0
try {{ Invoke-Expression ($statements -join "`n") }} catch {{
    if ('{outcome}' -in @('denied-wrapped', 'trust-wrapped', 'ambiguous-message', 'unrelated-command')) {{
        if ($_.Exception -ne $script:injectedError.Exception -or
            $_.FullyQualifiedErrorId -ne $script:injectedError.FullyQualifiedErrorId) {{ throw }}
        $caughtCode = -1
    }} else {{
        if ($_.Exception -isnot [ComponentModel.Win32Exception]) {{ throw }}
        $caughtCode = $_.Exception.NativeErrorCode
    }}
}}
$expectedCode = switch ('{outcome}') {{
    'denied' {{ 5 }} 'trust' {{ 1789 }}
    {{ $_ -in @('denied-wrapped', 'trust-wrapped', 'ambiguous-message', 'unrelated-command') }} {{ -1 }}
    default {{ 0 }}
}}
if ($caughtCode -ne $expectedCode -or $script:addCalls -ne 1) {{ throw "Wrong membership result: code=$caughtCode calls=$script:addCalls" }}
Write-Output 'membership verified'
""")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "membership verified"


@pytest.mark.parametrize("membership", ["standard", "admin", "error"])
def test_setup_admin_guard_checks_only_the_local_account(membership: str) -> None:
    result = setup_function_command(rf"""
$function = $ast.Find({{ param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Test-LocalAdministrator'
}}, $true)
if (-not $function) {{ throw 'Missing exact-member administrator check' }}
Invoke-Expression $function.Extent.Text
$UserName = 'exhibition'; $user = [pscustomobject]@{{ SID = 'S-1-5-21-1-2-3-1001' }}
$script:checked = $false; $script:disposed = $false
function Get-LocalGroupMember {{ throw [ComponentModel.Win32Exception]::new(1789) }}
function Get-LocalGroup {{
    param($SID)
    if ($SID -ne 'S-1-5-32-544') {{ throw 'Wrong administrator group' }}
    return [pscustomobject]@{{ Name = 'Localized administrators' }}
}}
$fakeGroup = [pscustomobject]@{{}}
$fakeGroup | Add-Member ScriptMethod Invoke {{
    param($Method, $Arguments)
    if ($Method -ne 'IsMember' -or $Arguments.Count -ne 1 -or
        $Arguments[0] -ne "WinNT://DOMAIN/$env:COMPUTERNAME/exhibition") {{ throw 'Guard must use the resolved local account path' }}
    $script:checked = $true
    if ('{membership}' -eq 'error') {{ throw 'Injected membership lookup failure' }}
    return '{membership}' -eq 'admin'
}}
$fakeGroup | Add-Member ScriptMethod Dispose {{ $script:disposed = $true }}
$fakeMember = [pscustomobject]@{{}}
$fakeMember | Add-Member ScriptMethod InvokeGet {{
    param($Property)
    if ($Property -ne 'ADsPath') {{ throw 'Wrong local account property' }}
    return "WinNT://DOMAIN/$env:COMPUTERNAME/exhibition"
}}
$fakeMember | Add-Member ScriptMethod Dispose {{ }}
function New-Object {{
    param($TypeName, $ArgumentList)
    if ($TypeName -eq 'DirectoryServices.DirectoryEntry' -and
        $ArgumentList -eq "WinNT://$env:COMPUTERNAME/exhibition,user") {{ return $fakeMember }}
    if ($TypeName -ne 'DirectoryServices.DirectoryEntry' -or
        $ArgumentList -ne "WinNT://$env:COMPUTERNAME/Localized administrators,group") {{ throw 'Wrong local group binding' }}
    return $fakeGroup
}}
$guard = $ast.Find({{ param($node)
    $node -is [System.Management.Automation.Language.IfStatementAst] -and
    $node.Extent.Text.Contains("throw 'The exhibition account is an administrator;")
}}, $true)
if (-not $guard) {{ throw 'Missing production administrator guard' }}
$caught = $false
try {{ Invoke-Expression $guard.Extent.Text }} catch {{
    if ('{membership}' -eq 'admin' -and $_.Exception.Message -notlike '*account is an administrator*') {{ throw }}
    if ('{membership}' -eq 'error' -and $_.Exception.Message -notlike '*Injected membership lookup failure*') {{ throw }}
    $caught = $true
}}
if ($caught -ne ('{membership}' -ne 'standard') -or -not $script:checked -or -not $script:disposed) {{ throw 'Wrong administrator guard result' }}
Write-Output 'guard verified'
""")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "guard verified"


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
    assert "    KbdInteractiveAuthentication no\n" in block
    assert "ChallengeResponseAuthentication" not in block
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


@pytest.mark.parametrize("scm,registry", [(False, False), (True, True), (True, False), (False, True)])
def test_ssh_registration_requires_registry_and_service_manager_agreement(scm: bool, registry: bool) -> None:
    result = setup_function_command(rf"""
function Get-CimInstance {{ param($ClassName, $Filter); if (${str(scm).lower()}) {{ return [pscustomobject]@{{ Name = 'sshd' }} }} }}
function Test-Path {{
    param($LiteralPath)
    if ($LiteralPath -ne 'HKLM:\SYSTEM\CurrentControlSet\Services\sshd') {{ throw 'Wrong service registry path' }}
    return ${str(registry).lower()}
}}
$registration = Get-SshServiceRegistration
if ([bool]$registration -ne ${str(scm).lower()}) {{ throw 'Wrong service lookup result' }}
Write-Output 'registration verified'
""")
    assert (result.returncode == 0) == (scm == registry), result.stdout + result.stderr
    if scm != registry:
        assert "registry and service manager state disagree" in result.stderr


@pytest.mark.parametrize("case", ["trusted", "wrong-path", "invalid-signature", "other-publisher"])
def test_missing_service_requires_canonical_microsoft_signed_sshd(case: str, tmp_path: Path) -> None:
    # Use a real regular-file object while isolating signature and canonical-path inputs.
    fixture = tmp_path / "sshd.exe"
    fixture.write_text("test executable", encoding="utf-8")
    fixture_path = str(fixture).replace("'", "''")
    result = setup_function_command(rf"""
$canonical = Join-Path $env:SystemRoot 'System32\OpenSSH\sshd.exe'
$fixture = Microsoft.PowerShell.Management\Get-Item -LiteralPath '{fixture_path}'
function Resolve-Path {{
    param($LiteralPath)
    return [pscustomobject]@{{ Path = $(if ('{case}' -eq 'wrong-path') {{ 'C:\project\sshd.exe' }} else {{ $canonical }}) }}
}}
function Get-Item {{ param($LiteralPath, [switch]$Force); return $fixture }}
function Get-AuthenticodeSignature {{
    param($LiteralPath)
    if ($LiteralPath -ne $canonical) {{ throw 'Signature was checked on the wrong binary' }}
    return [pscustomobject]@{{
        Status = $(if ('{case}' -eq 'invalid-signature') {{ 'HashMismatch' }} else {{ 'Valid' }})
        SignerCertificate = [pscustomobject]@{{ Subject = $(if ('{case}' -eq 'other-publisher') {{ 'CN=Publisher, O=Other Corporation' }} else {{ 'CN=Microsoft Windows, O=Microsoft Corporation, C=US' }}) }}
    }}
}}
$resolved = Resolve-TrustedSshd -Executable $canonical
if ($resolved -ne $canonical) {{ throw 'Wrong service executable returned' }}
Write-Output 'binary verified'
""")
    assert (result.returncode == 0) == (case == "trusted"), result.stdout + result.stderr
    if case == "wrong-path":
        assert "only be registered from the inbox" in result.stderr
    elif case != "trusted":
        assert "valid Microsoft signature" in result.stderr


@pytest.mark.parametrize("existing,failure", [
    ("missing", "none"), ("missing", "register"), ("missing", "start"),
    ("missing", "record"), ("missing", "late"),
    ("missing", "not-running"),
    ("running", "late"), ("stopped", "none"), ("prior-created", "late"),
])
def test_setup_registers_missing_ssh_service_and_rolls_back_only_its_creation(existing: str, failure: str) -> None:
    """Exercise production service setup/activation/catch statements with installed binaries."""
    result = setup_function_command(rf"""
$capability = [pscustomobject]@{{ State = 'Installed' }}
$sshd = Join-Path $env:SystemRoot 'System32\OpenSSH\sshd.exe'
$configPath = Join-Path $env:ProgramData 'ssh\sshd_config'
$script:exists = '{existing}' -ne 'missing'
$script:running = '{existing}' -in @('running', 'prior-created')
$service = if ($script:exists) {{ [pscustomobject]@{{ PathName = $sshd; State = $(if ($script:running) {{ 'Running' }} else {{ 'Stopped' }}); StartMode = 'Manual' }} }} else {{ $null }}
$record = [pscustomobject]@{{ completedPhase = 'deployment-acl'; originalService = $service }}
if ('{existing}' -eq 'prior-created') {{ $record | Add-Member NoteProperty createdService $true }}
$migration = $ast.EndBlock.Statements | Where-Object {{ $_.Extent.Text.StartsWith("if (`$record.PSObject.Properties.Name -notcontains 'createdService')") }}
Invoke-Expression $migration.Extent.Text
$createdServiceThisRun = $false
$accountActivationAttempted = $false; $passwordResetAttempted = $false
$sshChanges = @(); $script:published = $false
$script:created = 0; $script:deleted = 0; $script:started = 0; $script:stopped = 0; $script:automatic = $false
$script:recordFailed = $false
$script:trusted = $false
function Add-WindowsCapability {{ throw 'Installed capability must not be reinstalled' }}
function Get-CimInstance {{ param($ClassName, $Filter); if ($script:exists) {{ return [pscustomobject]@{{ PathName = $sshd; State = $(if ($script:running) {{ 'Running' }} else {{ 'Stopped' }}) }} }} }}
function Resolve-Path {{ param($LiteralPath); if ($LiteralPath -ne $sshd) {{ throw 'Wrong service executable' }}; return [pscustomobject]@{{ Path = $sshd }} }}
function Resolve-TrustedSshd {{ param($Executable); if ($Executable -ne $sshd) {{ throw 'Wrong service executable' }}; $script:trusted = $true; return $sshd }}
function Test-Path {{ param($LiteralPath); if ($LiteralPath -ne 'HKLM:\SYSTEM\CurrentControlSet\Services\sshd') {{ throw 'Unexpected registry lookup' }}; return $script:exists }}
function New-Service {{
    [CmdletBinding()]param($Name, $BinaryPathName, $DisplayName, $Description, $StartupType, $Credential)
    if (-not $script:trusted -or $Name -ne 'sshd' -or $BinaryPathName -ne ('"' + $sshd + '"') -or
        $DisplayName -ne 'OpenSSH SSH Server' -or -not $Description -or $StartupType -ne 'Manual' -or $Credential) {{ throw 'Incorrect service registration or non-LocalSystem account' }}
    if ($script:exists) {{ throw 'Existing service was registered again' }}
    if ('{failure}' -eq 'register') {{ throw 'Injected registration failure' }}
    $script:exists = $true; $script:created++
}}
function Set-Service {{
    param($Name, $StartupType)
    if (-not $script:exists) {{ throw [ComponentModel.Win32Exception]::new(1060) }}
    if ($StartupType -ne 'Automatic' -or -not $script:published -or -not $script:running) {{ throw 'Automatic startup enabled before verified SSH startup' }}
    if ('{existing}' -ne 'missing') {{ throw 'Pre-existing startup policy changed' }}
    $script:automatic = $true
}}
function Start-Service {{
    param($Name)
    if (-not $script:exists) {{ throw [ComponentModel.Win32Exception]::new(1060) }}
    if (-not $script:published) {{ throw 'Service started before SSH activation' }}
    $script:started++
    if ('{failure}' -eq 'start') {{ throw 'Injected service startup failure' }}
    $script:running = '{failure}' -ne 'not-running'
}}
function Restart-Service {{ param($Name); if (-not $script:running) {{ throw 'Stopped existing service restarted' }} }}
function Stop-Service {{ param($Name); if ('{existing}' -ne 'missing') {{ throw 'Pre-existing service stopped' }}; $script:running = $false; $script:stopped++ }}
function Get-Service {{ param($Name); if ($script:exists) {{ return [pscustomobject]@{{ Status = $(if ($script:running) {{ 'Running' }} else {{ 'Stopped' }}) }} }} }}
function Invoke-CheckedNative {{
    param($Executable, $Arguments)
    if ($Executable -ne 'sc.exe' -or ($Arguments -join ' ') -ne 'delete sshd' -or
        '{existing}' -ne 'missing' -or $script:running) {{ throw 'Unsafe service deletion' }}
    $script:exists = $false; $script:deleted++
}}
function Save-Record {{
    if ('{failure}' -eq 'record' -and $script:created -and -not $script:recordFailed) {{ $script:recordFailed = $true; throw 'Injected service record failure' }}
}}
function Restore-SshFile {{ throw 'No SSH file change was injected in this seam' }}
$outerTry = $ast.Find({{ param($node)
    $node -is [System.Management.Automation.Language.TryStatementAst] -and $node.Body.Extent.Text.Contains('$sshChanges += Publish-SshFile')
}}, $true)
$prepare = @($outerTry.Body.Statements | Where-Object {{
    $_.Extent.Text.StartsWith('if ($capability.State') -or
    ($_.Extent.Text.StartsWith('if (-not $service)') -and ($_.Extent.Text.Contains('New-Service') -or $_.Extent.Text.Contains('Set-Service')))
}})
$activate = $outerTry.Body.Statements | Where-Object {{ $_.Extent.Text.StartsWith('if ($service -and $service.State') }}
if (-not $prepare.Count -or -not $activate) {{ throw 'Missing production service transaction seam' }}
$caught = $false
try {{
    foreach ($statement in $prepare) {{ Invoke-Expression $statement.Extent.Text }}
    $script:published = $true
    Invoke-Expression $activate.Extent.Text
    if ('{failure}' -eq 'late') {{ throw 'Injected later setup failure' }}
}} catch {{
    $caught = $true
    $originalError = $_
    try {{ Invoke-Expression ($outerTry.CatchClauses[0].Body.Statements.Extent.Text -join "`n") }} catch {{
        if ($_.Exception.Message -ne $originalError.Exception.Message -and $_.Exception.Message -ne 'ScriptHalted') {{ throw }}
    }}
}}
if ($caught -ne ('{failure}' -ne 'none')) {{ throw "Unexpected service setup failure=$caught : $originalError" }}
if ('{existing}' -eq 'missing') {{
    $expectedCreated = if ('{failure}' -eq 'register') {{ 0 }} else {{ 1 }}
    if ($script:created -ne $expectedCreated) {{ throw 'Missing service was not registered exactly once' }}
    if ($script:exists -ne ('{failure}' -eq 'none')) {{ throw 'Wrong service state after setup/rollback' }}
    if ($expectedCreated -and $record.createdService -ne ('{failure}' -eq 'none')) {{ throw 'Wrong service ownership record' }}
    if ('{failure}' -eq 'none') {{
        if (-not $script:running -or $script:started -ne 1 -or -not $script:automatic) {{ throw 'New service did not start with automatic startup' }}
        # A successful rerun sees the service as pre-existing, even with createdService in its record.
        $service = Get-SshServiceRegistration
        $createdServiceThisRun = $false
        foreach ($statement in $prepare) {{ Invoke-Expression $statement.Extent.Text }}
        Invoke-Expression $activate.Extent.Text
        if ($script:created -ne 1 -or $script:started -ne 1 -or -not $record.createdService) {{ throw 'Rerun re-registered service or lost ownership history' }}
    }}
    if ('{failure}' -ne 'none' -and $script:deleted -ne $expectedCreated) {{ throw 'Created service was not rolled back' }}
}} elseif ($script:created -or $script:deleted -or $script:stopped -or -not $script:exists) {{ throw 'Pre-existing service changed ownership' }}
Write-Output 'service transaction verified'
""")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().endswith("service transaction verified")


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
    ("pending", "none"), ("pending", "restart"), ("pending", "password"),
    ("enabled", "none"), ("enabled", "restart"), ("enabled", "password"),
    ("enabled", "save"), ("disabled", "none"),
])
def test_account_activation_waits_for_ssh_and_preserves_existing_state(account_state: str, failure: str) -> None:
    result = setup_function_command(rf"""
$pending = '{account_state}' -in @('new', 'pending')
$script:accountEnabled = '{account_state}' -eq 'enabled'
$user = if ('{account_state}' -eq 'new') {{ $null }} else {{ [pscustomobject]@{{ Enabled = $script:accountEnabled; SID = 'S-1-5-21-1-2-3-1001' }} }}
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
$sshChanges = @(); $accountActivationAttempted = $false; $passwordResetAttempted = $false
$script:passwordCleared = $false
# The production tail runs after a new account has been created and recorded.
if (-not $user) {{ $user = [pscustomobject]@{{ Enabled = $false; SID = 'S-1-5-21-1-2-3-1001' }} }}
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
    if ($script:published -ne 2 -or -not $script:restarted -or -not $script:passwordCleared) {{ throw 'Account enabled before SSH and passwordless sign-in were ready' }}
    $script:accountEnabled = $true
}}
function Set-LocalUser {{
    [CmdletBinding()]param($SID, [Security.SecureString]$Password)
    if ($SID -ne $user.SID -or $Password.Length -ne 0) {{ throw 'Password reset must clear only the managed local account' }}
    if ($script:published -ne 2 -or -not $script:restarted) {{ throw 'Password cleared before SSH activation' }}
    if ('{failure}' -eq 'password') {{ throw 'Injected password reset failure' }}
    $script:passwordCleared = $true
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
if ($pending -and '{failure}' -eq 'save' -and ($script:disableCount -ne 1 -or $script:restoreCount -ne 2)) {{ throw 'Late failure did not disable before restoring both files' }}
if ('{account_state}' -eq 'enabled' -and '{failure}' -in @('password', 'save') -and $script:restoreCount -ne 0) {{ throw 'SSH restrictions rolled back after password reset on enabled account' }}
if ('{failure}' -eq 'none' -and -not $script:passwordCleared) {{ throw 'Successful setup left a password' }}
Write-Output 'state verified'
""")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().endswith("state verified")
