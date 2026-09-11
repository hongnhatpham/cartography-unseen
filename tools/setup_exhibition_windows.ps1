#requires -Version 5.1
[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidatePattern('^[a-z][a-z0-9_-]{0,19}$')][string]$UserName,
    [Parameter(Mandatory)][string]$DeploymentPath,
    [Parameter(Mandatory)][string[]]$PublicKeyFiles,
    [Parameter(Mandatory)][string[]]$ApprovedPeerAddresses,
    [Parameter(Mandatory)][datetime]$ExhibitionEndDate,
    [string]$TailscaleInterfaceAlias = 'Tailscale',
    [string]$MonitorCredentialPath,
    [string]$MonitorEndpoint,
    [switch]$Apply
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Invoke-CheckedNative {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Executable failed with exit code $LASTEXITCODE" }
}
function Set-PrivateAcl {
    param([string]$Path, [string]$ReaderSid, [switch]$Directory, [switch]$Modify)
    $acl = if ($Directory) { New-Object Security.AccessControl.DirectorySecurity } else { New-Object Security.AccessControl.FileSecurity }
    $acl.SetAccessRuleProtection($true, $false)
    $acl.SetOwner([Security.Principal.SecurityIdentifier]'S-1-5-32-544')
    $inherit = if ($Directory) { 'ContainerInherit,ObjectInherit' } else { 'None' }
    foreach ($sid in @('S-1-5-18', 'S-1-5-32-544')) {
        $rule = New-Object Security.AccessControl.FileSystemAccessRule([Security.Principal.SecurityIdentifier]$sid, 'FullControl', $inherit, 'None', 'Allow')
        $acl.AddAccessRule($rule)
    }
    if ($ReaderSid) {
        $rights = if ($Modify) { 'Modify' } else { 'ReadAndExecute' }
        $rule = New-Object Security.AccessControl.FileSystemAccessRule([Security.Principal.SecurityIdentifier]$ReaderSid, $rights, $inherit, 'None', 'Allow')
        $acl.AddAccessRule($rule)
    }
    Set-Acl -LiteralPath $Path -AclObject $acl
}
function Test-SshPort {
    param($Ports)
    foreach ($port in @($Ports)) {
        if ($port -eq 'Any' -or $port -eq '22') { return $true }
        if ($port -match '^(\d+)-(\d+)$' -and 22 -ge [int]$Matches[1] -and 22 -le [int]$Matches[2]) { return $true }
    }
    return $false
}
function Assert-TrustedState {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return }
    $item = Get-Item -LiteralPath $Path -Force
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Privileged state traverses a reparse point: $Path" }
    $acl = Get-Acl -LiteralPath $Path
    $trustedSids = @('S-1-5-18', 'S-1-5-32-544')
    $ownerSid = $acl.GetOwner([Security.Principal.SecurityIdentifier]).Value
    if ($ownerSid -notin $trustedSids) { throw "Privileged state must be owned by SYSTEM or Administrators: $Path" }
    $writeRights = [Security.AccessControl.FileSystemRights]::Write -bor [Security.AccessControl.FileSystemRights]::Delete -bor
        [Security.AccessControl.FileSystemRights]::DeleteSubdirectoriesAndFiles -bor [Security.AccessControl.FileSystemRights]::ChangePermissions -bor
        [Security.AccessControl.FileSystemRights]::TakeOwnership
    foreach ($ace in $acl.GetAccessRules($true, $true, [Security.Principal.SecurityIdentifier])) {
        if ($ace.AccessControlType -eq 'Allow' -and $ace.IdentityReference.Value -notin $trustedSids -and ($ace.FileSystemRights -band $writeRights)) {
            throw "Privileged state grants write/delete/ACL rights to another principal: $Path"
        }
    }
}
function Get-SshAllowRules {
    param([string]$SshdPath)
    @(Get-NetFirewallRule -PolicyStore ActiveStore -Enabled True -Direction Inbound -Action Allow | ForEach-Object {
        $rule = $_
        $port = $rule | Get-NetFirewallPortFilter
        $app = $rule | Get-NetFirewallApplicationFilter
        $svc = $rule | Get-NetFirewallServiceFilter
        if (($port.Protocol -in @('TCP', '6', 'Any', '256')) -and (Test-SshPort $port.LocalPort) -and
            ($app.Program -eq 'Any' -or $app.Program -eq $SshdPath -or $app.Program -like '*\sshd.exe') -and
            ($svc.Service -in @('Any', 'sshd'))) { $rule }
    })
}
function Assert-ProtectedParents {
    param([string]$Path)
    $trustedSids = @('S-1-5-18', 'S-1-5-32-544', [Security.Principal.WindowsIdentity]::GetCurrent().User.Value,
        'S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464') # TrustedInstaller
    $dangerous = [Security.AccessControl.FileSystemRights]::Delete -bor [Security.AccessControl.FileSystemRights]::DeleteSubdirectoriesAndFiles -bor
        [Security.AccessControl.FileSystemRights]::ChangePermissions -bor [Security.AccessControl.FileSystemRights]::TakeOwnership
    # The protected leaf need not exist yet; its existing ancestors still govern replacement.
    $parent = ([IO.DirectoryInfo]$Path).Parent
    while ($parent) {
        if ($parent.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Protected path ancestor is a reparse point: $($parent.FullName)" }
        $acl = Get-Acl -LiteralPath $parent.FullName
        if ($acl.GetOwner([Security.Principal.SecurityIdentifier]).Value -notin $trustedSids) {
            throw "Protected path ancestor is not administrator-controlled: $($parent.FullName)"
        }
        foreach ($ace in $acl.GetAccessRules($true, $true, [Security.Principal.SecurityIdentifier])) {
            if (($ace.PropagationFlags -band [Security.AccessControl.PropagationFlags]::InheritOnly) -eq 0 -and
                $ace.AccessControlType -eq 'Allow' -and $ace.IdentityReference.Value -notin $trustedSids -and ($ace.FileSystemRights -band $dangerous)) {
                throw "Protected path ancestor permits another principal to replace protected content: $($parent.FullName). Use an administrator-controlled parent."
            }
        }
        $parent = $parent.Parent
    }
}
function Get-MonitorPlan {
    param([string]$CredentialPath, [string]$Endpoint)
    if (-not $CredentialPath) {
        if ($Endpoint) { throw 'MonitorEndpoint requires MonitorCredentialPath.' }
        return [pscustomobject]@{ action = 'unchanged'; credentialSource = $null; endpointHost = $null }
    }
    $uri = $null
    if (-not [Uri]::TryCreate($Endpoint, [UriKind]::Absolute, [ref]$uri) -or $uri.Scheme -ne 'https' -or
        -not $uri.Host -or $uri.UserInfo -or $uri.Query -or $uri.Fragment -or $uri.AbsolutePath -ne '/api/v1/heartbeat') {
        throw 'MonitorEndpoint must be an HTTPS /api/v1/heartbeat URL without credentials, query or fragment.'
    }
    return [pscustomobject]@{ action = 'install-or-update'; credentialSource = $CredentialPath; endpointHost = $uri.Host }
}
function Read-MonitorCredential {
    param([string]$Path)
    try {
        $file = Get-Item -LiteralPath $Path -Force
        if ($file -isnot [IO.FileInfo] -or $file.Length -lt 1 -or $file.Length -gt 8192) { throw 'Invalid credential file.' }
        $cursor = $file
        while ($cursor) {
            if ($cursor.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Reparse credential path.' }
            $cursor = if ($cursor -is [IO.FileInfo]) { $cursor.Directory } else { $cursor.Parent }
        }
        $value = [IO.File]::ReadAllText($file.FullName) | ConvertFrom-Json
        if ($value -isnot [pscustomobject] -or $value.PSObject.Properties.Name -notcontains 'token' -or
            $value.PSObject.Properties.Name -notcontains 'machineId' -or $value.token -isnot [string] -or
            $value.token -notmatch '^[\x21-\x7e]{16,512}$' -or $value.machineId -isnot [string] -or
            $value.machineId -notmatch '^[A-Za-z0-9_-]{1,64}$') { throw 'Invalid credential structure.' }
        return [string]$value.token
    }
    catch { throw 'Monitor credential must be a regular, non-reparse JSON file containing valid machineId and token fields.' }
}
function New-ExhibitionMonitorTask {
    param([string]$Root, [string]$ConfigPath, [string]$StateDirectory)
    $arguments = '-I -B "{0}" --config "{1}" --state-directory "{2}" --snapshot-directory "{3}"' -f
        (Join-Path $Root 'tools\monitor_agent.py'), $ConfigPath, $StateDirectory, (Join-Path $Root 'cache\monitoring')
    $action = New-ScheduledTaskAction -Execute (Join-Path $Root 'runtime\python\python.exe') -Argument $arguments -WorkingDirectory $Root
    $principal = New-ScheduledTaskPrincipal -UserId 'S-1-5-18' -LogonType ServiceAccount -RunLevel Highest
    $settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) `
        -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1)
    New-ScheduledTask -Action $action -Principal $principal -Trigger (New-ScheduledTaskTrigger -AtStartup) -Settings $settings `
        -Description 'Optional telemetry collector; starts before console logon; protected credential; no remote command execution.'
}

$identity = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $identity.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run from elevated 64-bit Windows PowerShell, including for the read-only firewall plan.'
}
if (-not [Environment]::Is64BitProcess) { throw 'Use 64-bit Windows PowerShell.' }
$root = (Resolve-Path -LiteralPath $DeploymentPath).Path.TrimEnd('\')
if ($root -eq [IO.Path]::GetPathRoot($root).TrimEnd('\') -or $root -eq $env:USERPROFILE -or
    $root -eq $env:ProgramData -or $root -eq $env:SystemRoot -or $root -like "$env:SystemRoot\*") {
    throw 'DeploymentPath must be a dedicated project directory, not a drive, profile, Windows, or ProgramData root.'
}
if ($root.StartsWith('\\') -or $root -match '["\r\n]') { throw 'Use a local deployment path without quotes or newlines.' }
Assert-ProtectedParents $root
$cursor = Get-Item -LiteralPath $root
while ($null -ne $cursor) {
    if ($cursor.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Deployment path traverses a reparse point: $($cursor.FullName)" }
    $cursor = $cursor.Parent
}
if (@(Get-ChildItem -LiteralPath $root -Recurse -Force | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) {
    throw 'Deployment contains a reparse point. Use a plain dedicated copy before applying recursive ACLs.'
}
foreach ($required in @('tools\run_exhibition.ps1', 'tools\exhibition_status.ps1', 'app\main.py', 'config.json')) {
    if (-not (Test-Path -LiteralPath (Join-Path $root $required) -PathType Leaf)) { throw "Missing deployment file: $required" }
}
if ($ExhibitionEndDate -le (Get-Date)) { throw 'ExhibitionEndDate must be in the future; include the intended local expiry time.' }
$peers = @($ApprovedPeerAddresses | ForEach-Object {
    $ip = $null
    if (-not [Net.IPAddress]::TryParse($_, [ref]$ip) -or $_ -match '[/%,*\s]') { throw "Expected an exact peer IP address: $_" }
    $bytes = $ip.GetAddressBytes()
    $tail4 = $bytes.Length -eq 4 -and $bytes[0] -eq 100 -and $bytes[1] -ge 64 -and $bytes[1] -le 127
    $tail6 = $bytes.Length -eq 16 -and $ip.ToString().StartsWith('fd7a:115c:a1e0:', [StringComparison]::OrdinalIgnoreCase)
    if (-not ($tail4 -or $tail6)) { throw "Not a Tailscale address: $_" }
    $ip.ToString()
} | Sort-Object -Unique)
if (-not $peers.Count) { throw 'At least one approved peer is required.' }
$adapter = Get-NetAdapter -Name $TailscaleInterfaceAlias
if ($adapter.InterfaceDescription -notmatch 'Tailscale') { throw 'The selected adapter does not identify itself as Tailscale.' }
$keys = @($PublicKeyFiles | ForEach-Object {
    foreach ($line in (Get-Content -LiteralPath $_)) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        if ($line -notmatch '^(ssh-ed25519|ssh-rsa|ecdsa-sha2-nistp(?:256|384|521)) ([A-Za-z0-9+/]+={0,2})(?: [^\r\n]*)?$') {
            throw "Not a plain OpenSSH public key in $_. Private keys and authorized_keys options are rejected."
        }
        $null = [Convert]::FromBase64String($Matches[2])
        # Drop comments so deployment records do not copy personal key labels.
        "$($Matches[1]) $($Matches[2])"
    }
} | Sort-Object -Unique)
if (-not $keys.Count) { throw 'Supply at least one public key.' }
$stateParent = Join-Path $env:ProgramData 'CartographyExhibition'
$stateRoot = Join-Path $stateParent $UserName
# Validate before reading any record or following state-controlled paths. ProgramData itself
# permits ordinary users to create new directories; each existing managed ancestor must be trusted.
Assert-ProtectedParents $stateParent
Assert-TrustedState $stateParent
Assert-TrustedState $stateRoot
if (Test-Path -LiteralPath $stateRoot) {
    foreach ($item in Get-ChildItem -LiteralPath $stateRoot -Recurse -Force) { Assert-TrustedState $item.FullName }
}
$recordPath = Join-Path $stateRoot 'rollback.json'
$prior = if (Test-Path -LiteralPath $recordPath) { Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json } else { $null }
$taskName = "CartographyExhibition-$UserName"
$monitorTaskName = "CartographyExhibition-Monitor-$UserName"
$monitorPlan = Get-MonitorPlan -CredentialPath $MonitorCredentialPath -Endpoint $MonitorEndpoint
if ($MonitorCredentialPath) {
    foreach ($required in @('runtime\python\python.exe', 'tools\monitor_agent.py', 'app\monitoring.py')) {
        if (-not (Test-Path -LiteralPath (Join-Path $root $required) -PathType Leaf)) { throw "Missing monitoring deployment file: $required" }
    }
    $ownedMonitor = $prior -and $prior.PSObject.Properties.Name -contains 'monitorTaskName' -and $prior.monitorTaskName -eq $monitorTaskName
    if ((Get-ScheduledTask -TaskName $monitorTaskName -ErrorAction SilentlyContinue) -and -not $ownedMonitor) { throw 'Unmanaged monitoring task name collision.' }
}
$ruleName = "CartographyExhibition-SSH-$UserName"
$user = Get-LocalUser -Name $UserName -ErrorAction SilentlyContinue
if ($user -and (-not $prior -or $prior.userSid -ne $user.SID.Value)) { throw 'Refusing to adopt an existing unmanaged local account. Choose a new account name.' }
if ($prior -and $prior.deploymentPath -ne $root) { throw 'Existing deployment record uses a different path.' }
if ((Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) -and -not $prior) { throw 'Unmanaged scheduled task name collision.' }
if ($user -and @(Get-LocalGroupMember -SID 'S-1-5-32-544' | Where-Object SID -eq $user.SID).Count) { throw 'The exhibition account is an administrator; remove that membership before continuing.' }
$sshd = Join-Path $env:SystemRoot 'System32\OpenSSH\sshd.exe'
$capability = Get-WindowsCapability -Online -Name 'OpenSSH.Server~~~~0.0.1.0'
$existingRules = @(Get-SshAllowRules $sshd)
$unmanagedPolicy = @($existingRules | Where-Object PolicyStoreSourceType -ne 'Local')
if ($unmanagedPolicy.Count) { throw "SSH-capable inbound allow rules are controlled by policy: $($unmanagedPolicy.Name -join ', '). An administrator must narrow those policies first." }
if (@(Get-NetFirewallProfile -PolicyStore ActiveStore | Where-Object { -not $_.Enabled -or $_.DefaultInboundAction -eq 'Allow' }).Count) {
    throw 'All firewall profiles must be enabled with default inbound blocking before setup.'
}
$configPath = Join-Path $stateRoot 'sshd_config'
$keyPath = Join-Path $stateRoot 'authorized_keys'
$candidate = @"
# Cartography exhibition managed configuration. The default sshd_config is untouched.
Port 22
PubkeyAuthentication yes
PasswordAuthentication no
AuthenticationMethods publickey
PermitEmptyPasswords no
AllowUsers $UserName
AuthorizedKeysFile $($keyPath.Replace('\', '/'))
StrictModes yes
DisableForwarding yes
AllowTcpForwarding no
AllowAgentForwarding no
X11Forwarding no
PermitTunnel no
Subsystem sftp sftp-server.exe
"@
[pscustomobject]@{
    mode = $(if ($Apply) { 'Apply' } else { 'Read-only plan; rerun with -Apply after review' })
    account = "$env:COMPUTERNAME\$UserName"; standardUser = $true; expiresLocal = $ExhibitionEndDate.ToString('o')
    deployment = $root; deploymentAcl = 'SYSTEM/Administrators own all files with full control; exhibition user read+execute on code/tools/runtime/models'
    writableDirectories = @('cache', 'logs', 'journeys', 'screenshot'); activeConfiguration = 'cache\exhibition\config.json'
    privilegedState = $stateRoot; publicKeyCount = $keys.Count; sshConfiguration = $candidate
    installOpenSsh = ($capability.State -ne 'Installed'); task = $taskName
    interface = $adapter.Name; approvedPeers = $peers; disableInboundRules = @($existingRules | ForEach-Object Name)
    accountExists = [bool]$user; rollbackRecord = $recordPath
    monitoring = $monitorPlan
} | ConvertTo-Json -Depth 5 | Write-Output
if (-not $Apply) { return }
$monitorToken = if ($MonitorCredentialPath) { Read-MonitorCredential $MonitorCredentialPath } else { $null }

# The record precedes changes and is refreshed as each phase completes. It never contains a password.
if (-not (Test-Path -LiteralPath $stateParent)) {
    New-Item -ItemType Directory -Path $stateParent -ErrorAction Stop | Out-Null
    Set-PrivateAcl -Path $stateParent -Directory
}
Assert-TrustedState $stateParent
if (-not (Test-Path -LiteralPath $stateRoot)) {
    New-Item -ItemType Directory -Path $stateRoot -ErrorAction Stop | Out-Null
    Set-PrivateAcl -Path $stateRoot -Directory
}
Assert-TrustedState $stateRoot
Set-PrivateAcl -Path $stateRoot -Directory
$service = Get-CimInstance Win32_Service -Filter "Name='sshd'"
$record = if ($prior) { $prior } else {
    [pscustomobject]@{
        version = 1; deploymentPath = $root; userName = $UserName; userSid = $null; createdUser = $false
        taskName = $taskName; firewallRuleName = $ruleName; disabledFirewallRules = @()
        originalCapabilityState = [string]$capability.State; originalService = $service | Select-Object PathName, StartMode, State
        originalServiceRecovery = $null; originalServiceFailureFlag = $null; originalAclFile = (Join-Path $stateRoot 'deployment-acl.txt')
        originalTask = $null; completedPhase = 'begin'; updatedAt = $null
    }
}
function Save-Record {
    $record.updatedAt = (Get-Date).ToUniversalTime().ToString('o')
    $record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $recordPath -Encoding UTF8
}
Save-Record
try {
    if (-not $prior) {
        Invoke-CheckedNative icacls.exe @($root, '/save', $record.originalAclFile, '/t', '/q')
        if ($service) {
            $record.originalServiceRecovery = @(& sc.exe qfailure sshd)
            $record.originalServiceFailureFlag = @(& sc.exe qfailureflag sshd)
            Save-Record
        }
    }
    if (-not $user) {
        $password = Read-Host "Local password for $UserName (never saved by this script)" -AsSecureString
        try { $user = New-LocalUser -Name $UserName -Password $password -PasswordNeverExpires -AccountExpires $ExhibitionEndDate -Description 'Cartography exhibition standard account' }
        finally { if ($password) { $password.Dispose() }; Remove-Variable password -ErrorAction SilentlyContinue }
        $record.userSid = $user.SID.Value; $record.createdUser = $true; $record.completedPhase = 'account'; Save-Record
    }
    Set-LocalUser -Name $UserName -AccountExpires $ExhibitionEndDate -PasswordNeverExpires $true
    $usersGroup = Get-LocalGroup -SID 'S-1-5-32-545'
    if (-not @(Get-LocalGroupMember -Group $usersGroup | Where-Object SID -eq $user.SID).Count) {
        Add-LocalGroupMember -Group $usersGroup -Member $user
    }
    Set-PrivateAcl -Path $root -ReaderSid $user.SID.Value -Directory
    # Ownership also matters: a previous file owner could otherwise grant themselves write access.
    Invoke-CheckedNative icacls.exe @($root, '/setowner', '*S-1-5-32-544', '/t', '/q')
    foreach ($child in Get-ChildItem -LiteralPath $root -Force) {
        Invoke-CheckedNative icacls.exe @($child.FullName, '/reset', '/t', '/q')
    }
    foreach ($directory in @('cache', 'logs', 'journeys', 'screenshot')) {
        $mutablePath = Join-Path $root $directory
        if (-not (Test-Path -LiteralPath $mutablePath)) { New-Item -ItemType Directory -Path $mutablePath | Out-Null }
    }
    $mutableConfigDirectory = Join-Path $root 'cache\exhibition'
    if (-not (Test-Path -LiteralPath $mutableConfigDirectory)) { New-Item -ItemType Directory -Path $mutableConfigDirectory | Out-Null }
    $mutableConfig = Join-Path $mutableConfigDirectory 'config.json'
    if (-not (Test-Path -LiteralPath $mutableConfig)) {
        Copy-Item -LiteralPath (Join-Path $root 'config.json') -Destination $mutableConfig
    }
    foreach ($directory in @('cache', 'logs', 'journeys', 'screenshot')) {
        Set-PrivateAcl -Path (Join-Path $root $directory) -ReaderSid $user.SID.Value -Directory -Modify
    }
    $record.completedPhase = 'deployment-acl'; Save-Record
    if ($capability.State -ne 'Installed') {
        $installed = Add-WindowsCapability -Online -Name 'OpenSSH.Server~~~~0.0.1.0'
        if ($installed.RestartNeeded) { throw 'OpenSSH installation requires a reboot. Reboot and rerun this command.' }
    }
    Stop-Service sshd -ErrorAction SilentlyContinue
    # Generate host keys locally. These identify this server; they are not fleet client credentials.
    $hostKeyDirectory = Join-Path $env:ProgramData 'ssh'
    if (-not (Test-Path -LiteralPath $hostKeyDirectory)) {
        New-Item -ItemType Directory -Path $hostKeyDirectory -ErrorAction Stop | Out-Null
        Set-PrivateAcl -Path $hostKeyDirectory -Directory
    }
    Assert-TrustedState $hostKeyDirectory
    Invoke-CheckedNative (Join-Path (Split-Path $sshd) 'ssh-keygen.exe') @('-A')
    Set-PrivateAcl -Path $stateRoot -Directory -ReaderSid $user.SID.Value
    $keyCandidate = Join-Path $stateRoot 'authorized_keys.candidate'
    $keys | Set-Content -LiteralPath $keyCandidate -Encoding ascii
    Set-PrivateAcl -Path $keyCandidate -ReaderSid $user.SID.Value
    Invoke-CheckedNative (Join-Path (Split-Path $sshd) 'ssh-keygen.exe') @('-l', '-f', $keyCandidate)
    if (Test-Path -LiteralPath $keyPath) { Copy-Item -LiteralPath $keyPath -Destination (Join-Path $stateRoot 'authorized_keys.previous') -Force }
    Move-Item -LiteralPath $keyCandidate -Destination $keyPath -Force
    $candidatePath = Join-Path $stateRoot 'sshd_config.candidate'
    $candidate | Set-Content -LiteralPath $candidatePath -Encoding ascii
    Set-PrivateAcl -Path $candidatePath
    Invoke-CheckedNative $sshd @('-t', '-f', $candidatePath)
    if (Test-Path -LiteralPath $configPath) { Copy-Item -LiteralPath $configPath -Destination (Join-Path $stateRoot 'sshd_config.previous') -Force }
    Move-Item -LiteralPath $candidatePath -Destination $configPath -Force
    # Catch the broad default rule that capability installation may just have created.
    foreach ($rule in @(Get-SshAllowRules $sshd)) {
        if ($rule.Name -eq $ruleName) { continue }
        if ($rule.PolicyStoreSourceType -ne 'Local') { throw "Cannot narrow policy-owned rule $($rule.Name)" }
        if ($record.disabledFirewallRules -notcontains $rule.Name) {
            $record.disabledFirewallRules = @($record.disabledFirewallRules) + $rule.Name; Save-Record
        }
        Disable-NetFirewallRule -Name $rule.Name | Out-Null
    }
    if (Get-NetFirewallRule -Name $ruleName -ErrorAction SilentlyContinue) { Remove-NetFirewallRule -Name $ruleName }
    New-NetFirewallRule -Name $ruleName -DisplayName "Exhibition SSH ($UserName): approved Tailscale peers" `
        -Direction Inbound -Action Allow -Enabled True -Profile Any -Protocol TCP -LocalPort 22 `
        -RemoteAddress $peers -InterfaceAlias $adapter.Name -Program $sshd -Service sshd | Out-Null
    $serviceChange = Invoke-CimMethod -InputObject (Get-CimInstance Win32_Service -Filter "Name='sshd'") -MethodName Change `
        -Arguments @{ PathName = ('"{0}" -f "{1}"' -f $sshd, $configPath); StartMode = 'Automatic' }
    if ($serviceChange.ReturnValue -ne 0) { throw "Could not configure sshd service: Win32 error $($serviceChange.ReturnValue)" }
    Invoke-CheckedNative sc.exe @('failure', 'sshd', 'reset=', '86400', 'actions=', 'restart/5000/restart/15000/restart/60000')
    Invoke-CheckedNative sc.exe @('failureflag', 'sshd', '1')
    $record.completedPhase = 'ssh'; Save-Record
    $action = New-ScheduledTaskAction -Execute "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" `
        -Argument ('-NoLogo -NoProfile -ExecutionPolicy Bypass -File "{0}" -DeploymentPath "{1}"' -f (Join-Path $root 'tools\run_exhibition.ps1'), $root) -WorkingDirectory $root
    $principal = New-ScheduledTaskPrincipal -UserId $user.SID.Value -LogonType Interactive -RunLevel Limited
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:COMPUTERNAME\$UserName"
    $settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) `
        -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
    Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal -Trigger $trigger -Settings $settings `
        -Description 'Interactive exhibition supervisor; console logon required; no stored password.' -Force | Out-Null
    # Give this account query/run/stop rights on its own task without granting task-edit rights.
    $scheduler = New-Object -ComObject 'Schedule.Service'
    $scheduler.Connect()
    $registered = $scheduler.GetFolder('\').GetTask($taskName)
    $registered.SetSecurityDescriptor("D:P(A;;FA;;;SY)(A;;FA;;;BA)(A;;GRGX;;;$($user.SID.Value))", 0)
    if ($MonitorCredentialPath) {
        # Never put the credential or privileged SQLite outbox in the user's writable cache.
        $monitorDirectory = Join-Path $stateRoot 'monitor'
        if (-not (Test-Path -LiteralPath $monitorDirectory)) { New-Item -ItemType Directory -Path $monitorDirectory | Out-Null }
        Set-PrivateAcl -Path $monitorDirectory -Directory
        $monitorConfig = Join-Path $monitorDirectory 'config.json'
        $monitorCandidate = Join-Path $monitorDirectory 'config.pending.json'
        try {
            [IO.File]::WriteAllText($monitorCandidate, (@{ endpoint = $MonitorEndpoint; token = $monitorToken } | ConvertTo-Json -Compress), (New-Object Text.UTF8Encoding($false)))
            Set-PrivateAcl -Path $monitorCandidate
            Stop-ScheduledTask -TaskName $monitorTaskName -ErrorAction SilentlyContinue
            if (Test-Path -LiteralPath $monitorConfig) { [IO.File]::Replace($monitorCandidate, $monitorConfig, [NullString]::Value) }
            else { [IO.File]::Move($monitorCandidate, $monitorConfig) }
            Set-PrivateAcl -Path $monitorConfig
        }
        finally { $monitorToken = $null }
        $record | Add-Member -NotePropertyName monitorTaskName -NotePropertyValue $monitorTaskName -Force
        $record | Add-Member -NotePropertyName monitorConfigPath -NotePropertyValue $monitorConfig -Force
        Save-Record
        $monitorTask = New-ExhibitionMonitorTask -Root $root -ConfigPath $monitorConfig -StateDirectory $monitorDirectory
        Register-ScheduledTask -TaskName $monitorTaskName -InputObject $monitorTask -Force | Out-Null
        $scheduler.GetFolder('\').GetTask($monitorTaskName).SetSecurityDescriptor('D:P(A;;FA;;;SY)(A;;FA;;;BA)', 0)
    }
    Start-Service sshd
    if ($MonitorCredentialPath) { Start-ScheduledTask -TaskName $monitorTaskName }
    $record.completedPhase = 'complete'; Save-Record
    Write-Output "Configured. Log on locally as $UserName to start the interactive task. Rollback record: $recordPath"
}
catch {
    # Fail closed if provisioning did not complete. Local console access remains available.
    Stop-Service sshd -ErrorAction SilentlyContinue
    Save-Record
    throw
}
