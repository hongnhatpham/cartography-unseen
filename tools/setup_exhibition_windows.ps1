#requires -Version 5.1
[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidatePattern('^[a-z][a-z0-9_-]{0,19}$')][string]$UserName,
    [Parameter(Mandatory)][string]$DeploymentPath,
    [Parameter(Mandatory)][string[]]$PublicKeyFiles,
    [string[]]$ApprovedPeerAddresses = @('100.87.222.71', 'fd7a:115c:a1e0::b434:de48', '100.96.33.72', 'fd7a:115c:a1e0::9f34:2149'),
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
function Grant-ProjectAccess {
    param([string]$Path, [string]$UserSid)
    # Add access without resetting existing permissions or changing ownership.
    Invoke-CheckedNative icacls.exe @($Path, '/grant', "*${UserSid}:(OI)(CI)M", '/t', '/q')
}
function Test-LocalAdministrator {
    param([string]$Account)
    $administrators = Get-LocalGroup -SID 'S-1-5-32-544'
    $group = New-Object DirectoryServices.DirectoryEntry -ArgumentList "WinNT://$env:COMPUTERNAME/$($administrators.Name),group"
    # IsMember checks this local account without resolving unrelated domain members.
    try {
        $member = New-Object DirectoryServices.DirectoryEntry -ArgumentList "WinNT://$env:COMPUTERNAME/$Account,user"
        # WinNT may canonicalize the path to domain/computer/user on domain-joined machines.
        try { return [bool]$group.Invoke('IsMember', @($member.InvokeGet('ADsPath'))) }
        finally { $member.Dispose() }
    }
    finally { $group.Dispose() }
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
function Add-ExhibitionSshConfiguration {
    param([string]$Existing, [string]$Account, [string]$KeyPath)
    if ($Existing -match '(?im)^\s*Include\s') {
        throw 'SSH configuration uses Include directives. Integrate the exhibition Match User block manually so existing policies keep their order.'
    }
    $begin = "# BEGIN Cartography exhibition $Account"
    $end = "# END Cartography exhibition $Account"
    $pattern = '(?ms)^' + [regex]::Escape($begin) + '\r?\n.*?^' + [regex]::Escape($end) + '\r?\n?'
    $existingWithoutBlock = [regex]::Replace($Existing, $pattern, '')
    $block = @"
$begin
Match User $Account
    AllowUsers $Account
    PubkeyAuthentication yes
    PasswordAuthentication no
    KbdInteractiveAuthentication no
    AuthenticationMethods publickey
    AuthorizedKeysFile "$($KeyPath.Replace('\', '/'))"
    PubkeyAcceptedKeyTypes ssh-ed25519,ecdsa-sha2-nistp256,ecdsa-sha2-nistp384,ecdsa-sha2-nistp521,rsa-sha2-256,rsa-sha2-512
    DisableForwarding yes
    AllowTcpForwarding no
    AllowAgentForwarding no
    X11Forwarding no
    PermitTunnel no
$end
"@
    # Put this user's settings before existing Match blocks; leave every global directive intact.
    $firstMatch = [regex]::Match($existingWithoutBlock, '(?im)^\s*Match\s')
    if ($firstMatch.Success) {
        return $existingWithoutBlock.Insert($firstMatch.Index, "$block`r`n")
    }
    return $existingWithoutBlock.TrimEnd() + "`r`n$block`r`n"
}
function Get-SshServicePaths {
    param([string]$CommandLine)
    $defaultExecutable = Join-Path $env:SystemRoot 'System32\OpenSSH\sshd.exe'
    $defaultConfig = Join-Path $env:ProgramData 'ssh\sshd_config'
    if (-not $CommandLine) { return @{ executable = $defaultExecutable; config = $defaultConfig } }
    $expanded = [Environment]::ExpandEnvironmentVariables($CommandLine)
    if ($expanded -notmatch '^\s*(?:"(?<exe>[^"]+sshd\.exe)"|(?<exe>\S+sshd\.exe))(?:\s+-f\s+(?:"(?<config>[^"]+)"|(?<config>\S+)))?\s*$') {
        throw 'Existing sshd service has custom command-line options. Review its configuration manually; setup will not redirect it.'
    }
    return @{ executable = $Matches.exe; config = $(if ($Matches.ContainsKey('config')) { $Matches.config } else { $defaultConfig }) }
}
function Get-SshServiceRegistration {
    $service = Get-CimInstance Win32_Service -Filter "Name='sshd'"
    $registryExists = Test-Path -LiteralPath 'HKLM:\SYSTEM\CurrentControlSet\Services\sshd'
    if ([bool]$service -ne $registryExists) {
        throw 'sshd registry and service manager state disagree. Reboot or repair the service registration before rerunning setup.'
    }
    return $service
}
function Resolve-TrustedSshd {
    param([string]$Executable)
    $canonical = Join-Path $env:SystemRoot 'System32\OpenSSH\sshd.exe'
    $resolved = (Resolve-Path -LiteralPath $Executable).Path
    if ($resolved -ne $canonical) { throw 'Missing sshd service can only be registered from the inbox System32\OpenSSH\sshd.exe.' }
    $item = Get-Item -LiteralPath $resolved -Force
    if ($item -isnot [IO.FileInfo]) { throw 'The sshd service executable must be a file.' }
    $cursor = $item
    while ($cursor) {
        if ($cursor.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'The sshd service executable path must not traverse a reparse point.' }
        $cursor = if ($cursor -is [IO.FileInfo]) { $cursor.Directory } else { $cursor.Parent }
    }
    $signature = Get-AuthenticodeSignature -LiteralPath $resolved
    if ($signature.Status -ne 'Valid' -or -not $signature.SignerCertificate -or
        $signature.SignerCertificate.Subject -notmatch '(^|,\s*)O=Microsoft Corporation(,|$)') {
        throw 'The sshd service executable must have a valid Microsoft signature.'
    }
    return $resolved
}
function Restore-SshFile {
    param($Change)
    if ($Change.existed) {
        $restorePath = $Change.path + '.' + [Guid]::NewGuid().ToString('N') + '.restore'
        try {
            Copy-Item -LiteralPath $Change.backup -Destination $restorePath
            if (Test-Path -LiteralPath $Change.path) { [IO.File]::Replace($restorePath, $Change.path, [NullString]::Value) }
            else { [IO.File]::Move($restorePath, $Change.path) }
        }
        finally { if (Test-Path -LiteralPath $restorePath) { Remove-Item -LiteralPath $restorePath } }
    }
    elseif (Test-Path -LiteralPath $Change.path) { Remove-Item -LiteralPath $Change.path }
}
function Publish-SshFile {
    param([string]$CandidatePath, [string]$Path, [string]$BackupPath)
    $change = [pscustomobject]@{ path = $Path; backup = $BackupPath; existed = (Test-Path -LiteralPath $Path) }
    # Finish the backup before touching the live file. Candidates share the target volume.
    if ($change.existed) { Copy-Item -LiteralPath $Path -Destination $BackupPath -Force }
    try {
        if ($change.existed) { [IO.File]::Replace($CandidatePath, $Path, [NullString]::Value) }
        else { [IO.File]::Move($CandidatePath, $Path) }
    }
    catch { Restore-SshFile $change; throw }
    return $change
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
    param([string]$Root, [string]$ConfigPath, [string]$StateDirectory, [string]$UserId)
    $arguments = '-I -B "{0}" --config "{1}" --state-directory "{2}" --snapshot-directory "{3}"' -f
        (Join-Path $Root 'tools\monitor_agent.py'), $ConfigPath, $StateDirectory, (Join-Path $Root 'cache\monitoring')
    $action = New-ScheduledTaskAction -Execute (Join-Path $Root 'runtime\python\python.exe') -Argument $arguments -WorkingDirectory $Root
    $principal = New-ScheduledTaskPrincipal -UserId $UserId -LogonType Interactive -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) `
        -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1)
    New-ScheduledTask -Action $action -Principal $principal -Trigger (New-ScheduledTaskTrigger -AtLogOn -User $UserId) -Settings $settings `
        -Description 'Optional telemetry collector; starts at exhibition user logon; no stored password or remote command execution.'
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
$fleetAddresses = @('100.87.222.71', 'fd7a:115c:a1e0::b434:de48', '100.96.33.72', 'fd7a:115c:a1e0::9f34:2149')
$peers = @($ApprovedPeerAddresses | ForEach-Object {
    $ip = $null
    if (-not [Net.IPAddress]::TryParse($_, [ref]$ip) -or $_ -match '[/%,*\s]') { throw "Expected an exact peer IP address: $_" }
    $bytes = $ip.GetAddressBytes()
    $tail4 = $bytes.Length -eq 4 -and $bytes[0] -eq 100 -and $bytes[1] -ge 64 -and $bytes[1] -le 127
    $tail6 = $bytes.Length -eq 16 -and $ip.ToString().StartsWith('fd7a:115c:a1e0:', [StringComparison]::OrdinalIgnoreCase)
    if (-not ($tail4 -or $tail6)) { throw "Not a Tailscale address: $_" }
    if ($ip.ToString() -notin $fleetAddresses) { throw "Only aria and asus-rog are approved SSH source machines: $_" }
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
$existingMonitor = Get-ScheduledTask -TaskName $monitorTaskName -ErrorAction SilentlyContinue
if ($existingMonitor -and $existingMonitor.Principal.LogonType -ne 'Interactive' -and -not $MonitorCredentialPath) {
    throw 'An earlier monitor task runs without an interactive user. Supply MonitorCredentialPath and MonitorEndpoint to migrate it before making the project writable.'
}
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
$activationPending = -not $user -or ($prior -and $prior.PSObject.Properties.Name -contains 'accountActivationPending' -and $prior.accountActivationPending)
$activateAccount = $activationPending -and (-not $user -or -not $user.Enabled)
if ($prior -and $prior.deploymentPath -ne $root) { throw 'Existing deployment record uses a different path.' }
if ((Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) -and -not $prior) { throw 'Unmanaged scheduled task name collision.' }
if ($user -and (Test-LocalAdministrator -Account $UserName)) { throw 'The exhibition account is an administrator; remove that membership before continuing.' }
$service = Get-SshServiceRegistration
$sshPaths = Get-SshServicePaths -CommandLine $(if ($service) { $service.PathName } else { '' })
$sshd = $sshPaths.executable
$configPath = $sshPaths.config
$capability = Get-WindowsCapability -Online -Name 'OpenSSH.Server~~~~0.0.1.0'
if ((Get-NetFirewallRule -Name $ruleName -ErrorAction SilentlyContinue) -and (-not $prior -or $prior.firewallRuleName -ne $ruleName)) {
    throw 'Unmanaged firewall rule name collision.'
}
$keyPath = Join-Path $stateRoot 'authorized_keys'
$existingConfig = if (Test-Path -LiteralPath $configPath) { Get-Content -LiteralPath $configPath -Raw } else { "Subsystem sftp sftp-server.exe`r`n" }
$candidate = Add-ExhibitionSshConfiguration -Existing $existingConfig -Account $UserName -KeyPath $keyPath
$ports = @([regex]::Matches($existingConfig, '(?im)^\s*Port\s+(\d+)\s*(?:#.*)?$') | ForEach-Object { [int]$_.Groups[1].Value })
if (-not $ports.Count) { $ports = @(22) }
if ($ports.Count -ne 1) { throw 'Multiple SSH listener ports require a manual firewall plan.' }
$sshPort = $ports[0]
[pscustomobject]@{
    mode = $(if ($Apply) { 'Apply' } else { 'Read-only plan; rerun with -Apply after review' })
    account = "$env:COMPUTERNAME\$UserName"; standardUser = $true; accountNeverExpires = $true
    enableAccountAfterSsh = $activateAccount
    localSignIn = 'No password; clear any existing password after SSH activation'
    deployment = $root; deploymentAcl = 'Add exhibition user Modify access throughout the project; preserve existing access and ownership'
    writableDirectories = @($root); activeConfiguration = 'cache\exhibition\config.json'
    privilegedState = $stateRoot; publicKeyCount = $keys.Count; sshConfiguration = $candidate
    installOpenSsh = ($capability.State -ne 'Installed'); task = $taskName
    registerSshService = (-not [bool]$service)
    sshOperationalLogAvailable = [bool](Get-WinEvent -ListLog 'OpenSSH/Operational' -ErrorAction SilentlyContinue)
    interface = $adapter.Name; approvedPeers = $peers; sshPort = $sshPort; sshConfigPath = $configPath
    existingFirewallRules = 'unchanged'; existingServiceConfiguration = 'unchanged; restart running service to load the added user block'
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
$record = if ($prior) { $prior } else {
    [pscustomobject]@{
        version = 1; deploymentPath = $root; userName = $UserName; userSid = $null; createdUser = $false
        taskName = $taskName; firewallRuleName = $ruleName; disabledFirewallRules = @()
        originalCapabilityState = [string]$capability.State; originalService = $service | Select-Object PathName, StartMode, State
        createdService = $false
        originalServiceRecovery = $null; originalServiceFailureFlag = $null; originalAclFile = (Join-Path $stateRoot 'deployment-acl.txt')
        originalTask = $null; completedPhase = 'begin'; updatedAt = $null
    }
}
if ($record.PSObject.Properties.Name -notcontains 'createdService') {
    $record | Add-Member -NotePropertyName createdService -NotePropertyValue $false
}
function Save-Record {
    $record.updatedAt = (Get-Date).ToUniversalTime().ToString('o')
    $record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $recordPath -Encoding UTF8
}
Save-Record
$sshChanges = @()
$accountActivationAttempted = $false
$passwordResetAttempted = $false
$createdServiceThisRun = $false
$candidatePath = $null
$keyCandidate = $null
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
        $user = New-LocalUser -Name $UserName -NoPassword -Disabled -AccountNeverExpires -Description 'Cartography exhibition standard account'
        $record.userSid = $user.SID.Value; $record.createdUser = $true; $record.completedPhase = 'account'
        $record | Add-Member -NotePropertyName accountActivationPending -NotePropertyValue $true -Force
        Save-Record
    }
    Set-LocalUser -Name $UserName -AccountNeverExpires -PasswordNeverExpires $true
    # Add only the known local SID. Enumerating Users can fail on unrelated stale domain SIDs.
    try { Add-LocalGroupMember -SID 'S-1-5-32-545' -Member $user.SID.Value -ErrorAction Stop }
    catch {
        # Windows PowerShell can wrap the exception for -ErrorAction Stop; its stable error ID survives.
        if ($_.Exception -isnot [Microsoft.PowerShell.Commands.MemberExistsException] -and
            $_.FullyQualifiedErrorId -ne 'MemberExists,Microsoft.PowerShell.Commands.AddLocalGroupMemberCommand') { throw }
    }
    if ($MonitorCredentialPath -and $existingMonitor) {
        # Remove an earlier SYSTEM task before its executable becomes user-writable.
        Stop-ScheduledTask -TaskName $monitorTaskName
        Unregister-ScheduledTask -TaskName $monitorTaskName -Confirm:$false
    }
    Grant-ProjectAccess -Path $root -UserSid $user.SID.Value
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
    $record.completedPhase = 'deployment-acl'; Save-Record
    if ($capability.State -ne 'Installed') {
        $installed = Add-WindowsCapability -Online -Name 'OpenSSH.Server~~~~0.0.1.0'
        if ($installed.RestartNeeded) { throw 'OpenSSH installation requires a reboot. Reboot and rerun this command.' }
    }
    # Generate host keys locally. These identify this server; they are not fleet client credentials.
    $hostKeyDirectory = Join-Path $env:ProgramData 'ssh'
    if (-not (Test-Path -LiteralPath $hostKeyDirectory)) {
        New-Item -ItemType Directory -Path $hostKeyDirectory -ErrorAction Stop | Out-Null
        Set-PrivateAcl -Path $hostKeyDirectory -Directory
    }
    Assert-TrustedState $hostKeyDirectory
    Invoke-CheckedNative (Join-Path (Split-Path $sshd) 'ssh-keygen.exe') @('-A')
    Set-PrivateAcl -Path $stateRoot -Directory -ReaderSid $user.SID.Value
    $keyCandidate = $keyPath + '.' + [Guid]::NewGuid().ToString('N') + '.candidate'
    $keys | ForEach-Object { 'from="{0}" {1}' -f ($peers -join ','), $_ } | Set-Content -LiteralPath $keyCandidate -Encoding ascii
    Set-PrivateAcl -Path $keyCandidate -ReaderSid $user.SID.Value
    Invoke-CheckedNative (Join-Path (Split-Path $sshd) 'ssh-keygen.exe') @('-l', '-f', $keyCandidate)
    $candidatePath = $configPath + '.' + [Guid]::NewGuid().ToString('N') + '.candidate'
    [IO.File]::WriteAllText($candidatePath, $candidate, (New-Object Text.UTF8Encoding($false)))
    Set-PrivateAcl -Path $candidatePath
    Invoke-CheckedNative $sshd @('-t', '-f', $candidatePath)
    $configBackup = Join-Path $stateRoot 'sshd_config.previous'
    $record | Add-Member -NotePropertyName sshConfigPath -NotePropertyValue $configPath -Force
    $record | Add-Member -NotePropertyName sshConfigBackup -NotePropertyValue $configBackup -Force
    Save-Record
    if (Get-NetFirewallRule -Name $ruleName -ErrorAction SilentlyContinue) { Remove-NetFirewallRule -Name $ruleName }
    New-NetFirewallRule -Name $ruleName -DisplayName "Exhibition SSH ($UserName): approved Tailscale peers" `
        -Direction Inbound -Action Allow -Enabled True -Profile Any -Protocol TCP -LocalPort $sshPort `
        -RemoteAddress $peers -InterfaceAlias $adapter.Name -Program $sshd -Service sshd | Out-Null
    if (-not $service) {
        $registeredService = Get-SshServiceRegistration
        if ($registeredService -and $capability.State -eq 'Installed') {
            throw 'sshd was registered after the setup plan was read. Rerun setup to preserve that service.'
        }
        $trustedSshd = Resolve-TrustedSshd -Executable $sshd
        if ($registeredService) {
            # A capability installed by this run may already have registered its service.
            $installedPaths = Get-SshServicePaths -CommandLine $registeredService.PathName
            if ($installedPaths.executable -ne $trustedSshd -or $installedPaths.config -ne $configPath -or
                $registeredService.StartName -notin @('LocalSystem', 'NT AUTHORITY\SYSTEM')) {
                throw 'The newly installed sshd service has unexpected paths or account. Review it before rerunning setup.'
            }
        }
        else {
            # Omitting Credential and DependsOn uses LocalSystem with no dependencies.
            New-Service -Name sshd -BinaryPathName ('"{0}"' -f $trustedSshd) -DisplayName 'OpenSSH SSH Server' `
                -Description 'SSH protocol based service to provide secure encrypted communications between two untrusted hosts over an insecure network.' `
                -StartupType Manual -ErrorAction Stop | Out-Null
        }
        $createdServiceThisRun = $true
        $record.createdService = $true
        Save-Record
    }
    $record.completedPhase = 'ssh-staged'; Save-Record
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
        $monitorDirectory = Join-Path $root 'cache\exhibition-monitor'
        if (-not (Test-Path -LiteralPath $monitorDirectory)) { New-Item -ItemType Directory -Path $monitorDirectory | Out-Null }
        Set-PrivateAcl -Path $monitorDirectory -Directory -ReaderSid $user.SID.Value -Modify
        $monitorConfig = Join-Path $monitorDirectory 'config.json'
        $monitorCandidate = Join-Path $monitorDirectory 'config.pending.json'
        try {
            [IO.File]::WriteAllText($monitorCandidate, (@{ endpoint = $MonitorEndpoint; token = $monitorToken } | ConvertTo-Json -Compress), (New-Object Text.UTF8Encoding($false)))
            Set-PrivateAcl -Path $monitorCandidate -ReaderSid $user.SID.Value -Modify
            Stop-ScheduledTask -TaskName $monitorTaskName -ErrorAction SilentlyContinue
            if (Test-Path -LiteralPath $monitorConfig) { [IO.File]::Replace($monitorCandidate, $monitorConfig, [NullString]::Value) }
            else { [IO.File]::Move($monitorCandidate, $monitorConfig) }
            Set-PrivateAcl -Path $monitorConfig -ReaderSid $user.SID.Value -Modify
        }
        finally { $monitorToken = $null }
        $record | Add-Member -NotePropertyName monitorTaskName -NotePropertyValue $monitorTaskName -Force
        $record | Add-Member -NotePropertyName monitorConfigPath -NotePropertyValue $monitorConfig -Force
        Save-Record
        $monitorTask = New-ExhibitionMonitorTask -Root $root -ConfigPath $monitorConfig -StateDirectory $monitorDirectory -UserId $user.SID.Value
        Register-ScheduledTask -TaskName $monitorTaskName -InputObject $monitorTask -Force | Out-Null
        $scheduler.GetFolder('\').GetTask($monitorTaskName).SetSecurityDescriptor("D:P(A;;FA;;;SY)(A;;FA;;;BA)(A;;GRGX;;;$($user.SID.Value))", 0)
    }
    # Activate only after validation and task setup succeed. Retain both originals for recovery.
    $sshChanges += Publish-SshFile -CandidatePath $keyCandidate -Path $keyPath -BackupPath (Join-Path $stateRoot 'authorized_keys.previous')
    $sshChanges += Publish-SshFile -CandidatePath $candidatePath -Path $configPath -BackupPath $configBackup
    if ($service -and $service.State -eq 'Running') { Restart-Service sshd }
    elseif (-not $service) {
        Start-Service sshd
        if ((Get-Service sshd).Status -ne 'Running') { throw 'The newly registered sshd service did not reach Running state.' }
        Set-Service sshd -StartupType Automatic
    }
    # Windows PowerShell 5.1 accepts an empty SecureString; no plaintext password or prompt is needed.
    $blankPassword = New-Object Security.SecureString
    try {
        $passwordResetAttempted = $true
        Set-LocalUser -SID $user.SID -Password $blankPassword -ErrorAction Stop
    }
    finally { $blankPassword.Dispose() }
    if ($activateAccount) {
        $accountActivationAttempted = $true
        Enable-LocalUser -Name $UserName
    }
    if ($activationPending) {
        $record | Add-Member -NotePropertyName accountActivationPending -NotePropertyValue $false -Force
    }
    # Both tasks start at the user's next console logon. No interactive token is required during setup.
    $record.completedPhase = 'complete'; Save-Record
    Write-Output "Configured. Log on locally as $UserName to start the interactive task. Rollback record: $recordPath"
}
catch {
    # A persisted ownership record alone never authorizes removing a service on a later run.
    if ($createdServiceThisRun) {
        if (Get-Service sshd -ErrorAction SilentlyContinue) {
            Stop-Service sshd -ErrorAction Stop
            Invoke-CheckedNative sc.exe @('delete', 'sshd')
        }
        $record.createdService = $false
    }
    # Disable an account we just activated before rolling back its SSH restrictions.
    # If disabling fails, leave those restrictions in place rather than expose global login policy.
    if ($accountActivationAttempted) {
        Disable-LocalUser -Name $UserName
        $record | Add-Member -NotePropertyName accountActivationPending -NotePropertyValue $true -Force
    }
    # Clearing a password cannot be undone. Keep the new SSH restrictions for an already-enabled
    # account if a later step fails, without disabling that pre-existing account.
    if ($passwordResetAttempted -and -not $activateAccount -and $user.Enabled) {
        Save-Record
        throw
    }
    # Never stop a pre-existing SSH service when another setup phase fails.
    [array]::Reverse($sshChanges)
    foreach ($change in $sshChanges) {
        try { Restore-SshFile $change }
        catch { Write-Warning "Could not restore $($change.path). Original backup retained at $($change.backup)." }
    }
    if ($sshChanges.Count -and $service -and $service.State -eq 'Running' -and (Get-Service sshd).Status -ne 'Running') {
        Start-Service sshd
    }
    Save-Record
    throw
}
finally {
    foreach ($stagedPath in @($candidatePath, $keyCandidate)) {
        if ($stagedPath -and (Test-Path -LiteralPath $stagedPath)) { Remove-Item -LiteralPath $stagedPath }
    }
}
