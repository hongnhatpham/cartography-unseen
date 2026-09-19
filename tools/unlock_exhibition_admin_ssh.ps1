#requires -RunAsAdministrator
param([string]$UserName = 'hong-ssh')

$ErrorActionPreference = 'Stop'
$fleetRoot = Join-Path $env:ProgramData 'CartographyExhibition'
$stateRoot = Join-Path $fleetRoot $UserName
$keyPath = Join-Path $stateRoot 'authorized_keys'
$profileKeyPath = Join-Path $env:SystemDrive "Users\$UserName\.ssh\authorized_keys"
$configPath = Join-Path $env:ProgramData 'ssh\sshd_config'
$sshd = Join-Path $env:SystemRoot 'System32\OpenSSH\sshd.exe'
$ruleName = "CartographyExhibition-SSH-$UserName"
$approvedPeers = @('100.87.222.71', 'fd7a:115c:a1e0::b434:de48', '100.96.33.72', 'fd7a:115c:a1e0::9f34:2149')
$ariaKey = 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGB+badvJKcIRy/vjzcuq7UVcFug6YtRmFsW3KKCENZV aria-to-hp-zbook-2026-09-19'

foreach ($path in @($configPath, $sshd)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required file missing: $path" }
}

$keyExisted = Test-Path -LiteralPath $keyPath -PathType Leaf
if (-not $keyExisted -and -not (Test-Path -LiteralPath $profileKeyPath -PathType Leaf)) {
    throw "No existing SSH keys found at $keyPath or $profileKeyPath. Refusing to risk locking out the working SSH account."
}

$fleetRootExisted = Test-Path -LiteralPath $fleetRoot -PathType Container
$stateRootExisted = Test-Path -LiteralPath $stateRoot -PathType Container
foreach ($directory in @($fleetRoot, $stateRoot)) {
    if (Test-Path -LiteralPath $directory) {
        $item = Get-Item -LiteralPath $directory -Force
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw "Refusing reparse-point key directory: $directory"
        }
    }
}
$fleetRootAclBackup = if ($fleetRootExisted) { Get-Acl -LiteralPath $fleetRoot }
$stateRootAclBackup = if ($stateRootExisted) { Get-Acl -LiteralPath $stateRoot }

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$configBackup = "$configPath.before-admin-$stamp"
$keyBackup = "$keyPath.before-admin-$stamp"
Copy-Item -LiteralPath $configPath -Destination $configBackup
if ($keyExisted) {
    Copy-Item -LiteralPath $keyPath -Destination $keyBackup
    $keyAclBackup = Get-Acl -LiteralPath $keyPath
}

$administrators = Get-LocalGroup -SID 'S-1-5-32-544'
$wasAdministrator = [bool](Get-LocalGroupMember -Group $administrators.Name -Member $UserName -ErrorAction SilentlyContinue)
$originalDescription = (Get-LocalUser -Name $UserName).Description
$firewallRule = Get-NetFirewallRule -Name $ruleName -ErrorAction SilentlyContinue
$firewallRuleExisted = [bool]$firewallRule
if ($firewallRuleExisted) {
    $firewallFilter = $firewallRule | Get-NetFirewallAddressFilter
    $originalFirewallEnabled = $firewallRule.Enabled
    $originalFirewallAction = $firewallRule.Action
    $originalRemoteAddresses = @($firewallFilter.RemoteAddress)
}

try {
    $keys = @()
    foreach ($source in @($keyPath, $profileKeyPath)) {
        if (Test-Path -LiteralPath $source -PathType Leaf) {
            $keys += Get-Content -LiteralPath $source | Where-Object { $_.Trim() }
        }
    }
    $keys = @($keys | Select-Object -Unique)
    New-Item -ItemType Directory -Path $stateRoot -Force | Out-Null
    $administratorsSid = New-Object Security.Principal.SecurityIdentifier('S-1-5-32-544')
    $systemSid = New-Object Security.Principal.SecurityIdentifier('S-1-5-18')
    foreach ($directory in @($fleetRoot, $stateRoot)) {
        $directoryAcl = New-Object Security.AccessControl.DirectorySecurity
        $directoryAcl.SetOwner($administratorsSid)
        $directoryAcl.SetAccessRuleProtection($true, $false)
        foreach ($sid in @($administratorsSid, $systemSid)) {
            $directoryRule = New-Object Security.AccessControl.FileSystemAccessRule(
                $sid,
                [Security.AccessControl.FileSystemRights]::FullControl,
                [Security.AccessControl.InheritanceFlags]'ContainerInherit, ObjectInherit',
                [Security.AccessControl.PropagationFlags]::None,
                [Security.AccessControl.AccessControlType]::Allow
            )
            [void]$directoryAcl.AddAccessRule($directoryRule)
        }
        Set-Acl -LiteralPath $directory -AclObject $directoryAcl
    }

    $ariaBlob = ($ariaKey -split ' ')[1]
    $keys = @($keys | Where-Object { $_ -notmatch [regex]::Escape($ariaBlob) })
    $keys += 'from="100.87.222.71,fd7a:115c:a1e0::b434:de48" ' + $ariaKey
    Set-Content -LiteralPath $keyPath -Encoding ascii -Value $keys

    # sshd rejects administrator key files that ordinary users can modify.
    $acl = New-Object Security.AccessControl.FileSecurity
    $acl.SetOwner($administratorsSid)
    $acl.SetAccessRuleProtection($true, $false)
    foreach ($sid in @($administratorsSid, $systemSid)) {
        $rule = New-Object Security.AccessControl.FileSystemAccessRule(
            $sid,
            [Security.AccessControl.FileSystemRights]::FullControl,
            [Security.AccessControl.AccessControlType]::Allow
        )
        [void]$acl.AddAccessRule($rule)
    }
    Set-Acl -LiteralPath $keyPath -AclObject $acl

    $config = Get-Content -Raw -LiteralPath $configPath
    $begin = "# BEGIN Cartography exhibition $UserName"
    $end = "# END Cartography exhibition $UserName"
    $pattern = '(?ms)^' + [regex]::Escape($begin) + '\r?\n.*?^' + [regex]::Escape($end)
    $block = @"
$begin
Match User $UserName
    AllowUsers $UserName
    PubkeyAuthentication yes
    PasswordAuthentication no
    KbdInteractiveAuthentication no
    AuthenticationMethods publickey
    AuthorizedKeysFile "$($keyPath.Replace('\', '/'))"
$end
"@
    if ([regex]::IsMatch($config, $pattern)) {
        $candidate = [regex]::Replace($config, $pattern, $block)
    }
    else {
        # Upgrade the marker used by the earlier Aria-only setup when present.
        $legacyBegin = "# BEGIN ARIA MANAGED $($UserName.ToUpperInvariant())"
        $legacyEnd = "# END ARIA MANAGED $($UserName.ToUpperInvariant())"
        $legacyPattern = '(?ms)^' + [regex]::Escape($legacyBegin) + '\r?\n.*?^' + [regex]::Escape($legacyEnd)
        if ([regex]::IsMatch($config, $legacyPattern)) {
            $candidate = [regex]::Replace($config, $legacyPattern, $block)
        }
        else {
            # A user-specific block must precede the stock administrators block,
            # because OpenSSH keeps the first value it obtains for each setting.
            $administratorsPattern = '(?m)^Match\s+Group\s+administrators\s*$'
            if ([regex]::IsMatch($config, $administratorsPattern)) {
                $candidate = [regex]::Replace(
                    $config,
                    $administratorsPattern,
                    "$block`r`n`r`nMatch Group administrators",
                    1
                )
            }
            else {
                $candidate = $config.TrimEnd() + "`r`n`r`n$block`r`n"
            }
        }
    }
    $candidatePath = "$configPath.admin-candidate"
    [IO.File]::WriteAllText($candidatePath, $candidate, (New-Object Text.UTF8Encoding($false)))
    & $sshd -t -f $candidatePath
    if ($LASTEXITCODE -ne 0) { throw "sshd rejected the candidate configuration ($LASTEXITCODE)." }
    Copy-Item -LiteralPath $candidatePath -Destination $configPath -Force
    Remove-Item -LiteralPath $candidatePath -Force

    if (-not $wasAdministrator) {
        Add-LocalGroupMember -Group $administrators.Name -Member $UserName
    }
    Set-LocalUser -Name $UserName -Description 'Fleet-only exhibition SSH administrator'
    if ($firewallRuleExisted) {
        Set-NetFirewallRule -Name $ruleName -Enabled True -Action Allow
        $firewallRule | Get-NetFirewallAddressFilter | Set-NetFirewallAddressFilter -RemoteAddress $approvedPeers
    }
    else {
        New-NetFirewallRule `
            -Name $ruleName `
            -DisplayName "Cartography exhibition SSH ($UserName)" `
            -Direction Inbound `
            -Protocol TCP `
            -LocalPort 22 `
            -RemoteAddress $approvedPeers `
            -Action Allow `
            -Profile Any | Out-Null
    }

    Restart-Service sshd
    if ((Get-Service sshd).Status -ne 'Running') { throw 'sshd did not return to Running.' }
    Write-Output 'SUCCESS: fleet-only administrative SSH enabled for asus-rog and aria.'
    Write-Output "Backups: $configBackup ; $keyBackup"
}
catch {
    Copy-Item -LiteralPath $configBackup -Destination $configPath -Force
    if ($keyExisted) {
        Copy-Item -LiteralPath $keyBackup -Destination $keyPath -Force
        Set-Acl -LiteralPath $keyPath -AclObject $keyAclBackup
    }
    else {
        Remove-Item -LiteralPath $keyPath -Force -ErrorAction SilentlyContinue
    }
    if ($stateRootExisted) {
        Set-Acl -LiteralPath $stateRoot -AclObject $stateRootAclBackup
    }
    else {
        Remove-Item -LiteralPath $stateRoot -Force -ErrorAction SilentlyContinue
    }
    if ($fleetRootExisted) {
        Set-Acl -LiteralPath $fleetRoot -AclObject $fleetRootAclBackup
    }
    else {
        Remove-Item -LiteralPath $fleetRoot -Force -ErrorAction SilentlyContinue
    }
    if (-not $wasAdministrator) {
        Remove-LocalGroupMember -Group $administrators.Name -Member $UserName -ErrorAction SilentlyContinue
    }
    Set-LocalUser -Name $UserName -Description $originalDescription
    if ($firewallRuleExisted) {
        Set-NetFirewallRule -Name $ruleName -Enabled $originalFirewallEnabled -Action $originalFirewallAction
        Set-NetFirewallAddressFilter -InputObject $firewallFilter -RemoteAddress $originalRemoteAddresses
    }
    else {
        Remove-NetFirewallRule -Name $ruleName -ErrorAction SilentlyContinue
    }
    Restart-Service sshd -ErrorAction SilentlyContinue
    throw
}
