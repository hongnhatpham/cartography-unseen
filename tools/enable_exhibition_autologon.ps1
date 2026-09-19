#requires -RunAsAdministrator
param(
    [string]$UserName = 'exhibition',
    [switch]$BlankPasswordConfirmed,
    [switch]$ClearLegalNotice
)

$ErrorActionPreference = 'Stop'
$user = Get-LocalUser -Name $UserName -ErrorAction Stop
if (-not $user.Enabled) { throw "Local account '$UserName' is disabled." }
$administrators = Get-LocalGroup -SID 'S-1-5-32-544'
$administratorSids = @(
    $group = [ADSI]("WinNT://./{0},group" -f $administrators.Name)
    foreach ($member in @($group.psbase.Invoke('Members'))) {
        $sidBytes = $member.GetType().InvokeMember(
            'objectSid',
            [Reflection.BindingFlags]::GetProperty,
            $null,
            $member,
            $null
        )
        (New-Object Security.Principal.SecurityIdentifier($sidBytes, 0)).Value
    }
)
if ($administratorSids -contains $user.SID.Value) {
    throw "Refusing autologon for administrator account '$UserName'."
}

Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
public static class LocalLogonCheck {
    [DllImport("advapi32.dll", SetLastError=true, CharSet=CharSet.Unicode)]
    static extern bool LogonUser(string user, string domain, string password,
        int logonType, int provider, out IntPtr token);
    [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr handle);
    public static void RequireBlankInteractivePassword(string user, string domain) {
        IntPtr token;
        if (!LogonUser(user, domain, "", 2, 0, out token)) throw new Win32Exception();
        CloseHandle(token);
    }
}
'@
if (-not $BlankPasswordConfirmed) {
    [LocalLogonCheck]::RequireBlankInteractivePassword($UserName, $env:COMPUTERNAME)
}

$path = 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon'
$policyPath = 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System'
$caption = [string](Get-ItemPropertyValue -LiteralPath $policyPath -Name LegalNoticeCaption -ErrorAction SilentlyContinue)
$text = [string](Get-ItemPropertyValue -LiteralPath $policyPath -Name LegalNoticeText -ErrorAction SilentlyContinue)
if (($caption -or $text) -and -not $ClearLegalNotice) {
    throw 'Windows logon notice policy would block unattended sign-in. Rerun with -ClearLegalNotice only with policy-owner approval.'
}

$backupRoot = Join-Path $env:ProgramData 'CartographyExhibition'
New-Item -ItemType Directory -Force -Path $backupRoot | Out-Null
$backup = Join-Path $backupRoot ('autologon-before-{0}.json' -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
$names = @('AutoAdminLogon', 'DefaultUserName', 'DefaultDomainName')
$previous = @{}
$existing = Get-ItemProperty -LiteralPath $path
if ($null -ne $existing.PSObject.Properties['DefaultPassword'] -or
        [string]$existing.AutoAdminLogon -eq '1') {
    throw 'An existing autologon configuration is present; refusing to overwrite its credentials.'
}
foreach ($name in $names) {
    $property = $existing.PSObject.Properties[$name]
    $previous[$name] = @{ exists = $null -ne $property; value = $(if ($null -eq $property) { $null } else { $property.Value }) }
}
$backupData = @{
    winlogon = $previous
    legal_notice = @{ caption = $caption; text = $text; cleared = [bool]$ClearLegalNotice }
}
[IO.File]::WriteAllText($backup, ($backupData | ConvertTo-Json), (New-Object Text.UTF8Encoding($false)))

try {
    if ($ClearLegalNotice) {
        Set-ItemProperty -LiteralPath $policyPath -Name LegalNoticeCaption -Type String -Value ''
        Set-ItemProperty -LiteralPath $policyPath -Name LegalNoticeText -Type String -Value ''
    }
    Set-ItemProperty -LiteralPath $path -Name AutoAdminLogon -Type String -Value '0'
    Set-ItemProperty -LiteralPath $path -Name DefaultUserName -Type String -Value $UserName
    Set-ItemProperty -LiteralPath $path -Name DefaultDomainName -Type String -Value $env:COMPUTERNAME
    # An explicit empty value is required; omitting this value disables autologon.
    Set-ItemProperty -LiteralPath $path -Name DefaultPassword -Type String -Value ''
    Set-ItemProperty -LiteralPath $path -Name AutoAdminLogon -Type String -Value '1'
}
catch {
    $failure = $_
    try { Remove-ItemProperty -LiteralPath $path -Name DefaultPassword -ErrorAction Stop }
    catch { Write-Warning "Could not remove DefaultPassword during rollback: $($_.Exception.Message)" }
    foreach ($name in $names) {
        try {
            if ($previous[$name].exists) {
                Set-ItemProperty -LiteralPath $path -Name $name -Type String -Value ([string]$previous[$name].value)
            } else {
                Remove-ItemProperty -LiteralPath $path -Name $name -ErrorAction Stop
            }
        }
        catch { Write-Warning "Could not restore $name during rollback: $($_.Exception.Message)" }
    }
    if ($ClearLegalNotice) {
        try { Set-ItemProperty -LiteralPath $policyPath -Name LegalNoticeCaption -Type String -Value $caption }
        catch { Write-Warning "Could not restore LegalNoticeCaption: $($_.Exception.Message)" }
        try { Set-ItemProperty -LiteralPath $policyPath -Name LegalNoticeText -Type String -Value $text }
        catch { Write-Warning "Could not restore LegalNoticeText: $($_.Exception.Message)" }
    }
    Write-Warning "Autologon setup failed and previous values were restored. Backup: $backup"
    throw $failure
}

Write-Output "CONFIGURED: reboot twice and verify '$env:COMPUTERNAME\$UserName' signs in automatically both times."
Write-Output "Previous values: $backup"
