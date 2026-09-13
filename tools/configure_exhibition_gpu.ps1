#requires -Version 5.1
[CmdletBinding()]
param(
    [string]$DeploymentPath = (Split-Path -Parent $PSScriptRoot),
    [string]$RegistryPath = 'HKCU:\Software\Microsoft\DirectX\UserGpuPreferences'
)
$ErrorActionPreference = 'Stop'
$gpuRoot = (Resolve-Path -LiteralPath $DeploymentPath).Path
$executables = @('python.exe', 'pythonw.exe') | ForEach-Object {
    $candidate = Join-Path $gpuRoot "runtime\python\$_"
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { throw "Missing exhibition interpreter: $candidate" }
    $candidate
}
# Run in the exhibition account's logon task before creating either GL context.
# A packaged editor can virtualize HKCU; its registry writes are not sufficient.
if (-not (Test-Path -LiteralPath $RegistryPath)) {
    New-Item -Path $RegistryPath -Force | Out-Null
}
foreach ($executable in $executables) {
    $existing = (Get-Item -LiteralPath $RegistryPath).GetValue($executable, '')
    $options = @(([string]$existing -split ';') | Where-Object { $_ -and $_ -notmatch '^GpuPreference=' })
    $desired = 'GpuPreference=2;' + (($options | ForEach-Object { "$_;" }) -join '')
    if ($existing -ne $desired) {
        New-ItemProperty -LiteralPath $RegistryPath -Name $executable -Value $desired -PropertyType String -Force | Out-Null
    }
}
