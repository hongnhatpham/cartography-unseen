#requires -Version 5.1
[CmdletBinding()]
param(
    [string]$DeploymentPath = (Split-Path -Parent $PSScriptRoot),
    [string]$UserName = $env:USERNAME,
    [ValidateRange(10, 3600)][int]$MaxHeartbeatAgeSeconds = 30,
    [ValidateRange(0, 1024)][double]$MinimumFreeGiB = 5,
    [switch]$Json
)
$ErrorActionPreference = 'Stop'
try {
    $root = (Resolve-Path -LiteralPath $DeploymentPath).Path
    $monitoring = Join-Path $root 'cache\monitoring'
    $statusPath = Join-Path $monitoring 'supervisor.json'
    $supervisor = if (Test-Path -LiteralPath $statusPath) { Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json } else { $null }
    $task = Get-ScheduledTask -TaskName "CartographyExhibition-$UserName" -ErrorAction SilentlyContinue
    $service = Get-Service -Name sshd -ErrorAction SilentlyContinue
    $volume = Get-Volume -FilePath $root
    $freeGiB = [Math]::Round($volume.SizeRemaining / 1GB, 2)
    $heartbeatAge = if ($supervisor) { [Math]::Max(0, ((Get-Date).ToUniversalTime() - [datetime]::Parse($supervisor.updatedAt).ToUniversalTime()).TotalSeconds) } else { $null }
    $maintenance = Test-Path -LiteralPath (Join-Path $monitoring 'maintenance.stop')
    $fresh = $null -ne $heartbeatAge -and $heartbeatAge -le $MaxHeartbeatAgeSeconds
    $processAlive = $false
    if ($supervisor -and $supervisor.childPid) {
        $process = Get-Process -Id $supervisor.childPid -ErrorAction SilentlyContinue
        $processAlive = $null -ne $process -and $process.ProcessName -in @('python', 'pythonw')
    }
    $healthy = $fresh -and $supervisor.state -eq 'running' -and $processAlive -and $task.State -eq 'Running'
    $exitCode = if ($maintenance) { 3 } elseif (-not $healthy) { 2 } elseif ($freeGiB -lt $MinimumFreeGiB -or -not $service -or $service.Status -ne 'Running') { 1 } else { 0 }
    $report = [pscustomobject]@{
        exitCode = $exitCode; deploymentPath = $root; taskState = $(if ($task) { [string]$task.State } else { 'missing' })
        sshdState = $(if ($service) { [string]$service.Status } else { 'missing' }); freeGiB = $freeGiB
        maintenance = $maintenance; heartbeatAgeSeconds = $heartbeatAge; childProcessPresent = $processAlive
        supervisor = $supervisor
    }
    if ($Json) { $report | ConvertTo-Json -Depth 6 } else { $report | Format-List }
    exit $exitCode
}
catch { Write-Error $_ -ErrorAction Continue; exit 4 }
