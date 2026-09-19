param(
    [string]$Root = 'C:\Exhibition\Cartography',
    [datetime]$Until = (Get-Date).Date.AddHours(20),
    [int]$IntervalSeconds = 30,
    [switch]$Repair
)

$ErrorActionPreference = 'Stop'
$logDir = Join-Path $Root 'logs'
$logPath = Join-Path $logDir 'exhibition-watchdog.log'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Write-Event([string]$Level, [string]$Message) {
    $line = '{0:o} [{1}] {2}' -f (Get-Date), $Level, $Message
    $line | Tee-Object -FilePath $logPath -Append
}

function Read-JsonRetry([string]$Path) {
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        try { return Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json }
        catch {
            if ($attempt -eq 3) { throw }
            Start-Sleep -Milliseconds 100
        }
    }
}

function Test-State {
    $issues = [System.Collections.Generic.List[string]]::new()
    try { $config = Read-JsonRetry (Join-Path $Root 'cache\exhibition\config.json') }
    catch { $issues.Add("config unreadable: $($_.Exception.Message)"); return $issues }

    if (-not $config.journey_map) { $issues.Add('journey_map is disabled') }
    if (-not $config.fullscreen) { $issues.Add('fullscreen preference is disabled') }
    if (-not $config.map_sync_enabled) { $issues.Add('automatic uploads are disabled') }

    $monitoring = Join-Path $Root 'cache\monitoring'
    $now = [DateTimeOffset]::UtcNow
    try {
        $supervisor = Read-JsonRetry (Join-Path $monitoring 'supervisor.json')
        $age = ($now - [DateTimeOffset]::Parse([string]$supervisor.updatedAt)).TotalSeconds
        if ($age -gt 20) { $issues.Add("supervisor heartbeat stale: $([math]::Round($age)) seconds") }
        if ($supervisor.state -ne 'running') { $issues.Add("supervisor state is $($supervisor.state): $($supervisor.detail)") }
        if ([int]$supervisor.restarts -gt 0) { $issues.Add("app has restarted $($supervisor.restarts) time(s)") }
    } catch { $issues.Add("supervisor telemetry unavailable: $($_.Exception.Message)") }

    try {
        $artwork = Read-JsonRetry (Join-Path $monitoring 'artwork.json')
        $age = $now.ToUnixTimeSeconds() - [double]$artwork.updated_at
        if ($age -gt 20) { $issues.Add("artwork telemetry stale: $([math]::Round($age)) seconds") }
        if ($artwork.state -ne 'running') { $issues.Add("artwork state is $($artwork.state)") }
        if ([double]$artwork.display_fps -lt 45) { $issues.Add("display FPS low: $([math]::Round([double]$artwork.display_fps, 1))") }
        if ([double]$artwork.generation_fps -lt 2) { $issues.Add("generation FPS low: $([math]::Round([double]$artwork.generation_fps, 1))") }
        if ([double]$artwork.frame_age_ms -gt 500) { $issues.Add("AI frame stale: $([math]::Round([double]$artwork.frame_age_ms)) ms") }
    } catch { $issues.Add("artwork telemetry unavailable: $($_.Exception.Message)") }

    try {
        $uploader = Read-JsonRetry (Join-Path $monitoring 'uploader.json')
        $age = $now.ToUnixTimeSeconds() - [double]$uploader.updated_at
        if ($age -gt 120) { $issues.Add("uploader heartbeat stale: $([math]::Round($age)) seconds") }
        if ($uploader.error_code) { $issues.Add("uploader error: $($uploader.error_code)") }
        if ([int]$uploader.error_archives -gt 0 -or [int]$uploader.invalid_archives -gt 0) {
            $issues.Add("uploader archive failures: $($uploader.error_archives) error, $($uploader.invalid_archives) invalid")
        }
    } catch { $issues.Add("uploader telemetry unavailable: $($_.Exception.Message)") }

    $systemDrive = [IO.DriveInfo]::new([IO.Path]::GetPathRoot($Root))
    $freeGiB = [math]::Round($systemDrive.AvailableFreeSpace / 1GB, 2)
    $reserveGiB = [double]$config.map_min_free_gib
    if ($freeGiB -lt [math]::Max($reserveGiB, 5)) {
        $issues.Add("low storage: $freeGiB GiB free (capture reserve $reserveGiB GiB)")
    }

    return $issues
}

if ($Repair) {
    powercfg /change monitor-timeout-ac 0 | Out-Null
    powercfg /change monitor-timeout-dc 0 | Out-Null
    powercfg /change standby-timeout-ac 0 | Out-Null
    powercfg /change standby-timeout-dc 0 | Out-Null
    Write-Event INFO 'Power policy set to keep the machine and displays awake on AC and battery.'
}

$previous = $null
Write-Event INFO "Watch started; checking every $IntervalSeconds seconds until $($Until.ToString('o'))."
while ((Get-Date) -lt $Until) {
    try {
        $issues = @(Test-State)
        $state = if ($issues.Count) { $issues -join ' | ' } else { 'healthy' }
        if ($state -ne $previous) {
            Write-Event $(if ($issues.Count) { 'ALERT' } else { 'OK' }) $state
            $previous = $state
        }
    } catch {
        $state = "watchdog check failed: $($_.Exception.Message)"
        if ($state -ne $previous) { Write-Event ERROR $state; $previous = $state }
    }
    Start-Sleep -Seconds $IntervalSeconds
}
Write-Event INFO 'Watch window ended.'
