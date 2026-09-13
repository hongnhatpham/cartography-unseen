param(
    [string]$DeploymentPath = (Split-Path -Parent $PSScriptRoot),
    [string]$ResultsDirectory
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath $DeploymentPath).Path
$base = if ($ResultsDirectory) { [IO.Path]::GetFullPath($ResultsDirectory) } else { Join-Path (Split-Path -Parent $root) ('Commissioning\soak-' + (Get-Date -Format 'yyyyMMdd')) }
$runDirectory = Join-Path $base ('run-' + (Get-Date -Format 'HHmmss-fff'))
New-Item -ItemType Directory -Path $runDirectory | Out-Null
[IO.File]::WriteAllText((Join-Path $base 'active.json'), (@{directory=$runDirectory} | ConvertTo-Json), (New-Object Text.UTF8Encoding($false)))
$env:CARTOGRAPHY_SOAK_DIRECTORY = $base
$supervisorPath = "$root\tools\run_exhibition.ps1"
$stopFlag = "$root\cache\monitoring\maintenance.stop"
$taskName = "CartographyExhibition-$env:USERNAME"
if (Test-Path -LiteralPath "$runDirectory\app.json") { throw 'Choose a fresh run directory; previous evidence must be preserved' }
# Each run needs its own immutable configuration snapshot before stopping the
# live app. A missing snapshot previously let the test app exit immediately.
if (-not (Test-Path -LiteralPath "$runDirectory\config.json")) {
    Copy-Item -LiteralPath "$root\cache\exhibition\config.json" -Destination "$runDirectory\config.json"
}
$null = Get-Content -LiteralPath "$runDirectory\config.json" -Raw | ConvertFrom-Json
$original = [IO.File]::ReadAllText($supervisorPath)
if (-not $original.Contains("'-m app.main --config cache/exhibition/config.json'")) { throw 'Unexpected supervisor source; no change made' }
Copy-Item -LiteralPath $supervisorPath -Destination "$runDirectory\run_exhibition.original.ps1"
New-Item -ItemType File -Path $stopFlag -Force | Out-Null
$deadline = (Get-Date).AddSeconds(40)
while ((Get-ScheduledTask -TaskName $taskName).State -eq 'Running') {
    if ((Get-Date) -gt $deadline) { throw 'App did not stop cleanly for commissioning' }
    Start-Sleep -Seconds 1
}
try {
    $testSource = $original.Replace("'-m app.main --config cache/exhibition/config.json'", "'-m tools.soak_app'")
    # Task Scheduler does not inherit this shell's environment. Set the selected
    # output directory inside the temporary supervisor source as well.
    $escapedBase = $base.Replace("'", "''")
    $testSource = $testSource.Replace("`$ErrorActionPreference = 'Stop'", "`$ErrorActionPreference = 'Stop'`r`n`$env:CARTOGRAPHY_SOAK_DIRECTORY = '$escapedBase'")
    [IO.File]::WriteAllText($supervisorPath, $testSource, (New-Object Text.UTF8Encoding($false)))
    Remove-Item -LiteralPath $stopFlag
    Start-ScheduledTask -TaskName $taskName
    $deadline = (Get-Date).AddSeconds(180)
    while (-not (Test-Path -LiteralPath "$runDirectory\app.json")) {
        if ((Get-Date) -gt $deadline) { throw 'Commissioning app did not initialize' }
        Start-Sleep -Seconds 1
    }
    $watcher = Start-Process -FilePath "$root\runtime\python\pythonw.exe" -ArgumentList '-u tools/soak_watch.py' -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput "$runDirectory\watcher.stdout.log" -RedirectStandardError "$runDirectory\watcher.stderr.log"
    Write-Output "Commissioning app initialized; independent watcher PID $($watcher.Id); reports $runDirectory"
} finally {
    [IO.File]::WriteAllText($supervisorPath, $original, (New-Object Text.UTF8Encoding($false)))
}
