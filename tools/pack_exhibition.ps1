param(
    [switch]$Zip,
    [switch]$SkipVerify
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DistRoot = Join-Path $Root "dist"
$Target = Join-Path $DistRoot "RealtimeDiffusionArt"
$resolvedRoot = (Resolve-Path -LiteralPath $Root).Path
if (Test-Path -LiteralPath $Target) {
    $resolvedTarget = (Resolve-Path -LiteralPath $Target).Path
    if (-not $resolvedTarget.StartsWith($resolvedRoot + [IO.Path]::DirectorySeparatorChar)) {
        throw "Refusing to remove target outside project root: $resolvedTarget"
    }
    Remove-Item -LiteralPath $Target -Recurse -Force
}
New-Item -ItemType Directory -Path $Target -Force | Out-Null

$requiredFiles = @(
    "config.json",
    "prompts.json",
    "requirements-app.txt",
    "requirements-torch.txt",
    "run.bat",
    "run_debug.bat",
    "setup_first_run.bat",
    "README.md",
    "README_EXHIBITION.txt"
)
$requiredDirs = @("app", "shaders", "assets")
foreach ($file in $requiredFiles) {
    $source = Join-Path $Root $file
    if (-not (Test-Path -LiteralPath $source)) { throw "Missing required file: $source" }
    Copy-Item -LiteralPath $source -Destination $Target
}
foreach ($dir in $requiredDirs) {
    $source = Join-Path $Root $dir
    if (-not (Test-Path -LiteralPath $source)) { throw "Missing required directory: $source" }
    Copy-Item -LiteralPath $source -Destination $Target -Recurse
}
$reportTarget = Join-Path $Target "docs\performance"
New-Item -ItemType Directory -Path $reportTarget -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $Root "docs\performance\window-input-stutter-20260905.md") -Destination $reportTarget
Copy-Item -LiteralPath (Join-Path $Root "docs\performance\inference-freezes-20260905.md") -Destination $reportTarget
Copy-Item -LiteralPath (Join-Path $Root "docs\journey-map.md") -Destination (Join-Path $Target "docs")
New-Item -ItemType Directory -Path `
    (Join-Path $Target "cache"), `
    (Join-Path $Target "logs"), `
    (Join-Path $Target "models"), `
    (Join-Path $Target "runtime"), `
    (Join-Path $Target "tools"), `
    (Join-Path $Target "wheelhouse") -Force | Out-Null
$toolFiles = @(
    "benchmark.py",
    "soak_test.py",
    "replay_performance.py",
    "bootstrap_first_run.ps1",
    "download_models.py",
    "prepare_models.ps1",
    "prepare_runtime.ps1",
    "verify_offline.py"
)
foreach ($toolFile in $toolFiles) {
    Copy-Item -LiteralPath (Join-Path $Root "tools\$toolFile") -Destination (Join-Path $Target "tools")
}
Get-ChildItem -LiteralPath $Target -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
foreach ($model in @("sd_turbo", "taesd")) {
    $modelMetadata = Join-Path $Target "models\$model\.cache"
    if (Test-Path -LiteralPath $modelMetadata) { Remove-Item -LiteralPath $modelMetadata -Recurse -Force }
}

if (-not $SkipVerify) {
    $sourcePython = Join-Path $Root "runtime\python\python.exe"
    if (-not (Test-Path -LiteralPath $sourcePython)) { throw "Source Python runtime is missing" }
    & $sourcePython -m py_compile `
        (Join-Path $Target "app\main.py") `
        (Join-Path $Target "app\config.py") `
        (Join-Path $Target "app\renderer\proxy_renderer.py") `
        (Join-Path $Target "app\diffusion\worker.py")
    if ($LASTEXITCODE -ne 0) { throw "Packed source verification failed" }
}
Get-ChildItem -LiteralPath $Target -Recurse -Directory -Filter "__pycache__" |
    Remove-Item -Recurse -Force
if ($Zip) {
    $zipPath = Join-Path $DistRoot "RealtimeDiffusionArt.zip"
    if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath -Force }
    Compress-Archive -LiteralPath $Target -DestinationPath $zipPath -CompressionLevel Optimal
}
Write-Host "Exhibition package ready: $Target"
