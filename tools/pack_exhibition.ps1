param(
    [switch]$Zip,
    [switch]$SkipVerify,
    [switch]$PreparedOffline
)
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$DistRoot = Join-Path $Root 'dist'
$Target = Join-Path $DistRoot 'RealtimeDiffusionArt'

function Assert-ProjectPath([string]$Path) {
    $full = [IO.Path]::GetFullPath($Path)
    if (-not $full.StartsWith($Root + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing path outside project root: $full"
    }
    # Resolve-Path alone does not resolve Windows junction targets. Reject links
    # at every ancestor before traversing, copying, or removing anything.
    for ($part = $full; $part; $part = Split-Path -Parent $part) {
        if (Test-Path -LiteralPath $part) {
            if ((Get-Item -LiteralPath $part -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Refusing reparse point: $part"
            }
        }
    }
    return $full
}

function Assert-PlainTree([string]$Path) {
    $null = Assert-ProjectPath $Path
    foreach ($entry in Get-ChildItem -LiteralPath $Path -Force) {
        if ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw "Refusing reparse point: $($entry.FullName)"
        }
        if ($entry.PSIsContainer) { Assert-PlainTree $entry.FullName }
    }
}

function Test-PrivateOrGenerated([IO.FileSystemInfo]$Entry) {
    if ($Entry.Name -in @('.git', '.ssh', '.aws', '.cache', '__pycache__', 'cache', 'logs', 'journeys', 'restored-journeys', 'screenshot', 'screenshots', 'credentials', 'dashboard')) { return $true }
    if ($Entry.Name -match '^(\.env($|\.)|id_(rsa|dsa|ecdsa|ed25519)($|\.)|.*\.(key|pfx|p12)$|.*(credentials?|secrets?|tokens?).*\.(json|ini|txt|yaml|yml)$)') { return $true }
    # Certificate bundles are runtime dependencies; private PEM keys are not.
    if (-not $Entry.PSIsContainer -and $Entry.Extension -eq '.pem') {
        return [bool](Select-String -LiteralPath $Entry.FullName -Pattern '-----BEGIN .*PRIVATE KEY-----' -Quiet)
    }
    return $false
}

function Copy-PackageFile([string]$Relative) {
    $source = Assert-ProjectPath (Join-Path $Root $Relative)
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Missing required file: $source" }
    $destination = Assert-ProjectPath (Join-Path $Target $Relative)
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination
}

function Copy-PackageTree([string]$Relative) {
    $source = Assert-ProjectPath (Join-Path $Root $Relative)
    if (-not (Test-Path -LiteralPath $source -PathType Container)) { throw "Missing required directory: $source" }
    $destination = Assert-ProjectPath (Join-Path $Target $Relative)
    New-Item -ItemType Directory -Path $destination -Force | Out-Null
    foreach ($entry in Get-ChildItem -LiteralPath $source -Force) {
        $null = Assert-ProjectPath $entry.FullName
        if (Test-PrivateOrGenerated $entry) { continue }
        $child = Join-Path $Relative $entry.Name
        if ($entry.PSIsContainer) { Copy-PackageTree $child }
        else { Copy-PackageFile $child }
    }
}

if ($PreparedOffline -and $SkipVerify) {
    throw '-PreparedOffline requires real offline verification; remove -SkipVerify.'
}
$null = Assert-ProjectPath $Target
if (Test-Path -LiteralPath $Target) {
    Assert-PlainTree $Target
    Remove-Item -LiteralPath $Target -Recurse -Force
}
New-Item -ItemType Directory -Path $Target -Force | Out-Null

$requiredFiles = @(
    'config.json', 'prompts.json', 'requirements-app.txt', 'requirements-torch.txt',
    'requirements-monitor.txt', 'requirements-storage.txt',
    'run.bat', 'run_debug.bat', 'setup_first_run.bat', 'README.md', 'README_EXHIBITION.txt',
    'docs\performance\window-input-stutter-20260905.md',
    'docs\performance\inference-freezes-20260905.md',
    'docs\journey-map.md', 'docs\journey-storage.md', 'docs\exhibition-windows.md',
    'dashboard\README.md'
)
foreach ($file in $requiredFiles) { Copy-PackageFile $file }
foreach ($dir in @('app', 'shaders', 'assets')) { Copy-PackageTree $dir }
foreach ($dir in @('cache', 'logs', 'models', 'runtime', 'tools', 'wheelhouse')) {
    New-Item -ItemType Directory -Path (Join-Path $Target $dir) -Force | Out-Null
}
$toolFiles = @(
    'benchmark.py', 'soak_test.py', 'replay_performance.py',
    'bootstrap_first_run.ps1', 'download_models.py', 'prepare_models.ps1',
    'prepare_runtime.ps1', 'verify_offline.py',
    'setup_exhibition_windows.ps1', 'run_exhibition.ps1', 'exhibition_status.ps1',
    'monitor_agent.py', 'sync_journeys.py', 'configure_journey_sync.py',
    'restore_journey.py', 'export_journey_svg.py'
)
foreach ($file in $toolFiles) { Copy-PackageFile (Join-Path 'tools' $file) }

if ($PreparedOffline) {
    $settings = Get-Content -LiteralPath (Join-Path $Target 'config.json') -Raw | ConvertFrom-Json
    foreach ($model in @(@('model_path', 'models/sd_turbo'), @('taesd_path', 'models/taesd'))) {
        $configured = $settings.($model[0])
        if ($configured -and $configured.Replace('\', '/') -ne $model[1]) {
            throw "Prepared packages require $($model[0]) = $($model[1]); custom model paths are not portable."
        }
    }
    # Copy only the prepared interpreter, never runtime siblings or wheel caches.
    Copy-PackageTree 'runtime\python'
    # Match the pinned realtime models, without download metadata or extra weights.
    $modelFiles = @(
        'sd_turbo\model_index.json', 'sd_turbo\scheduler\scheduler_config.json',
        'sd_turbo\tokenizer\merges.txt', 'sd_turbo\tokenizer\special_tokens_map.json',
        'sd_turbo\tokenizer\tokenizer_config.json', 'sd_turbo\tokenizer\vocab.json',
        'sd_turbo\text_encoder\config.json', 'sd_turbo\text_encoder\model.fp16.safetensors',
        'sd_turbo\unet\config.json', 'sd_turbo\unet\diffusion_pytorch_model.fp16.safetensors',
        'taesd\config.json', 'taesd\diffusion_pytorch_model.safetensors'
    )
    foreach ($file in $modelFiles) { Copy-PackageFile (Join-Path 'models' $file) }
    $packedPython = Join-Path $Target 'runtime\python\python.exe'
    if (-not (Test-Path -LiteralPath (Join-Path $Target 'runtime\python\pythonw.exe'))) { throw 'Prepared runtime is missing pythonw.exe.' }
    & $packedPython -I -c 'import torch, pygame, moderngl, diffusers; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
    if ($LASTEXITCODE -ne 0) { throw 'Packed runtime verification failed.' }
    & $packedPython -I (Join-Path $Target 'tools\verify_offline.py') --root $Target
    if ($LASTEXITCODE -ne 0) { throw 'Packed offline generation failed; no install marker was written.' }
    # A new marker certifies this copy. Never reuse the source machine's cache.
    "Prepared offline package verified: $((Get-Date).ToString('o'))" |
        Set-Content -LiteralPath (Join-Path $Target 'cache\INSTALL_COMPLETE.txt') -Encoding UTF8
}
elseif (-not $SkipVerify) {
    $sourcePython = Assert-ProjectPath (Join-Path $Root 'runtime\python\python.exe')
    if (-not (Test-Path -LiteralPath $sourcePython)) { throw 'Source Python runtime is missing' }
    & $sourcePython -I -m py_compile `
        (Join-Path $Target 'app\main.py') `
        (Join-Path $Target 'app\config.py') `
        (Join-Path $Target 'app\renderer\proxy_renderer.py') `
        (Join-Path $Target 'app\diffusion\worker.py') `
        (Join-Path $Target 'app\monitoring.py') `
        (Join-Path $Target 'tools\monitor_agent.py')
    if ($LASTEXITCODE -ne 0) { throw 'Packed source verification failed' }
}
Get-ChildItem -LiteralPath $Target -Recurse -Directory -Filter '__pycache__' |
    Remove-Item -Recurse -Force
if ($Zip) {
    $zipPath = Assert-ProjectPath (Join-Path $DistRoot 'RealtimeDiffusionArt.zip')
    if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath -Force }
    Compress-Archive -LiteralPath $Target -DestinationPath $zipPath -CompressionLevel Optimal
}
$mode = if ($PreparedOffline) { 'prepared offline; verify again on the target GPU' } else { 'lightweight; first launch requires Internet downloads' }
Write-Host "Exhibition package ready: $Target ($mode)"
