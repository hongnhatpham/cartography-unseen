param(
    [string]$PythonVersion = "3.11.9",
    [switch]$Rebuild,
    [switch]$SkipWheelhouseDownload
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Runtime = Join-Path $Root "runtime\python"
$Wheelhouse = Join-Path $Root "wheelhouse"
$Staging = Join-Path $Root "cache\runtime_prepare"

if ($Rebuild -and (Test-Path -LiteralPath $Runtime)) {
    $resolvedRuntime = (Resolve-Path -LiteralPath $Runtime).Path
    $resolvedRoot = (Resolve-Path -LiteralPath $Root).Path
    if (-not $resolvedRuntime.StartsWith($resolvedRoot + [IO.Path]::DirectorySeparatorChar)) {
        throw "Refusing to remove runtime outside project root: $resolvedRuntime"
    }
    Remove-Item -LiteralPath $Runtime -Recurse -Force
}
New-Item -ItemType Directory -Path $Runtime, $Wheelhouse, $Staging -Force | Out-Null
$Python = Join-Path $Runtime "python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    $package = Join-Path $Staging "python.$PythonVersion.zip"
    if (-not (Test-Path -LiteralPath $package)) {
        Invoke-WebRequest -Uri "https://www.nuget.org/api/v2/package/python/$PythonVersion" -OutFile $package
    }
    $extract = Join-Path $Staging "python_nuget"
    if (Test-Path -LiteralPath $extract) { Remove-Item -LiteralPath $extract -Recurse -Force }
    Expand-Archive -LiteralPath $package -DestinationPath $extract
    Copy-Item -Path (Join-Path $extract "tools\*") -Destination $Runtime -Recurse -Force
}

& $Python -m ensurepip --upgrade
& $Python -m pip install --upgrade pip
if (-not $SkipWheelhouseDownload) {
    $torchWheel = Join-Path $Wheelhouse "torch-2.6.0+cu124-cp311-cp311-win_amd64.whl"
    $visionWheel = Join-Path $Wheelhouse "torchvision-0.21.0+cu124-cp311-cp311-win_amd64.whl"
    if (-not (Test-Path -LiteralPath $torchWheel)) {
        & curl.exe -L --fail --retry 20 --retry-delay 5 -C - -o $torchWheel "https://download.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp311-cp311-win_amd64.whl"
        if ($LASTEXITCODE -ne 0) { throw "Resumable PyTorch wheel download failed" }
    }
    if (-not (Test-Path -LiteralPath $visionWheel)) {
        & curl.exe -L --fail --retry 20 --retry-delay 5 -C - -o $visionWheel "https://download.pytorch.org/whl/cu124/torchvision-0.21.0%2Bcu124-cp311-cp311-win_amd64.whl"
        if ($LASTEXITCODE -ne 0) { throw "Resumable torchvision wheel download failed" }
    }
    & $Python -m pip download --dest $Wheelhouse --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple --constraint (Join-Path $Root "requirements-app.txt") -r (Join-Path $Root "requirements-torch.txt")
    if ($LASTEXITCODE -ne 0) { throw "PyTorch wheel download failed" }
    & $Python -m pip download --dest $Wheelhouse --find-links $Wheelhouse --constraint (Join-Path $Root "requirements-torch.txt") -r (Join-Path $Root "requirements-app.txt")
    if ($LASTEXITCODE -ne 0) { throw "Application wheel download failed" }
}
& $Python -m pip install --no-index --find-links $Wheelhouse -r (Join-Path $Root "requirements-torch.txt")
if ($LASTEXITCODE -ne 0) { throw "Offline PyTorch install failed" }
& $Python -m pip install --no-index --find-links $Wheelhouse -r (Join-Path $Root "requirements-app.txt")
if ($LASTEXITCODE -ne 0) { throw "Offline application install failed" }
& $Python -c "import torch, pygame, moderngl, diffusers; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw "Bundled runtime verification failed" }
Write-Host "Portable runtime prepared at $Runtime"
