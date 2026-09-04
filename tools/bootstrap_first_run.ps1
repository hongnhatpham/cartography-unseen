param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LogDirectory = Join-Path $Root "logs"
$Marker = Join-Path $Root "cache\INSTALL_COMPLETE.txt"
New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
$LogPath = Join-Path $LogDirectory ("first_run_setup_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

Start-Transcript -LiteralPath $LogPath -Force | Out-Null
try {
    Write-Host "Checking NVIDIA driver..." -ForegroundColor Cyan
    $NvidiaSmiCommand = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    $NvidiaSmiPath = if ($NvidiaSmiCommand) { $NvidiaSmiCommand.Source } else { $null }
    if (-not $NvidiaSmiPath) {
        $SystemNvidiaSmi = Join-Path $env:SystemRoot "System32\nvidia-smi.exe"
        if (Test-Path -LiteralPath $SystemNvidiaSmi) {
            $NvidiaSmiPath = $SystemNvidiaSmi
        }
    }
    if (-not $NvidiaSmiPath) {
        throw "No NVIDIA driver was detected. Install the current NVIDIA driver for this GPU, reboot if requested, then run run.bat again. The separate CUDA Toolkit is not required."
    }
    & $NvidiaSmiPath --query-gpu=name,driver_version,memory.total --format=csv,noheader
    if ($LASTEXITCODE -ne 0) {
        throw "The NVIDIA driver is installed but nvidia-smi could not query the GPU. Update or repair the NVIDIA driver."
    }

    if ($Force -and (Test-Path -LiteralPath $Marker)) {
        Remove-Item -LiteralPath $Marker -Force
    }

    Write-Host "`nPreparing portable Python and CUDA-enabled PyTorch..." -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot "prepare_runtime.ps1")
    if ($LASTEXITCODE -ne 0) { throw "Runtime preparation failed." }

    $Python = Join-Path $Root "runtime\python\python.exe"
    Write-Host "`nDownloading the pinned SD-Turbo and TAESD models..." -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot "prepare_models.ps1") -PythonExe $Python
    if ($LASTEXITCODE -ne 0) { throw "Model preparation failed." }

    Write-Host "`nVerifying CUDA and performing a real offline generation..." -ForegroundColor Cyan
    & $Python (Join-Path $PSScriptRoot "verify_offline.py") --root $Root
    if ($LASTEXITCODE -ne 0) { throw "Final offline verification failed." }

    $RuntimeSummary = & $Python -c "import torch; print(f'PyTorch {torch.__version__}; CUDA runtime {torch.version.cuda}; GPU {torch.cuda.get_device_name(0)}')"
    New-Item -ItemType Directory -Path (Split-Path -Parent $Marker) -Force | Out-Null
    @(
        "Realtime Diffusion Art installation complete"
        "Completed: $((Get-Date).ToString('o'))"
        $RuntimeSummary
    ) | Set-Content -LiteralPath $Marker -Encoding UTF8
    Write-Host "`nFIRST RUN SETUP COMPLETE" -ForegroundColor Green
    Write-Host $RuntimeSummary
}
catch {
    Write-Error $_
    exit 1
}
finally {
    Stop-Transcript | Out-Null
}
