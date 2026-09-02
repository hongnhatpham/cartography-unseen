param(
    [string]$PythonCommand = "py -3.11",
    [switch]$SkipTorch
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$parts = $PythonCommand -split " "
$pythonExe = $parts[0]
$pythonArgs = @($parts | Select-Object -Skip 1)
& $pythonExe @pythonArgs -m venv .venv
$venvPython = Join-Path $Root ".venv\Scripts\python.exe"
& $venvPython -m pip install --upgrade pip
if (-not $SkipTorch) {
    & $venvPython -m pip install --index-url https://download.pytorch.org/whl/cu124 -r requirements-torch.txt
}
& $venvPython -m pip install -r requirements-app.txt
Write-Host "Development environment ready."
Write-Host "Run: .venv\Scripts\python.exe -m app.main --debug"
