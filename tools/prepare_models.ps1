param(
    [string]$PythonExe = "",
    [string[]]$Only = @()
)
# Revisions are pinned inside tools\download_models.py, which is the single
# source of truth for what an offline machine needs.
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $PythonExe) {
    $bundled = Join-Path $Root "runtime\python\python.exe"
    $dev = Join-Path $Root ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $bundled) { $PythonExe = $bundled }
    elseif (Test-Path -LiteralPath $dev) { $PythonExe = $dev }
    else { $PythonExe = "python" }
}
$arguments = @((Join-Path $PSScriptRoot "download_models.py"), "--root", $Root)
foreach ($name in $Only) { $arguments += @("--only", $name) }
& $PythonExe @arguments
if ($LASTEXITCODE -ne 0) { throw "Model preparation failed with exit code $LASTEXITCODE" }
