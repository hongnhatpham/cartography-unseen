#requires -Version 5.1
[CmdletBinding()]
param(
    [string]$DeploymentPath = (Split-Path -Parent $PSScriptRoot),
    [ValidateRange(1, 60)][int]$HeartbeatSeconds = 5,
    [ValidateRange(1, 20)][int]$MaxConsecutiveFailures = 8
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$root = (Resolve-Path -LiteralPath $DeploymentPath).Path
$monitoring = Join-Path $root 'cache\monitoring'
$logDirectory = Join-Path $root 'logs\supervisor'
New-Item -ItemType Directory -Path $monitoring, $logDirectory -Force | Out-Null
$stopFlag = Join-Path $monitoring 'maintenance.stop'
$statusPath = Join-Path $monitoring 'supervisor.json'
$lockPath = Join-Path $monitoring 'supervisor.lock'
$logPath = Join-Path $logDirectory 'supervisor.log'
$lock = $null
$child = $null
$job = $null
$started = (Get-Date).ToUniversalTime().ToString('o')
$failures = 0
$restarts = 0
$lastExit = $null
$exitCode = 0
$lastState = 'validating'
$lastDetail = ''

function Write-SupervisorLog {
    param([string]$Message)
    if ((Test-Path -LiteralPath $logPath) -and (Get-Item -LiteralPath $logPath).Length -ge 1048576) {
        for ($index = 4; $index -ge 1; $index--) {
            $source = "$logPath.$index"
            if (Test-Path -LiteralPath $source) { Move-Item -LiteralPath $source -Destination "$logPath.$($index + 1)" -Force }
        }
        Move-Item -LiteralPath $logPath -Destination "$logPath.1" -Force
    }
    Add-Content -LiteralPath $logPath -Encoding UTF8 -Value ("{0} {1}" -f (Get-Date).ToUniversalTime().ToString('o'), $Message)
}
function Write-AtomicSupervisorFile {
    param([string]$Source, [string]$Destination)
    # ReplaceFile leaves both names intact for 1175; retry only these bounded
    # sharing conflicts. Persistent failures still reach the supervisor handler.
    for ($attempt = 0; $attempt -le 10; $attempt++) {
        try {
            if ([IO.File]::Exists($Destination)) { [IO.File]::Replace($Source, $Destination, [NullString]::Value) }
            else { [IO.File]::Move($Source, $Destination) }
            return
        } catch {
            $nativeCode = $_.Exception.GetBaseException().HResult -band 0xFFFF
            if ($attempt -eq 10 -or $nativeCode -notin @(5, 32, 33, 1175)) { throw }
            Start-Sleep -Milliseconds 25
        }
    }
}
function Write-SupervisorStatus {
    param([string]$State, [string]$Detail = '')
    $script:lastState = $State
    $script:lastDetail = $Detail
    $childId = if ($child -and -not $child.HasExited) { $child.Id } else { $null }
    $payload = [ordered]@{
        schemaVersion = 1; state = $State; detail = $Detail.Substring(0, [Math]::Min(1000, $Detail.Length))
        updatedAt = (Get-Date).ToUniversalTime().ToString('o'); startedAt = $started
        supervisorPid = $PID; childPid = $childId; restarts = $restarts; consecutiveFailures = $failures
        lastExitCode = $lastExit; maintenance = (Test-Path -LiteralPath $stopFlag); deploymentPath = $root
    } | ConvertTo-Json -Compress
    $temp = "$statusPath.$PID.tmp"
    [IO.File]::WriteAllText($temp, $payload, (New-Object Text.UTF8Encoding($false)))
    Write-AtomicSupervisorFile -Source $temp -Destination $statusPath
}
function Wait-WithHeartbeat {
    param([int]$Seconds, [string]$State)
    for ($elapsed = 0; $elapsed -lt $Seconds; $elapsed++) {
        if (Test-Path -LiteralPath $stopFlag) { return $false }
        if ($elapsed % $HeartbeatSeconds -eq 0) { Write-SupervisorStatus $State }
        Start-Sleep -Seconds 1
    }
    return $true
}

try {
    # File sharing exclusion works across console and SSH sessions and disappears on process death.
    try { $lock = [IO.File]::Open($lockPath, 'OpenOrCreate', 'ReadWrite', 'None') }
    catch { Write-Output 'Another supervisor owns this deployment.'; exit 4 }
    Write-SupervisorStatus 'validating'
    foreach ($required in @('runtime\python\python.exe', 'cache\INSTALL_COMPLETE.txt', 'tools\verify_offline.py', 'cache\exhibition\config.json', 'app\main.py')) {
        if (-not (Test-Path -LiteralPath (Join-Path $root $required) -PathType Leaf)) { throw "Missing $required. Complete offline commissioning before starting the task." }
    }
    if (Test-Path -LiteralPath $stopFlag) { Write-SupervisorStatus 'maintenance'; exit 0 }
    $env:HF_HUB_OFFLINE = '1'; $env:TRANSFORMERS_OFFLINE = '1'; $env:DIFFUSERS_OFFLINE = '1'
    $env:PYTHONNOUSERSITE = '1'; $env:PYTHONDONTWRITEBYTECODE = '1'; $env:PYGAME_HIDE_SUPPORT_PROMPT = '1'; $env:PYTHONUNBUFFERED = '1'
    $env:HF_HOME = Join-Path $root 'cache\huggingface'
    $env:HUGGINGFACE_HUB_CACHE = Join-Path $root 'cache\huggingface\hub'
    $env:TRANSFORMERS_CACHE = Join-Path $root 'cache\huggingface\transformers'
    $env:TORCH_HOME = Join-Path $root 'cache\torch'; $env:XDG_CACHE_HOME = Join-Path $root 'cache'
    Set-Location -LiteralPath $root
    $python = Join-Path $root 'runtime\python\python.exe'
    $validation = @'
from pathlib import Path
from app.config import AppConfig, configure_local_environment
from tools.verify_offline import missing_model_files
root = Path.cwd()
configure_local_environment(root, offline=True)
config = AppConfig.load(root / 'cache/exhibition/config.json')
missing = missing_model_files(root, config.backend_dict(root))
model = Path(config.backend_dict(root)['model_path'])
for name in ('model_index.json', 'tokenizer/merges.txt', 'text_encoder/model.fp16.safetensors', 'unet/diffusion_pytorch_model.fp16.safetensors'):
    path = model / name
    if not path.is_file() or path.stat().st_size == 0:
        missing.append(str(path))
if missing:
    raise SystemExit('Missing model files: ' + ', '.join(missing))
import torch, pygame, moderngl, diffusers
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable')
'@
    & $python -c $validation
    if ($LASTEXITCODE -ne 0) { throw "Offline prerequisite validation failed with exit code $LASTEXITCODE" }
    & (Join-Path $PSScriptRoot 'configure_exhibition_gpu.ps1') -DeploymentPath $root
    # A kill-on-close Job Object prevents orphan renderers if Task Scheduler kills this supervisor.
    Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
public static class ExhibitionJob {
    [StructLayout(LayoutKind.Sequential)] struct Basic {
        public long ProcessTime, JobTime; public uint Flags;
        public UIntPtr Min, Max; public uint Active; public UIntPtr Affinity; public uint Priority, Scheduling;
    }
    [StructLayout(LayoutKind.Sequential)] struct Io { public ulong A, B, C, D, E, F; }
    [StructLayout(LayoutKind.Sequential)] struct Extended {
        public Basic Basic; public Io Io; public UIntPtr ProcessMemory, JobMemory, PeakProcess, PeakJob;
    }
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)] static extern IntPtr CreateJobObject(IntPtr attr, string name);
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool SetInformationJobObject(IntPtr job, int kind, ref Extended info, uint length);
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
    [DllImport("kernel32.dll")] public static extern bool CloseHandle(IntPtr handle);
    public static IntPtr Create() {
        IntPtr job = CreateJobObject(IntPtr.Zero, null);
        if (job == IntPtr.Zero) throw new Win32Exception();
        var info = new Extended(); info.Basic.Flags = 0x2000;
        if (!SetInformationJobObject(job, 9, ref info, (uint)Marshal.SizeOf(info))) { CloseHandle(job); throw new Win32Exception(); }
        return job;
    }
    public static void Assign(IntPtr job, IntPtr process) {
        if (!AssignProcessToJobObject(job, process)) throw new Win32Exception();
    }
}
'@
    $job = [ExhibitionJob]::Create()
    while (-not (Test-Path -LiteralPath $stopFlag)) {
        $launchTime = Get-Date
        $info = New-Object Diagnostics.ProcessStartInfo
        $info.FileName = $python; $info.Arguments = '-m app.main --config cache/exhibition/config.json'; $info.WorkingDirectory = $root
        $info.UseShellExecute = $false
        $child = [Diagnostics.Process]::Start($info)
        try { [ExhibitionJob]::Assign($job, $child.Handle) }
        catch { if (-not $child.HasExited) { $child.Kill() }; throw }
        Write-SupervisorLog "Started app PID $($child.Id), restart $restarts"
        while (-not $child.HasExited -and -not (Test-Path -LiteralPath $stopFlag)) {
            Write-SupervisorStatus 'running'
            $null = $child.WaitForExit($HeartbeatSeconds * 1000)
        }
        if (Test-Path -LiteralPath $stopFlag) { break }
        $lastExit = $child.ExitCode
        $duration = ((Get-Date) - $launchTime).TotalSeconds
        $failures = if ($duration -ge 300) { 1 } else { $failures + 1 }
        $restarts++
        Write-SupervisorLog "App exited $lastExit after $([int]$duration)s; consecutive short runs $failures"
        $child.Dispose(); $child = $null
        if ($failures -ge $MaxConsecutiveFailures) {
            # Latch until an operator removes the flag; Task Scheduler restart cannot create a crash loop.
            New-Item -ItemType File -Path $stopFlag -Force | Out-Null
            Write-SupervisorStatus 'faulted' 'Repeated short app runs. Inspect logs, then remove maintenance.stop and start the task.'
            $exitCode = 2
            break
        }
        $backoff = [int][Math]::Min(60, [Math]::Pow(2, $failures))
        if (-not (Wait-WithHeartbeat -Seconds $backoff -State 'backoff')) { break }
    }
    if ($exitCode -eq 0) { Write-SupervisorStatus 'maintenance' 'Stopped by maintenance flag.' }
}
catch {
    $exitCode = 2
    if ($lock) {
        Write-SupervisorLog $_.Exception.Message
        Write-SupervisorStatus 'faulted' $_.Exception.Message
    }
    Write-Error $_ -ErrorAction Continue
}
finally {
    if ($child -and -not $child.HasExited) {
        $null = $child.CloseMainWindow()
        if (-not $child.WaitForExit(10000)) { $child.Kill(); $child.WaitForExit() }
    }
    if ($job) { $null = [ExhibitionJob]::CloseHandle($job) }
    if ($child) { $child.Dispose(); $child = $null }
    if ($lock) {
        try { Write-SupervisorStatus $lastState $lastDetail }
        finally { $lock.Dispose() }
    }
}
exit $exitCode
