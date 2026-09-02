@echo off
setlocal
cd /d "%~dp0"
title Realtime Diffusion Art - First Run Setup
echo.
echo ============================================================
echo  REALTIME DIFFUSION ART - FIRST RUN SETUP
echo ============================================================
echo.
echo This one-time setup downloads Python, CUDA-enabled PyTorch,
echo application libraries, and the pinned SD-Turbo model.
echo A compatible NVIDIA display driver and Internet are required.
echo The download can take a while, but interrupted downloads resume.
echo.
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\bootstrap_first_run.ps1"
if errorlevel 1 (
  echo.
  echo SETUP FAILED. Read the error above and README_EXHIBITION.txt.
  echo The setup log is in the logs folder.
  pause
  endlocal
  exit /b 1
)
echo.
echo Setup complete. Starting the artwork...
endlocal
exit /b 0
