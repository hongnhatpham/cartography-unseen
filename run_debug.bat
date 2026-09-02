@echo off
setlocal
cd /d "%~dp0"
set "PROJECT_ROOT=%CD%"
if not exist "%PROJECT_ROOT%\cache\INSTALL_COMPLETE.txt" (
  call "%PROJECT_ROOT%\setup_first_run.bat"
  if errorlevel 1 exit /b 1
)
set "HF_HOME=%PROJECT_ROOT%\cache\huggingface"
set "HUGGINGFACE_HUB_CACHE=%PROJECT_ROOT%\cache\huggingface\hub"
set "TRANSFORMERS_CACHE=%PROJECT_ROOT%\cache\huggingface\transformers"
set "TORCH_HOME=%PROJECT_ROOT%\cache\torch"
set "XDG_CACHE_HOME=%PROJECT_ROOT%\cache"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "DIFFUSERS_OFFLINE=1"
set "PYTHONNOUSERSITE=1"
set "PYGAME_HIDE_SUPPORT_PROMPT=1"
if not exist "%PROJECT_ROOT%\logs" mkdir "%PROJECT_ROOT%\logs"
if not exist "%PROJECT_ROOT%\runtime\python\python.exe" goto :install_missing
if not exist "%PROJECT_ROOT%\models\sd_turbo\model_index.json" goto :install_missing
"%PROJECT_ROOT%\runtime\python\python.exe" -m app.main --debug %*
if errorlevel 1 pause
endlocal
exit /b 0

:install_missing
del /q "%PROJECT_ROOT%\cache\INSTALL_COMPLETE.txt" >nul 2>&1
call "%PROJECT_ROOT%\setup_first_run.bat"
if errorlevel 1 exit /b 1
"%PROJECT_ROOT%\runtime\python\python.exe" -m app.main --debug %*
if errorlevel 1 pause
endlocal
