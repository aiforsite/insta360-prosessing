@echo off
REM Batch file to activate virtual environment and run processing_runner.py
REM Usage: run.bat [arguments]
REM Example: run.bat --test
REM Example: run.bat --reset

REM Get the directory where this batch file is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM ----------------------------
REM Logging configuration
REM ----------------------------
set "LOG_DIR=%SCRIPT_DIR%logs"
set "LOG_FILE=%LOG_DIR%\processing_runner.log"
set "MAX_LOG_BYTES=5242880"  REM 5 MB
set "RETAIN_LOGS=10"

if not exist "%LOG_DIR%" (
    mkdir "%LOG_DIR%" >nul 2>&1
)

REM Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found at venv\Scripts\activate.bat
    echo.
    echo Please create a virtual environment first:
    echo   python -m venv venv
    echo   venv\Scripts\activate.bat
    echo   pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if activation was successful
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Run the processing runner with any arguments passed to this batch file
echo Running processing_runner.py...
echo Logs: "%LOG_FILE%" (rotate at 5 MB, keep %RETAIN_LOGS% rotated files)

REM Startup-only log rotation/cleanup (server restarts nightly)
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "& { $ErrorActionPreference='SilentlyContinue'; " ^
  " $logDir = '%LOG_DIR%'; " ^
  " $logFile = '%LOG_FILE%'; " ^
  " $maxBytes = [int64]%MAX_LOG_BYTES%; " ^
  " $retain = [int]%RETAIN_LOGS%; " ^
  " if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }; " ^
  " $base = [System.IO.Path]::GetFileNameWithoutExtension($logFile); " ^
  " if (Test-Path $logFile) { " ^
  "   $len = (Get-Item -LiteralPath $logFile).Length; " ^
  "   if ($len -ge $maxBytes) { " ^
  "     $ts = Get-Date -Format 'yyyyMMdd_HHmmss'; " ^
  "     $rotated = Join-Path $logDir ($base + '_' + $ts + '.log'); " ^
  "     Move-Item -Force -LiteralPath $logFile -Destination $rotated; " ^
  "   } " ^
  " } " ^
  " $files = Get-ChildItem -Path $logDir -Filter ($base + '_*.log') -File | Sort-Object LastWriteTime -Descending; " ^
  " if ($files.Count -gt $retain) { $files | Select-Object -Skip $retain | ForEach-Object { Remove-Item -Force -LiteralPath $_.FullName } } " ^
  " }"

REM Append all output (stdout+stderr) to the log file for this run
echo.>> "%LOG_FILE%"
echo ===== RUN START %DATE% %TIME% args: %* =====>> "%LOG_FILE%"
python processing_runner.py %* >> "%LOG_FILE%" 2>&1
set EXIT_CODE=%ERRORLEVEL%
echo ===== RUN END %DATE% %TIME% exit_code: %EXIT_CODE% =====>> "%LOG_FILE%"

REM Store exit code (already captured above)

REM Deactivate virtual environment (optional, but clean)
call venv\Scripts\deactivate.bat

REM Exit with the same code as the Python script
if %EXIT_CODE% neq 0 (
    echo.
    echo Script exited with error code %EXIT_CODE%
    pause
)

exit /b %EXIT_CODE%
