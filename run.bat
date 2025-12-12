@echo off
REM Batch file to activate virtual environment and run processing_runner.py
REM Usage: run.bat [arguments]
REM Example: run.bat --test
REM Example: run.bat --reset

REM Get the directory where this batch file is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

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
python processing_runner.py %*

REM Store exit code
set EXIT_CODE=%ERRORLEVEL%

REM Deactivate virtual environment (optional, but clean)
call venv\Scripts\deactivate.bat

REM Exit with the same code as the Python script
if %EXIT_CODE% neq 0 (
    echo.
    echo Script exited with error code %EXIT_CODE%
    pause
)

exit /b %EXIT_CODE%
