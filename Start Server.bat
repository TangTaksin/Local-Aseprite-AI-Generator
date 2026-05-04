@echo off
setlocal enabledelayedexpansion

:: Local AI Generator - Simplified Setup Script
:: =====================================================

echo.
echo ========================================================
echo    Local AI Generator for Aseprite - Auto Setup
echo ========================================================
echo.

:: Get the directory where this batch file is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo Current directory: %SCRIPT_DIR%

:: Check if required files exist
echo [1/4] Checking required files...
if not exist "startup_script.py" (
    echo ✗ startup_script.py not found in current directory
    echo Please ensure all files are in the same folder
    pause
    exit /b 1
)

if not exist "sd_server.py" (
    if exist "paste.txt" (
        echo Found paste.txt - copying to sd_server.py...
        copy "paste.txt" "sd_server.py" >nul 2>&1
        echo ✓ Created sd_server.py from paste.txt
    ) else (
        echo ✗ Neither sd_server.py nor paste.txt found
        pause
        exit /b 1
    )
) else (
    echo ✓ sd_server.py found
)

:: Check if Python is installed
echo [2/4] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ Python is already installed
    python --version
) else (
    echo ✗ Python not found. 
    echo.
    echo PLEASE INSTALL PYTHON MANUALLY:
    echo 1. Go to https://python.org
    echo 2. Download Python 3.8 or newer
    echo 3. During installation, CHECK "Add Python to PATH"
    echo 4. Restart your computer after installation
    echo 5. Run this script again
    echo.
    pause
    exit /b 1
)

:: Create virtual environment
echo [3/4] Setting up virtual environment...
if exist "venv\Scripts\activate.bat" (
    echo ✓ Virtual environment already exists
) else (
    echo Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ✗ Failed to create virtual environment
        echo Try running as Administrator
        pause
        exit /b 1
    )
    echo ✓ Virtual environment created
)

:: Create necessary directories
echo [4/4] Setting up directories...
if not exist "loras" mkdir loras
if not exist "models" mkdir models
echo ✓ Directories created

echo.
echo ========================================================
echo              Basic Setup Complete!
echo ========================================================
echo.
echo Now starting Python setup and server...
echo Python will handle dependency installation and server startup.
echo.

:: Activate virtual environment and start Python setup
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ✗ Failed to activate virtual environment
    pause
    exit /b 1
)

echo ✓ Virtual environment activated
echo Starting Python setup script...
echo.

:: Let Python handle everything from here
python startup_script.py

:: If we get here, the server has stopped
echo.
echo ========================================================
echo                   SETUP/SERVER ENDED
echo ========================================================
echo.
pause
exit /b 0