@echo off
REM Production Trading System Startup Script
REM Windows batch file to run the trading system

echo ================================================
echo PRODUCTION TRADING SYSTEM
echo ================================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if virtual environment should be activated
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Set environment variables if .env file exists
if exist ".env" (
    echo Loading environment variables from .env file...
    for /f "usebackq tokens=1,2 delims==" %%a in (".env") do (
        set "%%a=%%b"
    )
)

REM Default command line arguments
set DRY_RUN=--dry-run
set LOG_LEVEL=--log-level INFO

REM Parse command line arguments
:parse_args
if "%1"=="--live" (
    set DRY_RUN=
    shift
    goto parse_args
)
if "%1"=="--debug" (
    set LOG_LEVEL=--log-level DEBUG
    shift
    goto parse_args
)
if "%1"=="--validate" (
    set VALIDATE_ONLY=--validate-only
    shift
    goto parse_args
)

REM Show configuration
echo.
echo Configuration:
if defined DRY_RUN (
    echo   Mode: DRY RUN ^(paper trading, no real trades^)
) else (
    echo   Mode: LIVE TRADING ^(real money!^)
)
echo   Log Level: %LOG_LEVEL:~12%

if not defined DRY_RUN (
    echo.
    echo WARNING: You are about to run LIVE TRADING!
    echo This will execute real trades with real money.
    echo Press Ctrl+C to cancel, or any key to continue...
    pause >nul
)

echo.
echo Starting trading system...
echo ================================================

REM Run the trading system
python run_production_trading.py %DRY_RUN% %LOG_LEVEL% %VALIDATE_ONLY% %*

REM Check exit code
if errorlevel 1 (
    echo.
    echo ERROR: Trading system failed with error code %errorlevel%
    echo Check the logs for details.
) else (
    echo.
    echo Trading system completed successfully.
)

echo.
echo Press any key to exit...
pause >nul
