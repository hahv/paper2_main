@echo off
setlocal

REM Get ESC character for ANSI colors
for /f %%a in ('echo prompt $E ^| cmd') do set "ESC=%%a"

REM ── VENV ────────────────────────────────────────────────────────────────────
echo %ESC%[93m[VENV] Activating Python virtual environment...%ESC%[0m
call C:\batch\.venv\Scripts\activate.bat
if errorlevel 1 (
    echo %ESC%[31m[ERROR] Failed to activate Python venv. Aborting.%ESC%[0m
    exit /b 1
)

REM ── STEP 1 ──────────────────────────────────────────────────────────────────
echo %ESC%[91m[STEP 1] Syncing blocks in zpaper2.md...%ESC%[0m
call C:\batch\batscript\sync_blocks.bat "%~dp0zpaper2.md"
if errorlevel 1 (
    echo %ESC%[31m[ERROR] sync_blocks.bat failed. Aborting.%ESC%[0m
    exit /b 1
)

REM ── STEP 2 ──────────────────────────────────────────────────────────────────
echo %ESC%[91m[STEP 2] Changing to main paper directory...%ESC%[0m
cd /d "G:\My Drive\1_PhD\Obsidian\Home\3. Writing\paperfire2"
if errorlevel 1 (
    echo %ESC%[31m[ERROR] Could not change directory. Aborting.%ESC%[0m
    exit /b 1
)

REM ── STEP 3 ──────────────────────────────────────────────────────────────────
echo %ESC%[91m[STEP 3] Building PDF...%ESC%[0m
call C:\batch\task\pandoc\pd.bat
if errorlevel 1 (
    echo %ESC%[31m[ERROR] pd failed.%ESC%[0m
    exit /b 1
)

echo %ESC%[92m[DONE] PDF built successfully.%ESC%[0m
endlocal