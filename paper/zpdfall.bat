@echo off
setlocal

REM Get ESC character for ANSI colors
for /f %%a in ('echo prompt $E ^| cmd') do set "ESC=%%a"

REM Step 1
echo %ESC%[91m[STEP 1] Syncing blocks in zpaper2.md...%ESC%[0m
call sync_blocks.bat ./zpaper2.md

REM Step 2
echo %ESC%[91m[STEP 2] Changing to main paper directory...%ESC%[0m
cd /d "G:\My Drive\1_PhD\Obsidian\Home\3. Writing\paperfire2"

REM Step 3
echo %ESC%[91m[STEP 3] Building PDF...%ESC%[0m
call pd

endlocal
