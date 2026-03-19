@echo off
setlocal

:: Check if argument is provided
if "%~1"=="" (
    echo Usage: topdf.bat "path\to\file.md"
    pause
    exit /b 1
)

:: Get components from the input path
set "INPUT_FILE=%~1"
set "FILE_DIR=%~dp1"
set "FILE_NAME=%~nx1"
set "FILE_BASE=%~n1"

:: Go to the parent folder of the .md file
cd /d "%FILE_DIR%"

:: Create out\ subdir if it doesn't exist
if not exist "out\" mkdir "out"

:: Run pandoc — output goes to <input_dir>\out\<name>.pdf
pandoc "%FILE_NAME%" -o "out\%FILE_BASE%.pdf" --include-in-header="preamble.tex" --pdf-engine=xelatex

if %errorlevel%==0 (
    echo.
    echo [OK] PDF generated: %FILE_DIR%out\%FILE_BASE%.pdf
    start "" "%FILE_DIR%out\%FILE_BASE%.pdf"
) else (
    echo.
    echo [ERROR] Pandoc failed. Check the output above.
)

pause
endlocal
