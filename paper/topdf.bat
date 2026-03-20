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

:: Build pandoc command
set "PANDOC_CMD=pandoc "%FILE_NAME%" -o "out\%FILE_BASE%.pdf" --include-in-header="0.tex\preamble.tex" --pdf-engine=xelatex --metadata link-citations=true -V colorlinks=true"

:: Auto-detect optional files in 2.ref\ subfolder
if exist "2.ref\refs.bib"  set "PANDOC_CMD=%PANDOC_CMD% --citeproc --bibliography="2.ref\refs.bib""
if exist "2.ref\style.csl" set "PANDOC_CMD=%PANDOC_CMD% --csl="2.ref\style.csl""

:: Run pandoc
%PANDOC_CMD%

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
