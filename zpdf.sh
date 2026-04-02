#!/bin/bash

# ---------------------------------------------------------------------------
# Script: zpdf.sh
# Description: Converts the main paper markdown to PDF using to_pdf.
#              By default, runs the standard build process inside WSL.
#
# Arguments:
#   -a, --all    Executes the Windows 'zpdfall.bat' file instead, which
#                syncs blocks and builds the full PDF project on Windows.
# ---------------------------------------------------------------------------
cd /mnt/e/SyncData/paper2_main/paper || exit 1

if [[ "$1" == "-a" || "$1" == "--all" ]]; then
    BAT_PATH=$(wslpath -w /mnt/e/SyncData/paper2_main/paper/zpdfall.bat)

    echo "Running Windows batch script: zpdfall.bat"
    powershell.exe -NoProfile -Command "
        \$env:PATH = [System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path','User');
        cmd.exe /c '$BAT_PATH'
    "
else
    echo "Running standard to_pdf..."
    to_pdf ./zpaper2.md
fi