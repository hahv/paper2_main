#!/bin/bash

# ==========================================
# 1. WSL STEP
# ==========================================
echo "--> [1/2] Running extraction in WSL..."

# Run the python script. If it fails, stop the whole script.
if ! python zz_fig_extract.py -f; then
    echo "Error: WSL extraction failed. Aborting."
    exit 1
fi
echo "WSL extraction successful!"


# ==========================================
# 2. WINDOWS STEPS
# ==========================================
echo "--> [2/2] Running quality analysis in Windows..."

# The backslash (\) at the end of each line tells Bash to combine these 
# lines into one continuous command before sending it to cmd.exe. 
# You can easily add, remove, or modify steps here.

cmd.exe /c " \
    cd /d E:\Dev\__halib && \
    call .\.venv\Scripts\activate.bat && \
    cd /d E:\SyncData\paper2_main && \
    python paper\3.fig\quality_analyses\mk_quality_fig.py -i fig_qualitative_failure.yaml && \
    python paper\3.fig\quality_analyses\mk_quality_fig.py -i fig_qualitative_success.yaml \
"

echo "--> Pipeline complete!"