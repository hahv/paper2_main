#!/bin/bash

# Utility functions for shell scripts

function set_optim_outdir() {
    local new_dir="$1"
    # Replace the OPTIM_OUTDIR value in src/common.py with the provided directory
    sed -i -E "s|OPTIM_OUTDIR = \".*\"|OPTIM_OUTDIR = \"$new_dir\"|" src/common.py
}
