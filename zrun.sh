#!/bin/bash

DRY_RUN=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --dry|-d)
            DRY_RUN=true
            shift
            ;;
    esac
done

CMD="python zbin/zrun_main.py -r config/zruns/zrun_cfg.yaml"

if [ "$DRY_RUN" = true ]; then
    CMD="$CMD --dry-run"
    echo "[DRY RUN] $CMD"
fi

eval $CMD
