#!/bin/bash

DRY_RUN=false

for arg in "$@"; do
    case $arg in
        --dry|-d)
            DRY_RUN=true
            ;;
    esac
done

CMD="python zbin/zrun_main.py \
    -r config/zruns/zrun_cfg.yaml \
    -opt \
    -precomputed ./zout/zoptim/MainPC__ds_UFireIndoor2__mt_no_temp_method__af4b0d32a3d2__20260316.172122"

if [ "$DRY_RUN" = true ]; then
    CMD="$CMD --dry_run"
fi

echo "[CMD] $CMD"
eval $CMD
