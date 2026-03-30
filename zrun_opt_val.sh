#!/bin/bash
source ./zutils.sh

SPLIT_ARG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --splittask|-split)
            SPLIT_ARG="-split"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Temporarily modify OPTIM_OUTDIR in common.py
set_optim_outdir "zout/zoptim"

./zrun.sh \
    -r config/zruns/zrun_cfg_opt_val.yaml \
    -opt \
    -pre ./zout/zoptim/MainPC__ds_UFireIndoorVal__mt_no_temp_method__af4b0d32a3d2__20260327.162459 \
    $SPLIT_ARG

# Revert OPTIM_OUTDIR back to the default
set_optim_outdir "zout/zoptim"