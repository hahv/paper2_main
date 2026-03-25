#!/bin/bash
#
# Typical Example Usages:
#   1. Normal run: bash zrun.sh
#   2. Dry run: bash zrun.sh -d
#   3. Optim mode (default precomputed): bash zrun.sh -opt
#   4. Optim mode (custom precomputed): bash zrun.sh -opt -pre ./custom/path
#

DRY_RUN=false
OPTIM=false
PRECOMPUTED_PATH="./zout/zoptim/MainPC__ds_UFireIndoor2__mt_no_temp_method__af4b0d32a3d2__20260316.172122"

function show_help() {
    cat << 'EOF'
Typical Example Usages:
  1. Normal run: bash zrun.sh
  2. Dry run: bash zrun.sh -d
  3. Optim mode (default precomputed): bash zrun.sh -opt
  4. Optim mode (custom precomputed): bash zrun.sh -opt -pre ./custom/path
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            exit 0
            ;;
        --dry|-d)
            DRY_RUN=true
            shift
            ;;
        --optim|-opt)
            OPTIM=true
            shift
            ;;
        --precom|-pre)
            PRECOMPUTED_PATH="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

CMD="python zbin/zrun_main.py -r config/zruns/zrun_cfg.yaml"

if [ "$OPTIM" = true ]; then
    echo "[INFO] Running in OPTIM mode"
    CMD="$CMD -opt -precomputed \"$PRECOMPUTED_PATH\""
else
    echo "[INFO] Running in NORMAL mode"
fi

if [ "$DRY_RUN" = true ]; then
    CMD="$CMD --dry_run"
fi

echo "[CMD] $CMD"
eval $CMD
