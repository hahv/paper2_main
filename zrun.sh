#!/bin/bash

# ──────────────────────────────────────────────
# Usage:
#   bash zrun.sh                            # normal run
#   bash zrun.sh -d                         # dry run
#   bash zrun.sh -p test                    # preset: test config
#   bash zrun.sh -p sanity                  # preset: optim sanity (UFireIndoor2)
#   bash zrun.sh -p val [-split]            # preset: optim val (UFireIndoorVal)
#   bash zrun.sh -opt -pre ./custom/path    # manual optim with custom precomputed
#   bash zrun.sh -r config/custom.yaml      # custom config
# ──────────────────────────────────────────────

# ── Defaults ──────────────────────────────────
DRY_RUN=false
OPTIM=false
SPLIT_TASK=false
HAS_PRECOM=false
PRESET="test"

CFG_DIR="config/zruns"
DEFAULT_CFG_TEST="$CFG_DIR/zrun_cfg_test.yaml"
DEFAULT_CFG_SANITY="$CFG_DIR/zrun_cfg_opt_test.yaml"
DEFAULT_CFG_VAL="$CFG_DIR/zrun_cfg_opt_val.yaml"

DEFAULT_PRECOM_SANITY="./zout/zoptim_sanity/MainPC__ds_UFireIndoor2__mt_no_temp_method__af4b0d32a3d2__20260316.172122"

DEFAULT_PRECOM_VAL="./zout/zoptim_val/MainPC__ds_UFireIndoorVal__mt_no_temp_method__af4b0d32a3d2__20260327.162459"
DEFAULT_PRECOM_NONE=""

DEFAULT_OUTDIR_SANITY="zout/zoptim_sanity"
DEFAULT_OUTDIR_VAL="zout/zoptim_val"

CONFIG_PATH="$DEFAULT_CFG_TEST"
PRECOMPUTED_PATH="$DEFAULT_PRECOM_SANITY"

# ── Inline utility ─────────────────────────────
function set_optim_outdir() {
  sed -i -E "s|OPTIM_OUTDIR = \".*\"|OPTIM_OUTDIR = \"$1\"|" src/common.py
  echo "[INFO] OPTIM_OUTDIR set to: $1"
}

function show_help() {
cat << 'EOF'
Usage: bash zrun.sh [OPTIONS]

Presets (-p):
  test      Use test dataset config
  sanity    Optim run on UFireIndoor2  (outdir: zout/zoptim_sanity)
  val       Optim run on UFireIndoorVal (outdir: zout/zoptim_val), supports -split

Options:
  -p,  --preset   <name>   Load a named preset (test | sanity | val)
  -r,  --run_cfg  <path>   Path to run config YAML
  -opt,--optim             Enable optim mode
  -pre,--precom   <path>   Path to precomputed output
  -split,--splittask       Enable split task mode
  -d,  --dry               Dry run (no execution)
  -h,  --help              Show this help
EOF
}

# ── Parse arguments ────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --help|-h)          show_help; exit 0 ;;
    --dry|-d)           DRY_RUN=true; shift ;;
    --optim|-opt)       OPTIM=true; shift ;;
    --preset|-p)        PRESET="$2"; shift 2 ;;
    --precom|-pre)      PRECOMPUTED_PATH="$2"; HAS_PRECOM=true; shift 2 ;;
    -r|--run_cfg)       CONFIG_PATH="$2"; shift 2 ;;
    -split|--splittask) SPLIT_TASK=true; shift ;;
    *)                  shift ;;
  esac
done

# ── Apply preset ───────────────────────────────
case "$PRESET" in
  test)
    CONFIG_PATH="$DEFAULT_CFG_TEST"
    echo "[INFO] Preset: test"
    ;;
  sanity)
    CONFIG_PATH="$DEFAULT_CFG_SANITY"
    OPTIM=true
    PRECOMPUTED_PATH="$DEFAULT_PRECOM_SANITY"
    set_optim_outdir "$DEFAULT_OUTDIR_SANITY"
    echo "[INFO] Preset: sanity"
    ;;
  val)
    CONFIG_PATH="$DEFAULT_CFG_VAL"
    OPTIM=true
    PRECOMPUTED_PATH="$DEFAULT_PRECOM_VAL"
    set_optim_outdir "$DEFAULT_OUTDIR_VAL"
    echo "[INFO] Preset: val"
    ;;
  "")
    ;;  # No preset, use parsed flags as-is
  *)
    echo "[ERROR] Unknown preset: '$PRESET'. Valid: test | sanity | val"; exit 1 ;;
esac

# ── Build command ──────────────────────────────
CMD="python zbin/zrun_main.py -r \"$CONFIG_PATH\""

if [ "$OPTIM" = true ]; then
  echo "[INFO] Mode: OPTIM"
  CMD="$CMD -opt -precomputed \"$PRECOMPUTED_PATH\""
else
  echo "[INFO] Mode: NORMAL"
  if [ "$HAS_PRECOM" = true ]; then
    CMD="$CMD -precomputed \"$PRECOMPUTED_PATH\""
  fi
fi

[ "$DRY_RUN"    = true ] && CMD="$CMD --dry_run"
[ "$SPLIT_TASK" = true ] && CMD="$CMD -split"

echo "[CMD] $CMD"
eval $CMD