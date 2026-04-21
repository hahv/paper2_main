#!/bin/bash
set -euo pipefail

# ──────────────────────────────────────────────
# Usage:
#   bash zrun.sh                            # normal run (test config, no precomputed)
#   bash zrun.sh -d                         # dry run (passes --dry_run to Python)
#   bash zrun.sh -p test                    # preset: test config + precomputed
#   bash zrun.sh -p sanity                  # preset: optim sanity (UFireIndoor2)
#   bash zrun.sh -p val [-split]            # preset: optim val (UFireIndoorVal)
#   bash zrun.sh -opt -pre ./custom/path    # manual optim with custom precomputed
#   bash zrun.sh -r config/custom.yaml      # custom config (overrides preset config)
# ──────────────────────────────────────────────

# ── Paths ──────────────────────────────────────
CFG_DIR="config/zruns"
DEFAULT_CFG_TEST="$CFG_DIR/zrun_cfg_test.yaml"
DEFAULT_CFG_SANITY="$CFG_DIR/zrun_cfg_opt_test.yaml"
DEFAULT_CFG_VAL="$CFG_DIR/zrun_cfg_opt_val.yaml"

DEFAULT_PRECOM_SANITY="./zout/zoptim_sanity/MainPC__ds_UFireIndoor2__mt_no_temp_method__af4b0d32a3d2__20260316.172122"
DEFAULT_PRECOM_VAL="./zout/zruns/_precomputed_NoTemp/MainPC__ds_UFireIndoorVal__mt_no_temp_method__af4b0d32a3d2__20260421.140547"
DEFAULT_PRECOM_TEST="./zout/zruns/_precomputed_NoTemp/MainPC__ds_UFireIndoorTest__mt_no_temp_method__af4b0d32a3d2__20260330.205208"


DEFAULT_OUTDIR_SANITY="zout/zoptim_sanity"
DEFAULT_OUTDIR_VAL="zout/zoptim_val"

# ── State ──────────────────────────────────────
DRY_RUN=false
OPTIM=false
SPLIT_TASK=false
PRESET=""

# Resolved values (set by preset, then optionally overridden by user flags)
CONFIG_PATH="$DEFAULT_CFG_TEST"
PRECOMPUTED_PATH=""

# Track explicit user-provided overrides (empty = not provided)
USER_CFG=""
USER_PRECOM=""

# ── Utilities ──────────────────────────────────
function set_optim_outdir() {
  sed -i -E "s|OPTIM_OUTDIR = \".*\"|OPTIM_OUTDIR = \"$1\"|" src/common.py
  echo "[INFO] OPTIM_OUTDIR set to: $1"
}

function show_help() {
cat << 'EOF'
Usage: bash zrun.sh [OPTIONS]

Presets (-p):
  test      Test config with default precomputed path (no optim)
  sanity    Optim run on UFireIndoor2   (outdir: zout/zoptim_sanity)
  val       Optim run on UFireIndoorVal (outdir: zout/zoptim_val), supports -split

Options:
  -p,  --preset   <name>   Load a named preset (test | sanity | val)
  -r,  --run_cfg  <path>   Path to run config YAML  [overrides preset config]
  -opt,--optim             Enable optim mode
  -pre,--precom   <path>   Path to precomputed output  [overrides preset default]
  -split,--splittask       Enable split task mode
  -d,  --dry               Dry run (passes --dry_run to Python)
  -h,  --help              Show this help
EOF
}

# ── Parse arguments ────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --help|-h)           show_help; exit 0 ;;
    --dry|-d)            DRY_RUN=true;         shift ;;
    --optim|-opt)        OPTIM=true;           shift ;;
    --preset|-p)         PRESET="$2";          shift 2 ;;
    --precom|-pre)       USER_PRECOM="$2";     shift 2 ;;
    -r|--run_cfg)        USER_CFG="$2";        shift 2 ;;
    -split|--splittask)  SPLIT_TASK=true;      shift ;;
    *)                   echo "[WARN] Unknown argument: '$1'"; shift ;;
  esac
done

# ── Apply preset (sets defaults; user flags override after) ──
case "$PRESET" in
  "")
    # No preset: plain run, no precomputed passed
    echo "[INFO] No preset — using test config (no precomputed)"
    ;;
  test)
    echo "[INFO] Preset: test"
    CONFIG_PATH="$DEFAULT_CFG_TEST"
    PRECOMPUTED_PATH="$DEFAULT_PRECOM_TEST"
    ;;
  sanity)
    echo "[INFO] Preset: sanity"
    CONFIG_PATH="$DEFAULT_CFG_SANITY"
    PRECOMPUTED_PATH="$DEFAULT_PRECOM_SANITY"
    OPTIM=true
    set_optim_outdir "$DEFAULT_OUTDIR_SANITY"
    ;;
  val)
    echo "[INFO] Preset: val"
    CONFIG_PATH="$DEFAULT_CFG_VAL"
    PRECOMPUTED_PATH="$DEFAULT_PRECOM_VAL"
    OPTIM=true
    set_optim_outdir "$DEFAULT_OUTDIR_VAL"
    ;;
  *)
    echo "[ERROR] Unknown preset: '$PRESET'. Valid: test | sanity | val"
    exit 1
    ;;
esac

# ── User flags override preset defaults ────────
[ -n "$USER_CFG" ]    && CONFIG_PATH="$USER_CFG"
[ -n "$USER_PRECOM" ] && PRECOMPUTED_PATH="$USER_PRECOM"

# ── Validate ───────────────────────────────────
if [ "$OPTIM" = true ] && [ -z "$PRECOMPUTED_PATH" ]; then
  echo "[ERROR] Optim mode requires a precomputed path. Provide -pre <path> or use a preset."
  exit 1
fi

# ── Build command (array — safe for paths with spaces) ──
CMD=(python zbin/zrun_main.py -r "$CONFIG_PATH")

if [ "$OPTIM" = true ]; then
  CMD+=(-opt -precomputed "$PRECOMPUTED_PATH")
elif [ -n "$PRECOMPUTED_PATH" ]; then
  CMD+=(-precomputed "$PRECOMPUTED_PATH")
fi

[ "$DRY_RUN"   = true ] && CMD+=(--dry_run)
[ "$SPLIT_TASK" = true ] && CMD+=(-split)

# ── Summary ────────────────────────────────────
echo "────────────────────────────────────────────"
printf "[CFG] %-14s %s\n" "config:"      "$CONFIG_PATH"
[ -n "$PRECOMPUTED_PATH" ] && \
  printf "[CFG] %-14s %s\n" "precomputed:" "$PRECOMPUTED_PATH"
[ "$OPTIM"      = true  ] && printf "[CFG] %-14s %s\n" "mode:"     "OPTIM"   \
                          || printf "[CFG] %-14s %s\n" "mode:"     "NORMAL"
[ "$SPLIT_TASK" = true  ] && printf "[CFG] %-14s %s\n" "split:"    "enabled"
[ "$DRY_RUN"    = true  ] && printf "[CFG] %-14s %s\n" "dry_run:"  "enabled"
echo "[CMD] ${CMD[*]}"
echo "────────────────────────────────────────────"

# ── Execute ────────────────────────────────────
"${CMD[@]}"