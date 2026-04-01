#!/bin/bash

# ──────────────────────────────────────────────
# Usage:
#   bash zreport.sh                         # normal report
#   bash zreport.sh -opt                    # optim report (default indir)
#   bash zreport.sh -opt -in ./custom/path  # optim report (custom indir)
#   bash zreport.sh -p sanity               # preset: sanity optim report
#   bash zreport.sh -p val [-in ./override] # preset: val optim report
#   bash zreport.sh -f                      # force regenerate CSVs
#   bash zreport.sh -d                      # dry run
# ──────────────────────────────────────────────

# ── Defaults ──────────────────────────────────
OPTIM=false
DRY_RUN=false
FORCE=false
INDIR=""
PRESET=""

DEFAULT_INDIR_VAL="zout/zoptim_val"
DEFAULT_INDIR_SANITY="zout/zoptim_sanity"
DEFAULT_INDIR_NORMAL="./zout/reports/baseline_compare_perf"

function show_help() {
cat << 'EOF'
Usage: bash zreport.sh [OPTIONS]

Presets (-p):
  sanity    Optim report for sanity run  (indir: zout/zoptim_sanity)
  val       Optim report for val run     (indir: zout/zoptim_val)

Options:
  -p,  --preset  <name>   Load a named preset (sanity | val)
  -opt,--optim            Enable optim report mode
  -in, --indir   <path>   Input (and output) directory
  -f,  --force            Force regenerate CSVs
  -d,  --dry              Dry run (no execution)
  -h,  --help             Show this help
EOF
}

# ── Parse arguments ────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --help|-h)    show_help; exit 0 ;;
    --optim|-opt) OPTIM=true; shift ;;
    --dry|-d)     DRY_RUN=true; shift ;;
    --force|-f)   FORCE=true; shift ;;
    --preset|-p)  PRESET="$2"; shift 2 ;;
    --indir|-in)  INDIR="$2"; shift 2 ;;
    *)            shift ;;
  esac
done

# ── Apply preset ───────────────────────────────
case "$PRESET" in
  sanity)
    OPTIM=true
    [ -z "$INDIR" ] && INDIR="$DEFAULT_INDIR_SANITY"
    echo "[INFO] Preset: sanity"
    ;;
  val)
    OPTIM=true
    [ -z "$INDIR" ] && INDIR="$DEFAULT_INDIR_VAL"
    echo "[INFO] Preset: val"
    ;;
  "")
    ;;
  *)
    echo "[ERROR] Unknown preset: '$PRESET'. Valid: sanity | val"; exit 1 ;;
esac

# ── Resolve default dirs ───────────────────────
if [ "$OPTIM" = true ]; then
  echo "[INFO] Mode: OPTIM"
  [ -z "$INDIR" ] && INDIR="$DEFAULT_INDIR_VAL"
  CMD="python zbin/rp/run_report.py -i \"$INDIR\" -o \"$INDIR\" -opt"
else
  echo "[INFO] Mode: NORMAL"
  [ -z "$INDIR" ] && INDIR="$DEFAULT_INDIR_NORMAL"
  CMD="python zbin/rp/run_report.py -i \"$INDIR\" -o \"$INDIR\""
fi

[ "$DRY_RUN" = true ] && CMD="$CMD --dry_run"
[ "$FORCE"   = true ] && CMD="$CMD -f"

echo "[CMD] $CMD"
eval $CMD