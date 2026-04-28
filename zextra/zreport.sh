#!/bin/bash
set -euo pipefail

# ──────────────────────────────────────────────
# Usage:
#   bash zreport.sh                         # normal report (default indir)
#   bash zreport.sh -opt                    # optim report (default: zout/zoptim_val)
#   bash zreport.sh -opt -in ./custom/path  # optim report (custom indir)
#   bash zreport.sh -p sanity               # preset: sanity optim report
#   bash zreport.sh -p val [-in ./override] # preset: val optim report (override indir)
#   bash zreport.sh -f                      # force regenerate CSVs
#   bash zreport.sh -d                      # dry run
# ──────────────────────────────────────────────

# ── Paths ──────────────────────────────────────
DEFAULT_INDIR_VAL="zout/zoptim_val"
DEFAULT_INDIR_SANITY="zout/zoptim_sanity"
DEFAULT_INDIR_NORMAL="./zout/reports/baseline_compare_perf"

# ── State ──────────────────────────────────────
OPTIM=false
DRY_RUN=false
FORCE=false
PRESET=""
USER_INDIR=""   # explicit -in value; empty = not provided

# ── Utilities ──────────────────────────────────
function show_help() {
cat << 'EOF'
Usage: bash zreport.sh [OPTIONS]

Presets (-p):
  sanity    Optim report for sanity run  (indir: zout/zoptim_sanity)
  val       Optim report for val run     (indir: zout/zoptim_val)

Options:
  -p,  --preset  <name>   Load a named preset (sanity | val)
  -opt,--optim            Enable optim report mode
  -in, --indir   <path>   Input/output directory  [overrides preset default]
  -f,  --force            Force regenerate CSVs
  -d,  --dry              Dry run (no execution)
  -h,  --help             Show this help
EOF
}

# ── Parse arguments ────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --help|-h)    show_help; exit 0 ;;
    --optim|-opt) OPTIM=true;       shift ;;
    --dry|-d)     DRY_RUN=true;     shift ;;
    --force|-f)   FORCE=true;       shift ;;
    --preset|-p)  PRESET="$2";      shift 2 ;;
    --indir|-in)  USER_INDIR="$2";  shift 2 ;;
    *)            echo "[WARN] Unknown argument: '$1'"; shift ;;
  esac
done

# ── Apply preset (sets defaults; user -in overrides after) ──
case "$PRESET" in
  "")
    ;;
  sanity)
    echo "[INFO] Preset: sanity"
    OPTIM=true
    INDIR="$DEFAULT_INDIR_SANITY"
    ;;
  val)
    echo "[INFO] Preset: val"
    OPTIM=true
    INDIR="$DEFAULT_INDIR_VAL"
    ;;
  *)
    echo "[ERROR] Unknown preset: '$PRESET'. Valid: sanity | val"
    exit 1
    ;;
esac

# ── User -in overrides preset default ─────────
if [ -n "$USER_INDIR" ]; then
  INDIR="$USER_INDIR"
elif [ -z "${INDIR:-}" ]; then
  # No preset and no -in: resolve from mode
  if [ "$OPTIM" = true ]; then
    INDIR="$DEFAULT_INDIR_VAL"
    echo "[INFO] No indir specified — defaulting to: $INDIR"
  else
    INDIR="$DEFAULT_INDIR_NORMAL"
  fi
fi

# ── Build command (array — safe for paths with spaces) ──
CMD=(python zbin/rp/run_report.py -i "$INDIR" -o "$INDIR")

if [ "$OPTIM" = true ]; then
  CMD+=(-opt)
fi

[ "$DRY_RUN" = true ] && CMD+=(--dry_run)
[ "$FORCE"   = true ] && CMD+=(-f)

# ── Summary ────────────────────────────────────
echo "────────────────────────────────────────────"
printf "[CFG] %-10s %s\n" "mode:"    "$( [ "$OPTIM" = true ] && echo 'OPTIM' || echo 'NORMAL' )"
printf "[CFG] %-10s %s\n" "indir:"   "$INDIR"
[ "$FORCE"   = true ] && printf "[CFG] %-10s %s\n" "force:"   "enabled"
[ "$DRY_RUN" = true ] && printf "[CFG] %-10s %s\n" "dry_run:" "enabled"
echo "[CMD] ${CMD[*]}"
echo "────────────────────────────────────────────"

# ── Execute ────────────────────────────────────
"${CMD[@]}"