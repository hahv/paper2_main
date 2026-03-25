#!/bin/bash
#
# Typical Example Usages:
#   1. Normal report: bash zreport.sh
#   2. Optim report (default path): bash zreport.sh -opt
#   3. Optim report (custom path): bash zreport.sh -opt -in ./my/custom/path
#   4. Force regenerate CSVs: bash zreport.sh -f
#   5. Dry run: bash zreport.sh -opt -d
#

OPTIM=false
DRY_RUN=false
FORCE=false
INDIR=""

function show_help() {
    cat << 'EOF'
Typical Example Usages:
  1. Normal report: bash zreport.sh
  2. Optim report (default path): bash zreport.sh -opt
  3. Optim report (custom path): bash zreport.sh -opt -in ./my/custom/path
  4. Force regenerate CSVs: bash zreport.sh -f
  5. Dry run: bash zreport.sh -opt -d
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            exit 0
            ;;
        --optim|-opt)
            OPTIM=true
            shift
            ;;
        --dry|-d)
            DRY_RUN=true
            shift
            ;;
        --force|-f)
            FORCE=true
            shift
            ;;
        --indir|-in)
            INDIR="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

if [ "$OPTIM" = true ]; then
    echo "[INFO] Running in OPTIM mode"
    if [ -z "$INDIR" ]; then
        INDIR="./zout/zoptim"
    fi
    CMD="python zbin/rp/run_report.py -i \"$INDIR\" -o \"$INDIR\" -opt"
else
    echo "[INFO] Running in NORMAL mode"
    if [ -z "$INDIR" ]; then
        INDIR="./zout/reports/baseline_compare_perf"
    fi
    CMD="python zbin/rp/run_report.py -i \"$INDIR\" -o \"$INDIR\""
fi

if [ "$DRY_RUN" = true ]; then
    CMD="$CMD --dry_run"
fi

if [ "$FORCE" = true ]; then
    CMD="$CMD -f"
fi

echo "[CMD] $CMD"
eval $CMD