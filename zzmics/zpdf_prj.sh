BASE_DIR="/mnt/e/SyncData/paper2_main/paper"
TEX_FILE="/mnt/e/SyncData/paper2_main/paper/out/zpaper2.tex"
OUT_DIR="/mnt/e/SyncData/paper2_main/paper/out"

python -m halib.utils.tex_op \
    -b "$BASE_DIR" \
    -m "$TEX_FILE" \
    -o "$OUT_DIR"