from idna.idnadata import scripts
from pylab import False_
import shutil
import random
from halib import *
from halib.common.common import seed_everything
from tap import Tap

# vis
from halib.exp.viz.plot import PlotHelper as plth


class PlotVideoArgs(Tap):
    in_dir: str = "./zout/reports/viz/vdb_screenshots"  # Input dir
    seed: int = 1  # Random seed for reproducibility
    num_frame_per_cate: int = (
        5  # Number of frames to sample from each category for visualization
    )


RAW_SRC_IMG_DIR = "raw"
SOURCE_CATE = ["fire", "smoke", "none"]
TARGET_CATE = ["firesmoke", "none"]
mapping_src_to_tgt = {"fire": "firesmoke", "smoke": "firesmoke", "none": "none"}
FILE_NAME_SEP = "__"
TEMP_SRC_DIR = "src"


@log_func(log_time=True)
def mk_grid_plot(in_dir, num_frame_per_cate):
    """
    1. Scan raw/ frames, parse (src_cate, data_source) from filenames.
    2. Map each src_cate → target_cate and group candidates.
    3. Round-robin select num_frame_per_cate images per target_cate for diversity.
    4. Copy selected images into src/{target_cate}/.
    5. Build a grid figure (cols = target categories, rows = frames) and save it.
    Returns the path of the saved grid image.
    """
    raw_dir = os.path.join(in_dir, RAW_SRC_IMG_DIR)
    src_dir = os.path.join(in_dir, TEMP_SRC_DIR)
    # !FORCE CLEANUP
    if os.path.exists(src_dir):
        shutil.rmtree(src_dir)
    os.makedirs(src_dir, exist_ok=True)

    # ── Step 1: Collect all frame images grouped by (target_cate, src_cate, data_source) ──
    # Structure: {target_cate: {(src_cate, data_source): [img_path, ...]}}
    # Filename format: {data_source}__{src_cate}__{video_name}__frm{idx}.jpg
    # Stored under:    raw/{src_cate}/
    grouped = {tgt: {} for tgt in TARGET_CATE}

    def raw_cate_dir_to_src_cate(raw_cate_dir):
        for src_cate in SOURCE_CATE:
            if src_cate in raw_cate_dir:
                return src_cate
        return None

    for raw_cate_dir in sorted(os.listdir(raw_dir)):
        if not os.path.isdir(os.path.join(raw_dir, raw_cate_dir)):
            continue
        src_cate = raw_cate_dir_to_src_cate(raw_cate_dir)
        assert src_cate is not None, (
            f"src_cate not found in directory name: {raw_cate_dir}"
        )
        tgt_cate = mapping_src_to_tgt[src_cate]
        full_cate_dir = os.path.join(raw_dir, raw_cate_dir)
        for fname in sorted(os.listdir(full_cate_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            # Parse data_source from the first segment of the filename
            parts = fname.split(FILE_NAME_SEP)
            if len(parts) < 2:
                continue
            data_source = parts[0]
            key = (src_cate, data_source)
            grouped[tgt_cate].setdefault(key, []).append(
                os.path.join(full_cate_dir, fname)
            )

    # ── Step 2: Create src/ subdirectory for each target category ──
    for tgt_cate in TARGET_CATE:
        os.makedirs(os.path.join(src_dir, f"{tgt_cate}"), exist_ok=True)

    # ── Step 3: Diversified selection via round-robin across (src_cate, data_source) groups ──
    # Cycles through every unique (src_cate, data_source) key in order so that
    # the selected frames are spread across as many sources as possible.
    selected = {tgt: [] for tgt in TARGET_CATE}

    for tgt_cate in TARGET_CATE:
        groups = grouped[tgt_cate]
        if not groups:
            pprint(f"[warn] no frames found for target category: {tgt_cate}")
            continue

        # Sort keys first for a stable base, then shuffle once for diversity
        group_keys = sorted(groups.keys())
        random.shuffle(group_keys)  # one-time shuffle; seeded via seed_everything
        # Keep a per-group index pointer instead of mutating the lists
        group_ptrs = {k: 0 for k in group_keys}
        exhausted = set()
        count = 0
        key_cycle_idx = 0

        while count < num_frame_per_cate and len(exhausted) < len(group_keys):
            k = group_keys[key_cycle_idx % len(group_keys)]
            key_cycle_idx += 1
            if k in exhausted:
                continue
            ptr = group_ptrs[k]
            if ptr >= len(groups[k]):
                exhausted.add(k)
                continue
            selected[tgt_cate].append((k, groups[k][ptr]))
            group_ptrs[k] += 1
            count += 1

        pprint(
            f"[{tgt_cate}] selected {len(selected[tgt_cate])}/{num_frame_per_cate} frames "
            f"from {len(group_keys)} (src_cate, data_source) groups"
        )

    # ── Step 4: Copy selected frames into src/{target_cate}/ ──
    for tgt_cate, items in selected.items():
        tgt_subdir = os.path.join(src_dir, tgt_cate)
        for (src_cate, data_source), img_path in items:
            shutil.copy2(img_path, tgt_subdir)
    # !add "row_" prefix for all folder names in src to allow PlotHelper work properly
    for tgt_cate in TARGET_CATE:
        src_subdir = os.path.join(src_dir, tgt_cate)
        new_subdir = os.path.join(src_dir, f"row_{tgt_cate}")
        os.rename(src_subdir, new_subdir)

    # !! WE DO NOT RUN STEP 5, since there are some unusual problems with the generated outputs (png, jpg, pdf, etc.) of plotly (using kaleido as backend). It works fine in Windows, but in Linux the generated images have wrong layouts (with wrong spacing -- too much space between images).
    # ! To fix this, we instead run the .bat file to generate the grid plot in Windows: `scripts/run_vdb_plot.bat`
    # # ── Step 5: Build grid figure ──
    # outfile = os.path.join(in_dir, "vdb_grid_plot.pdf")
    # def fmt_row_label_func(x):
    #     return x.replace("row_", "").upper()
    # def fmt_col_label_func(x):
    #     return ""
    # df = plth.get_img_grid_df(src_dir)
    # df.columns = [f'sample_{i}' for i in range(len(df.columns))]
    # df.to_csv(os.path.join(in_dir, "vdb_grid_input.csv"), index=False, sep=";", encoding="utf-8")
    # plth.plot_image_grid(
    #     df,
    #     save_path=outfile,
    #     img_width=300,
    #     img_height=300,
    #     img_stack_padding_px=5,
    #     img_stack_direction="horizontal",
    #     img_scale_mode="fit",
    #     title="Image Grid Test",
    #     fig_margin=dict(l=0, r=10, t=30, b=10),
    #     outline_color="#000000",
    #     outline_size=2,
    #     cell_margin_px=5,
    #     row_line_size=2,
    #     col_line_size=2,
    #     tickfont=dict(size=13, family="Arial", color="black"),
    #     format_row_label_func=fmt_row_label_func,
    #     format_col_label_func=fmt_col_label_func,
    #     show=False,
    # )
    # return outfile
    return src_dir  # return the src_dir for downstream plotting


def main():
    args = PlotVideoArgs().parse_args()
    seed_everything(args.seed)
    outdir = mk_grid_plot(args.in_dir, args.num_frame_per_cate)
    with ConsoleLog("VDB Grid Plot Src"):
        pprint(f"Saved all selected frames to dir ⏬")
        pprint_local_path(outdir, get_wins_path=True)


if __name__ == "__main__":
    main()
