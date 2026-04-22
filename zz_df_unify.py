from halib import *
from halib.common import *
import os

# ── DataFrame building ────────────────────────────────────────────────────────

def get_df(indir: str) -> pd.DataFrame:
    """Merge per-experiment timeline CSVs into one unified DataFrame."""
    TARGET_FILE = "_timeline_report_raw.csv"
    FIXED_COLS = ["video", "video_path", "num_frames", "frame_idx", "gt_label"]
    JOIN_COLS = ["video", "frame_idx", "num_frames", "gt_label"]
    exp_dirs = fs.list_dirs(indir)
    exp_dirs = [
        d for d in exp_dirs if os.path.exists(os.path.join(indir, d, TARGET_FILE))
    ]
    assert len(exp_dirs) > 0, f"No valid experiment dirs found in {indir}"

    df_list = []
    method_unique_list = []
    for exp_dir in exp_dirs:
        csv_path = os.path.join(indir, exp_dir, TARGET_FILE)
        df = pd.read_csv(
            csv_path, sep=";", encoding="utf-8", keep_default_na=False, na_values=[""]
        )

        # Last column of each CSV must be the method name
        method_name = df.columns.tolist()[-1]
        assert method_name not in method_unique_list, (
            f"Duplicate method name '{method_name}' found in multiple CSVs"
        )
        method_unique_list.append(method_name)

        extra_cols = [c for c in df.columns if c not in FIXED_COLS]
        predict_col = [c for c in extra_cols if c in method_name]
        assert len(predict_col) == 1, (
            f"Expected exactly one predict column for '{method_name}', found: {predict_col}"
        )
        # df.rename(columns={predict_col[0]: method_name}, inplace=True)
        # csvfile.fn_display_df(df.head(5))

        # make gt_label consistent (e.g. "None" vs "none")
        df["gt_label"] = df["gt_label"].str.lower()

        df = df[FIXED_COLS + [method_name]].set_index(JOIN_COLS)
        pprint(df.columns.tolist())
        # make method_name col values consistent (lowercase, remove spaces)
        df[method_name] = df[method_name].str.lower().str.strip()

        # Store abs path of this exp dir as a sibling column
        df[f"{method_name}_dir"] = os.path.abspath(os.path.join(indir, exp_dir))
        df_list.append(df)

    assert len(df_list) > 0, "No experiments matched exp_valid_types."
    unified_df = pd.concat(df_list, axis=1).reset_index()
    unified_df = unified_df.loc[:, ~unified_df.columns.duplicated()]

    # Normalize paths for the current machine
    from halib.system.path import normalize_paths

    for col in [
        c for c in unified_df.columns if c.endswith("_dir") or c.endswith("_path")
    ]:
        unified_df[col] = unified_df[col].apply(normalize_paths)

    return unified_df