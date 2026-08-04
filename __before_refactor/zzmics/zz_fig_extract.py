from src.common import seed_everything
from halib import *
from halib.common import *
from tap import Tap
import os
import shutil
import yaml
import cv2
import numpy as np
from dataclasses import dataclass, field


ROWS = ["row_01_rgb", "row_02_mask"]
FRAME_FILENAME = "frame.jpg"
PLACEHOLDER_COLOR = "#526975"
RGB_VIDEO_SUFFIX = "_out.mp4"
MASK_VIDEO_SUFFIX = "_fgmask_out.mp4"


# ── Placeholder ───────────────────────────────────────────────────────────────


def create_placeholder_img(
    img_name: str = "placeholder",
    size: tuple[int, int] = (640, 640),
    color: str = PLACEHOLDER_COLOR,
    outdir: str | None = None,
) -> np.ndarray:
    """Create a solid-color placeholder image with centered label text."""
    width, height = size
    hex_color = color.lstrip("#")
    bgr_color = tuple(int(hex_color[i : i + 2], 16) for i in (4, 2, 0))

    img = np.full((height, width, 3), bgr_color, dtype=np.uint8)
    line_color = (200, 200, 200)
    cv2.rectangle(img, (0, 0), (width - 1, height - 1), line_color, 4)
    cv2.line(img, (0, 0), (width, height), line_color, 2)
    cv2.line(img, (width, 0), (0, height), line_color, 2)

    font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 3, 5
    words = img_name.replace("_", " ").replace("-", " ").split()
    lines, curr_line = [], ""
    for word in words:
        test_line = f"{curr_line} {word}".strip()
        (text_w, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        if text_w > width - 100 and curr_line:
            lines.append(curr_line)
            curr_line = word
        else:
            curr_line = test_line
    if curr_line:
        lines.append(curr_line)

    text_height = cv2.getTextSize("Ay", font, font_scale, thickness)[0][1]
    line_spacing = text_height + 30
    y_start = (height - len(lines) * line_spacing) // 2 + text_height // 2
    for i, line in enumerate(lines):
        (tw, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        x = (width - tw) // 2
        y = y_start + i * line_spacing
        cv2.putText(
            img, line, (x, y), font, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA
        )
        cv2.putText(
            img, line, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
        )

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        cv2.imwrite(os.path.join(outdir, f"{img_name}.png"), img)
    return img


# ── Args ──────────────────────────────────────────────────────────────────────


class FigExtractArgs(Tap):
    yaml_path: str = "./zz_fig_extract.yaml"  # Path to the fig-extract YAML config file
    force_recreate: bool = False  # Remove and recreate outdir/src from scratch

    def configure(self):
        self.add_argument("-i", "--yaml_path")
        self.add_argument("-f", "--force_recreate", action="store_true")


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class Condition:
    col_name: str
    raw_value: str | list[str] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.raw_value, str):
            self.raw_value = [self.raw_value]
        for v in self.raw_value:
            if self.col_name == 'gt_label':
                valid_gt = ["fire_smoke", "none"]
                assert str(v).lower() in valid_gt, f"gt_label value must be in {valid_gt}, got: {v}"
            else:
                valid_raw = ["fire", "smokeonly", "none", "skipped"]
                assert str(v).lower() in valid_raw, f"raw_label value must be in {valid_raw}, got: {v}"


@dataclass
class SubCase:
    case_name: str
    case_desc: str
    num_cases: int
    conditions: list[Condition] = field(default_factory=list)
    prefer: list[dict] | None = None  # [{video: "x.mp4", frames: [1, 2]}, ...]


@dataclass
class Case:
    name: str
    sub_cases: list[SubCase]


# ── Parsing ───────────────────────────────────────────────────────────────────


def parse_condition(raw: dict) -> Condition:
    return Condition(
        col_name=raw["col_name"],
        raw_value=raw["raw_value"],
    )


def parse_sub_case(raw: dict) -> SubCase:
    return SubCase(
        case_name=raw["case_name"],
        case_desc=raw["case_desc"],
        num_cases=raw["num_cases"],
        conditions=[parse_condition(c) for c in raw.get("conditions", [])],
        prefer=raw.get("prefer"),
    )


def load_config(yaml_path: str) -> tuple[str, list[Case], dict]:
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    outdir = cfg["outdir"]
    cases = [
        Case(name=c["name"], sub_cases=[parse_sub_case(s) for s in c["sub_cases"]])
        for c in cfg["list_cases"]
    ]
    return outdir, cases, cfg


# ── Folder helpers ────────────────────────────────────────────────────────────


def get_sub_case_folders(col_idx: int, sub_case: SubCase) -> list[str]:
    """e.g. col_02_failure_wasted_infer_01, col_02_failure_wasted_infer_02"""
    base = f"col_{col_idx:02d}_{sub_case.case_name}"
    return [f"{base}_{i:02d}" for i in range(1, sub_case.num_cases + 1)]


CASE_DIR = {}


def create_case_dirs(outdir: str, case: Case) -> None:
    case_dir = os.path.join(outdir, "src", case.name)
    global CASE_DIR
    CASE_DIR[case.name] = case_dir
    for row in ROWS:
        for col_idx, sub_case in enumerate(case.sub_cases, start=1):
            for folder_name in get_sub_case_folders(col_idx, sub_case):
                folder = os.path.join(case_dir, row, folder_name)
                os.makedirs(folder, exist_ok=True)
                print(f"  created: {folder}")


def create_all_dirs(outdir: str, cases: list[Case]) -> None:
    for case in cases:
        create_case_dirs(outdir, case)


# ── Pretty-print ──────────────────────────────────────────────────────────────


def _fmt_condition(c: Condition) -> str:
    return f"{c.col_name}[{c.raw_value}]"


def print_tree(outdir: str, cases: list[Case]) -> None:
    print("\nFolder structure:")
    for case in cases:
        print(f"\n{os.path.join(outdir, case.name)}/")
        for row in ROWS:
            print(f"  {row}/")
            for col_idx, sub_case in enumerate(case.sub_cases, start=1):
                cond_str = "  &  ".join(_fmt_condition(c) for c in sub_case.conditions)
                for folder_name in get_sub_case_folders(col_idx, sub_case):
                    print(f"    {folder_name}/  ← {cond_str}")


# ── DataFrame building ────────────────────────────────────────────────────────


def get_df(config: dict) -> pd.DataFrame:
    """Merge per-experiment timeline CSVs into one unified DataFrame."""
    TARGET_FILE = "_timeline_report_raw.csv"
    FIXED_COLS = ["video", "video_path", "num_frames", "frame_idx", "gt_label"]
    JOIN_COLS = ["video", "frame_idx", "num_frames", "gt_label"]
    indir = config["indir"]
    exp_valid_types: list[str] = config.get("exp_valid_types", [])

    exp_dirs = fs.list_dirs(indir)
    exp_dirs = [
        d for d in exp_dirs if os.path.exists(os.path.join(indir, d, TARGET_FILE))
    ]
    assert len(exp_dirs) > 0, f"No valid experiment dirs found in {indir}"

    df_list = []
    method_unique_list = []
    for exp_dir in exp_dirs:
        csv_path = os.path.join(indir, exp_dir, TARGET_FILE)
        df = pd.read_csv(csv_path, sep=";", encoding="utf-8", keep_default_na=False, na_values=[""])

        # Last column of each CSV must be the method name
        method_name = df.columns.tolist()[-1]
        assert method_name not in method_unique_list, (
            f"Duplicate method name '{method_name}' found in multiple CSVs"
        )
        method_unique_list.append(method_name)

        # Skip experiments not declared in exp_valid_types
        if exp_valid_types:
            assert method_name in exp_valid_types, (
                f"Method '{method_name}' from {csv_path} not in exp_valid_types: {exp_valid_types}"
            )

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


# ── Condition filtering ───────────────────────────────────────────────────────


def _match_label(value, target) -> bool:
    """Match cell value against a string or list target (compared as str)."""
    if isinstance(target, list):
        return str(value) in [str(t) for t in target]
    return str(value) == str(target)


def filter_df_for_subcase(df: pd.DataFrame, sub_case: SubCase) -> pd.DataFrame:
    """Return rows where ALL conditions are satisfied."""
    mask = pd.Series(True, index=df.index)
    for cond in sub_case.conditions:
        if cond.col_name not in df.columns:
            print(f"  [WARN] Column '{cond.col_name}' missing in df — condition skipped")
            continue
        mask &= df[cond.col_name].apply(lambda v: _match_label(v, cond.raw_value))
    return df[mask].copy()


# ── Row sampling ──────────────────────────────────────────────────────────────


def sample_rows(df: pd.DataFrame, sub_case: SubCase) -> list[dict]:
    """Prefer-first sampling: use prefer video/frames first, then random fill."""
    n = sub_case.num_cases
    selected: list[dict] = []

    if sub_case.prefer:
        for pref in sub_case.prefer:
            subset = df[df["video"] == pref.get("video", "")]
            if pref.get("frames"):
                subset = subset[subset["frame_idx"].isin(pref["frames"])]
            selected.extend(subset.to_dict("records"))
            if len(selected) >= n:
                break

    used = {(r["video"], r["frame_idx"]) for r in selected}
    used_videos = {r["video"] for r in selected}
    remaining = df[~df.apply(lambda r: (r["video"], r["frame_idx"]) in used, axis=1)].copy()
    need = n - len(selected)
    if need > 0 and len(remaining) > 0:
        # seed_everything() at main() ensures this is deterministic across runs
        remaining = remaining.sample(frac=1.0)  # shuffle to randomize

        # Priority: 1. Unseen videos first, 2. Round-robin across videos
        remaining["__used"] = remaining["video"].isin(used_videos)
        remaining["__g_idx"] = remaining.groupby("video").cumcount()
        remaining = remaining.sort_values(["__used", "__g_idx"])

        sampled = remaining.head(need).drop(columns=["__used", "__g_idx"])
        selected.extend(sampled.to_dict("records"))

    return selected[:n]


# ── Video search & frame extraction ──────────────────────────────────────────


def find_video_file(exp_dir: str, video_stem: str, suffix: str) -> str | None:
    """
    Recursively search exp_dir for '{video_stem}{suffix}'.
    e.g. suffix='_out.mp4' or '_fgmask_out.mp4'
    Returns the full path of the first match, or None.
    """
    if not exp_dir or not os.path.isdir(exp_dir):
        return None
    target = f"{video_stem}{suffix}"
    for root, _, files in os.walk(exp_dir):
        if target in files:
            return os.path.join(root, target)
    return None


def extract_rgb_frame(video_path: str, frame_idx: int) -> np.ndarray | None:
    """Seek to frame_idx (1-indexed) in video and return the BGR frame, or None on failure."""
    if not video_path or not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def get_rgb_dir_col(sub_case: SubCase) -> str | None:
    """
    Resolve the _dir column for RGB video lookup.
    RGB lives in the exp that uses gt_label (the baseline/no_temp exp).
    e.g. returns 'no_temp_method_dir'
    """
    for cond in sub_case.conditions:
        if cond.col_name != "gt_label":
            return f"{cond.col_name}_dir"
    return None


def get_mask_dir_col(sub_case: SubCase) -> str | None:
    """
    Resolve the _dir column for mask video lookup.
    Mask lives in the exp that uses raw_label (the skip/motion exp).
    e.g. returns 'temp_method_motion_block.AccMotionDet_dir'
    """
    for cond in sub_case.conditions:
        if cond.col_name != "gt_label":
            return f"{cond.col_name}_dir"
    return None

# ── Main extraction loop ──────────────────────────────────────────────────────


def extract_all_cases(outdir: str, cases: list[Case], df: pd.DataFrame) -> None:
    """For each sub_case: filter → sample → extract RGB + mask → save (placeholder if missing)."""
    src_dir = os.path.join(outdir, "src")
    global CASE_DIR
    from halib.filetype import textfile

    for case in cases:  # case: success, failure
        case_dir: str = CASE_DIR.get(case.name)  # ty:ignore[invalid-assignment]
        assert os.path.exists(case_dir), f"Case dir '{case_dir}' does not exist"
        anno_file = os.path.join(case_dir, "col_anno.txt")

        # sub_case: success_skip, success_infer, failure_false_skip,
        # failure_wasted_infer, failure_model_error_1
        sub_case_anno = []
        col = 0
        for col_idx, sub_case in enumerate(case.sub_cases, start=1):
            console.rule(
                f"[cyan]{case.name} / {sub_case.case_name}[/cyan]  (need {sub_case.num_cases})"
            )

            filtered = filter_df_for_subcase(df, sub_case)
            # ! save filter result for debugging
            filtered_outfile = os.path.join(
                case_dir, f"filtered_{sub_case.case_name}.csv"
            )
            filtered.to_csv(filtered_outfile, index=False, encoding="utf-8", sep=";")

            rows = sample_rows(filtered, sub_case) if len(filtered) > 0 else []
            if len(rows) < sub_case.num_cases:
                print(
                    f"  [WARN] Only {len(rows)}/{sub_case.num_cases} rows available — rest will be placeholders"
                )

            # Dynamically resolve which exp dir holds RGB vs mask videos
            rgb_dir_col = get_rgb_dir_col(sub_case)
            mask_dir_col = get_mask_dir_col(sub_case)
            if rgb_dir_col and rgb_dir_col not in df.columns:
                print(
                    f"  [WARN] RGB dir col '{rgb_dir_col}' not in df — RGB will be placeholders"
                )
                rgb_dir_col = None
            if mask_dir_col and mask_dir_col not in df.columns:
                print(
                    f"  [WARN] Mask dir col '{mask_dir_col}' not in df — masks will be placeholders"
                )
                mask_dir_col = None

            for i, folder_name in enumerate(get_sub_case_folders(col_idx, sub_case)):
                col += 1
                sub_case_anno.append(f"col_{col}---{sub_case.case_desc} ")

                # No matching row → full placeholder pair
                if i >= len(rows):
                    file_name = f"{sub_case.case_name}_placeholder_{i + 1}.jpg"
                    rgb_dst = os.path.join(
                        src_dir, case.name, "row_01_rgb", folder_name, file_name
                    )
                    mask_dst = os.path.join(
                        src_dir, case.name, "row_02_mask", folder_name, file_name
                    )

                    ph = create_placeholder_img(
                        img_name=f"{sub_case.case_name} placeholder {i + 1}",
                        outdir=None,
                    )
                    cv2.imwrite(rgb_dst, ph)
                    cv2.imwrite(mask_dst, ph)
                    print(f"  [PLACEHOLDER] {folder_name}")
                    continue

                row = rows[i]
                tag = f"{row['video']}@{row['frame_idx']}"
                file_name = f"{sub_case.case_name}_{tag}.jpg"
                rgb_dst = os.path.join(
                    src_dir, case.name, "row_01_rgb", folder_name, file_name
                )
                mask_dst = os.path.join(
                    src_dir, case.name, "row_02_mask", folder_name, file_name
                )

                video_stem = os.path.splitext(row["video"])[0]

                # ── RGB: seek into {stem}_out.mp4 inside the baseline exp dir
                rgb_exp_dir = row.get(rgb_dir_col, "") if rgb_dir_col else ""
                rgb_video = find_video_file(rgb_exp_dir, video_stem, RGB_VIDEO_SUFFIX)
                rgb = (
                    extract_rgb_frame(rgb_video, row["frame_idx"])
                    if rgb_video
                    else None
                )
                if rgb is None:
                    print(
                        f"  [WARN] RGB missing ({tag}, video='{rgb_video}') — placeholder used"
                    )
                    rgb = create_placeholder_img(
                        img_name=f"RGB missing {tag}", outdir=None
                    )
                cv2.imwrite(rgb_dst, rgb)
                print(f"  [RGB]  {tag} → {rgb_dst}")

                # ── Mask: seek into {stem}_fgmask_out.mp4 inside the motion exp dir
                mask_exp_dir = row.get(mask_dir_col, "") if mask_dir_col else ""
                mask_video = find_video_file(
                    mask_exp_dir, video_stem, MASK_VIDEO_SUFFIX
                )
                mask = (
                    extract_rgb_frame(mask_video, row["frame_idx"])
                    if mask_video
                    else None
                )
                if mask is None:
                    print(
                        f"  [WARN] Mask missing ({tag}, video='{mask_video}') — placeholder used"
                    )
                    mask = create_placeholder_img(
                        img_name=f"MASK missing {tag}", outdir=None
                    )
                cv2.imwrite(mask_dst, mask)
                print(f"  [MASK] {tag} → {mask_dst}")
        textfile.write(sub_case_anno, anno_file)
        print(f"  Annotation saved to {anno_file}")


# ── Entry point ───────────────────────────────────────────────────────────────


def main(args: FigExtractArgs) -> None:
    outdir, cases, cfg = load_config(args.yaml_path)
    seed = cfg.get("random_seed", 42)
    seed_everything(seed)
    os.makedirs(outdir, exist_ok=True)

    src_dir = os.path.join(outdir, "src")
    if os.path.exists(src_dir):
        if args.force_recreate:
            console.rule("[red]Force recreate — removing existing src/")
            shutil.rmtree(src_dir)
        else:
            console.rule("[yellow]src/ already exists — use -f to force recreate")

    create_all_dirs(outdir, cases)
    print_tree(outdir, cases)

    console.rule("[bold green]Building unified DataFrame")
    df = get_df(cfg)
    outfile_top_5 = os.path.abspath(os.path.join(src_dir, "head5_unified_df.csv"))
    outfile_full = os.path.abspath(os.path.join(src_dir, "unified_df.csv"))
    df.head(5).to_csv(outfile_top_5, sep=";", encoding="utf-8", index=False)
    df.to_csv(outfile_full, sep=";", encoding="utf-8", index=False)

    with ConsoleLog(f"Flatten unified_df.csv"):
        os.system(f"xan flatten {outfile_top_5}")
    with ConsoleLog("Extract frames"):
        extract_all_cases(outdir, cases, df)


if __name__ == "__main__":
    main(FigExtractArgs().parse_args())
