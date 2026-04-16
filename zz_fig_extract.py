from src.common import seed_everything
from halib import *
from halib.common import *
from tap import Tap
import os
import yaml
import cv2
import numpy as np
from dataclasses import dataclass, field
from halib.filetype import yamlfile

ROWS = ["row_01_rgb", "row_02_mask"]


def create_placeholder_img(
    img_name="placeholder", size=(1920, 1080), color="#526975", outdir="."
):
    width, height = size
    # Convert hex color string to BGR tuple for OpenCV
    color = color.lstrip("#")
    bgr_color = tuple(int(color[i : i + 2], 16) for i in (4, 2, 0))

    # Create solid background image
    img = np.full((height, width, 3), bgr_color, dtype=np.uint8)

    # Draw rectangle border and diagonals
    line_color = (200, 200, 200)
    cv2.rectangle(img, (0, 0), (width - 1, height - 1), line_color, 4)
    cv2.line(img, (0, 0), (width, height), line_color, 2)
    cv2.line(img, (width, 0), (0, height), line_color, 2)

    # Simple word wrapping
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    thickness = 5
    words = img_name.replace("_", " ").replace("-", " ").split()

    lines = []
    curr_line = ""
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

    # Calculate text block dimensions to center it
    text_height = cv2.getTextSize("Ay", font, font_scale, thickness)[0][1]
    line_spacing = text_height + 30
    total_height = len(lines) * line_spacing
    y_start = (height - total_height) // 2 + text_height // 2

    # Draw text lines
    for i, line in enumerate(lines):
        (tw, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        x = (width - tw) // 2
        y = y_start + (i * line_spacing)
        # Draw soft shadow/outline then main text text
        cv2.putText(
            img, line, (x, y), font, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA
        )
        cv2.putText(
            img, line, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
        )

    # Save the output file
    os.makedirs(outdir, exist_ok=True)
    out_file = os.path.join(outdir, f"{img_name}.png")
    cv2.imwrite(out_file, img)
    print(f"Created placeholder: {out_file}")
    return img


# ── Args ─────────────────────────────────────────────────────────────────────


class FigExtractArgs(Tap):
    yaml_path: str = "./zz_fig_extract.yaml"  # Path to the fig-extract YAML config file
    force_recreate: bool = (
        False  # Whether to force re-creation of folders if they already exist
    )

    def configure(self):
        self.add_argument(
            "-i",
            "--yaml_path",
            type=str,
            default="./zz_fig_extract.yaml",
            help="Path to the fig-extract YAML config file. Default is './zz_fig_extract.yaml'.",
        )
        self.add_argument(
            "-f",
            "--force_recreate",
            action="store_true",
            help="Whether to force re-creation of folders if they already exist. Default is False.",
        )


# ── Data model ───────────────────────────────────────────────────────────────


@dataclass
class Condition:
    exp: str
    gt_label: str | None = None
    raw_label: str | list[str] | None = None


@dataclass
class SubCase:
    case_name: str
    case_desc: str
    num_cases: int
    conditions: list[Condition] = field(default_factory=list)
    prefer: list[dict] | None = None


@dataclass
class Case:
    name: str
    sub_cases: list[SubCase]


# ── Parsing ───────────────────────────────────────────────────────────────────


def parse_condition(raw: dict) -> Condition:
    return Condition(
        exp=raw["exp"],
        gt_label=raw.get("gt_label"),
        raw_label=raw.get("raw_label"),
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
    cfg = None
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    outdir = cfg["outdir"]
    cases = [
        Case(
            name=c["name"],
            sub_cases=[parse_sub_case(s) for s in c["sub_cases"]],
        )
        for c in cfg["list_cases"]
    ]
    return outdir, cases, cfg


# ── Folder creation ───────────────────────────────────────────────────────────


def get_sub_case_folders(col_idx: int, sub_case: SubCase) -> list[str]:
    """Return the list of folder names for a sub_case, one per num_cases."""
    base = f"col_{col_idx:02d}_{sub_case.case_name}"
    return [f"{base}_{i:02d}" for i in range(1, sub_case.num_cases + 1)]


def create_case_dirs(outdir: str, case: Case) -> None:
    case_dir = os.path.join(outdir, "src", case.name)
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
    label = f"gt={c.gt_label}" if c.gt_label else f"raw={c.raw_label}"
    return f"{c.exp}[{label}]"


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


def dir2method(dir_path: str) -> str:
    METHOD_CFG_FILE = "__method_cfg.yaml"
    cfg_path = os.path.join(dir_path, METHOD_CFG_FILE)
    if not os.path.exists(cfg_path):
        assert False, f"Method config file not found: {cfg_path}"
    method_cfg = yamlfile.load_yaml(cfg_path, to_dict=True)
    method_name = method_cfg.get("method_name")
    extra_cfgs = method_cfg.get("extra_cfgs", {})
    if extra_cfgs:
        motion = ""
        try:
            motion_full = extra_cfgs["skip_proc"]["params"]["motion"]["name"]
            motion = motion_full.split(".")[-1]
        except KeyError:
            pass
        if motion:
            method_name += f".{motion}"
    return method_name


def get_df(config: dict) -> pd.DataFrame:
    """Find related csv files in indir, process them and return the final
    DataFrame that can be used for figure generation.
    """
    TARGET_FILE = "_timeline_report_raw.csv"
    FIXED_COLS = ["video", "video_path", "num_frames", "frame_idx", "gt_label"]
    JOIN_COLS = ["video", "frame_idx", "num_frames", "gt_label"]
    indir = config["indir"]
    exp_dirs = fs.list_dirs(indir)
    exp_dirs = [
        d for d in exp_dirs if os.path.exists(os.path.join(indir, d, TARGET_FILE))
    ]
    df_list = []
    for i, exp_dir in enumerate(exp_dirs):
        method_name = dir2method(os.path.join(indir, exp_dir))
        csv_path = os.path.join(indir, exp_dir, TARGET_FILE)
        df = pd.read_csv(csv_path, sep=";", encoding="utf-8")
        col_names = df.columns.tolist()
        extra_cols = [c for c in col_names if c not in FIXED_COLS]
        predict_col = [c for c in extra_cols if c in method_name]
        assert len(predict_col) == 1, (
            f"Expected exactly one predict column for method '{method_name}' in {csv_path}, but found: {predict_col}"
        )
        predict_col = predict_col[0]
        df.rename(columns={predict_col: method_name}, inplace=True)

        # Only keep FIXED_COLS and the method_name column to avoid conflicting paths like /mnt/d vs /mnt/e
        df = df[FIXED_COLS + [method_name]]
        df.set_index(JOIN_COLS, inplace=True)
        method_dir_path = os.path.abspath(os.path.join(indir, exp_dir))
        method_dir_col = f"{method_name}_dir"
        df[method_dir_col] = method_dir_path
        # df.to_csv(f"./zz_fig_extract.{method_name}.csv", index=True, sep=";",
        # encoding="utf-8") # debug step
        # nomalize all path basing on current working dir in current machine
        from halib.system.path import normalize_paths
        path_cols = [c for c in df.columns if c.endswith("_dir") or c.endswith("_path")]
        for pc in path_cols:
            df[pc] = df[pc].apply(lambda p: normalize_paths(p))
        df_list.append(df)

    assert len(df_list) > 0, f"No valid csv files found in {indir}"
    unified_df = pd.concat(df_list, axis=1).reset_index()
    unified_df = unified_df.loc[:, ~unified_df.columns.duplicated()]
    return unified_df


# ── Entry point ───────────────────────────────────────────────────────────────


def main(args: FigExtractArgs) -> None:
    outdir, cases, cfg = load_config(args.yaml_path)
    seed_everything(
        cfg.get("random_seed", 42)
    )  # Set random seed from config if provided
    os.makedirs(outdir, exist_ok=True)

    if args.force_recreate and os.path.exists(f"{outdir}/src"):
        import shutil

        console.rule("[red]Force Recreate Enabled - Removing Existing 'src/' Folder")
        shutil.rmtree(f"{outdir}/src")
    else:
        console.rule(
            "[green][Done][/green] Already exists folders under 'src/'. Use --force_recreate to remove and recreate them."
        )
        return
    create_all_dirs(outdir, cases)
    print_tree(outdir, cases)


if __name__ == "__main__":
    # main(Args().parse_args())
    # create_placeholder_img(
    #     img_name="test_placeholder", size=(1920, 1080), color="#526975"
    # )
    # pprint(
    #     dir2method(
    #         "zout/reports/baseline_vs_accMotion/MainPC__ds_UFireIndoorTest__mt_no_temp_method__af4b0d32a3d2__20260330.205208"
    #     )
    # )
    args = FigExtractArgs().parse_args()
    outdir, cases, cfg = load_config(args.yaml_path)
    df = get_df(cfg)
    df = df.head(5)
    df.to_csv("./zz_fig_extract.csv", index=False, sep=";", encoding="utf-8")
