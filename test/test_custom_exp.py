import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))  # Add parent directory to sys.path

from pathlib import Path

import pandas as pd
from halib import *
from tap import *

from src.config import *
from src.exp import Paper2Exp
from src.external_exp import ExternalExpRunner

# ---------------------------------------------------------------------------
# Paths — adjust to your local setup
# ---------------------------------------------------------------------------

TEST_EXP_DIR_FIRENET = "./test/custom_exp/firenet"
TEST_EXP_DIR_YOLO = "./test/custom_exp/yolov5l_notemp"

TABLE_MODE = "pfc"
TABLE_DECIMALS = 2
VIDEO_NAME_LIMIT = 40

DATASET_DIR = "./datasets/UFireIndoorFull"
EXTERNAL_CFG = "config/zruns/run_external.yaml"
EXTERNAL_CFG_MINI = (
    "config/zruns/run_external_mini.yaml"  # for quick testing with fewer frames/videos
)


class CustomArgs(Tap):
    parent_dir: str = r"./zout/zruns/_baseline/UFireIndoorFull"
    test_mini: bool = False  # If True, use EXTERNAL_CFG_MINI which runs on a smaller subset of frames/videos for faster testing.
    test_yolo: bool = False  # If True, test on the YOLOv5l_notemp experiment instead of the FireNet experiment.

    def configure(self):
        self.add_argument("-indir", "--parent_dir")
        self.add_argument("-m", "--test_mini")
        self.add_argument("-y", "--test_yolo")


# ---------------------------------------------------------------------------
# Approach 2 (new): Paper2Exp.from_custom_exp with placeholder config
# Loader type is auto-detected from the experiment directory name.
# tl_type is always no_skip (set in config/methods/external_method.yaml).
# ---------------------------------------------------------------------------


def _external_cfg_fn(exp_dir_path: str) -> str:
    return EXTERNAL_CFG


def _external_cfg_mini_fn(exp_dir_path: str) -> str:
    return EXTERNAL_CFG_MINI


def test_exp_from_custom_dir(
    custom_exp_dir: str = TEST_EXP_DIR_FIRENET, cfg_find_fn=_external_cfg_fn
):
    """Run the full Paper2Exp pipeline for a firenet external experiment."""
    exp = Paper2Exp.from_custom_exp(
        exp_dir_path=custom_exp_dir,
        expDir_to_cfgFile_fn=cfg_find_fn,
    )
    exp.run_exp()


def test_exp_vs_external_exp(
    test_dir=TEST_EXP_DIR_FIRENET, cfg_find_fn=_external_cfg_fn, dataset_dir=DATASET_DIR
):
    """
    Cross-check Paper2Exp.from_custom_exp against ExternalExpRunner on the same
    experiment directory.  Both pipelines must yield identical per-frame and
    per-video metric values (within floating-point tolerance).

    ExternalExpRunner computes metrics directly with numpy; Paper2Exp goes
    through TorchMetricsBackend via the full run_exp() pipeline.  The
    comparison covers accuracy, F1, precision, recall (TPR) and FPR.
    FPS is intentionally excluded because wall-clock elapsed time may
    differ by small amounts between runs.
    """
    exp_dir = Path(test_dir).resolve()
    cfg_name = exp_dir.name  # "firenet"

    metric_cols = [
        "metric_accuracy",
        "metric_f1_score",
        "metric_precision",
        "metric_recall (TPR)",
        "metric_FPR (False Alarm Rate)",
        "metric_FPS",
    ]

    # ------------------------------------------------------------------
    # Step 1 — ExternalExpRunner: reference metrics (computed in-memory,
    #           _results.csv files are also written as a side-effect so that
    #           Paper2Exp can read them next).
    # ------------------------------------------------------------------
    console.rule("[bold]ExternalExpRunner — ref metrics[/bold]")
    runner = ExternalExpRunner.from_dir(
        exp_dir=str(exp_dir),
        dataset_dir=dataset_dir,
    )
    all_dfs = runner._load_and_write_normalized_csvs()
    ref_pf = runner._compute_per_frame_metrics(all_dfs)
    ref_pv = runner._compute_per_video_metrics(all_dfs)

    # ------------------------------------------------------------------
    # Step 2 — Paper2Exp.from_custom_exp: run full pipeline, which writes
    #   _{cfg_name}__per_frame.csv  and  _{cfg_name}__per_video.csv
    #   into exp_dir.
    # ------------------------------------------------------------------
    console.rule("[bold]Paper2Exp — run full pipeline[/bold]")
    exp = Paper2Exp.from_custom_exp(
        exp_dir_path=str(exp_dir),
        expDir_to_cfgFile_fn=cfg_find_fn,
    )
    exp.run_exp()

    # halib's save_results_to_csv strips ".csv" and appends "__perf.csv"
    # so the actual files are  _firenet__per_frame__perf.csv  (not  _firenet__per_frame.csv)
    p2_pf_csv = exp_dir / f"_{cfg_name}__per_frame__perf.csv"
    p2_pv_csv = exp_dir / f"_{cfg_name}__per_video__perf.csv"
    assert p2_pf_csv.exists(), f"Paper2Exp did not write expected CSV: {p2_pf_csv}"
    assert p2_pv_csv.exists(), f"Paper2Exp did not write expected CSV: {p2_pv_csv}"

    p2_pf_df = pd.read_csv(str(p2_pf_csv), sep=";")
    p2_pv_df = pd.read_csv(str(p2_pv_csv), sep=";")

    # ------------------------------------------------------------------
    # Step 3 — Compare metric columns
    # ------------------------------------------------------------------
    tol = 1e-4
    console.rule("[bold]Metric comparison — per_frame[/bold]")
    for col in metric_cols:
        ref_val = ref_pf[col]
        p2_val = float(p2_pf_df.iloc[0][col])
        console.print(f"  {col}:  ref={ref_val:.6f}  paper2={p2_val:.6f}")
        assert abs(ref_val - p2_val) < tol, (
            f"per_frame '{col}' mismatch: "
            f"ExternalExpRunner={ref_val:.6f}  Paper2Exp={p2_val:.6f}"
        )

    console.rule("[bold]Metric comparison — per_video[/bold]")
    for col in metric_cols:
        ref_val = ref_pv[col]
        p2_val = float(p2_pv_df.iloc[0][col])
        console.print(f"  {col}:  ref={ref_val:.6f}  paper2={p2_val:.6f}")
        assert abs(ref_val - p2_val) < tol, (
            f"per_video '{col}' mismatch: "
            f"ExternalExpRunner={ref_val:.6f}  Paper2Exp={p2_val:.6f}"
        )
    console.rule(
        "[bold green]✓ Paper2Exp.from_custom_exp and ExternalExpRunner are consistent[/bold green]"
    )


def gen_perf_report_custom_exps():
    """Batch: run Paper2Exp.from_custom_exp for every directory under parent_dir."""
    args = CustomArgs().parse_args()
    custom_exp_dirs = fs.list_dirs(args.parent_dir)
    for exp_dir in tqdm(custom_exp_dirs):
        exp_dir_name = fs.get_dir_name(exp_dir)
        console.rule(f"Gen report for exp <<{exp_dir_name}>>")
        exp = Paper2Exp.from_custom_exp(
            exp_dir_path=exp_dir,
            expDir_to_cfgFile_fn=_external_cfg_fn,
        )
        exp.run_exp()


if __name__ == "__main__":
    args = CustomArgs().parse_args()

    if args.test_mini:
        test_dir = "./test/custom_exp/firenet_mini" if not args.test_yolo else "./test/custom_exp/yolov5l_notemp_mini"
        dataset_dir = "./datasets/UFireIndoor2"
        cfg_fn = _external_cfg_mini_fn
    else:
        test_dir =  TEST_EXP_DIR_FIRENET if not args.test_yolo else TEST_EXP_DIR_YOLO
        dataset_dir = DATASET_DIR
        cfg_fn = _external_cfg_fn

    pprint_box(f"{test_dir=}, {dataset_dir=}")

    # test if Paper2Exp.from_custom_exp can successfully run an external experiment end-to-end and write results to CSV
    # test_exp_from_custom_dir()

    # test if the metrics computed by Paper2Exp.from_custom_exp match those computed by ExternalExpRunner on the same experiment directory (cross-checking the two implementations)
    test_exp_vs_external_exp(test_dir=test_dir, cfg_find_fn=cfg_fn, dataset_dir=dataset_dir)

    # Optional: batch generate performance reports for all custom experiments under a parent directory
    # gen_perf_report_custom_exps()
