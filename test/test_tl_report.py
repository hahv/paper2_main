from pathlib import Path
import sys
from tap import Tap
from typing import Literal

sys.path.append("/mnt/e/SyncData/paper2_main")
from halib import *
from src.results.timeline.tl_report import TlReportGen

# =========================================================================================
# Usage Examples:
#
# 1. Run generation test based on a directory (automatically detects single vs multi):
#    python test/test_tl_report.py --task gen
#
# 2. Run generation test on a specific custom directory (auto detects if single exp or parent of exps):
#    python test/test_tl_report.py --task gen --in_dir /path/to/exp_or_parent_dir
#
# 3. Run CSV reconstruction test:
#    python test/test_tl_report.py --task csv
#
# 4. Run MULTIPLE comparison test (combines multiple exps into one report):
#    python test/test_tl_report.py --task compare
#
# 5. Change output table mode (p, fc, pfc) or decimals:
#    python test/test_tl_report.py --task gen --table_mode p --table_decimals 4
# =========================================================================================

class TestArgs(Tap):
    # What tests to run
    task: Literal["gen", "csv", "compare"] = "gen"

    # Directories
    in_dir: str = "./zout/test_data/test_single_exp/MainPC__ds_UFireIndoor2__mt_no_temp_method__af4b0d32a3d2__20260211.104731"

    # Formatting options
    table_decimals: int = 2
    table_mode: Literal["p", "fc", "pfc"] = "pfc" # p=percent, fc=frame count, pfc=both
    video_name_limit: int = 40

    def configure(self):
        self.add_argument("-i", "--in_dir")
        self.add_argument("-t", "--task")

def test_gen(args: TestArgs):
    """
    Test generating a report based on the provided directory.
    Automatically detects if it is a single experiment folder (contains __config.yaml)
    or a parent folder of multiple experiments.
    """
    path = Path(args.in_dir)
    if not path.exists():
        print(f"Error: Directory {path} does not exist.")
        return

    # Heuristic: A single experiment usually has __config.yaml or CSV files directly in it.
    if (path / "__config.yaml").exists() or any(path.glob("*_results.csv")):
        print(f"\n[Test] Generating Single Exp Report for: {args.in_dir}")
        TlReportGen.gen_TlReport_exp(
            args.in_dir,
            table_mode=args.table_mode,
            table_decimals=args.table_decimals,
            video_name_limit=args.video_name_limit,
        )
    else:
        print(f"\n[Test] Generating Multiple Individual Exp Reports for: {args.in_dir}")
        TlReportGen.gen_TlReport_muti_exps(
            args.in_dir,
            table_mode=args.table_mode,
            table_decimals=args.table_decimals,
            video_name_limit=args.video_name_limit,
        )

def test_gen_from_csv(args: TestArgs):
    """Test reconstructing a report from an existing CSV."""
    print(f"\n[Test] Reconstruct from CSV")
    csv_path = f"{args.in_dir}/_timeline_report.csv"
    output_html_path = f"{args.in_dir}/_timeline_report_reconstructed.html"

    csv_parent_dir = Path(csv_path).parent
    TlReportGen.tlReport_from_csv(
        csv_path, output_html_path, title=f"[Reconstructed] {csv_parent_dir.name}"
    )

def test_gen_compare_exps(args: TestArgs):
    """
    Test generating a SINGLE comparison report combining multiple experiment directories.
    """
    print(f"\n[Test] Compare Multiple Experiments")
    root_path = Path(args.in_dir)
    exp_dirs = [str(p) for p in root_path.iterdir() if p.is_dir()]
    assert len(exp_dirs) >= 2, "Need at least two experiment directories to compare."

    # 2. Define Output Path
    output_html = f"{root_path}/comparison_report.html"

    # 3. Run Comparison Generation
    TlReportGen.gen_TlReport_compare(
        exp_dirs=exp_dirs,
        output_path=output_html,
        title="Compare Multiple Experiments",
        table_mode=args.table_mode,
        table_decimals=args.table_decimals,
    )

if __name__ == "__main__":
    args = TestArgs().parse_args()

    if args.task == "gen":
        test_gen(args)

    if args.task == "csv":
        test_gen_from_csv(args)

    if args.task == "compare":
        test_gen_compare_exps(args)
