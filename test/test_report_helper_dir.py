from pathlib import Path
import sys

sys.path.append("/mnt/e/SyncData/paper2_main")
from halib import *
from src.results.timeline.tl_report import TlReportGen

TEST_DIR = "./zout/test_data"
SINGLE_EXP_DIR = f"{TEST_DIR}/test_single_exp/MainPC__ds_UFireIndoor2__mt_no_temp_method__af4b0d32a3d2__20260211.104731"
MULTIPLE_EXPS_DIR = f"{TEST_DIR}/test_exp_vs_baseline"
TEST_COMPARE_DIR = f"{TEST_DIR}/test_prof_old_model_vs_new_model_vs_newest"

TABLE_DECIMALS = 2
TABLE_MODE = "pfc"  # percent + frame count
VIDEO_NAME_LIMIT = 40  # Max characters for video names in the report table


def test_gen_single_dir(single_exp_dir=SINGLE_EXP_DIR):
    """Test generating a report for one specific experiment directory."""
    print(f"\n[Test] Single Exp: {single_exp_dir}")
    TlReportGen.gen_TlReport_exp(
        single_exp_dir,
        table_mode=TABLE_MODE,
        table_decimals=TABLE_DECIMALS,
        video_name_limit=VIDEO_NAME_LIMIT,
    )


def test_gen_multiple_dirs(multi_exp_dir=MULTIPLE_EXPS_DIR):
    """Test generating individual reports for ALL experiments in the root folder."""
    print(f"\n[Test] Batch Gen for Root: {multi_exp_dir}")
    TlReportGen.gen_TlReport_muti_exps(
        multi_exp_dir,
        table_mode=TABLE_MODE,
        table_decimals=TABLE_DECIMALS,
        video_name_limit=VIDEO_NAME_LIMIT,
    )


def test_gen_from_csv():
    """Test reconstructing a report from an existing CSV."""
    print(f"\n[Test] Reconstruct from CSV")
    csv_path = f"{SINGLE_EXP_DIR}/_timeline_report.csv"
    output_html_path = f"{SINGLE_EXP_DIR}/_timeline_report_reconstructed.html"

    csv_parent_dir = Path(csv_path).parent
    TlReportGen.tlReport_from_csv(
        csv_path, output_html_path, title=f"[Reconstructed] {csv_parent_dir.name}"
    )


def test_gen_compare_exps():
    """
    Test generating a SINGLE comparison report combining multiple experiment directories.
    """
    print(f"\n[Test] Compare Multiple Experiments")
    root_path = Path(TEST_COMPARE_DIR)
    exp_dirs = [str(p) for p in root_path.iterdir() if p.is_dir()]
    assert len(exp_dirs) >= 2, "Need at least two experiment directories to compare."

    # 2. Define Output Path
    output_html = f"{root_path}/comparison_report.html"

    # 3. Run Comparison Generation
    TlReportGen.gen_TlReport_compare(
        exp_dirs=exp_dirs,
        output_path=output_html,
        title="Compare Multiple Experiments",
        table_mode=TABLE_MODE,
        table_decimals=TABLE_DECIMALS,
    )


if __name__ == "__main__":
    # 1. Single Experiment
    # test_gen_single_dir()

    # 2. Batch Processing (Individual Reports)
    # test_gen_multiple_dirs()

    # 3. CSV Reconstruction
    # test_gen_from_csv()

    # 4. Comparison Report (New)
    test_gen_compare_exps()
