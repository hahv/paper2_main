from pathlib import Path
import sys

sys.path.append("/mnt/e/SyncData/paper2_main")
from halib import *
from src.results.timeline.tl_report import TlReportGen

ALL_EXP_DIR = "./zout/zruns"
SINGLE_EXP_DIR = "./zout/zruns/MainPC__ds_UFireIndoor2__mt_no_temp_method__af4b0d32a3d2__20260211.104731"
TABLE_DECIMALS = 4


def test_gen_from_single_dir():
    exp_dir = SINGLE_EXP_DIR
    TlReportGen.gen_TlReport_exp(exp_dir, table_mode="p", table_decimals=TABLE_DECIMALS)


def test_gen_from_muti_dir():
    all_exp_dir = ALL_EXP_DIR
    TlReportGen.gen_TlReport_muti_exps(all_exp_dir, table_mode="p")


def test_gen_from_csv():
    csv_path = f"{SINGLE_EXP_DIR}/_timeline_report.csv"
    output_html_path = f"{SINGLE_EXP_DIR}/_timeline_report_reconstructed.html"
    # get name of parent dir of the csv file
    csv_parent_dir = Path(csv_path).parent
    TlReportGen.tlReport_from_csv(
        csv_path, output_html_path, title=f"[Reconstructed] {csv_parent_dir.name}"
    )


if __name__ == "__main__":
    test_gen_from_single_dir()
    # test_gen_from_muti_dir()
    # test_gen_from_csv()
