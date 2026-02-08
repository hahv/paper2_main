from pathlib import Path
import sys

sys.path.append("/mnt/e/SyncData/paper2_main")
from halib import *
from src.results.timeline.report_helper import TimelineReportGen

def test_gen_from_dir():
    all_exp_dir = "./zout/zruns"
    TimelineReportGen.gen_TlReport_muti_exps(all_exp_dir, table_mode="p")

def test_gen_from_csv():
    csv_path = "./zout/zruns/MainPC__ds_UFireIndoorVal__mt_temp_method_motion_block__230bd6dcd6b2__20260204.045007/timeline_report.csv"
    output_html_path = "./zout/timeline_report_reconstructed.html"
    # get name of parent dir of the csv file
    csv_parent_dir = Path(csv_path).parent
    TimelineReportGen.timeline_from_tlreport_df(
        csv_path, output_html_path, title=f"[Reconstructed] {csv_parent_dir.name}"
    )
    pprint_local_path(output_html_path, get_wins_path=True)

if __name__ == "__main__":
    test_gen_from_dir()
    test_gen_from_csv()