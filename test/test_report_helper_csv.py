import sys

sys.path.append("/mnt/e/SyncData/paper2_main")
from collections import OrderedDict
from halib import *
from src.results.timeline.tl_report import TlReportGen


baseline_dir = (
    "MainPC__ds_UFireIndoorVal__mt_no_temp_method__af4b0d32a3d2__20260204.045007"
)
temp_dir = "MainPC__ds_UFireIndoorVal__mt_temp_method_motion_block__8a43c37ef1c3__20260204.045008"

test_dict = OrderedDict(
    {
        "Temp Method Motion Block": temp_dir,
        "Baseline No Temp Method": baseline_dir,
    }
)
list_items = list(test_dict.items())
list_items = list_items[:-1]  # Test only first n-1 items
DO_NORMALIZE = True

for name, dir in list_items:
    console.rule()
    print(f"{name}: {dir}")
    df, timeline_types = TlReportGen.get_tl_df_by_exp_dir(
        f"./zout/zruns/{dir}",
    )

    pprint(timeline_types)
    csvfile.fn_display_df(df.head(10))
    unique_by_cols = TlReportGen.get_unique_values(df)
    pprint(unique_by_cols)
    outfile = os.path.abspath(f"./zout/out.csv")
    df.to_csv(outfile, sep=";", index=False, encoding="utf-8")
    pprint_local_path(outfile, get_wins_path=True)
