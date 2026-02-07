from pprint import pp
from halib import *
from src.results.timeline.report_helper import TimelineReportGen

df, timeline_types, unique_by_cols = TimelineReportGen.get_timeline_csv_path_df(
    "./zout/zruns/MainPC__ds_UFireIndoorVal__mt_no_temp_method__af4b0d32a3d2__20260204.045007"
)
csvfile.fn_display_df(df.head(10))
pprint(timeline_types)
pprint(unique_by_cols)
