from pprint import pp
from halib import *
from src.results.timeline.report_helper import TimelineReportGen

df, timeline_types, unique_by_cols = TimelineReportGen.get_timeline_csv_path_df(
    "./zout/zruns/MainPC__ds_UFireIndoorVal__mt_temp_method_motion_block__8a43c37ef1c3__20260204.045008"
)
csvfile.fn_display_df(df.head(10))
pprint(timeline_types)
pprint(unique_by_cols)
