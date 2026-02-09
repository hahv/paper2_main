from halib import *
from src.results.timeline.report_helper import TlReportGen

exp_dir = r"./zout/zruns/4GPU_SV__ds_UFireIndoorFull__mt_no_temp_method__af4b0d32a3d2__20260209.142136"
TlReportGen.gen_TlReport_exp(exp_dir)
# TlReportGen.tlreport_from_csv(f"{exp_dir}/timeline_report.csv")
