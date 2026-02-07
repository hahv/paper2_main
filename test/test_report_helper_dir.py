import sys

sys.path.append("/mnt/e/SyncData/paper2_main")
from halib import *
from src.results.timeline.report_helper import TimelineReportGen

all_exp_dir = "./zout/zruns"
TimelineReportGen.gen_TlReport_muti_exps(all_exp_dir, table_mode="p")