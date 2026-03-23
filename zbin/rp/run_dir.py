import sys

PROJECT_PATH = "/mnt/e/SyncData/paper2_main"
sys.path.append(PROJECT_PATH)  # Add project directory to sys.path

from halib import *
from tap import *

from zbin.rp.run_report import gen_exp_perf_csv

class CustomArgs(Tap):
    indir: str = "./zout/zruns/4GPU_SV__ds_UFireIndoorFull__mt_no_temp_method__af4b0d32a3d2__20260303.180228"
def main():
    # Parse arguments
    args = CustomArgs().parse_args()
    indir = args.indir
    console.rule(f"Gen perf for exp {indir}")
    gen_exp_perf_csv(indir)
if __name__ == "__main__":
    main()