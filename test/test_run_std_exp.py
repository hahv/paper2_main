import os
import sys

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

from halib import *  # noqa: F403
from halib.system.path import normalize_paths
from tap import Tap

from src.exp import Paper2Exp


class RunStdExpArgs(Tap):
    exp_dir: str = "/mnt/e/SyncData/paper2_main/zout/zruns/4GPU_SV__ds_UFireIndoorFull__mt_no_temp_method__af4b0d32a3d2__20260209.142136"

    def configure(self):
        self.add_argument("-e", "--exp_dir")

def run_standard_exp(exp_dir: str):
    experiment = Paper2Exp.from_standard_exp(exp_dir)
    metric_rs = experiment.run_exp()
    return metric_rs


def main():
    args = RunStdExpArgs().parse_args()
    exp_dir = normalize_paths(args.exp_dir)
    run_standard_exp(exp_dir)

if __name__ == "__main__":
    main()
