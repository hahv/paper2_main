import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))  # Add parent directory to sys.path

from halib import *
from tap import *
from src.config import *
from src.exp import Paper2Exp


class CustomArgs(Tap):
    parent_dir: str = r"./zout/zruns/_baseline/UFireIndoorFull"
    # indir: str = r"./zout/zruns/_baseline/UFireIndoorFull/firenet"


def custom_exp_dir_to_cfg_file_fn(exp_dir_path: str) -> str:
    # This is a custom function to determine the config file path based on the experiment directory path.
    # For example, if the experiment directory contains "firenet", we can return a specific config file for firenet.
    # if "firenet" in exp_dir_path:
    #     return f"./config/zruns/run_firenet.yaml"
    # Add more conditions here if there are different types of experiments with different config files.

    # Default config file if no specific condition is met
    return f"config/zruns/run_base.yaml"


def main():
    # Parse arguments
    args = CustomArgs().parse_args()
    testdir = fs.list_dirs(args.parent_dir)[0]  # Assuming there's only one subdirectory
    exp = Paper2Exp.from_custom_exp(exp_dir_path=testdir, expDir_to_cfgFile_fn=custom_exp_dir_to_cfg_file_fn)
    pprint(exp.full_cfg)


if __name__ == "__main__":
    main()
