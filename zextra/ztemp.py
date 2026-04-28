from pathlib import Path
import sys
from tap import Tap
from typing import Literal

sys.path.append("/mnt/e/SyncData/paper2_main")
from halib import *
from halib.system.path import *
from src.utils import split_task_by_cfg


class CustomArgs(Tap):
    arg_str: str = "MyProject"
    arg_int: int = 32
    verbose: bool = False  # use --verbose to set True


def main():
    args = CustomArgs().parse_args()
    print(f"Parsed arguments: {args}")
    pc_abbr = get_PC_abbr_name()
    print(f"Current PC abbreviation: {pc_abbr}")

    split_result = split_task_by_cfg("config/zruns/ztask_split.yaml", total_exps=100)
    print(f"Split result: {split_result}")


if __name__ == "__main__":
    main()
