import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))  # Add parent directory to sys.path

from halib import *
from tap import *
from src.config import *
from src.exp import Paper2Exp
from src.external_exp import ExternalExpRunner

# ---------------------------------------------------------------------------
# Paths — adjust to your local setup
# ---------------------------------------------------------------------------
# Dataset directory that contains the video files and __labels.csv GT files
DATASET_DIR = "./datasets/UFireIndoorFull"

# Pre-existing external experiment directories (no Paper2Exp config required)
FIRENET_EXP_DIR = "./test/custom_exp/firenet"
YOLO_EXP_DIR = "./test/custom_exp/yolov5l_notemp"

TABLE_MODE = "pfc"
TABLE_DECIMALS = 2
VIDEO_NAME_LIMIT = 40


# ---------------------------------------------------------------------------
# Existing Paper2Exp.from_custom_exp demo
# ---------------------------------------------------------------------------

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


def demo_paper2exp_from_custom():
    args = CustomArgs().parse_args()
    testdir = fs.list_dirs(args.parent_dir)[0]  # Assuming there's only one subdirectory
    exp = Paper2Exp.from_custom_exp(exp_dir_path=testdir, expDir_to_cfgFile_fn=custom_exp_dir_to_cfg_file_fn)
    pprint(exp.full_cfg)


# ---------------------------------------------------------------------------
# ExternalExpRunner demos
# ---------------------------------------------------------------------------

def test_external_firenet():
    """
    Run the full pipeline (normalize CSVs, compute __perf*.csv, generate timeline)
    for a pre-existing firenet experiment directory.
    """
    runner = ExternalExpRunner.from_cls_model_dir(
        exp_dir=FIRENET_EXP_DIR,
        dataset_dir=DATASET_DIR,
        exp_name="firenet",
        tl_type="no_skip",
    )
    runner.run(
        table_mode=TABLE_MODE,
        table_decimals=TABLE_DECIMALS,
        video_name_limit=VIDEO_NAME_LIMIT,
    )


def test_external_yolo():
    """
    Run the full pipeline for a pre-existing YOLO OD experiment directory.
    Handles sparse _od.csv files (only detected frames have rows; empty = no detections).
    """
    runner = ExternalExpRunner.from_yolo_dir(
        exp_dir=YOLO_EXP_DIR,
        dataset_dir=DATASET_DIR,
        exp_name="yolov5l_notemp",
        tl_type="no_skip",
    )
    runner.run(
        table_mode=TABLE_MODE,
        table_decimals=TABLE_DECIMALS,
        video_name_limit=VIDEO_NAME_LIMIT,
    )


if __name__ == "__main__":
    # Uncomment the test you want to run:

    # --- Original Paper2Exp from custom dir ---
    # demo_paper2exp_from_custom()

    # --- External experiment: firenet classification model ---
    # test_external_firenet()

    # --- External experiment: YOLO object-detection model ---
    test_external_yolo()
