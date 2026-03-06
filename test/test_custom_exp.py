import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))  # Add parent directory to sys.path

from halib import *
from tap import *
from src.config import *
from src.exp import Paper2Exp
# ---------------------------------------------------------------------------
# Paths — adjust to your local setup
# ---------------------------------------------------------------------------

FIRENET_EXP_DIR = "./test/custom_exp/firenet"
YOLO_EXP_DIR = "./test/custom_exp/yolov5l_notemp"

TABLE_MODE = "pfc"
TABLE_DECIMALS = 2
VIDEO_NAME_LIMIT = 40

DATASET_DIR = "./datasets/UFireIndoorFull"
EXTERNAL_CFG = "config/zruns/run_external.yaml"

class CustomArgs(Tap):
    parent_dir: str = r"./zout/zruns/_baseline/UFireIndoorFull"

# ---------------------------------------------------------------------------
# Approach 2 (new): Paper2Exp.from_custom_exp with placeholder config
# Loader type is auto-detected from the experiment directory name.
# tl_type is always no_skip (set in config/methods/external_method.yaml).
# ---------------------------------------------------------------------------

def _external_cfg_fn(exp_dir_path: str) -> str:
    return EXTERNAL_CFG


def test_paper2exp_firenet():
    """Run the full Paper2Exp pipeline for a firenet external experiment."""
    exp = Paper2Exp.from_custom_exp(
        exp_dir_path=FIRENET_EXP_DIR,
        expDir_to_cfgFile_fn=_external_cfg_fn,
    )
    exp.run_exp()


def test_paper2exp_yolo():
    """Run the full Paper2Exp pipeline for a YOLO OD external experiment."""
    exp = Paper2Exp.from_custom_exp(
        exp_dir_path=YOLO_EXP_DIR,
        expDir_to_cfgFile_fn=_external_cfg_fn,
    )
    exp.run_exp()


def gen_perf_report_custom_exps():
    """Batch: run Paper2Exp.from_custom_exp for every directory under parent_dir."""
    args = CustomArgs().parse_args()
    custom_exp_dirs = fs.list_dirs(args.parent_dir)
    for exp_dir in tqdm(custom_exp_dirs):
        exp_dir_name = fs.get_dir_name(exp_dir)
        console.rule(f'Gen report for exp <<{exp_dir_name}>>')
        exp = Paper2Exp.from_custom_exp(
            exp_dir_path=exp_dir,
            expDir_to_cfgFile_fn=_external_cfg_fn,
        )
        exp.run_exp()


if __name__ == "__main__":
    # --- Generate performance reports for all custom experiments in the parent directory ---
    test_paper2exp_firenet()
    # test_paper2exp_yolo()
    # gen_perf_report_custom_exps()
