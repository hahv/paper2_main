# import os

# os.environ["OPENCV_FFMPEG_DEBUG"] = "1"
# os.environ["OPENCV_LOG_LEVEL"] = "VERBOSE"


from halib import *
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tap import *
from typing import List
from halib.exp.core.param_gen import ParamGen
from src.config import Config
from src.exp import Paper2Exp


class MultipleExpArgs(Tap):
    base_yaml: str = r"./config/zruns/_run_base.yaml"
    sweep_yaml: str = r"./config/zruns/_run_multi_exps.yaml"


def main():
    args = MultipleExpArgs().parse_args()
    base_yaml = args.base_yaml
    sweep_yaml = args.sweep_yaml

    ls_run_dicts = ParamGen.from_files(
        sweep_yaml=sweep_yaml, base_yaml=base_yaml
    ).expand()
    ls_run_cfgs: List[Config] = [
        Config.from_custom_yaml_file_or_str(cfg) for cfg in ls_run_dicts
    ]

    num_cfgs = len(ls_run_cfgs)
    assert num_cfgs > 0, "No configs to run!"

    console.rule(f"Total {num_cfgs} configs to run")
    for idx, config in tqdm(enumerate(ls_run_cfgs)):
        console.rule(f"Run [{idx + 1}/{num_cfgs}] - {config.cfg_name}")
        single_exp = Paper2Exp(config)
        single_exp.run_exp()


if __name__ == "__main__":
    main()
