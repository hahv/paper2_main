from halib import *  # noqa: F403
from halib.filetype import yamlfile
from argparse import ArgumentParser
from src.config import Config
from typing import List
from itertools import product

from src.exp import Paper2Exp
from halib.exp.core.param_gen import ParamGen
from tap import *

class MultipleExpArgs(Tap):
    indir: str = "./config/zruns"  # input dir of run configs

def update_base_cfg_fn(base_cfg: dict, combination_item: dict):
    for key, value in combination_item.items():
        selected_field = f"{key}_selector"
        base_cfg[selected_field][f"selected_{key}"] = value
    return base_cfg

def get_cfg_run_list(base_cfg_dict: dict, param_file: str) -> List[Config]:
    """
    Given a search space and a base config dict, generate a list of Config objects.
    """
    ls_dict_cfgs = ParamGen.expand_from_file(
        base_cfg=base_cfg_dict,
        params_file=param_file,
        update_base_cfg_fn=update_base_cfg_fn,
    )
    ls_run_cfgs: List[Config] = []
    for cfg in ls_dict_cfgs:
        cfg_obj = Config.from_custom_yaml_file_or_str(cfg)
        ls_run_cfgs.append(cfg_obj)
    return ls_run_cfgs


def main():
    args = MultipleExpArgs().parse_args()
    indir = args.indir
    run_files = fs.filter_files_by_extension(
        directory=indir, ext="yaml", recursive=False
    )
    num_cfgs = len(run_files)

    base_yaml = "config/zruns/__base.yaml"
    base_cfg_dict = yamlfile.load_yaml(base_yaml, to_dict=True)
    # ! force base cfg to have time_stamp, to make sure each run using the same base cfg gets a unique time_stamp
    base_cfg_dict["general"]["time_stamp"] = now_str()
    if "__base__" in base_cfg_dict:
        del base_cfg_dict["__base__"]

    param_file = r"./config/zruns/_run_gen.yaml"
    ls_run_cfgs: List[Config] = get_cfg_run_list(base_cfg_dict, param_file)
    assert len(ls_run_cfgs) > 0, "No configs to run!"

    console.rule(f"Total {len(ls_run_cfgs)} configs to run")
    for idx, config in tqdm(enumerate(ls_run_cfgs)):
        console.rule(f"Run [{idx + 1}/{num_cfgs}] - {config.cfg_name}")
        single_exp = Paper2Exp(config)
        single_exp.run_exp()


if __name__ == "__main__":
    main()
