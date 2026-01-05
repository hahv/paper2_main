from halib import *  # noqa: F403
from halib.filetype import yamlfile
from argparse import ArgumentParser
from src.config import Config
from typing import List
from itertools import product

from src.exp import Paper2Exp
from halib.exp.core.param_gen import ParamGen


def parse_args():
    parser = ArgumentParser(description="desc text")
    parser.add_argument(
        "-indir", "--indir", type=str, help="exp dir", default="./config/zruns"
    )
    return parser.parse_args()


def get_cfg_run_list(search_space: dict, base_cfg_dict: dict) -> List[Config]:
    """
    Given a search space and a base config dict, generate a list of Config objects.
    """
    keys = search_space.keys()
    values = search_space.values()
    # list all combinations
    cfg_list_dicts = [dict(zip(keys, combo)) for combo in product(*values)]
    # pprint(cfg_list_dicts)
    # replace values in base_cfg_dict and create Config objects
    cfg_list: List[Config] = []
    for cfg_combine in cfg_list_dicts:
        # create a new config dict
        new_cfg_dict = base_cfg_dict.copy()
        for key, value in cfg_combine.items():
            # set the selected_* field
            selected_field = f"{key}_selector"
            new_cfg_dict[selected_field][f"selected_{key}"] = value
        # create Config object
        cfg_obj = Config.from_custom_yaml_file_or_str(new_cfg_dict)
        cfg_list.append(cfg_obj)
    return cfg_list


def main():
    args = parse_args()
    indir = args.indir
    run_files = fs.filter_files_by_extension(
        directory=indir, ext="yaml", recursive=False
    )
    num_cfgs = len(run_files)

    RUN_SPACE = ParamGen.build_from_file("./config/zruns/_run_gen.yaml")
    base_yaml = "config/zruns/__base.yaml"
    base_cfg_dict = yamlfile.load_yaml(base_yaml, to_dict=True)
    # ! force base cfg to have time_stamp, to make sure each run using the same base cfg gets a unique time_stamp
    base_cfg_dict["general"]["time_stamp"] = now_str()
    if "__base__" in base_cfg_dict:
        del base_cfg_dict["__base__"]

    ls_run_cfgs: List[Config] = get_cfg_run_list(RUN_SPACE, base_cfg_dict)
    assert len(ls_run_cfgs) > 0, "No configs to run!"

    console.rule(f"Total {len(ls_run_cfgs)} configs to run")
    for idx, config in tqdm(enumerate(ls_run_cfgs)):
        console.rule(f"Run [{idx + 1}/{num_cfgs}] - {config.cfg_name}")
        single_exp = Paper2Exp(config)
        single_exp.run_exp()


if __name__ == "__main__":
    main()
