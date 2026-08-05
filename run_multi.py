# import os

# os.environ["OPENCV_FFMPEG_DEBUG"] = "1"
# os.environ["OPENCV_LOG_LEVEL"] = "VERBOSE"

from halib import *
import sys

# ! add <<prj_root>> to sys.path so we can import modules from src folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tap import *
from typing import List, Dict, Any
from halib.exp.core.param_gen import ParamGen
from src.config import Config
from src.exp import MyExp

class RunMultiArgs(Tap):
    base_yaml: str = r"./config/run_base.yaml"
    sweep_yaml: str = r"./config/run_multi.yaml"
    pre_computed_no_skip_dir: str = ""

    def configure(self):
        self.add_argument("-b", "--base_yaml")
        self.add_argument("-s", "--sweep_yaml")
        self.add_argument(
            "-pc",
            "--pre_computed_no_skip_dir",
            help="Path to precomputed inferences to skip actual model execution",
        )

def _get_cfg_path(method_name: str, prefix: str) -> str | None:
    """Helper to resolve paths for optim or wandb config files."""
    BASE_CFG_OPTIM = "config/zruns/optim"

    if "temp_baseline" in method_name:
        suffix = method_name.replace("temp_baseline_", "")
        file_name = f"{prefix}baseline_{suffix}.yaml"
    elif "temp_method" in method_name:
        suffix = method_name.replace("temp_method_", "")
        file_name = f"{prefix}{suffix}.yaml"
    else:
        return None
    # pprint(locals())
    # assert 0, "Debugging: Check the method name parsing logic and the generated file name."

    cfg_path = os.path.join(BASE_CFG_OPTIM, file_name)
    return cfg_path if os.path.exists(cfg_path) else None


def get_opt_cfg(method_name: str) -> str | None:
    return _get_cfg_path(method_name, "opt_")

def main():
    args = RunMultiArgs().parse_args()
    base_yaml = args.base_yaml
    sweep_yaml = args.sweep_yaml

    ls_run_dicts = ParamGen.from_files(
        sweep_yaml=sweep_yaml, base_yaml=base_yaml
    ).expand()

    initial_ls_run_cfgs: List[Config] = [
        Config.from_custom_yaml_file_or_str(cfg) for cfg in ls_run_dicts
    ]

    all_optim_run_cfgs: List[Config] = []
    cfg_stats = {}


    for idx, cfg_item in enumerate(initial_ls_run_cfgs):
        method_name = cfg_item.methodCfg.name

        opt_cfg_path = None
        use_optim_mode = cfg_item.general.is_optim_mode

        if method_name not in cfg_stats:
            cfg_stats[method_name] = 0
        if opt_cfg_path is None:
            if use_optim_mode:
                console.log(
                    f"[yellow]No optimization config found for method {method_name}, skipping hyper-parameter optimization.[/yellow]"
                )
            all_optim_run_cfgs.append(cfg_item)
            cfg_stats[method_name] += 1
        else:
            console.log(
                f"[green]Found optimization config for method {method_name}: {opt_cfg_path}[/green]"
            )
            optim_param_gen = ParamGen.from_files(
                sweep_yaml=opt_cfg_path, base_yaml=None
            )

            # ! Even we declare the search space in the optim config
            # ! (in `config/optim`), we may want to filter out some invalid
            # ! combinations of hyperparams, this func allows us to do that.
            def filter_fn(flatten_dict: Dict[str, Any]) -> bool:
                if method_name == "temp_method_pcheck_prof_win_vote_skip_eager":
                    window_size = flatten_dict.get(
                        "extra_cfgs.skip_proc.params.window_size", 16
                    )
                    fd_period = flatten_dict.get(
                        "extra_cfgs.skip_proc.params.fd_period", 16
                    )
                    return window_size >= fd_period
                elif method_name == "temp_method_pcheck_streak_count_skip_eager":
                    n_chk = flatten_dict.get(
                        "extra_cfgs.skip_proc.params.n_chk", 50
                    )
                    w_clr = flatten_dict.get(
                        "extra_cfgs.skip_proc.params.w_clr", 7
                    )
                    # fire_confirm_k = flatten_dict.get(
                    #     "extra_cfgs.skip_proc.params.fire_confirm_k", 1
                    # )
                    return (
                        w_clr >= n_chk
                    )
                else:
                    return True

            optim_cfgs = optim_param_gen.expand(filter_fn=filter_fn)
            for optim_param_set in optim_cfgs:
                base_cfg = Config.from_custom_yaml_file_or_str(
                    cfg_item.original_yaml_str
                )
                # ! only update the content of extra_cfgs
                optim_params = optim_param_set["extra_cfgs"]
                base_cfg.update_optim_params(optim_params)

                # ! force base_cfg to be in optimization mode
                base_cfg.general.is_optim_mode = use_optim_mode
                # ! also modifed the output dir to be under optim dir (if in optim mode)
                if use_optim_mode:
                    base_cfg.update_for_optim_mode()
                all_optim_run_cfgs.append(base_cfg)
            cfg_stats[method_name] += len(optim_cfgs)

    num_cfg_str = f"Total {len(all_optim_run_cfgs)} configs to run"
    console.rule(num_cfg_str)

    with ConsoleLog("Config Stats"):
        df_stats = pd.DataFrame.from_dict(
            cfg_stats,
            orient="index",
            columns=["Num Configs"],
        )
        csvfile.fn_display_df(df_stats)

    # with ConsoleLog('All cfgs:'):
    #     for idx, config in tqdm(enumerate(all_optim_run_cfgs)):
    #         pprint(config.methodCfg)
    
    for idx, config in tqdm(enumerate(all_optim_run_cfgs)):
        current_cfg: Config = config
        if args.pre_computed_no_skip_dir:
            current_cfg.inferCfg.pre_computed_no_skip_dir = (
                args.pre_computed_no_skip_dir
            )
            with ConsoleLog("Using pre-computed", characters="*"):
                pprint(args.pre_computed_no_skip_dir)

        cfg_run_status = f"Running config {idx + 1}/{len(all_optim_run_cfgs)}"
        console.rule(cfg_run_status)

        single_exp = MyExp(current_cfg)
        single_exp.run_exp()

if __name__ == "__main__":
    main()
