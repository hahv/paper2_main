# import os

# os.environ["OPENCV_FFMPEG_DEBUG"] = "1"
# os.environ["OPENCV_LOG_LEVEL"] = "VERBOSE"
from lightning.pytorch.loggers import WandbLogger

from halib import *
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tap import *
from typing import List
from halib.exp.core.param_gen import ParamGen
from src.config import Config
from src.exp import Paper2Exp
from src.utils import clear_slack_channel

from halib.filetype import yamlfile


class RunOptimArgs(Tap):
    base_yaml: str = r"./config/zruns/run_base.yaml"
    sweep_yaml: str = r"./config/zruns/run_optim.yaml"
    clean_slack: bool = False


def get_opt_cfg(method_name: str):
    BASE_CFG_OPTIM = "config/zruns/optim"
    # temp_method_motion_block, temp_method_rule_block
    opt_cfg = os.path.join(
        BASE_CFG_OPTIM, f"opt_{method_name.replace('temp_method_', '')}.yaml"
    )
    if not os.path.exists(opt_cfg):
        return None
    return opt_cfg


def get_wandb_params_cfg(method_name: str):
    BASE_CFG_OPTIM = "config/zruns/optim"
    # temp_method_motion_block, temp_method_rule_block
    opt_cfg = os.path.join(
        BASE_CFG_OPTIM, f"keep_opt_{method_name.replace('temp_method_', '')}.yaml"
    )
    if not os.path.exists(opt_cfg):
        return None
    return opt_cfg


def get_sweep_cfgs(base_cfg: dict, method_name: str) -> List[Config]:
    ls_cfg: List[Config] = []
    return ls_cfg


def send_slack_noti(logger: WandbLogger, message: str):
    logger.experiment.alert("Run Optim", message, level="INFO", wait_duration=0.001)
    time.sleep(0.5)


def main():
    args = RunOptimArgs().parse_args()
    base_yaml = args.base_yaml
    sweep_yaml = args.sweep_yaml
    clean_slack = args.clean_slack

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
        if cfg_item.general.is_optim_mode:
            opt_cfg_path = get_opt_cfg(method_name)  # ty:ignore[invalid-argument-type]

        if method_name not in cfg_stats:
            cfg_stats[method_name] = 0
        if opt_cfg_path is None:
            if cfg_item.general.is_optim_mode:
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
            optim_cfgs = optim_param_gen.expand()
            for optim_param_set in optim_cfgs:
                base_cfg = Config.from_custom_yaml_file_or_str(
                    cfg_item.original_yaml_str  # ty:ignore[invalid-argument-type]
                )
                # ! only update the content of extra_cfgs
                optim_params = optim_param_set["extra_cfgs"]
                base_cfg.update_optim_params(optim_params)
                all_optim_run_cfgs.append(base_cfg)
            cfg_stats[method_name] += len(optim_cfgs)

    num_cfg_str = f"Total {len(all_optim_run_cfgs)} configs to run"
    console.rule(num_cfg_str)

    # ! Clear slack channel for new run (to make it less noisy)
    if clean_slack:
        clear_slack_channel()
    with ConsoleLog("Config Stats"):
        df_stats = pd.DataFrame.from_dict(
            cfg_stats,
            orient="index",
            columns=["Num Configs"],  # ty:ignore[invalid-argument-type]
        )
        csvfile.fn_display_df(df_stats)

    cfg_stats_str = "Config Stats:\n" + str(cfg_stats)
    ls_meta_info_to_send = [num_cfg_str, cfg_stats_str]

    did_send_meta = False

    for idx, config in tqdm(enumerate(all_optim_run_cfgs)):
        current_cfg: Config = config
        cfg_name = current_cfg.get_cfg_name()
        cfg_wandb_logger: WandbLogger = current_cfg.get_wandb_logger(name=cfg_name)

        if not did_send_meta:
            for cfg_run_status in ls_meta_info_to_send:
                send_slack_noti(cfg_wandb_logger, cfg_run_status)
            did_send_meta = True

        cfg_run_status = f"Running config {idx + 1}/{len(all_optim_run_cfgs)}"
        console.rule(cfg_run_status)

        # exp_dict = current_cfg.methodCfg.get_dict()
        # pprint(exp_dict)

        send_slack_noti(cfg_wandb_logger, cfg_run_status)
        cfg_wandb_logger.experiment.log({"msg": cfg_run_status})

        method_name = current_cfg.methodCfg.name
        wandb_params_cfg_path = get_wandb_params_cfg(method_name)
        wandb_params_dict = {}
        if wandb_params_cfg_path:
            wandb_params_cfg_dict = yamlfile.load_yaml(
                wandb_params_cfg_path, to_dict=True
            )
            wandb_params_dict = current_cfg.methodCfg.get_wandb_dict(
                config_mask=wandb_params_cfg_dict["extra_cfgs"]
            )
        single_exp = Paper2Exp(current_cfg, wandb_logger=cfg_wandb_logger)
        single_exp.run_exp()

        # ! log the selected hyper-parameters to wandb
        if len(wandb_params_dict) > 0:
            cfg_wandb_logger.log_hyperparams(wandb_params_dict)

        # ! TODO: need to save perf metrics too

        cfg_wandb_logger.experiment.finish()


if __name__ == "__main__":
    main()
