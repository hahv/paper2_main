# This script is use Optuna to optimize the hyperparameters
# extra_cfgs:
#   diff_thres: 5           # pixel diff threshold for temporal stabilization
#   frame_diff_cfg: "config/FrameDifference.xml"
#   scale_factor: 1.0      # scale factor for frame diff
#   blk_size: 64             # block size for activity check
#   blk_act_thres: 0.05       # % pixels active in block to mark block active
#   frm_act_thres: 0.5       # % active blocks to mark frame active
#   firesmoke_cls_thres: 0.3      # fire/smoke classification threshold
#   min_roi: 0.7             # min ROI size (relative to frame) for BIG model
#   tiny_model: "models/tinycnn.pth"  # tiny CNN model path

# We need to optimize:
# scale_factor
# blk_size
# blk_act_thres
# frm_act_thres
# firesmoke_cls_thres
# min_roi

from halib import *
from halib.filetype import yamlfile
from argparse import ArgumentParser
import numpy as np

import optuna
from optuna.trial import Trial

from halib import *
from halib.research.params_gen import ParamGen
from temporal.config import *
from temporal.our_exp import OurExp


def parse_args():
    parser = ArgumentParser(description="desc text")
    parser.add_argument(
        "-cfg", "--cfg", type=str, help="config file", default=r"./config/base.yaml"
    )
    parser.add_argument(
        "-optcfg",
        "--optcfg",
        type=str,
        help="optimization config file",
        default=r"./config/opt_cfg.yaml",
    )

    return parser.parse_args()


def prepre_trial_dict(trial: Trial):

    method_cfg_dict = {
        "scale_factor": trial.suggest_categorical(
            "scale_factor", SEARCH_SPACE["scale_factor"]
        ),
        "blk_act_thres": trial.suggest_categorical(
            "blk_act_thres", SEARCH_SPACE["blk_act_thres"]
        ),
        "frm_act_thres": trial.suggest_categorical(
            "frm_act_thres", SEARCH_SPACE["frm_act_thres"]
        ),
        "firesmoke_cls_thres": trial.suggest_categorical(
            "firesmoke_cls_thres", SEARCH_SPACE["firesmoke_cls_thres"]
        ),
    }
    return method_cfg_dict


def run_pipeline(exp_cfg_file, method_cfg_dict):
    config = Config.from_custom_yaml_file(exp_cfg_file)
    experiment = OurExp(config)
    assert (
        config.method_cfg.method_used.name == "temp_stabilize"
    ), "Only temporal stabilization method is supported in this optimization script."
    config.method_cfg.method_used.extra_cfgs.update(
        method_cfg_dict
    )
    # update the method config with new hyperparameters
    metric_rs = experiment.run_exp(
        do_calc_metrics=config.infer_cfg.calc_metrics
    )
    pprint(metric_rs)
    return metric_rs

SEARCH_SPACE = None
current_exp_cfg_file = None

# Define evaluation function
def objective(trial: Trial):
    global SEARCH_SPACE
    num_trials = calc_num_trials(SEARCH_SPACE)

    trial_param_set = {}
    for params in SEARCH_SPACE:
        print(f"{params}: {SEARCH_SPACE[params]}")
        value = trial.suggest_categorical(params, SEARCH_SPACE[params])
        trial_param_set[params] = value

    # ---- Run your pipeline with these hyperparams ----
    global current_exp_cfg_file
    metrics = None
    with ConsoleLog(f"Running trial {trial.number+1}/{num_trials}"):
        print(f"param set :")
        pprint(trial_param_set)
        metrics = run_pipeline(exp_cfg_file=current_exp_cfg_file, method_cfg_dict=trial_param_set
        )

    # Return metric to maximize (e.g., F1 score)
    return np.random.rand()

def calc_num_trials(search_space):
    count = 1
    for param in search_space:
        count *= len(search_space[param])
    return count

def main():
    # Load base config for experiments
    global current_exp_cfg_file
    current_exp_cfg_file = parse_args().cfg
    assert os.path.exists(
        current_exp_cfg_file
    ), f"Config file {current_exp_cfg_file} does not exist."

    # Load optimization config
    opt_cfg_file = parse_args().optcfg
    global SEARCH_SPACE
    SEARCH_SPACE = ParamGen.build_from_file(opt_cfg_file)

    # Create a GridSampler with your space
    sampler = optuna.samplers.GridSampler(SEARCH_SPACE)

    # Persistent storage
    storage_url = "sqlite:////mnt/e/NextCloud/paper2_main/zout/tune/optuna_study.db"

    study = optuna.create_study(
        study_name="fire_smoke_opt_study",
        direction="maximize",
        sampler=sampler,
        storage=storage_url,
        load_if_exists=True,
    )
    # Run optimization
    study.optimize(objective)

    print("Best params:", study.best_params)
    print("Best value:", study.best_value)


if __name__ == "__main__":
    main()
