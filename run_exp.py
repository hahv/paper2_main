from halib import *
import sys

# ! add <<prj_root>> to sys.path so we can import modules from src folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tap import Tap
from src.config import Config
from src.exp import MyExp

class RunExp(Tap):
    cfg: str = r"config/run_base.yaml"  # config file path


def run_single_exp(exp_cfg_file, method_cfg_dict=None):
    # ! load all config for the Experiment from single yaml file 
    cfg = Config.from_custom_yaml_file(exp_cfg_file)
    experiment = MyExp(cfg)
    if method_cfg_dict is not None:
        cfg.methodCfg.extra_cfgs.update(method_cfg_dict)  # ty:ignore[unresolved-attribute]
    
    # update the method config with new hyperparameters
    metric_rs = experiment.run_exp(
        do_calc_metrics=cfg.inferCfg.calc_metrics, outdir=cfg.get_outdir()
    )
    return metric_rs

def main():
    args = RunExp().parse_args()
    cfg_file = args.cfg
    run_single_exp(cfg_file)

if __name__ == "__main__":
    main()
