from halib import *
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tap import Tap
from src.config import Config
from src.exp import Paper2Exp


class RunExp(Tap):
    cfg: str = r"config/zruns/__base.yaml"  # config file path

def run_single_exp(exp_cfg_file, method_cfg_dict=None):
    cfg = Config.from_custom_yaml_file(exp_cfg_file)
    experiment = Paper2Exp(cfg)
    if method_cfg_dict is not None:
        cfg.methodCfg.extra_cfgs.update(method_cfg_dict)  # ty:ignore[possibly-missing-attribute]
    # update the method config with new hyperparameters
    metric_rs = experiment.run_exp(do_calc_metrics=cfg.inferCfg.calc_metrics, outdir=cfg.get_outdir())
    # pprint(metric_rs)
    return metric_rs


def main():
    args = RunExp().parse_args()
    cfg_file = args.cfg
    run_single_exp(cfg_file)

if __name__ == "__main__":
    main()
