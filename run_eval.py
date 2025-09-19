from halib import *
from temporal.config import *
from temporal.our_exp import OurExp

from halib import *
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(
        description="desc text")
    parser.add_argument(
        "-cfg", "--cfg", type=str, help="config file path", default=r"config/base_eb0.yaml"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg_file = args.cfg
    config = Config.from_custom_yaml_file(cfg_file)
    experiment = OurExp(config)
    experiment.run_exp(
        do_calc_metrics=config.infer_cfg.calc_metrics, outdir=config.get_outdir()
    )

if __name__ == "__main__":
    main()
