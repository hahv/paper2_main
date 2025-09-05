from halib import *
from temporal.config import *
from temporal.our_exp import OurExp

def main():
    config = Config.from_custom_yaml_file(r"config/base.yaml")
    experiment = OurExp(config)
    experiment.run_exp(
        do_calc_metrics=config.infer_cfg.calc_metrics, outdir=config.get_outdir()
    )
if __name__ == "__main__":
    main()
