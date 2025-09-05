from halib import *
from temporal.config import *
from temporal.our_exp import OurExp

def main():
    config = Config.from_custom_yaml_file(r"config/base.yaml")
    experiment = OurExp(config)
    cal_metrics = config.infer_cfg.calc_metrics
    cal_metrics = False # ! test
    exp_results = experiment.run_exp(
        do_calc_metrics=cal_metrics,
        outdir=config.get_outdir(),
        return_df=True,
    )
    with ConsoleLog("Final results:"):
        if isinstance(exp_results, tuple) and len(exp_results) == 2 and isinstance(exp_results[0], pd.DataFrame):
            csvfile.fn_display_df(exp_results[0])
            pprint_local_path(exp_results[1])
        else:
            pprint(exp_results)

if __name__ == "__main__":
    main()
