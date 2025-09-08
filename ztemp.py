from halib import *
from argparse import ArgumentParser
from temporal.our_exp import *
from temporal.metric_src.test_metric_src import *


def main():
    cfg = None
    with timebudget("load cfg"):
        cfg = Config.from_custom_yaml_file(r"./config/base.yaml")
    # current_mt = MethodFactory.create_method(cfg)
    # pprint(current_mt)
    with timebudget("metric src"):
        metric_src = TestDSMetricSrc(cfg)
        metric_data = metric_src.get_data_metrics(
            indir=r"./zout/perf/MainPC__ds_Test__mt_no_temp__20250902.181041"
        )
        pprint(metric_data)


if __name__ == "__main__":
    main()
