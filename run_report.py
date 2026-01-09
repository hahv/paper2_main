from halib import * # noqa: F403
from halib.exp.perf.perfcalc import PerfCalc, PerfTB
from tap import *

class ReportArgs(Tap):
    indir: str = "./zout/zruns"  # output dir of runs
    metricDir: str = "config/metrics"  # metric config directory
    reportDir: str = "./zout/reports"  # report output directory
    now: bool = False  # whether to use current timestamp for report dir


def default_exp_csv_filter_fn(exp_file_name: str) -> bool:
        """
        Default filter function for experiments.
        Returns True if the experiment name does not start with "test_" or "debug_".
        """
        return "__perf.csv" in exp_file_name

def main():
    args = ReportArgs().parse_args()
    indir = args.indir
    metric_dir = args.metricDir
    report_dir = args.reportDir

    if args.now:
        report_dir = os.path.join(report_dir, now_str())
    os.makedirs(report_dir, exist_ok=True)

    metric_files = fs.filter_files_by_extension(metric_dir, ".yaml")
    assert metric_files, f"No metric files found in {metric_dir}"
    metricSet_names = [fs.get_file_name(f,split_file_ext=True)[0].replace("_metric", "") for f in metric_files]
    SEP= "__"
    for metricSet_name in metricSet_names:
        pattern = f"{metricSet_name}{SEP}perf"
        pprint (f" Find {pattern} in {indir}")
        # FIX: Capture 'pattern' as a default argument (p=pattern)
        # This freezes the value of 'pattern' at the moment the lambda is created.
        def exp_csv_filter_fn(csv_file_name, p=pattern):
            return p in csv_file_name
        perfTb_by_metric: PerfTB = PerfCalc.get_perftb_for_multi_exps(
            indir, exp_csv_filter_fn=exp_csv_filter_fn
        )
        perfTb_by_metric.to_csv(
            os.path.join(report_dir, f"perf_report__{metricSet_name}.csv"))
        perfTb_by_metric.plot(save_path=os.path.join(report_dir, f"perf_report__{metricSet_name}.png"))

if __name__ == "__main__":
    main()
