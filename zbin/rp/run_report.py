import os
import sys

project_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # Assuming zbin is one level below the project root
sys.path.append(project_dir)  # Add project directory to sys.path

from halib import *  # noqa: F403
from halib.exp.perf.perfcalc import PerfCalc, PerfTB
from tap import *
from loguru import logger as llogger
from src.exp import Paper2Exp
from src.common import GlobalConst


class ReportArgs(Tap):
    indir: str = "./zout/zruns"  # output dir of runs
    metricDir: str = "config/metrics"  # metric config directory
    outdir: str = "./zout/reports"  # report output directory
    now: bool = False  # whether to use current timestamp for report dir

    def configure(self):
        self.add_argument("-i", "--indir")
        self.add_argument("-m", "--metricDir")
        self.add_argument("-o", "--outdir")
        self.add_argument("-now", "--now", action="store_true")


def default_exp_csv_filter_fn(exp_file_name: str) -> bool:
    """
    Default filter function for experiments.
    Returns True if the experiment name does not start with "test_" or "debug_".
    """
    return "__perf.csv" in exp_file_name


def having_perf_csv(exp_dir: str) -> bool:
    """
    Checks if the given experiment directory contains any CSV files that match the default performance CSV filter.
    """
    csv_files = fs.filter_files_by_extension(exp_dir, ".csv")
    return any(default_exp_csv_filter_fn(csv_file) for csv_file in csv_files)


def gen_exp_perf_csv(exp_dir: str):
    """
    Generates performance CSV files for the given experiment directory.
    This is a placeholder function and should be implemented with the actual logic to generate CSV files.
    """
    SEP = "__"
    patterns = [f"{SEP}mt_", f"{SEP}ds_"]
    exp_name = fs.get_dir_name(exp_dir)
    standard_exp = all(p in exp_name for p in patterns)

    def get_cfg_fn(exp_dir_path: str) -> str:
        return GlobalConst.EXTERNAL_CFG

    if not standard_exp:
        custom_exp: Paper2Exp = Paper2Exp.from_custom_exp(
            exp_dir_path=exp_dir, find_cfgFile_func=get_cfg_fn
        )
        custom_exp.run_exp()
        # check again if the CSV is generated
        if having_perf_csv(exp_dir):
            llogger.info(f"Successfully generated performance CSV for {exp_dir}")
            return True
        else:
            llogger.warning(f"Failed to generate performance CSV for {exp_dir}")
            return False
    else:
        raise Exception(
            f"Standard exp {exp_name} is missing performance CSV. Please run the experiment with the appropriate settings to generate the CSV."
        )


def prepare_exp_dir(in_dir: str):
    """
    Prepares the experiment directory for reporting.
    Generates performance CSV files if they don't exist.
    Warns if CSV generation fails.
    """
    exp_dirs = fs.list_dirs(in_dir)
    exp_dirs = [os.path.join(in_dir, d) for d in exp_dirs]
    with ConsoleLog("Preparing exp dirs for reporting..."):
        for exp_dir in tqdm(exp_dirs):
            assert os.path.isdir(exp_dir), f"{exp_dir} is not a directory"
            if not having_perf_csv(exp_dir):
                llogger.info(f"Generating performance CSV for {exp_dir}...")
                try:
                    did_gen = gen_exp_perf_csv(exp_dir)
                    if not did_gen:
                        raise Exception("CSV generation failed")
                except Exception as e:
                    llogger.error(f"Error occurred while generating performance CSV for {exp_dir}: {e}")
                    with ConsoleLog(f"Error details for {exp_dir}"):
                        pprint_stack_trace()

def report_perf(indir: str, metric_dir: str, report_dir: str):
    metric_files = fs.filter_files_by_extension(metric_dir, ".yaml")
    assert metric_files, f"No metric files found in {metric_dir}"
    metricSet_names = [
        fs.get_file_name(f, split_file_ext=True)[0].replace("_metric", "")
        for f in metric_files
    ]
    SEP = "__"
    box_info = {
        "Input Directory": indir,
        "Metric Directory": metric_dir,
        "Report Directory": report_dir,
        "Metric Set Names": metricSet_names,
    }
    pprint_box(box_info, title="Report Generation Parameters")
    for metricSet_name in metricSet_names:
        pattern = f"{metricSet_name}{SEP}perf"
        pprint(f" Find {pattern} in {indir}")

        # FIX: Capture 'pattern' as a default argument (p=pattern)
        # This freezes the value of 'pattern' at the moment the lambda is created.
        def exp_csv_filter_fn(csv_file_name, p=pattern):
            return p in csv_file_name

        perfTb_by_metric: PerfTB = PerfCalc.get_perftb_for_multi_exps(
            indir, exp_csv_filter_fn=exp_csv_filter_fn
        )
        outfile = os.path.join(report_dir, f"perf_report__{metricSet_name}.csv")
        perfTb_by_metric.to_csv(outfile)
        pprint_local_path(
            outfile,
            get_wins_path=True,
            tag_or_box_title="Save perfTb to ⏬:",
        )
        perfTb_by_metric.plot(
            save_path=os.path.join(report_dir, f"perf_report__{metricSet_name}.svg")
        )


def main():
    args = ReportArgs().parse_args()
    indir = args.indir
    metric_dir = args.metricDir
    report_dir = args.outdir

    if args.now:
        report_dir = os.path.join(report_dir, now_str())
    os.makedirs(report_dir, exist_ok=True)

    # First prepare the experiment directories by generating performance CSV files if they don't exist.
    prepare_exp_dir(indir)

    # Then generate the performance report.
    report_perf(indir, metric_dir, report_dir)


if __name__ == "__main__":
    main()
