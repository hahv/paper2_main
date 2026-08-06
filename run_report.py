from torch.ao.quantization.fx.utils import return_arg_list
import os
import sys

project_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # Assuming zbin is one level below the project root
sys.path.append(project_dir)  # Add project directory to sys.path

from halib import *  # noqa: F403
from halib.exp.perf.perfcalc import PerfCalc, PerfTB
from halib.filetype import yamlfile
from tap import *
from loguru import logger as llogger
from src.exp import MyExp
from typing import cast

class ReportArgs(Tap):
    indir: str = "./zout"  # input directory containing multiple experiment directories
    # ! experiments in `indir` should use the same metric set defined in `metric_cfg_file` for performance evaluation
    metric_cfg_file: str = "config/metrics/video_metric.yaml"
    outdir: str = "./zout/__report" # output dir for the performance report
    skip_plot: bool = False  # whether to skip plotting the performance metrics
    is_optim_report: bool = False  # whether to generate an optimization report instead of a performance report
    def configure(self):
        self.add_argument("-i", "--indir")
        self.add_argument("-m", "--metric_cfg_file")
        self.add_argument("-o", "--outdir")
        self.add_argument("-noplot", "--skip_plot", action="store_true")
        self.add_argument("-opt", "--is_optim_report", action="store_true")

#! ================= PERFORMANCE REPORT =================
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


def gen_exp_perf_csv(exp_dir: str) -> bool:
    """
    Generates performance CSV files for the given experiment directory.
    This is a placeholder function and should be implemented with the actual logic to generate CSV files.
    """
    SEP = "__"
    patterns = [f"{SEP}mt_", f"{SEP}ds_"]
    exp_name = fs.get_dir_name(exp_dir)
    standard_exp = all(p in exp_name for p in patterns)
    assert not standard_exp, (
        f"Exp '{exp_name}' does not have a valid directory name for standard experiment"
    )
    standard_exp: MyExp = MyExp.from_standard_exp(exp_dir_path=exp_dir)
    pprint(f"[Perf Report Gen] for standard exp: {exp_name}")
    standard_exp.run_exp()
    if having_perf_csv(exp_dir):
        llogger.info(f"Successfully generated performance CSV for {exp_dir}")
        return True
    else:
        llogger.warning(f"Failed to generate performance CSV for {exp_dir}")
        return False

def prepare_exp_dir(in_dir: str, report_dir: str):
    """
    Prepares the experiment directory for reporting.
    Generates performance CSV files if they don't exist.
    Warns if CSV generation fails.
    """
    exp_dirs = fs.list_dirs(in_dir)
    exp_dirs = [os.path.join(in_dir, d) for d in exp_dirs]
    # remove outdir from exp_dirs if it exists    
    exp_dirs = [d for d in exp_dirs if os.path.abspath(d) != os.path.abspath(report_dir)]
    with ConsoleLog("Preparing exp dirs for reporting..."):
        for exp_dir in tqdm(exp_dirs):
            assert os.path.isdir(exp_dir), f"{exp_dir} is not a directory"
            # ! If the perf CSV file does not exist, generate it
            if not having_perf_csv(exp_dir):
                llogger.info(f"Generating performance CSV for {exp_dir}...")
                try:
                    did_gen = gen_exp_perf_csv(exp_dir)
                    if not did_gen:
                        raise Exception("CSV generation failed")
                except Exception as e:
                    llogger.error(
                        f"Error occurred while generating performance CSV for {exp_dir}: {e}"
                    )
                    with ConsoleLog(f"Error details for {exp_dir}"):
                        pprint_stack_trace()


def report_perf(
    indir: str,
    metric_cfg_file: str,
    report_dir: str,
    is_report_optim: bool = False,
    save_csv: bool = True,
    skip_plot: bool = False,
):
    assert os.path.exists(metric_cfg_file), (
        f"No metric files found in {metric_cfg_file}"
    )
    metric_cfg_dict = yamlfile.load_yaml(metric_cfg_file, to_dict=True)
    metricSet_names = metric_cfg_dict["extra_cfgs"]["mode"]
    SEP = "__"
    box_info = {
        "Input Directory": indir,
        "Metric Directory": metric_cfg_file,
        "Report Directory": report_dir,
        "Metric Set Names": metricSet_names,
    }
    pprint_box(box_info, title="Report Generation Parameters")

    os.makedirs(report_dir, exist_ok=True)

    # !First prepare the experiment directories by generating performance CSV files if they don't exist.
    prepare_exp_dir(indir, report_dir)
    metricSet_df_dict = {}
    for metricSet_name in metricSet_names:
        pattern = f"{metricSet_name}{SEP}perf"
        # pprint(f" Find {pattern} in {indir}")
        # FIX: Capture 'pattern' as a default argument (p=pattern)
        # This freezes the value of 'pattern' at the moment the lambda is created.
        def exp_csv_filter_fn(csv_file_name, p=pattern):
            return p in csv_file_name
        
        raw_df = None
        if is_report_optim:
            perfTb_by_metric, raw_df = PerfCalc.get_perftb_for_multi_exps(
                indir,
                exp_csv_filter_fn=exp_csv_filter_fn,
                show_all_cols=True,
                return_raw_df=True,
            )  # ty:ignore[not-iterable]
        else:
            perfTb_by_metric = PerfCalc.get_perftb_for_multi_exps(
                indir, exp_csv_filter_fn=exp_csv_filter_fn
            )
        outfile = os.path.join(report_dir, f"perf_report__{metricSet_name}.csv")
        # Explicitly cast to PerfTB for downstream code
        perfTb_by_metric = cast(PerfTB, perfTb_by_metric)
        perf_metric_df = perfTb_by_metric.to_csv(outfile)
        pprint_local_path(
            outfile,
            get_wins_path=True,
            tag_or_box_title="Save perfTb to ⏬:",
        )
        if not skip_plot:
            perfTb_by_metric.plot(
                save_path=os.path.join(report_dir, f"perf_report__{metricSet_name}.svg")
            )
        metricSet_df_dict[metricSet_name] = perf_metric_df
        
        # ! for optim case, we also save the raw_df for further analysis
        if is_report_optim and raw_df is not None:
            raw_outfile = os.path.join(report_dir, f"full_perf_report__{metricSet_name}.csv")
            raw_df.to_csv(raw_outfile, index=False, sep=";")
            pprint_local_path(
                raw_outfile,
                get_wins_path=True,
                tag_or_box_title="Save raw perf df to ⏬:",
            )
    return metricSet_df_dict

def main():
    args = ReportArgs().parse_args()
    indir = args.indir
    metric_dir = args.metric_cfg_file
    report_dir = args.outdir
    is_optim_report = args.is_optim_report
    report_perf(
        indir, metric_dir, report_dir, is_report_optim=is_optim_report, skip_plot=args.skip_plot
    )

if __name__ == "__main__":
    main()
