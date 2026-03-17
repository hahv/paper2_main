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
from src.exp import Paper2Exp
from src.common import GlobalConst
from src.param_select import WeightedSelect

from typing import Union


class ReportArgs(Tap):
    indir: str = "./zout/zruns"  # output dir of runs
    metric_cfg_file: str = "config/metrics/per_frame_metric.yaml"  # metric config directory  # metric config directory
    outdir: str = "./zout/reports"  # report output directory
    now: bool = False  # whether to use current timestamp for report dir
    skip_plot: bool = (
        False  # whether to skip plotting the performance report (only save CSV)
    )
    is_optim_report: bool = False  # whether this report is for optimization (affects how the report is generated)

    def configure(self):
        self.add_argument("-i", "--indir")
        self.add_argument("-m", "--metric_cfg_file")
        self.add_argument("-o", "--outdir")
        self.add_argument("-now", "--now", action="store_true")
        self.add_argument("-noplot", "--skip_plot", action="store_true")
        # The line `self.add_argument("-opt", "--is_optim_report", action="store_true")` in the `ReportArgs` class is defining a command-line argument for the script.
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
    else:
        standard_exp: Paper2Exp = Paper2Exp.from_standard_exp(exp_dir_path=exp_dir)
        pprint(f"[Perf Report Gen] for standard exp: {exp_name}")
        standard_exp.run_exp()

        # check again if the CSV is generated
    if having_perf_csv(exp_dir):
        llogger.info(f"Successfully generated performance CSV for {exp_dir}")
        return True
    else:
        llogger.warning(f"Failed to generate performance CSV for {exp_dir}")
        return False


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
                    llogger.error(
                        f"Error occurred while generating performance CSV for {exp_dir}: {e}"
                    )
                    with ConsoleLog(f"Error details for {exp_dir}"):
                        pprint_stack_trace()


def report_perf(
    indir: str,
    metric_cfg_file: str,
    report_dir: str,
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
    prepare_exp_dir(indir)
    metricSet_df_dict = {}
    for metricSet_name in metricSet_names:
        pattern = f"{metricSet_name}{SEP}perf"
        # pprint(f" Find {pattern} in {indir}")

        # FIX: Capture 'pattern' as a default argument (p=pattern)
        # This freezes the value of 'pattern' at the moment the lambda is created.
        def exp_csv_filter_fn(csv_file_name, p=pattern):
            return p in csv_file_name

        perfTb_by_metric: PerfTB = PerfCalc.get_perftb_for_multi_exps(
            indir, exp_csv_filter_fn=exp_csv_filter_fn
        )
        outfile = os.path.join(report_dir, f"perf_report__{metricSet_name}.csv")
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
    return metricSet_df_dict


#! ================= OPTIMIZATION REPORT =================
def get_total_skip_rate(csv_path: str) -> dict:
    assert os.path.exists(csv_path), f"File at path '{csv_path}' does not exist"

    default_result = {GlobalConst.COL_PARAM_SKIP_RATE: 0.0}

    try:
        # Read the CSV with the first two rows as multi-index header
        df = pd.read_csv(csv_path, sep=";", header=[0, 1], encoding="utf-8")
    except Exception:
        return default_result

    if df.empty:
        return default_result

    # Find the target column
    target_col = None
    for col in df.columns:
        # col is a tuple: (level_0, level_1)
        level_0, level_1 = str(col[0]).strip(), str(col[1]).strip()
        if level_0.startswith("temp_method_") and level_1 == "Correct Skip":
            target_col = col
            break

    if not target_col:
        return default_result

    # The first column usually contains the row labels like 'TOTAL', 'aihub__lb_fire__0182.mp4', etc.
    first_col = df.columns[0]

    # Find the row where the first column is 'TOTAL'
    total_row = df[df[first_col].astype(str).str.strip() == "TOTAL"]

    if total_row.empty:
        return default_result

    correct_skip_val = total_row[target_col].iloc[0]
    method_name = target_col[
        0
    ].strip()  # Extract method name from level_0 of the column header

    def get_percent(text):
        """
        Extracts the number before '%' from a string like '48.7500% (351)'
        Returns float or 0.0 if not found.
        """
        if not isinstance(text, str):
            return 0.0
        match = re.search(r"(\d+\.?\d*)%", text)
        if match:
            return float(match.group(1)) / 100  # convert to decimal
        return 0.0

    return {
        GlobalConst.METHOD_NAME: method_name,
        GlobalConst.COL_PARAM_SKIP_RATE: get_percent(correct_skip_val),
    }


def norm_optim_df(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"experiment", GlobalConst.COL_PARAM_SKIP_RATE}
    having_required_cols = all(col in df.columns for col in required_cols)
    if not having_required_cols:
        raise ValueError("DataFrame is missing required columns for optimization.")
    NOTEMP_BASELINE = "mt_no_temp_method"
    # df must have a row with column "experiment" that contains NOTEMP_BASELINE in their string (string.contains; not exact match)
    baseline_row = df[
        df["experiment"].str.contains(NOTEMP_BASELINE, case=False, na=False)
    ]
    if baseline_row.empty:
        raise ValueError(
            f"No baseline row found containing '{NOTEMP_BASELINE}' in 'experiment' column."
        )
    # move the baseline row to the top of the DataFrame
    baseline_index = baseline_row.index[0]
    df = pd.concat(
        [df.loc[[baseline_index]], df.drop(index=baseline_index)], ignore_index=True
    )
    return df


def prepare_optim_df(
    indir: str, metric_cfg_file: str, report_dir: str, selected_metricSet="per_frame"
):
    # first get the "per_frame" df
    metricSet_df_dict = report_perf(
        indir, metric_cfg_file, report_dir, save_csv=True, skip_plot=True
    )
    assert selected_metricSet in metricSet_df_dict, (
        f"Selected metric set '{selected_metricSet}' not found in generated metric sets: {list(metricSet_df_dict.keys())}"
    )
    perf_df = metricSet_df_dict[selected_metricSet]

    # # perf_df have two level header, can we just keep level 2
    perf_df.columns = perf_df.columns.get_level_values(1)
    # Turn index → column "experiment" and put it as first column
    perf_df = perf_df.reset_index(names="experiment")
    perf_df = perf_df[
        ["experiment"] + [c for c in perf_df.columns if c != "experiment"]
    ]

    ls_df = []
    for subdir in fs.list_dirs(indir):
        exp_dir = os.path.join(indir, subdir)
        if not os.path.isdir(exp_dir):
            continue

        # find the perf CSV file in this experiment directory for selected_metricSet
        csv_files = fs.filter_files_by_extension(exp_dir, ".csv")
        perf_csv_files = [
            f
            for f in csv_files
            if f"{selected_metricSet}" in f
            and "__perf" in f
            and subdir in f
            and subdir in f  # exp dir name in perf csv file
        ]
        assert len(perf_csv_files) == 1, (
            f"Expected exactly one perf CSV file for {exp_dir} with pattern '{selected_metricSet}' and '__perf', but found {len(perf_csv_files)}: {perf_csv_files}"
        )

        perf_csv_file_path = perf_csv_files[0]
        exp_perf_df = pd.read_csv(perf_csv_file_path, sep=";", encoding="utf-8")

        # Keep non-metric columns and drop dataset
        drop_cols = [col for col in exp_perf_df.columns if "metric_" in col] + [
            "dataset"
        ]
        exp_perf_df.drop(
            columns=[col for col in drop_cols if col in exp_perf_df.columns],
            inplace=True,
        )

        assert len(exp_perf_df) == 1, (
            f"Expected only one row in perf CSV, but got {len(exp_perf_df)} rows in {perf_csv_file_path}"
        )

        tl_report_csv = os.path.join(exp_dir, GlobalConst.TL_CSV_FILE_NAME)
        skip_rate_data = get_total_skip_rate(tl_report_csv)

        exp_perf_df[GlobalConst.COL_PARAM_SKIP_RATE] = skip_rate_data.get(
            GlobalConst.COL_PARAM_SKIP_RATE, 0.0
        )
        # set GlobalConst.SKIP_RATE col as type float
        exp_perf_df[GlobalConst.COL_PARAM_SKIP_RATE] = exp_perf_df[
            GlobalConst.COL_PARAM_SKIP_RATE
        ].astype(float)
        ls_df.append(exp_perf_df)

    extra_data_df = pd.concat(ls_df, ignore_index=True) if ls_df else None
    final_df = perf_df.copy()
    # !debug start
    # perf_df.to_csv(
    # os.path.join(report_dir, f"perf_report__{selected_metricSet}_for_optim.csv"),
    # sep=";",
    # encoding="utf-8",
    # index=False,
    # )
    # extra_data_df.to_csv(  # ty:ignore[unresolved-attribute, unused-ignore-comment]
    # os.path.join(report_dir, f"extra_data__{selected_metricSet}_for_optim.csv"),
    # sep=";",
    # encoding="utf-8",
    # index=False,
    # )
    # !debug end

    if extra_data_df is not None:
        final_df = pd.merge(
            perf_df,
            extra_data_df,
            on="experiment",
            how="inner",
            suffixes=("", "___DROP___"),
        ).filter(regex="^(?!.*___DROP___).*$")

    final_df = norm_optim_df(final_df)
    final_outfile = os.path.join(
        report_dir, f"___raw_rp_optim__{selected_metricSet}.csv"
    )
    final_df.to_csv(final_outfile, sep=";", encoding="utf-8", index=False)
    pprint_local_path(
        final_outfile,
        get_wins_path=True,
        tag_or_box_title="Save optimization report to ⏬:",
    )
    return final_df


def report_optim_by_csv(
    optim_csv_path: Union[str, pd.DataFrame],
    param_select_cfg=r"config/zruns/optim/__param_select.yaml",
    shorten=True,
):
    if isinstance(optim_csv_path, str):
        optim_df = pd.read_csv(optim_csv_path, sep=";", encoding="utf-8")
    else:
        optim_df = optim_csv_path.copy()
    param_select_dict = yamlfile.load_yaml(param_select_cfg, to_dict=True)
    weighted_select = WeightedSelect(optim_df, context=param_select_dict)
    chosen_param_df = weighted_select.choose_params()
    return chosen_param_df

def report_optim(
    indir: str,
    metric_cfg_file: str,
    report_dir: str,
    selected_metricSet="per_frame",
    param_select_cfg=r"config/zruns/optim/__param_select.yaml",
    shorten=True,
):
    optim_df = prepare_optim_df(indir, metric_cfg_file, report_dir, selected_metricSet)
    chosen_param_df = report_optim_by_csv(
        optim_df,
        param_select_cfg=param_select_cfg,
        shorten=shorten,
    )
    chosen_param_outfile = os.path.join(
        report_dir, f"___chosen_rp_optim__{selected_metricSet}.csv"
    )
    chosen_param_df.to_csv(chosen_param_outfile, sep=";", encoding="utf-8", index=False)
    # pprint_local_path(
    #     chosen_param_outfile,
    #     get_wins_path=True,
    #     tag_or_box_title="Save chosen parameters for optimization to ⏬:",
    # )
    return chosen_param_df


def main():
    args = ReportArgs().parse_args()
    indir = args.indir
    metric_dir = args.metric_cfg_file
    report_dir = args.outdir

    if args.now:
        report_dir = os.path.join(report_dir, now_str())

    if args.is_optim_report:
        # ! If it's an optimization report, we might want to generate a different type of report that focuses on optimization results.
        report_optim(indir, metric_dir, report_dir)
    else:
        # Then generate the performance report.
        report_perf(indir, metric_dir, report_dir, skip_plot=args.skip_plot)


if __name__ == "__main__":
    main()
