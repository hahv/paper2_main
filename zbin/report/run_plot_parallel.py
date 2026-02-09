from halib import *  # noqa: F403
from halib.utils.plotly_op import PlotlyUtils
from collections import OrderedDict
from pathlib import Path
from tap import *


class CustomArgs(Tap):
    # --- Basic Types ---
    infile: Path = Path("./zout/wandb_export.csv")  # Path to the CSV file
    csv_sep: str = ";"  # CSV separator


def exp_id_formatter_func(exp_id: str) -> str:
    return exp_id[:6] if len(exp_id) >= 6 else exp_id


ALL_COLS = [
    "Name",
    "State",
    "Notes",
    "User",
    "Tags",
    "Created",
    "Runtime",
    "Sweep",
    "method_name",
    "skip_proc.params.block_ratio_th",
    "skip_proc.params.block_size_orig",
    "skip_proc.params.min_roi_ratio",
    "skip_proc.params.motion.name",
    "skip_proc.params.motion.params.diff_frame_th",
    "skip_proc.params.scale_factor",
    "msg",
    "per_frame_metric_FPR (False Alarm Rate)",
    "per_frame_metric_FPS",
    "per_frame_metric_accuracy",
    "per_frame_metric_f1_score",
    "per_frame_metric_precision",
    "per_frame_metric_recall (TPR)",
    "per_video_metric_FPR (False Alarm Rate)",
    "per_video_metric_FPS",
    "per_video_metric_accuracy",
    "per_video_metric_f1_score",
    "per_video_metric_precision",
    "per_video_metric_recall (TPR)",
    "trainer/global_step",
]

METRIC_MAP = OrderedDict(
    {
        "metric_accuracy": "acc",
        "metric_f1_score": "f1",
        "metric_precision": "prec",
        "metric_recall (TPR)": "rec",
        "metric_FPR (False Alarm Rate)": "fpr",
        "metric_FPS": "fps",
    }
)

EXCLUDE_COLS = [
    "Notes",
    "State",
    "User",
    "Tags",
    "Created",
    "Runtime",
    "Sweep",
    "method_name",
    "msg",
    "trainer/global_step",
    "skip_proc.params.motion.name",
]


def main():
    args = CustomArgs().parse_args()
    df = pd.read_csv(args.infile, sep=args.csv_sep, encoding="utf-8")
    # Drop rows if df["Name"] is NaN
    # df = df.dropna(subset=["Name"])
    # csvfile.fn_display_df(df.head(5))

    # 1. Column Filtering and Param Shortening
    df = df.drop(columns=[c for c in EXCLUDE_COLS if c in df.columns])

    # Shorten ".params.xxx" to "xxx"
    df.rename(
        columns=lambda x: x.split(".params.")[-1] if ".params." in x else x,
        inplace=True,
    )

    # Clean invalid rows
    df = df.dropna(subset=["per_video_metric_FPS"])

    modes = ["per_video", "per_frame"]
    common_cols = [c for c in df.columns if not c.startswith(tuple(modes))]

    console.rule(f"[red]Number of experiments: <{len(df)}> [/red]")

    for mode in modes:
        # 2. Extract and Normalize Mode Data
        mode_prefix = f"{mode}_metric_"
        mode_specific_cols = [c for c in df.columns if c.startswith(mode)]

        # Select columns and strip the mode prefix
        df_mode = df[common_cols + mode_specific_cols].copy()
        df_mode.rename(
            columns=lambda x: x.replace(mode_prefix, "metric_"), inplace=True
        )

        # 3. Final Metric Renaming and Ordering
        df_mode.rename(columns=METRIC_MAP, inplace=True)

        # Maintain consistent column order: [Common Params] -> [Ordered Metrics]
        metric_cols = [v for v in METRIC_MAP.values() if v in df_mode.columns]
        non_metric_cols = [c for c in df_mode.columns if c not in metric_cols]
        df_mode = df_mode[non_metric_cols + metric_cols]
        # with ConsoleLog(f"Final columns for mode={mode}:"):
        #     pprint(df_mode.index.tolist())

        # 4. Visualization
        console.rule(f"[Mode={mode}] Parallel Plot")
        PlotlyUtils.parallel_plot(
            df_or_csv_file=df_mode,
            exclude_dims=["Name"],
            exp_id_formatter=exp_id_formatter_func,
            color="acc",  # renamed from metric_accuracy
            outdir="zout/reports",
            outfile=f"parallel_plot_{mode}.html",
            title=f"Parallel Plot (Mode={mode})",
            plot_width=1500,
            plot_bar_height=1200,
        )


if __name__ == "__main__":
    main()
