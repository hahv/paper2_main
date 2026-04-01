from halib import *
from tap import *
from pathlib import Path


DEFAULT_PERF_CSV = "./zfps_check.csv"


class CustomArgs(Tap):
    in_path: str = DEFAULT_PERF_CSV
    show_df: bool = False  # use --show_df to set True

    def configure(self):
        self.add_argument(
            "-i",
            "--in_path",
            type=str,
            default=DEFAULT_PERF_CSV,
            help="Input CSV file with columns: total_frames, skipped_frames, dl_avg_infer_ms, skip_module_avg_ms.",
        )
        self.add_argument(
            "--show_df",
            action="store_true",
            help="Display the resulting DataFrame.",
        )


def calculate_and_print(row_dict, show_df=False):
    N = row_dict["total_frames"]
    N_skip = row_dict["skipped_frames"]
    t_dl = row_dict["dl_avg_infer_ms"]
    t_skip_mod = row_dict["skip_module_avg_ms"]

    run_ratio = (N - N_skip) / N
    time_no_skip = t_dl
    time_skip = (t_dl + t_skip_mod) * run_ratio + t_skip_mod * (1 - run_ratio)
    fps_no_skip = 1000 / time_no_skip
    fps_skip = 1000 / time_skip
    speedup = fps_skip / fps_no_skip

    report_dict = {}
    report_dict.update(row_dict)
    report_dict["run_ratio"] = run_ratio
    report_dict["time_no_skip"] = time_no_skip
    report_dict["time_skip"] = time_skip
    report_dict["fps_no_skip"] = fps_no_skip
    report_dict["fps_skip"] = fps_skip
    report_dict["speedup"] = speedup
    report_df = pd.DataFrame([report_dict])
    if show_df:
        # convert this to a dataframe
        from halib.filetype import csvfile

        csvfile.fn_display_df(report_df)
    else:
        pprint_box(report_dict, title="FPS Check Report")
    report_df.to_csv(
        "paper/3.fig/fps_increase/zfps_check_report.csv",
        index=False,
        sep=";",
        encoding="utf-8",
    )


def main():
    args = CustomArgs().parse_args()

    in_path = Path(args.in_path).resolve()
    assert in_path.exists(), f"CSV file not found: {in_path}"
    assert in_path.suffix == ".csv", f"Expected a .csv file, got: {in_path}"

    df = pd.read_csv(in_path, sep=";", encoding="utf-8")
    # get the first row
    row = df.iloc[0]
    # convert row to dict
    calculate_and_print(row.to_dict(), show_df=args.show_df)


if __name__ == "__main__":
    main()
