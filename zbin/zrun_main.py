import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # To make sure it work in muti-gpu env

from halib import *
from tap import *
from halib.exp.core.param_gen import ParamGen

"""
Usage examples:
    # Run everything (default)
    python zbin/zrun_exp.py

    # Filter by model name (e.g., YOLO)
    python zbin/zrun_exp.py -f yolo

    # Filter by sweep parameter (e.g., specific learning rate)
    python zbin/zrun_exp.py -f lr_01

    # Dry run to see which commands would execute
    python zbin/zrun_exp.py -f yolo -d

    # Run with line profiler and precomputed inferences
    python zbin/zrun_exp.py -p -pc /path/to/inferences
"""


class CustomArgs(Tap):
    run_cfg_yaml: str = "config/zruns/zrun_cfg.yaml"
    filter: str = ""  # Replaced base/sweep patterns with a single filter
    use_line_profiler: bool = False
    is_optim_mode: bool = False
    pre_computed_no_skip_dir: str = ""
    dry_run: bool = False

    def configure(self):
        self.add_argument("-r", "--run_cfg_yaml")
        self.add_argument(
            "-f", "--filter", help="Substring filter for the command line"
        )
        self.add_argument("-p", "--use_line_profiler")
        self.add_argument("-opt", "--is_optim_mode", action="store_true")
        self.add_argument("-precomputed", "--pre_computed_no_skip_dir")
        self.add_argument(
            "-d",
            "--dry_run",
            action="store_true",
            help="Print commands without running",
        )


def main():
    # Parse arguments
    args = CustomArgs().parse_args()
    run_cfg_yaml = args.run_cfg_yaml
    run_cfgs = ParamGen.from_files(sweep_yaml=run_cfg_yaml).expand()
    cmd_str = "zbin/run_multi.py --base_yaml {base_yaml} --sweep_yaml {sweep_yaml}"
    for run_cfg in run_cfgs:
        base_yaml = run_cfg["base_yaml"]
        sweep_yaml = run_cfg["sweep_yaml"]

        cmd_str_run = cmd_str.format(base_yaml=base_yaml, sweep_yaml=sweep_yaml)

        if args.filter and (args.filter not in cmd_str_run):
            continue

        if args.is_optim_mode:
            cmd_str_run += " --is_optim_mode"
        if args.pre_computed_no_skip_dir:
            cmd_str_run += (
                f" --pre_computed_no_skip_dir {args.pre_computed_no_skip_dir}"
            )
        console.rule(f"Running command: {cmd_str_run}")
        with ConsoleLog("Run command", characters="▶"):
            pprint(cmd_str_run)
        pprint_box(run_cfg, title="Run Config")
        if args.use_line_profiler:
            cmd_str_run = f"python -m kernprof -l {cmd_str_run}"
        else:
            cmd_str_run = f"python {cmd_str_run}"

        if args.dry_run:
            console.log(f"[yellow]DRY RUN:[/yellow] {cmd_str_run}")
            continue

        os.system(cmd_str_run)


if __name__ == "__main__":
    main()
