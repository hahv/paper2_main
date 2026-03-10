import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # To make sure it work in muti-gpu env

from halib import *
from tap import *
from halib.exp.core.param_gen import ParamGen
from halib.filetype.yamlfile import *


class CustomArgs(Tap):
    run_cfg_yaml: str = "zbin/zrun_cfg.yaml"
    base_cfg_pattern: str = ""
    sweep_cfg_pattern: str = ""
    use_line_profiler: bool = False

    def configure(self):
        self.add_argument("-r", "--run_cfg_yaml")
        self.add_argument("-b", "--base_cfg_pattern")
        self.add_argument("-s", "--sweep_cfg_pattern")
        self.add_argument("-p", "--use_line_profiler")


def main():
    # Parse arguments
    args = CustomArgs().parse_args()
    run_cfg_yaml = args.run_cfg_yaml
    run_cfgs = ParamGen.from_files(sweep_yaml=run_cfg_yaml).expand()
    cmd_str = "zbin/run_multi.py --base_yaml {base_yaml} --sweep_yaml {sweep_yaml}"
    for run_cfg in run_cfgs:
        base_yaml = run_cfg["base_yaml"]
        sweep_yaml = run_cfg["sweep_yaml"]
        base_yaml_fname = fs.get_file_name(base_yaml)
        sweep_yaml_fname = fs.get_file_name(sweep_yaml)

        match_base = (not args.base_cfg_pattern) or (
            args.base_cfg_pattern in base_yaml_fname
        )
        match_sweep = (not args.sweep_cfg_pattern) or (
            args.sweep_cfg_pattern in sweep_yaml_fname
        )

        should_run = match_base and match_sweep

        if not should_run:
            continue
        cmd_str_run = cmd_str.format(base_yaml=base_yaml, sweep_yaml=sweep_yaml)
        console.rule(f"Running command: {cmd_str_run}")
        pprint_box(run_cfg, title="Run Config")
        if args.use_line_profiler:
            cmd_str_run = f"python -m kernprof -l {cmd_str_run}"
        else:
            cmd_str_run = f"python {cmd_str_run}"
        os.system(cmd_str_run)


if __name__ == "__main__":
    main()
