from halib import *
from argparse import ArgumentParser
from temporal.det_proposed import ProposedDetector
from temporal.config_bk import Config

from halib.research.perfcalc import PerfCalc
from halib.research.perftb import PerfTB


def parse_args():
    parser = ArgumentParser(
        description="desc text")
    parser.add_argument(
        "-indir", "--indir", type=str, help="arg1 description", default="./zout"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    indir = args.indir
    exp_dirs = []
    subdirs = fs.list_dirs(indir)
    # ignore unrelated subdirectories
    for subdir in subdirs:
        if not subdir.startswith('__'):
            exp_dirs.append(os.path.abspath(os.path.join(indir, subdir)))

    RAW_ALL_PERF_DIR = "./zreport/raw"
    # IGNORE_DS = ["pexels"]
    IGNORE_DS = [] # no ignored datasets
    # clear old performance files
    perf_files = fs.filter_files_by_extension(RAW_ALL_PERF_DIR, ".csv")
    for perf_file in perf_files:
        fs.delete_file(perf_file)  # clear old performance files
    for i, exp_dir in enumerate(exp_dirs):
        console.rule(f'[{i+1}/{len(exp_dirs)}] Proc: {exp_dir}')
        try:
            cfg = Config.from_custom_yaml_file(os.path.join(exp_dir, "__config.yaml"))
            cfg.infer.calc_perf_metrics = True  # ensure performance metrics

        except Exception as e:
            console.print(f"[red]Error processing {exp_dir}: {e}")
            continue
        if cfg.dataset.target_ds_name in IGNORE_DS:
            console.print(f"[yellow]Skipping dataset {cfg.dataset.target_ds_name} in {exp_dir}")
            continue
        assert os.path.exists(cfg.get_output_dir()), f"Output directory {cfg.get_output_dir()} does not exist."
        # pprint(cfg)
        detector = ProposedDetector(cfg)
        outdir = cfg.get_output_dir()
        outdir_name = os.path.basename(outdir)
        outfile = os.path.abspath(os.path.join(RAW_ALL_PERF_DIR, f"{outdir_name}.csv"))
        # pprint(outfile)
        detector.calculate_exp_perf_metrics(outfile=outfile)

    perftb = PerfCalc.gen_perf_report_for_multip_exps(indir=RAW_ALL_PERF_DIR)
    perftb.display()
    perftb.to_csv(f'./zreport/report_all.csv')
    def custom_sort_fn(method_ls):
        # Custom sorting function to sort by dataset name
        sorted_list = ['no_temp', 'temp_tpt', 'temp_stabilize']
        sorted_ls = []
        for method in sorted_list:
            if method in method_ls:
                sorted_ls.append(method)
        assert len(sorted_ls) == len(method_ls), "Custom sort function did not return all methods."
        return sorted_ls
    perftb.plot(save_path='./zreport/report_all.svg', open_plot=True, custom_sort_exp_fn=custom_sort_fn)

if __name__ == "__main__":
    main()
