from halib import *
from tap import *
from pathlib import Path
from halib.utils.csv_op import *

DEFAULT_TB_DIR = "./paper/4.table"


class CustomArgs(Tap):
    in_path: str = DEFAULT_TB_DIR

    def configure(self):
        self.add_argument(
            "-i",
            "--in_path",
            type=str,
            default=DEFAULT_TB_DIR,
            help="Input path: either a yaml config file or a directory containing yaml files. Default is './paper/4.table'. If a yaml file is provided, it must be inside the default directory. If a directory is provided, it will look for yaml files inside the 'raw' subdirectory.",
        )


def main():
    # Parse arguments
    args = CustomArgs().parse_args()

    in_path = Path(args.in_path).resolve()
    default_tb_path = Path(DEFAULT_TB_DIR).resolve()

    if in_path.is_file():
        assert in_path.is_relative_to(default_tb_path), (
            f"Config file {in_path} must be inside {DEFAULT_TB_DIR}"
        )
        tb_dir = DEFAULT_TB_DIR
        os.chdir(tb_dir)
        yaml_files = [str(in_path)]
    else:
        tb_dir = args.in_path
        os.chdir(tb_dir)
        indir = "./raw"
        yaml_files = fs.filter_files_by_extension(indir, ".yaml", recursive=False)
        assert len(yaml_files) > 0, f"No yaml files found in {indir}"

    outdir = "./out"

    for yaml_file in tqdm(yaml_files):
        # for each yaml file, find the corresponding csv file
        yaml_path = Path(yaml_file)
        yaml_filename = yaml_path.stem
        csv_file = yaml_path.parent / f"{yaml_filename}.csv"
        if not csv_file.exists():
            print(f"Warning: No csv file found for {yaml_file}, skipping...")
            continue

        cmd = f"python -m halib.utils.csv_op -cfg {yaml_file} -i {csv_file} -o {outdir}"
        console.rule(f"Processing {csv_file}")
        os.system(cmd)
    # after processing all files, rename all csv files in the output directory to have name but remove "_raw_" prefix and "_processed_" postfix
    out_csv_files = fs.filter_files_by_extension(outdir, ".csv", recursive=False)
    for out_csv_file in out_csv_files:
        out_csv_filename = Path(out_csv_file).stem
        new_csv_filename = out_csv_filename.replace("_raw_", "").replace(
            "_processed", ""
        )
        new_csv_file = Path(outdir) / f"{new_csv_filename}.csv"
        os.rename(out_csv_file, new_csv_file)


if __name__ == "__main__":
    main()
