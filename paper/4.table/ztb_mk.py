from halib import *
from tap import *
from pathlib import Path
from halib.utils.csv_op import *


class CustomArgs(Tap):
    tb_dir: str = "./paper/4.table"


def main():
    # Parse arguments
    args = CustomArgs().parse_args()
    # python -m halib.utils.csv_op -cfg test\test_csv_ops\users.yaml -i test\test_csv_ops\users.csv -o test\test_csv_ops
    tb_dir = args.tb_dir
    # change to the table directory
    os.chdir(tb_dir)
    indir = "./raw"
    outdir = "./out"

    # first list all yaml files in the input directory
    yaml_files = fs.filter_files_by_extension(indir, ".yaml", recursive=False)
    assert len(yaml_files) > 0, f"No yaml files found in {indir}"
    for yaml_file in tqdm(yaml_files):
        # for each yaml file, find the corresponding csv file
        yaml_filename = Path(yaml_file).stem
        csv_file = Path(indir) / f"{yaml_filename}.csv"
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
