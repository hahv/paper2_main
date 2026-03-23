from halib import *
from tap import *
from pathlib import Path


class CustomArgs(Tap):
    indir: str = "./paper/4.table/raw"
    outdir: str = "./paper/4.table/out"


def main():
    # Parse arguments
    args = CustomArgs().parse_args()
    # python -m halib.utils.csv_op -cfg test\test_csv_ops\users.yaml -i test\test_csv_ops\users.csv -o test\test_csv_ops
    indir = args.indir
    outdir = args.outdir

    # first list all csv files in the input directory
    csv_files = fs.filter_files_by_extension(indir, ".csv", recursive=False)
    assert len(csv_files) > 0, f"No csv files found in {indir}"
    for csv_file in tqdm(csv_files):
        # for each csv file, find the corresponding yaml config file
        csv_filename = Path(csv_file).stem
        yaml_file = Path(indir) / f"{csv_filename}.yaml"
        if not yaml_file.exists():
            print(f"Warning: No yaml config file found for {csv_file}, skipping...")
            continue

        cmd = f"python -m halib.utils.csv_op -cfg {yaml_file} -i {csv_file} -o {outdir}"
        console.rule(f"Processing {csv_file}")
        os.system(cmd)

if __name__ == "__main__":
    main()
