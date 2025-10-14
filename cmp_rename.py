from halib import *
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(
        description="desc text")
    parser.add_argument(
        "-indir",
        "--indir",
        type=str,
        help="input directory",
        default=r"/mnt/e/SyncData/paper2_baseline/zout/DFire_Val",
    )
    parser.add_argument('-p', '--prefix', type=str, help='prefix for the trial name', default='trial')
    return parser.parse_args()


def main():
    args = parse_args()
    indir = args.indir
    prefix = args.prefix
    all_dirs = os.listdir(indir)
    all_dirs = sorted(all_dirs) # asc order
    for idx, d in enumerate(tqdm(all_dirs)):
        full_path = os.path.join(indir, d)
        new_name = f'{prefix}_{idx+1:02d}'
        new_full_path = os.path.join(indir, new_name)
        if os.path.isdir(full_path):
            print(f'Renaming {full_path} to {new_full_path}')
            os.rename(full_path, new_full_path)
if __name__ == "__main__":
    main()
