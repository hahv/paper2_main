from halib import *
from argparse import ArgumentParser
from src.db_anno.newdb_labeler import *
import sys


def parse_args():
    parser = ArgumentParser(description="desc text")
    parser.add_argument(
        "-cfg",
        "--cfg",
        type=str,
        help="arg1 description",
        default="config/db_anno/__db_cfg.yaml",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg_yaml = args.cfg
    anno_provider = NewDBLabeler(cfg_yaml)
    anno_provider.process_labeling(max_workers=0)


if __name__ == "__main__":
    main()
