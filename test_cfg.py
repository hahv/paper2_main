from halib import *
from argparse import ArgumentParser
from temporal.config import *


def parse_args():
    parser = ArgumentParser(description="desc text")
    parser.add_argument("-cfg", "--cfg", type=str, default="./config/base.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg_file = args.cfg
    cfg = Config.from_custom_yaml_file(cfg_file)
    pprint(cfg)


if __name__ == "__main__":
    main()
