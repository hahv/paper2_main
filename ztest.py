from cfg_ds import *
from halib import *
from argparse import ArgumentParser


def main():
    ds_cfg = DsConfig.from_yaml_file("./config/zds_cfg.yaml")
    pprint(ds_cfg)


if __name__ == "__main__":
    main()