from halib import *
from argparse import ArgumentParser

"""
    This scripts is used to make image dataset of blocks for fire/smoke detection. Each image is a block (e.g. 64x64 pixels) can ether labeled as fire/smoke or not.
    Step to build the dataset  using yaml config file:
        1. Collect Object detection image dataset that contain bboxes of fire/smoke.
        2. Import them into `FiftyOne` object detection dataset.
        3. Process all Fiftyone dataset images -> Random create blocks of fire/smoke (using the bboxes info) and also non-fire/smoke blocks (randomly crop from the image, but not overlapping with any fire/smoke bboxes) -> The final output is a folder contain:
            - fire_smoke: folder contain all fire/smoke blocks (how much?)
            - non_fire_smoke: folder contain all non-fire/smoke blocks (how much?)
        4. Stratified split the dataset into train/val/test sets. Final output is a folder contain:
            - train: folder contain train set
            - val: folder contain val set
            - test: folder contain test set
            Each of train/val/test folder contain:
                    - fire_smoke: folder contain all fire/smoke blocks
                    - non_fire_smoke: folder contain all non-fire/smoke blocks
"""

def parse_args():
    parser = ArgumentParser(
        description="desc text")
    parser.add_argument('-cfg', '--cfg', type=str, default="./config/zds_cfg.yaml",
                        help='Configuration file')


def main():
    args = parse_args()
    arg1 = args.argument1
    arg2 = args.argument2
    arglist = args.arglist

    pass


if __name__ == "__main__":
    main()