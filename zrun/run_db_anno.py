from halib import *
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tap import *
from src.db_anno.newdb_labeler import *

class DBAnnoArgs(Tap):
    cfg: str = r"config/db_anno/__db_cfg.yaml"  # config file path

def main():
    args = DBAnnoArgs().parse_args()
    cfg_yaml = args.cfg
    anno_provider = NewDBLabeler(cfg_yaml)
    anno_provider.process_labeling(max_workers=0)


if __name__ == "__main__":
    main()
