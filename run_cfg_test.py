from src.config import Config
from halib import *
from tap import *

class CustomArgs(Tap):
    cfg_file: str = "config/run_base.yaml"

def main():
    # Parse arguments
    args = CustomArgs().parse_args()
    cfg = Config.from_yaml_file(args.cfg_file)
    pprint_box(cfg, title="Config")
    
if __name__ == "__main__":
    main()