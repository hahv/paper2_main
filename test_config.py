from halib import *
from tap import *
from src.config import Config

class CustomArgs(Tap):
    cfg_path: str = "config/run_base.yaml"  # Path to the config file

def main():
    # Parse arguments
    args = CustomArgs().parse_args()
    config = Config.from_custom_yaml_file_or_str(args.cfg_path)
    pprint_box(config, title="Loaded Cfg")
    
if __name__ == "__main__":
    main()