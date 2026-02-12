from halib import *
from tap import *

class CustomArgs(Tap):
    arg_str: str = "MyProject"
    arg_int: int = 32
    verbose: bool = False # use --verbose to set True

def main():
    # Parse arguments
    args = CustomArgs().parse_args()
    console.rule("Parsed args")
    with ConsoleLog("arg values"):
        pprint(args.arg_str)
        pprint(args.arg_int)
        pprint(args.verbose)
if __name__ == "__main__":
    main()