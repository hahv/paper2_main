from halib import * # noqa: F403
from tap import *

class CustomArgs(Tap):
    # --- Basic Types ---
    arg_str: str = "MyProject"  # Description acts as help text
def main():
    # Parse arguments
    args = CustomArgs().parse_args()
    print(f"Argument value: {args.arg_str}")

if __name__ == "__main__":
    main()