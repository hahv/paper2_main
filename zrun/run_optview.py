from halib import *
import click
import os
from tap import *

class OptViewArgs(Tap):
    indir: str = "./zout/tune"  # input dir of optuna dashboard
    ext: str = ".db"  # target ext
    port: int = 10000  # port number

def open_optuna_dashboard(filepath, sqlite3_file_index, port):
    start_port = port
    port_used = start_port + sqlite3_file_index
    optuna_cmd = "optuna-dashboard"
    cmd = f"{optuna_cmd} sqlite:///{os.path.abspath(filepath)} --port {port_used}"
    pprint(f"cmd: {cmd}")
    os.system(cmd)


def main():
    args = OptViewArgs().parse_args()
    indir = args.indir
    target_ext = args.ext
    start_port = args.port
    all_sqlite3_files = fs.filter_files_by_extension(indir, target_ext)
    idx_files = [(idx, f) for idx, f in enumerate(all_sqlite3_files)]
    pprint(idx_files)
    # use package click for user input a number(int)
    selected_index = click.prompt(
        f"Enter index of sqlite3 file to open (0-{len(all_sqlite3_files) - 1})",
        type=int,
    )
    assert selected_index < len(all_sqlite3_files), (
        f"Index {selected_index} is out of range"
    )
    chosen_sqlite3_file = all_sqlite3_files[selected_index]
    open_optuna_dashboard(
        filepath=chosen_sqlite3_file, sqlite3_file_index=selected_index, port=start_port
    )


if __name__ == "__main__":
    main()
