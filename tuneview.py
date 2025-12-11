from halib import *
from halib.system import filesys as fs
import click
from argparse import ArgumentParser
import os

def parse_args():
    parser = ArgumentParser(
        description="desc text")
    parser.add_argument('-indir', '--indir', type=str,
                        help='input dir of optuna dashboard', default='./zout/tune')
    parser.add_argument('-ext', '--ext', type=str,
                        help='target ext', default='.db') # can be .sqlite3
    parser.add_argument('-port', '--port', type=int,
                        help='port number', default=10000)
    return parser.parse_args()

def open_optuna_dashboard(filepath, sqlite3_file_index, port):
    start_port = port
    port_used = start_port + sqlite3_file_index
    optuna_cmd = 'optuna-dashboard'
    cmd = f'{optuna_cmd} sqlite:///{os.path.abspath(filepath)} --port {port_used}'
    pprint(f'cmd: {cmd}')
    os.system(cmd)


def main():
    args = parse_args()
    target_ext = args.ext
    start_port = args.port
    all_sqlite3_files = fs.filter_files_by_extension(args.indir, target_ext)
    idx_files = [(idx, f) for idx, f in enumerate(all_sqlite3_files)]
    pprint(idx_files)
    # use package click for user input a number(int)
    selected_index = click.prompt(f"Enter index of sqlite3 file to open (0-{len(all_sqlite3_files)-1})", type=int)
    assert selected_index < len(all_sqlite3_files), f'Index {selected_index} is out of range'
    chosen_sqlite3_file = all_sqlite3_files[selected_index]
    open_optuna_dashboard(filepath=chosen_sqlite3_file, sqlite3_file_index=selected_index, port=start_port)

if __name__ == "__main__":
    main()
