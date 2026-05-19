import os
from glob import glob
from rich.pretty import pprint
from typing import List, Optional

def main():

    # Common environment configuration
    lib_path = "E:/Dev/__halib"
    venv_python = "E:/Dev/__halib/.venv/Scripts/python.exe"
    base_env = f"set PYTHONPATH={lib_path} &&"

    print("Updating Citations in CSV/TEX...")
    bib_tex = r"paper/out/zpaper2.tex" # .tex file generated.
    csv_base_dir = r"paper/4.table/out"

    # list all csv in chapter3_base_dir and chapter4_base_dir (including subdirs)
    files = glob(f"{csv_base_dir}/*.csv", recursive=False)

    print(f"Found {len(files)} CSV files to update citations in.")
    pprint(files)

    outdir = r"paper/4.table/out/update_cite"
    files_str = " ".join([f'"{f}"' for f in files])

    cite_cmd = f'{base_env} "{venv_python}" -m halib.utils.pdcite_op --bib_tex "{bib_tex}" --files {files_str} --outdir "{outdir}"'
    print(f"Executing:/n{cite_cmd}/n")
    os.system(cite_cmd)

if __name__ == "__main__":
    main()
