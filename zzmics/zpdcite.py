from halib import *
import os

def main():
    # ! run the pdcite_op script with the config file
    BIB_TEX = r"./paper/out/zpaper2.tex"
    FILES = [
        r"./paper/4.table/out/tb_ufireindoor.csv",
        r"./paper/4.table/out/tb_perf_per_frame.csv",
    ]
    OUTDIR = r"./paper/pdcite_out"

    files_str = " ".join([f'"{f}"' for f in FILES])
    cmd_str = f'python -m halib.utils.pdcite_op --bib_tex "{BIB_TEX}" --files {files_str} --outdir "{OUTDIR}"'
    pprint_box(cmd_str, title="pdcite_op command")
    os.system(cmd_str)

if __name__ == "__main__":
    main()
