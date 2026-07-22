#!/usr/bin/env python3
"""
csv_to_txt_labels.py

Converts CSV label files (sep=';') with columns:
    frame_idx;video_path;label

into TXT files where:
    - the TXT filename is derived from the video file name found in `video_path`
      (extension replaced with .txt)
    - each line has the format: "<frame_idx> <label>"

The sub-directory structure under `indir` is mirrored under `outdir`.

Usage:
    python csv_to_txt_labels.py --indir /path/to/indir --outdir /path/to/outdir
"""

import os
import sys
from pathlib import Path

import pandas as pd
from tap import Tap
from halib import *


class Args(Tap):
    indir: str = "./datasets/UFireIndoorFull" # Input directory containing CSV files (searched recursively)
    outdir: str = "./zlabels_updated" # Output directory for generated TXT files (mirrors indir structure)


def find_csv_files(indir: Path):
    """Recursively yield all .csv files under indir, along with their path relative to indir."""
    for root, _dirs, files in os.walk(indir):
        for fname in files:
            if fname.lower().endswith(".csv"):
                full_path = Path(root) / fname
                rel_dir = Path(root).relative_to(indir)
                yield full_path, rel_dir


def get_output_filename_from_video_path(video_path: str) -> str:
    """
    Given a video_path value like:
        /mnt/d/zdataset_paper2/.../aihub__lb_none__0203.mp4
    Return the corresponding txt filename:
        aihub__lb_none__0203.txt
    """
    video_name = os.path.basename(str(video_path).strip())
    return f"{video_name}.txt"


def convert_csv_to_txt(csv_path: Path, out_dir: Path):
    """Read a single CSV file (sep=';') with pandas and write the corresponding
    TXT file."""
    try:
        df = pd.read_csv(csv_path, sep=";", keep_default_na=False, dtype={"frame_idx": str, "video_path": str, "label": str})
        
    except Exception as e:
        console.print(f"  [yellow][SKIP] {csv_path} could not be read: {e} [/yellow]")
        return

    required_cols = {"frame_idx", "video_path", "label"}
    if not required_cols.issubset(set(df.columns)):
        console.print(
            f"  [yellow][SKIP] {csv_path} missing required columns "
            f"{required_cols}, found {list(df.columns)} [/yellow]"
        )
        return

    if df.empty:
        console.print(f"  [yellow][SKIP] {csv_path} is empty, nothing to write. [/yellow]")
        return

    df["frame_idx"] = df["frame_idx"].str.strip()
    df["video_path"] = df["video_path"].str.strip()
    df["label"] = df["label"].str.strip()

    output_filename = get_output_filename_from_video_path(df["video_path"].iloc[0])

    console.rule(f"{output_filename}")

    out_dir.mkdir(parents=True, exist_ok=True)
    final_out_path = out_dir / output_filename
    
    # 1. Assert that no missing values (NaN) exist in either column
    assert df["frame_idx"].notna().all(), f"Data error: Missing values found in 'frame_idx' in {csv_path}"
    assert df["label"].notna().all(), f"Data error: Missing values found in 'label' in {csv_path}"
    # 2. Convert both columns to strings and concatenate
    lines_series = df["frame_idx"].astype(str) + " " + df["label"].astype(str)

    # 3. Convert the pandas Series to a standard Python list
    lines = lines_series.tolist()

    final_out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"  [OK] {csv_path} -> {final_out_path}")


def main():
    args = Args().parse_args()

    indir = Path(args.indir).resolve()
    outdir = Path(args.outdir).resolve()

    if not indir.is_dir():
        print(
            f"Error: indir '{indir}' does not exist or is not a directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    # get folder name of indir to create a subfolder in outdir
    indir_name = indir.name
    outdir = outdir / indir_name
    outdir.mkdir(parents=True, exist_ok=True)

    csv_files = list(find_csv_files(indir))
    if not csv_files:
        print(f"No CSV files found under '{indir}'.")
        return
    console.rule(f"[{len(csv_files)} files] CSV to TXT Conversion")

    print(f"Found {len(csv_files)} CSV file(s) under '{indir}'. Converting...")
    for csv_path, rel_dir in tqdm(csv_files, desc="Converting CSV files"):
        out_dir = outdir / rel_dir
        convert_csv_to_txt(csv_path, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
