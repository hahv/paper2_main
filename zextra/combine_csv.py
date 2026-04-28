import os
import glob
from tap import Tap

"""
Combine multiple CSV files into a single plain-text report.

This script scans an input directory for ``.csv`` files, sorts them
alphabetically, and writes their raw contents into one output text file.
Each CSV block is preceded by a banner line in the format:

    ===FILE_NAME: <csv_filename>===

Features:
- Deterministic ordering (alphabetical by filename)
- Configurable CSV separator and text encoding via CLI arguments
- Graceful handling when no CSV files are found

CLI arguments (Tap):
- input_dir (str): Directory containing source CSV files.
- output_file (str): Destination text file path.
  Default: ``combined_output.txt``
- separator (str): Declared CSV delimiter (default ``;``).
  Note: current implementation copies raw file text and does not parse rows.
- encoding (str): File encoding for reading/writing text.
  Default: ``utf-8``

Primary function:
- combine_csv_files(args: Args) -> None
  Reads all CSV files in ``args.input_dir`` and appends them to
  ``args.output_file`` with per-file header banners and blank-line separation.
"""

class Args(Tap):
    input_dir: str  # Path to the folder containing CSV files
    output_file: str = "combined_output.txt"  # Path to the output TXT file
    separator: str = ";"  # CSV separator (default: semicolon)
    encoding: str = "utf-8"  # File encoding


def combine_csv_files(args: Args) -> None:
    # Find all CSV files in the input directory, sorted alphabetically
    csv_files = sorted(glob.glob(os.path.join(args.input_dir, "*.csv")))

    if not csv_files:
        print(f"No CSV files found in: {args.input_dir}")
        return

    with open(args.output_file, "w", encoding=args.encoding) as out_file:
        for i, csv_path in enumerate(csv_files):
            file_name = os.path.basename(csv_path)

            # Write the file header banner
            out_file.write(f"===FILE_NAME: {file_name}===\n")

            # Write the raw CSV content
            with open(csv_path, "r", encoding=args.encoding) as csv_file:
                content = csv_file.read()
                out_file.write(content)

            # Add two blank lines between files (but not after the last one)
            if i < len(csv_files) - 1:
                out_file.write("\n\n")

    print(f"Done! {len(csv_files)} file(s) combined into: {args.output_file}")


if __name__ == "__main__":
    args = Args().parse_args()
    combine_csv_files(args)
