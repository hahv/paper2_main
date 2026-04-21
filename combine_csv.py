import os
import glob
from tap import Tap


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
