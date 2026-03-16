import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))  # Add parent directory to sys.path
from halib import *
import re
import pandas as pd
from tap import Tap
from src.common import GlobalConst


class ParseReportArgs(Tap):
    csv_path: str


def get_total_skip_rate(csv_path: str) -> dict:
    if not os.path.exists(csv_path):
        return {}

    try:
        # Read the CSV with the first two rows as multi-index header
        df = pd.read_csv(csv_path, sep=";", header=[0, 1], encoding="utf-8")
    except Exception as e:
        return {}

    if df.empty:
        return {}

    # Find the target column
    target_col = None
    for col in df.columns:
        # col is a tuple: (level_0, level_1)
        level_0, level_1 = str(col[0]).strip(), str(col[1]).strip()
        if level_0.startswith("temp_method_") and level_1 == "Correct Skip":
            target_col = col
            break

    if not target_col:
        return {}

    # The first column usually contains the row labels like 'TOTAL', 'aihub__lb_fire__0182.mp4', etc.
    first_col = df.columns[0]

    # Find the row where the first column is 'TOTAL'
    total_row = df[df[first_col].astype(str).str.strip() == "TOTAL"]

    if total_row.empty:
        return {}

    correct_skip_val = total_row[target_col].iloc[0]
    method_name = target_col[0].strip()  # Extract method name from level_0 of the column header

    def get_percent(text):
        """
        Extracts the number before '%' from a string like '48.7500% (351)'
        Returns float or None if not found.
        """
        match = re.search(r"(\d+\.?\d*)%", text)
        assert match, f"Could not extract percentage from text: {text}"
        return float(match.group(1)) / 100  # convert to decimal

    data = {
        GlobalConst.METHOD_NAME: method_name,
        GlobalConst.COL_PARAM_SKIP_RATE: get_percent(correct_skip_val),
    }
    return data

def main():
    args = ParseReportArgs().parse_args()
    result = get_total_skip_rate(args.csv_path)
    pprint(result)

if __name__ == "__main__":
    main()
