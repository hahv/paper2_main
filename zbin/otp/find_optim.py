from tap import Tap
import pandas as pd
import sys
from halib import *


class FindOptimArgs(Tap):
    csv_file: str = r"zbin/find_optim.csv"  # Path to the input CSV file containing grid search results
    recall_threshold: float = 0.99  # Minimum Recall safety constraint (default: 0.99)


def main():
    args = FindOptimArgs().parse_args()
    print("--- Running Algorithm 1: Grid Search Optimization ---")

    # Load the grid search results
    try:
        df = pd.read_csv(args.csv_file, sep=";", encoding="utf-8")
    except FileNotFoundError:
        print(f"Error: The file '{args.csv_file}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    # Ensure required columns exist based on the paper outline
    required_cols = ["exp_id", "Recall", "Filter_Rate"]
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Missing required column '{col}' in the CSV.")
            print(f"Please ensure your CSV has the columns: {', '.join(required_cols)}")
            sys.exit(1)

    print(f"Total configurations evaluated: {len(df)}")
    print(f"Safety Constraint (Recall >= {args.recall_threshold})\n")

    # Step 1: Filter out configurations that fail the safety constraint
    valid_configs = df[df["Recall"] >= args.recall_threshold].copy()

    if valid_configs.empty:
        print(
            f"Failed: No configurations met the safety constraint of Recall >= {args.recall_threshold}."
        )
        sys.exit(0)

    # Step 2: Compute FPR and SkipScore for valid configurations
    # Formula: SkipScore = Recall * Filter_Rate * (1 - FPR)
    valid_configs["FPR"] = 1.0 - valid_configs["Filter_Rate"]
    valid_configs["SkipScore"] = (
        valid_configs["Recall"]
        * valid_configs["Filter_Rate"]
        * (1.0 - valid_configs["FPR"])
    )

    # Step 3: Sort to find the best configurations
    valid_configs = valid_configs.sort_values(
        by="SkipScore", ascending=False
    ).reset_index(drop=True)

    # Output the results
    console.rule("Top Cfgs Meeting Safety Constraint ===")

    # Optional: Format the output nicely for readability
    display_df = valid_configs[["exp_id", "Recall", "Filter_Rate", "SkipScore"]].copy()
    display_df["Recall"] = display_df["Recall"].apply(lambda x: f"{x:.4f}")
    display_df["Filter_Rate"] = display_df["Filter_Rate"].apply(lambda x: f"{x:.4f}")
    display_df["SkipScore"] = display_df["SkipScore"].apply(lambda x: f"{x:.4f}")

    print(display_df.head(5).to_string(index=False))
    csvfile.fn_display_df(display_df.head(5))

    # Extract the absolute best configuration (h_best)
    best_exp = valid_configs.head(1)
    console.rule("Best Configuration (h_best)")
    csvfile.fn_display_df(best_exp[["exp_id", "Recall", "Filter_Rate", "SkipScore"]])


if __name__ == "__main__":
    main()
