import sys
from unittest.mock import patch

# Adjust path if necessary for your environment
sys.path.append("/mnt/e/SyncData/paper2_main")

from halib import *

# Import your actual classes
from src.results.timeline.tl_converter import TlProcessor, TlConfig

# ==========================================
# 1. SETUP MOCK CONFIGURATION
# ==========================================
# This matches the structure expected by your TimelineConfig class
MOCK_YAML_CONTENT = {
    "gt": {
        "legend_title": "1. GT (Ground Truth)",
        "include_col_in_table": False,
        "labels_colors": {"FireSmoke": "#fc6065", "None": "#2ecc71"},
    },
    "no_skip": {
        "legend_title": "2. NO SKIP (Baseline)",
        "include_col_in_table": True,
        "labels_colors": {
            "Correct": "#2ecc71",
            "False Alarm (FP)": "#f1c40f",
            "Miss (FN)": "#e74c3c",
        },
    },
    "skip": {
        "legend_title": "3. SKIP (Proposed)",
        "include_col_in_table": True,
        "labels_colors": {
            "Miss (FN)": "#8e44ad",
            "Waste (FP)": "#f1c40f",
            "True Proc. (TP)": "#2c3e50",
            "True Skip (TN)": "#00b894",
        },
    },
}


def generate_mock_data():
    """Generates a DataFrame with 2 videos and mixed scenarios."""
    data = {
        "video": ["vid_A"] * 4 + ["vid_B"] * 4,
        "frame_id": [0, 1, 2, 3, 0, 1, 2, 3],
        "gt_label": ["Fire", "Fire", "None", "None", "Fire", "None", "None", "Fire"],
        # Scenario 1: Proposed Method (Logic Type: 'skip')
        # Expect: Miss, True Proc, True Skip, Waste
        "algo_v3": [
            "Skipped",  # GT=Fire -> Miss
            "Processed",  # GT=Fire -> True Proc
            "Skipped",  # GT=None -> True Skip
            "Processed",  # GT=None -> Waste
            "Processed",  # GT=Fire -> True Proc
            "Skipped",  # GT=None -> True Skip
            "Processed",  # GT=None -> Waste
            "Skipped",  # GT=Fire -> Miss
        ],
        # Scenario 2: Baseline Method (Logic Type: 'no_skip')
        # Expect: Miss, Correct, Correct, False Alarm
        "baseline": [
            "None",  # GT=Fire -> Miss
            "Fire",  # GT=Fire -> Correct
            "None",  # GT=None -> Correct
            "Fire",  # GT=None -> FP
            "Fire",  # GT=Fire -> Correct
            "None",  # GT=None -> Correct
            "Fire",  # GT=None -> FP
            "None",  # GT=Fire -> Miss
        ],
    }
    return pd.DataFrame(data)


def main():
    # 1. Create Mock Data
    df = generate_mock_data()

    print("\n" + "=" * 40)
    print("--- 1. INPUT DATAFRAME (Head) ---")
    print("=" * 40)
    csvfile.fn_display_df(df)

    # 2. Define Mapping (Column -> Logic Type)
    # Note: 'gt_label' acts as both the source for ground truth AND a visual bar
    cols_to_timeline_types = {
        "gt_label": "gt",  # Will map to TLGtParser
        "algo_v3": "skip",  # Will map to SkipParser
        "baseline": "no_skip",  # Will map to NoSkipParser
    }

    # 3. Run Processor (With Mocked Config)
    # We patch 'load' so it returns our dict instead of reading a file
    print("\n[INFO] Processing Dataframe with MOCK configuration...")

    with patch.object(TlConfig, "load", return_value=MOCK_YAML_CONTENT):
        final_df, stats_df, styles_map = TlProcessor.proc_dataframe(
            df, cols_to_timeline_types
        )

    # ==========================================
    # 4. INSPECT RESULTS
    # ==========================================

    print("\n" + "=" * 40)
    print("--- 2. FINAL PROCESSED DATAFRAME (Frame Level) ---")
    print("=" * 40)
    print("Note: Values here are the transformed Labels (e.g., 'Miss (FN)')")
    csvfile.fn_display_df(final_df)

    print("\n" + "=" * 40)
    print("--- 3. STATS DATAFRAME (Video Level Summary) ---")
    print("=" * 40)
    print("Mode: 'pfc' (Percent + Frame Count)")
    # This shows the Pivot Table with MultiIndex Columns
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    csvfile.fn_display_df(stats_df)

    print("\n" + "=" * 40)
    print("--- 4. STYLES MAP (For Visualization) ---")
    print("=" * 40)
    print(f"Keys found: {list(styles_map.keys())}")
    console.rule()
    pprint(styles_map)


if __name__ == "__main__":
    main()
