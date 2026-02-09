import sys
import os
import numpy as np
from unittest.mock import patch
import pandas as pd
from halib.filetype import yamlfile

# Adjust path if necessary for your environment
sys.path.append("/mnt/e/SyncData/paper2_main")

from halib import *
from src.results.timeline.data_parser import TimelineConfig
from src.results.timeline.report_helper import TlReportGen

# ==========================================
# 1. SETUP MOCK CONFIGURATION (Same as test_data_parser.py)
# ==========================================
MOCK_YAML_CONTENT = yamlfile.load_yaml("/mnt/e/SyncData/paper2_main/config/mics/timeline_cfg.yaml")
# ==========================================
# 2. DYNAMIC MOCK DATA GENERATION
# ==========================================

def get_labels_for_type(timeline_type):
    """Retrieve label keys from the local MOCK config."""
    if timeline_type not in MOCK_YAML_CONTENT:
        return []
    # Handle new config nesting under "timeline"
    type_cfg = MOCK_YAML_CONTENT[timeline_type]
    if "timeline" in type_cfg and "labels_colors" in type_cfg["timeline"]:
        return list(type_cfg["timeline"]["labels_colors"].keys())
    # Fallback for old flat structure if needed
    return list(type_cfg.get("labels_colors", {}).keys())

def rand_mock_data_column(total_frames, timeline_type) -> list:
    """Generates a random sequence of labels logic."""
    label_options = get_labels_for_type(timeline_type)
    if not label_options:
         # Fallback if type not found or just mapped logic labels
         # But wait, 'rand_mock_data_by_timeline_type' in timeline.py produces OUTPUT labels.
         # The 'TimelineProcessor' expects RAW method output, and then parses it.
         # However, 'TLParser' logic IS what parses raw -> final.

         # Wait, looking at test_data_parser.py logic:
         # 'algo_v3' (Skip) has raw values: 'Skipped', 'Processed'
         # 'baseline' (NoSkip) has raw values: 'Fire', 'None' (Predictions)
         # 'gt_label' (GT) has raw values: 'Fire', 'None'

         # So we need to generate RAW method outputs, not the final labels.
         pass

    # We need to simulate RAW outputs appropriate for the logic:
    if timeline_type == "gt":
        # GT Raw: "Fire", "Smoke", "None"
        options = ["Fire", "None", "None", "None"] # bias towards None
    elif timeline_type == "skip":
        # Skip Raw: "Skipped", "Processed"
        options = ["Skipped", "Processed"]
    elif timeline_type == "no_skip":
        # NoSkip Raw: "Fire", "Smoke", "None" (Predictions)
        options = ["Fire", "None", "None"]
    else:
        options = ["Unknown"]

    # Generate segments for visual coherence
    num_segments = np.random.randint(5, 15)
    split_indices = np.sort(
        np.random.choice(range(1, total_frames), num_segments - 1, replace=False)
    )
    boundaries = [0] + list(split_indices) + [total_frames]

    all_labels = []
    for i in range(num_segments):
        seg_label = np.random.choice(options)
        seg_length = boundaries[i + 1] - boundaries[i]
        all_labels.extend([seg_label] * seg_length)

    return all_labels

def gen_random_video_df(video_name, total_frames, cols_map):
    """Generates a DataFrame for a single video with all required columns."""
    data = {
        "video": [video_name] * total_frames,
        "frame_id": list(range(total_frames)),
    }

    # Generate GT first (it drives the logic sometimes, but here we just want random viz)
    # Note: In real 'TimelineProcessor' logic, the parsers compare Method vs GT.
    # So purely random columns might lead to weird stats, but it validates the visualization pipeline.

    for col_name, t_type in cols_map.items():
        data[col_name] = rand_mock_data_column(total_frames, t_type)

    return pd.DataFrame(data)

def generate_multi_video_mock_data(cols_map, num_videos=3):
    frames_list = []
    for i in range(num_videos):
        # vary frame counts
        count = np.random.randint(50, 200)
        vid_name = f"video_{i+1:02d}"
        df = gen_random_video_df(vid_name, count, cols_map)
        frames_list.append(df)

    return pd.concat(frames_list, ignore_index=True)

NUM_VIDEOS = 5

def main():
    # 1. Define Mapping
    from collections import OrderedDict
    cols_to_timeline_types = OrderedDict({
        "gt_label": "gt",
        "algo_v3": "skip",
        "baseline": "no_skip",
    })
    output_html = "./zout/reports/test_timeline_report_dynamic.html"
    print(f"[INFO] Generating report to {output_html}...")

    # 2. Initialize Generator
    with patch.object(TimelineConfig, "load", return_value=MOCK_YAML_CONTENT):
        # 3. Generate Dynamic Mock Data
        # We must generate data inside the patch context if we used config-dependent generation,
        # but our generic generation above uses MOCK_YAML_CONTENT directly.
        df = generate_multi_video_mock_data(cols_to_timeline_types, num_videos=NUM_VIDEOS)

        print(f"[INFO] Generated DataFrame with {len(df)} rows across {df['video'].nunique()} videos.")

        generator = TlReportGen(cols_to_timeline_types)
        generator.generate(df, output_html, title="Dynamic Timeline Report v2", table_mode="p")

    print("[SUCCESS] Report generated.")

    if os.path.exists(output_html):
        print(f"File {output_html} exists. Size: {os.path.getsize(output_html)} bytes")
        pprint_local_path(output_html, get_wins_path=True)
    else:
        print(f"[ERROR] File {output_html} was not created.")

if __name__ == "__main__":
    main()
