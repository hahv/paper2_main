from dataclass_wizard import GT
import pandas as pd
import numpy as np
import os
from halib import *

# ============================================================================
# 1. METRICS & VISUALIZATION ENGINE
# ============================================================================
SKIP_COLOR_MAP = {
    "Miss (FN)": "#e91e63",
    "Waste (FP)": "#f1c40f",
    "True Skip (TN)": "#2fecc4",
    "True Proc. (TP)": "#16a085",
}
GT_COLOR_MAP = {
    "FireSmoke": "#e74c3c",
    "None": "#2ecc71",
}


def calculate_video_stats(video_name, df):
    total = len(df)
    if total == 0:
        return None

    # Boolean Masks
    is_fire = df["gt_label"].isin(["Fire", "Smoke"])
    is_safe = ~is_fire
    is_skipped = df["action"] == "Skipped"
    is_proc = df["action"] == "Processed"

    # Counts
    risk_count = (is_fire & is_skipped).sum()  # FN
    waste_count = (is_safe & is_proc).sum()  # FP
    true_skip_count = (is_safe & is_skipped).sum()  # TN
    hit_count = (is_fire & is_proc).sum()  # TP
    total_skip_count = is_skipped.sum()

    def to_pct(val):
        return f"{(val / total) * 100:.1f}%"

    return {
        "VIDEO NAME": video_name,
        "FRAMES": total,
        "SKIP %": to_pct(total_skip_count),
        "Miss (FN)": to_pct(risk_count),
        "Waste (FP)": to_pct(waste_count),
        "True Skip (TN)": to_pct(true_skip_count),
        "True Proc. (TP)": to_pct(hit_count),
        "_risk_raw": risk_count,
        "_df": df,
    }


def generate_timeline_html(df, width_px=300):
    """
    Generates a lightweight HTML/CSS stacked bar chart (2 Strips Only).
    1. Ground Truth (GT)
    2. Result Status (RES)
    """

    # --- 1. Define Colors ---

    # Strip 1: Ground Truth
    gt_colors = np.where(
        df["gt_label"].isin(["Fire", "Smoke"]),
        GT_COLOR_MAP["FireSmoke"],
        GT_COLOR_MAP["None"],
    )

    # Strip 2: Result Status (Integrated)
    is_fire = df["gt_label"].isin(["Fire", "Smoke"])
    is_skipped = df["action"] == "Skipped"

    cond_fn = is_fire & is_skipped  # Miss
    cond_fp = (~is_fire) & (~is_skipped)  # Waste
    cond_tp = is_fire & (~is_skipped)  # Hit
    cond_tn = (~is_fire) & is_skipped  # True Skip

    # ! Legend for Your Report
    # Miss (FN): Fire was present, but you Skipped it. (Safety Failure)
    # Waste (FP): No fire was present, but you Processed it. (Efficiency Failure)
    # True Skip (TN): No fire was present, and you Skipped it. (Efficiency Success)
    # Hit (TP): Fire was present, and you Processed it. (Safety Success)

    status_colors = np.select(
        [cond_fn, cond_fp, cond_tp, cond_tn],
        [
            SKIP_COLOR_MAP["Miss (FN)"],
            SKIP_COLOR_MAP["Waste (FP)"],
            SKIP_COLOR_MAP["True Proc. (TP)"],
            SKIP_COLOR_MAP["True Skip (TN)"],
        ],
        default="#95a5a6",
    )

    # --- 2. RLE Compression ---
    def rle_encoding(colors):
        if len(colors) == 0:
            return []
        changes = np.concatenate(([0], np.where(colors[:-1] != colors[1:])[0] + 1))
        segments = []
        for i in range(len(changes)):
            start = changes[i]
            end = changes[i + 1] if i + 1 < len(changes) else len(colors)
            length = end - start
            pct = (length / len(colors)) * 100
            segments.append((colors[start], pct))
        return segments

    gt_segs = rle_encoding(gt_colors)
    stat_segs = rle_encoding(status_colors)

    # --- 3. HTML Generation ---
    def make_strip(segments, label, height_px=15):
        html = f'<div style="display:flex; width:100%; height:{height_px}px; margin-bottom:2px;" title="{label}">'
        for color, pct in segments:
            html += (
                f'<div style="background:{color}; width:{pct}%; height:100%;"></div>'
            )
        html += "</div>"
        return html

    return (
        f'<div style="width:{width_px}px; background:#eee; padding:2px; border:1px solid #ccc;">'
        # Labels
        f'<div style="display:flex; justify-content:space-between; font-size:9px; color:#555; line-height:10px; margin-bottom:1px;">'
        f"<span>GT</span><span>PROC</span></div>"
        # Two Bars Only (GT and PROC)
        f"{make_strip(gt_segs, 'GT (TOP): Red=Fire, Green=Safe', height_px=15)}"
        f"{make_strip(stat_segs, 'PROC (BOTTOM): Purple=Miss, Gold=Waste, Navy=Hit, Teal=TrueSkip', height_px=15)}"
        f"</div>"
    )


# ============================================================================
# 2. REPORT GENERATOR
# ============================================================================


def generate_final_report(stats_list, output_file="final_report.html"):
    if not stats_list:
        return

    df_report = pd.DataFrame(stats_list)
    df_report["VISUALIZATION (Timeline)"] = df_report["_df"].apply(
        generate_timeline_html
    )
    df_display = df_report.drop(columns=["_df", "_risk_raw"])

    def highlight_risk(s):
        is_risk = s.name == "Miss (FN)"
        return [
            "color: red; font-weight: bold;" if (is_risk and v != "0.0%") else ""
            for v in s
        ]

    styler = (
        df_display.style.apply(highlight_risk)
        .set_properties(**{"text-align": "center", "vertical-align": "middle"})
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#2c3e50"),
                        ("color", "white"),
                        ("text-align", "center"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [("border", "1px solid #ddd"), ("padding", "8px")],
                },
                {
                    "selector": "tr:nth-child(even)",
                    "props": [("background-color", "#f2f2f2")],
                },
            ]
        )
        .hide(axis="index")
    )

    html_out = styler.to_html(escape=False)

    full_html = f"""
    <html>
    <head>
        <title>Skip Logic Efficiency Report</title>
        <style>
            body {{ font-family: "Segoe UI", sans-serif; padding: 20px; }}
            table {{ border-collapse: collapse; width: 100%; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            h2 {{ color: #2c3e50; }}
            .legend-box {{ background: #f9f9f9; padding: 15px; border: 1px solid #ccc; display: inline-block; margin-top: 20px; border-radius: 5px; }}
            .dot {{ height: 12px; width: 12px; display: inline-block; border-radius: 3px; margin-right: 5px; vertical-align: middle; }}
            .legend-item {{ margin-bottom: 8px; font-size: 14px; }}
        </style>
    </head>
    <body>
        <h2>Skip Logic Performance Report</h2>
        {html_out}
        <br>
        <div class="legend-box">
            <b>Timeline Legend</b><hr>

            <div class="legend-item"><b>1. GT (Ground Truth - Top Bar)</b></div>
            <div class="legend-item"><span class="dot" style="background:{GT_COLOR_MAP["FireSmoke"]}"></span>Fire/Smoke</div>
            <div class="legend-item"><span class="dot" style="background:{GT_COLOR_MAP["None"]}"></span>Safe</div>
            <br>

            <div class="legend-item"><b>2. PROC (Processing Status - Bottom Bar)</b></div>
            <div class="legend-item"><span class="dot" style="background:{SKIP_COLOR_MAP["Miss (FN)"]}"></span><b>Miss (FN)</b>: Unsafe Skip</div>
            <div class="legend-item"><span class="dot" style="background:{SKIP_COLOR_MAP["Waste (FP)"]}"></span><b>Waste (FP)</b>: Unneeded Proc</div>
            <div class="legend-item"><span class="dot" style="background:{SKIP_COLOR_MAP["True Skip (TN)"]}"></span><b>True Skip (TN)</b>: Correct Skip</div>
            <div class="legend-item"><span class="dot" style="background:{SKIP_COLOR_MAP["True Proc. (TP)"]}"></span><b>True Proc. (TP)</b>: Valid Processed</div>
        </div>
    </body>
    </html>
    """

    with open(output_file, "w") as f:
        f.write(full_html)

    print(f"✅ Report successfully generated: ⏬")
    pprint_local_path(output_file, get_wins_path=True)


# ============================================================================
# 3. MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    # Dummy Generator included for completeness
    def generate_dummy_data_scenario(scenario_type="perfect", frames=500):
        data = []
        fire_start, fire_end = 200, 300
        for i in range(frames):
            is_fire_event = fire_start <= i <= fire_end
            gt = "Fire" if is_fire_event else "Safe"
            if scenario_type == "perfect":
                if is_fire_event:
                    action = "Processed"
                else:
                    action = "Skipped" if np.random.rand() > 0.1 else "Processed"
            elif scenario_type == "dangerous_miss":
                if is_fire_event:
                    action = "Skipped" if np.random.rand() > 0.2 else "Processed"
                else:
                    action = "Skipped"
            elif scenario_type == "inefficient":
                action = "Processed" if np.random.rand() > 0.2 else "Skipped"
            data.append({"frame_idx": i, "action": action, "gt_label": gt})
        return pd.DataFrame(data)

    OUT_FILE = "./zout/reports/viz_skip.html"
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    def get_rand_num_frames():
        return np.random.randint(400, 601)

    videos = [
        (
            "Scenario_A_Ideal.mp4",
            generate_dummy_data_scenario("perfect", get_rand_num_frames()),
        ),
        (
            "Scenario_B_Miss.mp4",
            generate_dummy_data_scenario("dangerous_miss", get_rand_num_frames()),
        ),
        (
            "Scenario_C_Slow.mp4",
            generate_dummy_data_scenario("inefficient", get_rand_num_frames()),
        ),
    ]

    stats_list = []
    for name, df in videos:
        stats = calculate_video_stats(name, df)
        if stats:
            stats_list.append(stats)

    generate_final_report(stats_list, OUT_FILE)
