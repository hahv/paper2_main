from halib import *
from pathlib import Path
from tap import *
from typing import List, Optional, Literal
from halib.filetype import yamlfile


class CustomArgs(Tap):
    # --- Basic Types ---
    vis_cfg: Optional[Path] = Path("./vis_skip_cfg.yaml")


def calculate_video_stats(video_name, df):
    total = len(df)
    if total == 0:
        return None

    # Logic Stats (Same as before)
    is_fire = df["gt_label"].isin(["Fire", "Smoke"])
    is_safe = ~is_fire
    is_skipped = df["action"] == "Skipped"
    is_proc = df["action"] == "Processed"

    risk_count = (is_fire & is_skipped).sum()
    waste_count = (is_safe & is_proc).sum()
    true_skip_count = (is_safe & is_skipped).sum()
    hit_count = (is_fire & is_proc).sum()
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


def generate_timeline_html(df, vis_cfg_dict, width_px=350):
    """
    Generates a 3-Row Timeline: GT, BASE (Accuracy), OURS (Efficiency)
    """
    GT_COLOR_MAP = vis_cfg_dict["gt_color_map"]
    BASE_COLOR_MAP = vis_cfg_dict["no_skip_color_map"]
    SKIP_COLOR_MAP = vis_cfg_dict["skip_color_map"]

    # --- STRIP 1: GT ---
    gt_colors = np.where(
        df["gt_label"].isin(["Fire", "Smoke"]),
        GT_COLOR_MAP["FireSmoke"]["color"],
        GT_COLOR_MAP["None"]["color"],
    )

    # --- STRIP 2: BASELINE (Accuracy of Heavy Model) ---
    # Logic: Compare GT vs PRED (simulated heavy model prediction)
    # Correct (Green) = (GT==Fire & Pred==Fire) OR (GT==Safe & Pred==Safe)
    # FP (Gold)       = (GT==Safe & Pred==Fire)
    # FN (Red)        = (GT==Fire & Pred==Safe)

    is_gt_fire = df["gt_label"].isin(["Fire", "Smoke"])
    is_pred_fire = df["pred_label"].isin(["Fire", "Smoke"])

    cond_correct = is_gt_fire == is_pred_fire
    cond_fp = (~is_gt_fire) & (is_pred_fire)
    cond_fn = (is_gt_fire) & (~is_pred_fire)

    base_colors = np.select(
        [cond_fp, cond_fn, cond_correct],
        [
            BASE_COLOR_MAP["False Alarm (FP)"]["color"],
            BASE_COLOR_MAP["Miss (FN)"]["color"],
            BASE_COLOR_MAP["Correct"]["color"],
        ],
        default="#95a5a6",
    )

    # --- STRIP 3: OURS (Efficiency of Skip Logic) ---
    is_skipped = df["action"] == "Skipped"
    cond_skip_fn = is_gt_fire & is_skipped
    cond_skip_fp = (~is_gt_fire) & (~is_skipped)
    cond_skip_tp = is_gt_fire & (~is_skipped)
    cond_skip_tn = (~is_gt_fire) & is_skipped

    status_colors = np.select(
        [cond_skip_fn, cond_skip_fp, cond_skip_tp, cond_skip_tn],
        [
            SKIP_COLOR_MAP["Miss (FN)"]["color"],
            SKIP_COLOR_MAP["Waste (FP)"]["color"],
            SKIP_COLOR_MAP["True Proc. (TP)"]["color"],
            SKIP_COLOR_MAP["True Skip (TN)"]["color"],
        ],
        default="#95a5a6",
    )

    # --- HTML HELPER ---
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
    base_segs = rle_encoding(base_colors)
    stat_segs = rle_encoding(status_colors)

    def make_bar(segments, height_px=10):
        bar_html = f'<div style="display:flex; width:100%; height:{height_px}px;">'
        for color, pct in segments:
            bar_html += (
                f'<div style="background:{color}; width:{pct}%; height:100%;"></div>'
            )
        bar_html += "</div>"
        return bar_html

    def make_row(label, segments, tooltip):
        return (
            f"<tr>"
            f'<td style="width:40px; padding-right:5px; text-align:right; font-size:10px; font-weight:bold; color:#555; border:none;">{label}</td>'
            f'<td style="width:{width_px}px; border:none; padding:1px 0;">'
            f'<div style="width:100%; background:#eee; border:1px solid #ccc;" title="{tooltip}">'
            f"{make_bar(segments)}"
            f"</div>"
            f"</td>"
            f"</tr>"
        )

    return (
        f'<table style="margin:0 auto; border:none; width:auto; background:transparent;">'
        f"{make_row('GT', gt_segs, 'Ground Truth')}"
        f"{make_row('BASE', base_segs, 'Baseline: Green=Correct, Gold=FP, Red=FN')}"
        f"{make_row('OURS', stat_segs, 'Our Logic: Navy=Hit, Teal=Skip, Gold=Waste')}"
        f"</table>"
    )


# ============================================================================
# 2. REPORT GENERATOR
# ============================================================================


def generate_final_report(stats_list, vis_cfg_yaml, output_file="final_report.html"):
    if not stats_list:
        return

    df_report = pd.DataFrame(stats_list)
    vis_cfg_dict = yamlfile.load_yaml(vis_cfg_yaml, to_dict=True)
    df_report["VISUALIZATION (Timeline)"] = df_report["_df"].apply(
        generate_timeline_html, vis_cfg_dict=vis_cfg_dict
    )
    df_display = df_report.drop(columns=["_df", "_risk_raw"])

    def highlight_risk(s):
        is_risk = s.name == "Miss (FN)"
        return [
            "color: #e74c3c; font-weight: bold;" if (is_risk and v != "0.0%") else ""
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
                    "props": [("background-color", "#f8f9fa")],
                },
            ]
        )
        .hide(axis="index")
    )

    html_out = styler.to_html(escape=False)

    LEGEND_SECTIONS = vis_cfg_dict.get("legion_sections")
    # update maps to actual maps (not names)
    for section in LEGEND_SECTIONS:
        map_name = section["map_name"]
        section["map"] = vis_cfg_dict[map_name]

    # --- DYNAMIC LEGEND GENERATION ---
    def _make_legend_html(legend_sections):
        html_parts = []
        html_parts.append('<div class="legend-box">')
        html_parts.append('<h3 style="margin-top:0;">Timeline Legend</h3>')
        html_parts.append(
            '<hr style="border:0; border-top:1px solid #eee; margin-bottom:15px;">'
        )

        # Loop through sections
        for section in legend_sections:
            html_parts.append('<div class="legend-section">')
            html_parts.append(f'<span class="legend-title">{section["title"]}</span>')

            # Loop through items in the color map
            # Assumes your maps are dicts like: {"Label": "#hex"} OR {"Label": {"color": "#hex", "desc": "..."}}
            for label, val in section["map"].items():
                # Handle both simple string format and complex dict format
                if isinstance(val, dict):
                    color = val.get("color", "#ccc")
                    desc = val.get(
                        "desc", label
                    )  # Use label if no description provided
                else:
                    color = val
                    desc = label

                text = f"<b>{label}</b>"
                if desc != label:
                    text += f": {desc}"
                html_parts.append(
                    f'<div class="legend-item">'
                    f'<span class="dot" style="background:{color}"></span>{text}'
                    f"</div>"
                )
            html_parts.append("</div>")  # End section

        html_parts.append("</div>")  # End box
        return "".join(html_parts)

    legend_html = _make_legend_html(LEGEND_SECTIONS)
    full_html = f"""
    <html>
    <head>
        <title>Skip Logic Efficiency Report</title>
        <style>
            @media print {{ body {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }} }}
            body {{ font-family: 'CMU Serif', 'Times New Roman', serif; padding: 40px; color: #333; }}
            table {{ border-collapse: collapse; width: 100%; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
            h2 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
            .legend-box {{ background: #fff; padding: 20px; border: 1px solid #ddd; display: inline-block; margin-top: 30px; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
            .dot {{ height: 12px; width: 12px; display: inline-block; border-radius: 2px; margin-right: 8px; vertical-align: middle; }}
            .legend-item {{ margin-bottom: 5px; font-size: 14px; line-height: 1.6; }}
            .legend-section {{ margin-bottom: 15px; }}
            .legend-title {{ font-weight: bold; margin-bottom: 5px; display: block; color: #555; }}
        </style>
    </head>
    <body>
        <h2>Skip Logic Performance Report</h2>
        {html_out}
        <br>
        {legend_html}
    </body>
    </html>
    """

    with open(output_file, "w") as f:
        f.write(full_html)

    print(f"✅ Report successfully generated: ⏬")
    pprint_local_path(output_file, get_wins_path=True)


def generate_dummy_data_scenario(scenario_type="perfect", frames=500):
    data = []
    fire_start, fire_end = 200, 300
    for i in range(frames):
        is_fire_event = fire_start <= i <= fire_end
        gt = "Fire" if is_fire_event else "Safe"

        # --- SIMULATE HEAVY MODEL PREDICTION (With Errors) ---
        # Assume 95% Accuracy to make colors show up
        if gt == "Fire":
            pred = "Fire" if np.random.rand() > 0.05 else "Safe"  # 5% Miss
        else:
            pred = "Safe" if np.random.rand() > 0.05 else "Fire"  # 5% False Alarm

        # --- SIMULATE ACTION (Skip Logic) ---
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

        data.append(
            {"frame_idx": i, "action": action, "gt_label": gt, "pred_label": pred}
        )
    return pd.DataFrame(data)


def main():
    # Parse arguments
    args = CustomArgs().parse_args()
    OUT_FILE = "./zout/reports/viz_skip_v3.html"
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    scenarios = ["perfect", "dangerous_miss", "inefficient"]
    videos = []

    # Generate some random videos
    for i in range(3):
        scenario = scenarios[i]
        video_name = f"Video_{i + 1:02d}_{scenario}.mp4"
        df = generate_dummy_data_scenario(scenario, 500)
        videos.append((video_name, df))

    stats_list = []
    for name, df in videos:
        stats = calculate_video_stats(name, df)
        if stats:
            stats_list.append(stats)

    generate_final_report(stats_list, vis_cfg_yaml=args.vis_cfg, output_file=OUT_FILE)


if __name__ == "__main__":
    main()
