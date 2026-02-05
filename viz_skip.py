from halib import *
from pathlib import Path
from tap import *
from typing import List, Optional, Literal, Dict, Any
from halib.filetype import yamlfile
import pandas as pd
import numpy as np
import os


class CustomArgs(Tap):
    # --- Basic Types ---
    vis_cfg: Path = Path("./vis_skip_cfg.yaml")


class SkipLogicReport:
    """
    A unified class to generate Skip Logic Performance Reports.
    """

    @staticmethod
    def calculate_video_stats(video_name: str, df: pd.DataFrame) -> Optional[Dict]:
        """Calculates performance metrics (Miss, Waste, etc.) for a single video."""
        total = len(df)
        if total == 0:
            return None

        # Boolean Masks
        is_fire = df["gt_label"].isin(["Fire", "Smoke"])
        is_safe = ~is_fire
        is_skipped = df["action"] == "Skipped"
        is_proc = df["action"] == "Processed"

        # Counts
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

    @staticmethod
    def generate_timeline_html(
        df: pd.DataFrame,
        vis_cfg_dict: Dict,
        active_bars: Optional[List[str]] = None,
        width_px: int = 350,
    ) -> str:
        """
        Generates the HTML visualization column (Timeline).
        """
        if active_bars is None:
            active_bars = list(vis_cfg_dict.get("bar_types", {}).keys())

        # --- LOGIC HANDLERS ---
        def _logic_gt(df, cmap):
            return np.where(
                df["gt_label"].isin(["Fire", "Smoke"]),
                cmap["FireSmoke"]["color"],
                cmap["None"]["color"],
            )

        def _logic_no_skip(df, cmap):
            is_gt_fire = df["gt_label"].isin(["Fire", "Smoke"])
            is_pred_fire = df["pred_label"].isin(["Fire", "Smoke"])
            cond_correct = is_gt_fire == is_pred_fire
            cond_fp = (~is_gt_fire) & (is_pred_fire)
            cond_fn = (is_gt_fire) & (~is_pred_fire)
            return np.select(
                [cond_fp, cond_fn, cond_correct],
                [
                    cmap["False Alarm (FP)"]["color"],
                    cmap["Miss (FN)"]["color"],
                    cmap["Correct"]["color"],
                ],
                default="#95a5a6",
            )

        def _logic_skip(df, cmap):
            is_gt_fire = df["gt_label"].isin(["Fire", "Smoke"])
            is_skipped = df["action"] == "Skipped"
            cond_fn = is_gt_fire & is_skipped
            cond_fp = (~is_gt_fire) & (~is_skipped)
            cond_tp = is_gt_fire & (~is_skipped)
            cond_tn = (~is_gt_fire) & is_skipped
            return np.select(
                [cond_fn, cond_fp, cond_tp, cond_tn],
                [
                    cmap["Miss (FN)"]["color"],
                    cmap["Waste (FP)"]["color"],
                    cmap["True Proc. (TP)"]["color"],
                    cmap["True Skip (TN)"]["color"],
                ],
                default="#95a5a6",
            )

        HANDLERS = {"gt": _logic_gt, "no_skip": _logic_no_skip, "skip": _logic_skip}

        # --- HELPER FUNCTIONS ---
        def rle_encoding(colors):
            if len(colors) == 0:
                return []
            changes = np.concatenate(([0], np.where(colors[:-1] != colors[1:])[0] + 1))
            segments = []
            for i in range(len(changes)):
                start = changes[i]
                end = changes[i + 1] if i + 1 < len(changes) else len(colors)
                pct = ((end - start) / len(colors)) * 100
                segments.append((colors[start], pct))
            return segments

        def make_bar_html(segments, height_px=10):
            bar_inner = "".join(
                [
                    f'<div style="background:{c}; width:{p}%; height:100%;"></div>'
                    for c, p in segments
                ]
            )
            return f'<div style="display:flex; width:100%; height:{height_px}px;">{bar_inner}</div>'

        rows_html = ""
        for bar_key in active_bars:
            bar_conf = vis_cfg_dict["bar_types"].get(bar_key)
            if not bar_conf:
                continue

            cmap = vis_cfg_dict.get(bar_conf["map_name"])
            handler = HANDLERS.get(bar_key)

            if not cmap or not handler:
                continue

            try:
                colors = handler(df, cmap)
                segments = rle_encoding(colors)
                row_label = bar_conf.get("short_label", bar_key.upper())
                tooltip = bar_conf.get("legend_title", "")

                rows_html += (
                    f"<tr>"
                    f'<td style="width:40px; padding-right:5px; text-align:right; font-size:10px; font-weight:bold; color:#555; border:none;">{row_label}</td>'
                    f'<td style="width:{width_px}px; border:none; padding:1px 0;">'
                    f'<div style="width:100%; background:#eee; border:1px solid #ccc;" title="{tooltip}">'
                    f"{make_bar_html(segments)}"
                    f"</div>"
                    f"</td>"
                    f"</tr>"
                )
            except Exception as e:
                print(f"Error generating bar '{bar_key}': {e}")
                continue

        return f'<table style="margin:0 auto; border:none; width:auto; background:transparent;">{rows_html}</table>'

    @staticmethod
    def _apply_table_styles(df: pd.DataFrame) -> str:
        """Applies CSS styling to the DataFrame and returns HTML."""

        def highlight_risk(s):
            is_risk = s.name == "Miss (FN)"
            return [
                "color: #e74c3c; font-weight: bold;"
                if (is_risk and v != "0.0%")
                else ""
                for v in s
            ]

        styler = (
            df.style.apply(highlight_risk)
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
        return styler.to_html(escape=False)

    @staticmethod
    def _make_legend_html(bar_types_cfg: Dict) -> str:
        """Generates the HTML for the legend section."""
        html_parts = []
        html_parts.append('<div class="legend-box">')
        html_parts.append('<h3 style="margin-top:0;">Timeline Legend</h3>')
        html_parts.append(
            '<hr style="border:0; border-top:1px solid #eee; margin-bottom:10px;">'
        )

        html_parts.append('<div class="legend-grid">')

        for bar_type_key, bar_type_cfg in bar_types_cfg.items():
            html_parts.append('<div class="legend-section">')
            html_parts.append(
                f'<span class="legend-title">{bar_type_cfg["legend_title"]}</span>'
            )

            for label, val in bar_type_cfg["map"].items():
                if isinstance(val, dict):
                    color = val.get("color", "#ccc")
                    desc = val.get("desc", label)
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
            html_parts.append("</div>")

        html_parts.append("</div>")
        html_parts.append("</div>")
        return "".join(html_parts)

    @classmethod
    def generate_report(
        cls,
        stats_list: List[Dict],
        vis_cfg_yaml: Path,
        output_file: str = "final_report.html",
    ):
        """
        Orchestrates the full report generation process.
        """
        if not stats_list:
            print("No stats to report.")
            return

        vis_cfg_dict = yamlfile.load_yaml(vis_cfg_yaml, to_dict=True)
        df_report = pd.DataFrame(stats_list)

        df_report["VISUALIZATION (Timeline)"] = df_report["_df"].apply(
            cls.generate_timeline_html, vis_cfg_dict=vis_cfg_dict
        )

        df_display = df_report.drop(columns=["_df", "_risk_raw"])
        html_table = cls._apply_table_styles(df_display)

        bar_types_cfg = vis_cfg_dict.get("bar_types", {})
        for bar_type_key in bar_types_cfg:
            map_name = bar_types_cfg[bar_type_key]["map_name"]
            bar_types_cfg[bar_type_key]["map"] = vis_cfg_dict.get(map_name, {})

        legend_html = cls._make_legend_html(bar_types_cfg)

        full_html = f"""
        <html>
        <head>
            <title>Skip Logic Efficiency Report</title>
            <style>
                @media print {{ body {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }} }}
                
                body {{ 
                    font-family: 'CMU Serif', 'Times New Roman', serif; 
                    color: #333; 
                    margin: 0;
                    padding: 0;
                    height: 100vh;
                    display: flex;
                    flex-direction: column;
                    overflow: hidden; 
                }}

                /* --- SMALLER LEGEND PANE --- */
                #top-pane {{
                    flex: 0 0 auto;
                    background: #fff;
                    z-index: 20;
                    padding: 15px 40px; /* Reduced padding */
                    border-bottom: 4px solid #eee;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                }}

                #bottom-pane {{
                    flex: 1 1 auto;
                    overflow: auto;
                    padding: 20px 40px;
                    background: #fdfdfd;
                }}

                .legend-grid {{
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 15px; /* Tighter gap */
                }}
                
                /* --- COMPACT FONT SIZES --- */
                h2 {{ 
                    color: #2c3e50; 
                    text-align: center; 
                    margin-top: 0; 
                    margin-bottom: 10px; /* Reduced margin */
                    font-size: 20px;     /* Smaller Header */
                }}
                h3 {{
                    font-size: 14px;
                    margin-bottom: 5px;
                    color: #555;
                }}
                
                .legend-box {{ width: 100%; }}
                
                /* Smaller dots and text */
                .dot {{ 
                    height: 10px; width: 10px; /* Smaller dots */
                    display: inline-block; 
                    border-radius: 2px; 
                    margin-right: 6px; 
                    vertical-align: middle; 
                }}
                .legend-item {{ 
                    margin-bottom: 3px; 
                    font-size: 11px; /* Smaller font */
                    line-height: 1.4; 
                    white-space: nowrap; 
                }}
                .legend-title {{ 
                    font-weight: bold; 
                    margin-bottom: 5px; 
                    display: block; 
                    color: #555; 
                    border-bottom: 1px solid #eee; 
                    padding-bottom: 2px;
                    font-size: 12px; /* Smaller title */
                }}

                /* --- TABLE STYLING --- */
                table {{ 
                    border-collapse: separate; 
                    border-spacing: 0;
                    width: 100%; 
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1); 
                    font-size: 13px; /* Slightly smaller table text */
                }}
                
                th {{
                    position: sticky;
                    top: 0;
                    z-index: 10;
                    background-color: #2c3e50 !important;
                    color: white;
                    padding: 8px 12px; /* Compact header padding */
                    border-bottom: 2px solid #ddd;
                }}
            </style>
        </head>
        <body>
            <div id="top-pane">
                <h2>Skip Logic Performance Report</h2>
                {legend_html}
            </div>
            <div id="bottom-pane">
                {html_table}
            </div>
        </body>
        </html>
        """

        with open(output_file, "w") as f:
            f.write(full_html)

        print(f"✅ Report successfully generated: ⏬")
        pprint_local_path(output_file, get_wins_path=True)


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def generate_dummy_data_scenario(scenario_type="perfect", total_frames=500):
    data = []
    rand_num_fire_frames = int(total_frames * np.random.uniform(0.1, 0.7))
    fire_start = max(0, np.random.randint(0, total_frames - rand_num_fire_frames))
    fire_end = min(fire_start + rand_num_fire_frames - 1, total_frames - 1)

    for i in range(total_frames):
        is_fire_event = fire_start <= i <= fire_end
        gt = "Fire" if is_fire_event else "Safe"

        if gt == "Fire":
            pred = "Fire" if np.random.rand() > 0.05 else "Safe"
        else:
            pred = "Safe" if np.random.rand() > 0.05 else "Fire"

        if scenario_type == "perfect":
            action = (
                "Processed"
                if is_fire_event
                else ("Skipped" if np.random.rand() > 0.1 else "Processed")
            )
        elif scenario_type == "dangerous_miss":
            action = (
                ("Skipped" if np.random.rand() > 0.2 else "Processed")
                if is_fire_event
                else "Skipped"
            )
        elif scenario_type == "inefficient":
            action = "Processed" if np.random.rand() > 0.2 else "Skipped"

        data.append(
            {"frame_idx": i, "action": action, "gt_label": gt, "pred_label": pred}
        )
    return pd.DataFrame(data)


def main():
    args = CustomArgs().parse_args()
    OUT_FILE = "./zout/reports/viz_skip_v3.html"
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    scenarios = ["perfect", "dangerous_miss", "inefficient"]
    videos = []
    for i in range(10):
        scenario = np.random.choice(scenarios)
        video_name = f"Video_{i + 1:02d}_{scenario}.mp4"
        df = generate_dummy_data_scenario(scenario, np.random.randint(400, 800))
        videos.append((video_name, df))

    stats_list = []
    for name, df in videos:
        stats = SkipLogicReport.calculate_video_stats(name, df)
        if stats:
            stats_list.append(stats)

    SkipLogicReport.generate_report(
        stats_list, vis_cfg_yaml=args.vis_cfg, output_file=OUT_FILE
    )


if __name__ == "__main__":
    main()
