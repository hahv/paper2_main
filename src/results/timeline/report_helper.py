import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from halib import *  # Assuming this is available as in original file
from src.results.timeline.data_parser import TimelineProcessor

class TimelineReportGenerator:
    """
    Generates an HTML report for timeline visualization using TimelineProcessor.
    """
    def __init__(self, cols_to_types: Dict[str, str]):
        self.cols_to_types = cols_to_types

    def run(self, df: pd.DataFrame, output_path: str, title: str = "Timeline Report"):
        """
        Main entry point: Process dataframe and generate HTML report.
        """
        # 1. Process Data using the Logic Core
        final_df, stats_df, styles_map = TimelineProcessor.proc_dataframe(
            df, self.cols_to_types
        )

        # 2. Prepare Report Rows
        # Group frame-level data by video for visualization
        final_df_reset = final_df.reset_index()
        video_groups = {
            vid: grp.sort_values("frame_id")
            for vid, grp in final_df_reset.groupby("video")
        }

        report_rows = []

        # stats_df has 'video' as index (and includes a TOTAL row)
        # We will parse stats matrix into a flat dictionary for the report table
        for video_name, row in stats_df.iterrows():
            row_dict = {"VIDEO NAME": video_name}

            # Flatten MultiIndex Columns from stats_df: (Method, Outcome) -> "METHOD<br>Outcome"
            for (method_col, outcome), val in row.items():  # ty:ignore[not-iterable]
                # Clean up value (it comes as string "pct% (count)" from data_parser if mode='pfc')
                header = f"{method_col.upper()}<br><span style='font-size:10px; font-weight:normal'>{outcome}</span>"
                row_dict[header] = val

            # Create Timeline Visualization
            if video_name in video_groups:
                vid_df = video_groups[video_name]
                row_dict["FRAMES"] = len(vid_df)
                viz_html = self.generate_timeline_html(vid_df, styles_map)
                row_dict["VISUALIZATION"] = viz_html
            elif video_name == "TOTAL":
                row_dict["FRAMES"] = sum(len(v) for v in video_groups.values())
                row_dict["VISUALIZATION"] = "-"
            else:
                row_dict["FRAMES"] = 0
                row_dict["VISUALIZATION"] = "-" # For Total row or missing data

            report_rows.append(row_dict)

        # Create DataFrame for Report
        df_report = pd.DataFrame(report_rows)

        # Reorder columns: VIDEO NAME, FRAMES, ...stats..., VISUALIZATION
        cols = list(df_report.columns)
        priorities = ["VIDEO NAME", "FRAMES"]
        end_cols = ["VISUALIZATION"]

        mid_cols = [c for c in cols if c not in priorities and c not in end_cols]
        final_cols = priorities + mid_cols + end_cols

        # Ensure columns exist before selecting
        final_cols = [c for c in final_cols if c in df_report.columns]

        df_report = df_report[final_cols]

        # 3. Render HTML
        self.render_html(df_report, styles_map, output_path, title)
        print(f"[INFO] Report generated at: {output_path}")

    def generate_timeline_html(self, vid_df: pd.DataFrame, styles_map: Dict) -> str:
        """Generates the stacked bar HTML for a single video."""
        rows_html = ""

        # Maintain order from config
        for col_name, type_key in self.cols_to_types.items():
            if col_name not in vid_df.columns:
                continue

            style_cfg = styles_map.get(col_name, {})
            labels = vid_df[col_name].values

            # Resolve labels to colors
            labels_colors = style_cfg.get("labels_colors", {})

            def get_color(lbl):
                entry = labels_colors.get(lbl, "#ccc")
                if isinstance(entry, dict):
                    return entry.get("color", "#ccc")
                return entry

            colors = [get_color(l) for l in labels]
            segments = self._rle_encoding(colors)
            bar_html = self._make_bar_html(segments)

            short_label = col_name.upper()

            rows_html += f"""
                <tr>
                    <td style="width:50px; font-size:10px; font-weight:bold; border:none; text-align:right; padding-right:8px; color:#555; white-space:nowrap;">{short_label}</td>
                    <td style="width:300px; border:none; padding:2px 0;">
                        <div style="width:100%; border:1px solid #ccc; background:#eee;">{bar_html}</div>
                    </td>
                </tr>"""

        return f'<table style="background:transparent; border:none; margin:0;">{rows_html}</table>'

    def render_html(self, df_report: pd.DataFrame, styles_map: Dict, output_file: str, title: str):
        # 1. Styler Configuration
        styler = (
            df_report.style.set_properties(
                **{"text-align": "center", "vertical-align": "middle", "font-size": "13px"}
            )
            .set_table_styles(
                [
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#2c3e50"),
                            ("color", "white"),
                            ("padding", "10px"),
                            ("white-space", "nowrap"),
                        ],
                    },
                    {
                        "selector": "td",
                        "props": [("border", "1px solid #ddd"), ("padding", "6px")],
                    },
                    {
                        "selector": "tr:nth-child(even)",
                        "props": [("background-color", "#f9f9f9")],
                    },
                ]
            )
            .hide(axis="index")
        )

        # Highlight "Miss" columns in Red if they have "Miss" in header and value is not 0
        def highlight_miss(s):
            is_miss_col = "Miss" in s.name or "FN" in s.name
            if not is_miss_col:
                return ["" for _ in s]

            results = []
            for v in s:
                # Check if value starts with "0.0%" or "0 " -> No red
                # Value format is typically "10.5% (20)"
                if str(v).startswith("0.0%") or str(v).startswith("0 "):
                    results.append("")
                else:
                    results.append("color: #e74c3c; font-weight: bold;")
            return results

        styler.apply(highlight_miss)

        html_table = styler.to_html(escape=False)
        legend_html = self._make_legend_html(styles_map)

        # 2. Final HTML Assembly
        full_html = f"""
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: 'Segoe UI', sans-serif; margin: 0; padding: 0; height: 100vh; display: flex; flex-direction: column; overflow: hidden; color: #333; }}

                #top-pane {{ flex: 0 0 auto; background: #fff; padding: 15px 30px; border-bottom: 4px solid #eee; z-index: 20; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }}
                #bottom-pane {{ flex: 1 1 auto; overflow: auto; padding: 20px 30px; background: #fdfdfd; }}

                .legend-grid {{ display: flex; flex-wrap: wrap; gap: 30px; }}
                .legend-section {{ font-size: 12px; margin-bottom: 10px; }}
                .legend-title {{ font-weight: bold; border-bottom: 2px solid #eee; margin-bottom: 8px; padding-bottom: 4px; color: #444; font-size: 13px; }}
                .legend-item {{ margin-bottom: 4px; display: flex; align-items: center; }}
                .dot {{ height: 10px; width: 10px; border-radius: 2px; margin-right: 8px; display: inline-block; flex-shrink: 0; }}

                h2 {{ margin: 0 0 15px 0; color: #2c3e50; font-size: 22px; }}

                table {{ border-collapse: separate; border-spacing: 0; min-width: 100%; }}
                th {{ position: sticky; top: 0; z-index: 10; border-bottom: 2px solid #ccc; }}
            </style>
        </head>
        <body>
            <div id="top-pane">
                <h2>{title}</h2>
                {legend_html}
            </div>
            <div id="bottom-pane">
                {html_table}
            </div>
        </body>
        </html>
        """

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(full_html)

    def _make_legend_html(self, styles_map: Dict) -> str:
        html = '<div class="legend-grid">'

        # Iterate over configured columns
        for col_name, type_key in self.cols_to_types.items():
            style_cfg = styles_map.get(col_name, {})
            title = style_cfg.get("legend_title", col_name)
            labels_colors = style_cfg.get("labels_colors", {})

            html += f'<div class="legend-section"><div class="legend-title">{title}</div>'

            for label, val in labels_colors.items():
                if isinstance(val, dict):
                    color = val.get("color", "#ccc")
                    desc = val.get("desc", label)
                else:
                    color = val
                    desc = label

                html += f'<div class="legend-item"><span class="dot" style="background:{color}"></span><span>{desc}</span></div>'
            html += "</div>"

        html += "</div>"
        return html

    @staticmethod
    def _rle_encoding(arr):
        """Run Length Encoding for efficient HTML generation"""
        if len(arr) == 0:
            return []
        arr = np.array(arr)
        changes = np.concatenate(([0], np.where(arr[:-1] != arr[1:])[0] + 1))
        lengths = np.diff(np.append(changes, len(arr)))
        return [(arr[c], (lengths[i] / len(arr)) * 100) for i, c in enumerate(changes)]

    @staticmethod
    def _make_bar_html(segments):
        inner = "".join(
            [
                f'<div style="background:{c}; width:{p}%; height:12px;"></div>'
                for c, p in segments
            ]
        )
        return f'<div style="display:flex; width:100%;">{inner}</div>'
