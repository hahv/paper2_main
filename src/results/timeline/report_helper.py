import pandas as pd
import numpy as np
from typing import Dict, Literal
from halib import *  # Assuming this is available as in original file
from src.results.timeline.data_parser import TimelineProcessor


class TimelineReportGenerator:
    """
    Generates an HTML report for timeline visualization using TimelineProcessor.
    """

    def __init__(self, cols_to_types: Dict[str, str]):
        self.cols_to_types = cols_to_types

    def run(
        self,
        df: pd.DataFrame,
        output_path: str,
        title: str = "Timeline Report",
        table_mode: Literal["p", "fc", "pfc"] = "pfc",
    ):
        """
        Main entry point: Process dataframe and generate HTML report.
        """
        # 1. Process Data using the Logic Core
        final_df, stats_df, styles_map = TimelineProcessor.proc_dataframe(
            df, self.cols_to_types, table_mode=table_mode
        )

        # 2. Add "FRAMES" and "VISUALIZATION"
        # We need to construct metadata for each video in stats_df
        # Use final_df to get per-video frame data
        final_df_reset = final_df.reset_index()
        video_groups = {
            vid: grp.sort_values("frame_id")
            for vid, grp in final_df_reset.groupby("video")
        }

        # Prepare list for new columns (using dict for alignment)
        frames_list = []
        viz_list = []

        # Iterate stats_df.index to exact order (includes 'TOTAL')
        for video_name in stats_df.index:
            if video_name in video_groups:
                vid_df = video_groups[video_name]
                frames = len(vid_df)
                viz = self.generate_timeline_html(vid_df, styles_map)
            elif video_name == "TOTAL":
                frames = sum(len(v) for v in video_groups.values())
                viz = "-"
            else:
                frames = 0
                viz = "-"

            frames_list.append(frames)
            viz_list.append(viz)

        # 3. Construct Final Report DataFrame with MultiIndex Columns
        # Filter existing stats_df columns based on config `include`
        cols_to_keep = []
        for col in stats_df.columns:
            # col is tuple (Method, Outcome)
            method_key = col[0]
            if styles_map.get(method_key, {}).get("table", {}).get("include", True):
                cols_to_keep.append(col)

        report_df = stats_df[cols_to_keep].copy()

        # Add Metadata columns with (" ", "Column Name") structure to match MultiIndex
        # Using a single space " " as the top level grouping for general info
        report_df[(" ", "FRAMES")] = frames_list
        report_df[(" ", "VISUALIZATION")] = viz_list
        report_df[(" ", "VIDEO NAME")] = report_df.index

        # Reset index (drop=True since we already copied it to a column)
        report_df.reset_index(drop=True, inplace=True)

        # Ensure we maintain MultiIndex columns
        if not isinstance(report_df.columns, pd.MultiIndex):
            # Fallback if something flattened it (unlikely with this approach)
            report_df.columns = pd.MultiIndex.from_tuples(report_df.columns)

        # 4. Reorder Columns
        # Desired: VIDEO NAME, FRAMES, [Method1...], [Method2...], VISUALIZATION

        # Get Method Columns in order (preserve relative order from stats_df)
        method_cols = [c for c in cols_to_keep if c in report_df.columns]

        final_cols = (
            [
                (" ", "VIDEO NAME"),
                (" ", "FRAMES"),
            ]
            + method_cols
            + [(" ", "VISUALIZATION")]
        )

        # Select and reorder
        report_df = report_df[final_cols]

        # 3. Render HTML
        self.render_html(report_df, styles_map, output_path, title)
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
            # Handle new nested config structure
            labels_colors = style_cfg.get("timeline", {}).get(
                "labels_colors"
            ) or style_cfg.get("labels_colors", {})

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

    def render_html(
        self, df_report: pd.DataFrame, styles_map: Dict, output_file: str, title: str
    ):
        # 1. Styler Configuration
        styler = (
            df_report.style.set_properties(
                **{
                    "text-align": "center",
                    "vertical-align": "middle",
                    "font-size": "13px",
                }
            )
            .set_table_styles(
                [
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", "#2c3e50"),
                            ("color", "white"),
                            ("padding", "8px"),
                            (
                                "border",
                                "1px solid #fff",
                            ),  # White border for header grid
                            ("white-space", "nowrap"),
                            ("text-align", "center"),
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

        # Highlight Rules logic updated for MultiIndex columns
        def highlight_cells(s):
            # s.name is tuple: (Method, Outcome)
            if not isinstance(s.name, tuple):
                return ["" for _ in s]

            method_key, outcome = s.name

            # Skip metadata columns
            if method_key.strip() == "":
                return ["" for _ in s]

            # Find matching config
            # methods in styles_map keys match the Method part of column
            method_cfg = styles_map.get(method_key)
            if not method_cfg:
                return ["" for _ in s]

            rules = method_cfg.get("table", {}).get("highlight_rules", {})
            rule = rules.get(outcome)

            if not rule:
                return ["" for _ in s]

            condition = rule.get("condition", "")
            color = rule.get("color", "red")

            # Parse condition e.g. "< 20"
            import operator

            op_map = {
                "<": operator.lt,
                "<=": operator.le,
                ">": operator.gt,
                ">=": operator.ge,
                "==": operator.eq,
                "!=": operator.ne,
            }

            parts = condition.strip().split()
            if len(parts) != 2 or parts[0] not in op_map:
                return ["" for _ in s]

            op = op_map[parts[0]]
            try:
                threshold = float(parts[1])
            except ValueError:
                return ["" for _ in s]

            styles = []
            for val in s:
                # Parse value: "10.5% (20)" -> 10.5
                try:
                    pct = float(str(val).split("%")[0])
                    if op(pct, threshold):
                        styles.append(f"color: {color}; font-weight: bold;")
                    else:
                        styles.append("")
                except (ValueError, IndexError):
                    styles.append("")
            return styles

        styler.apply(highlight_cells)

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
            # Handle new nested config structure
            title = style_cfg.get("meta", {}).get("legend_title") or style_cfg.get(
                "legend_title", col_name
            )
            labels_colors = style_cfg.get("timeline", {}).get(
                "labels_colors"
            ) or style_cfg.get("labels_colors", {})

            html += (
                f'<div class="legend-section"><div class="legend-title">{title}</div>'
            )

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
