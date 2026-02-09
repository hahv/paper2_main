import pandas as pd
import numpy as np
from typing import Dict, Literal, Callable, Optional, Union
from halib import *
from src.results.timeline.data_parser import TimelineProcessor, TimelineConfig
from src.config import Config
from pathlib import Path
from halib.filetype import yamlfile


class TlReportGen:
    """
    Generates an HTML report for timeline visualization using TimelineProcessor.
    """

    FIX_COLUMNS = ["video", "video_path", "frame_id", "gt_label"]
    CSV_PATH_DF_COLUMNS = ["video", "gt_csv_path"]
    NOTEMP_MT_PATTERN = "mt_no_temp_method"

    def __init__(self, cols_to_types: Dict[str, str]):
        self.cols_to_types = cols_to_types

    @staticmethod
    def simplify_exp_name(exp_name: str) -> str:
        """
        Default function to shorten experiment names for column naming.
        Example: "mt_temp_method_motion_block" -> "temp_method_motion_block"
        """
        SEP = "__"
        parts = exp_name.split(SEP)
        # only get the part after 'mt_'
        for part in parts:
            if part.startswith("mt_"):
                return part[3:]  # Remove 'mt_' prefix
        return exp_name  # Fallback to full name if no 'mt_' part found

    @staticmethod
    def default_sort_func_tlreport(df: pd.DataFrame) -> pd.DataFrame:
        sort_cols = []
        ascending_vals = []

        # 1. Priority: no_temp_method
        # Check for columns under 'no_temp_method'
        if "no_temp_method" in df.columns.get_level_values(0):
            for sub_col in ["Miss (FN)", "False Alarm (FP)"]:
                col_tuple = ("no_temp_method", sub_col)
                if col_tuple in df.columns:
                    sort_cols.append(col_tuple)
                    ascending_vals.append(False)

        # 2. Priority: temp_method*
        # Check for columns starting with 'temp_method'
        temp_cols = [
            c
            for c in df.columns.get_level_values(0).unique()
            if str(c).startswith("temp_method")
        ]

        for t_col in temp_cols:
            for sub_col in ["Miss (FN)", "Waste (FP)"]:
                col_tuple = (t_col, sub_col)
                if col_tuple in df.columns:
                    sort_cols.append(col_tuple)
                    ascending_vals.append(False)

        # Separate TOTAL row to keep it at top
        total_col = (" ", "VIDEO NAME")
        if total_col in df.columns:
            is_total = df[total_col] == "TOTAL"
            df_total = df[is_total]
            df_rest = df[~is_total]
        else:
            df_total = pd.DataFrame()
            df_rest = df

        if sort_cols:
            df_rest = df_rest.sort_values(by=sort_cols, ascending=ascending_vals)

        if not df_total.empty:
            return pd.concat([df_total, df_rest], ignore_index=True)

        return df_rest

    @staticmethod
    # ! col_name is the shortened name
    def col_name_to_timeline_type(col_name: str) -> str:
        # first find the part that starts with 'mt_'
        if col_name.startswith("no_temp_method"):
            return "no_skip"
        elif col_name.startswith("temp_method"):
            return "skip"
        else:
            raise ValueError(f"Cannot determine timeline type for col_name={col_name}")

    @staticmethod
    def get_timeline_csv_path_df(
        exp_dir: str,
        shorten_exp_name_func: Callable[[str], str] = simplify_exp_name,
        exp_name_to_timeline_type: Callable[[str], str] = col_name_to_timeline_type,
        do_normalize: bool = True,
    ) -> tuple[pd.DataFrame, Dict[str, str]]:
        """
        Loads GT, Experiment, and Baseline data into a single frame-level DataFrame and a mapping of columns to timeline types.
        1. Identifies the baseline 'no_temp' experiment directory.
        2. Loads frame-level labels from CSVs for GT, Experiment, and Baseline.
        3. Merges data on frame indices and constructs the final DataFrame.
        4. Calculates timeline types for each column based on experiment names.

        Return:
            combined_df (pd.DataFrame): Merged DataFrame with frame-level labels.
            timeline_types_by_col (Dict[str, str]): Mapping of column names to timeline types
        """

        def _find_bl_notemp_dir(exp_path: Path, ds_name: str) -> Path | None:
            """Heuristic to find the latest 'no_temp' sibling directory."""
            if TlReportGen.NOTEMP_MT_PATTERN in exp_path.name:
                return None  # Current is already no_temp_method
            candidates = [
                p
                for p in exp_path.parent.iterdir()
                if p.is_dir()
                and p != exp_path
                and TlReportGen.NOTEMP_MT_PATTERN in p.name
                and f"ds_{ds_name}" in p.name
            ]
            # Return latest by name (lexicographical sort usually works for timestamps)
            return max(candidates, key=lambda p: p.name) if candidates else None

        def _load_frame_series(
            path: Path, col_name: str, required: bool = False
        ) -> pd.DataFrame:
            """
            Helper to read a 2-column CSV (frame_idx, value) and standardise it.
            """
            if not path.exists():
                if required:
                    raise FileNotFoundError(f"Required CSV not found: {path}")
                return pd.DataFrame()

            # Efficiently read only needed columns
            # GT uses 'label', Predictions use 'pred_label'
            val_col = "label" if col_name == "gt_label" else "pred_label"
            # [Context: infer results of exp]
            # ! pred_label column is set to "None", so if we do not specify dtype for it, it will be loaded as NaN type, but we need str type here
            df = pd.read_csv(
                path,
                sep=";",
                usecols=["frame_idx", val_col],
                encoding="utf-8",
                dtype={val_col: str},
                keep_default_na=False,
            )  # ty:ignore[no-matching-overload]

            # all unique values is val_col, must not any nan values
            if df[val_col].isna().any():
                raise ValueError(
                    f"Column '{val_col}' in {path} contains NaN values for col_name={col_name}"
                )

            # convert to lower case of val_col
            df[val_col] = df[val_col].str.lower()

            # Standardise to 'frame_id' + 'col_name'
            return df.rename(columns={"frame_idx": "frame_id", val_col: col_name})

        exp_path = Path(exp_dir)

        # 1. Setup: Load Config to find Dataset Path
        config_file = exp_path / "__config.yaml"
        if not config_file.exists():
            raise FileNotFoundError(f"Config file missing: {config_file}")

        cfg_data = yamlfile.load_yaml(str(config_file), to_dict=True)
        exp_cfg = Config.from_custom_yaml_file_or_str(cfg_data.get("original-yaml-str"))
        dataset_dir = Path(exp_cfg.dbsetCfg.dir_path)  # ty:ignore[invalid-argument-type]
        assert dataset_dir.exists(), f"Dataset dir not found: {dataset_dir}"

        # 2. Discovery: Find Baseline Dir & Video Files
        baseline_dir = _find_bl_notemp_dir(
            exp_path,
            exp_cfg.dbsetCfg.name,  # ty:ignore[invalid-argument-type]
        )

        video_files = fs.filter_files_by_extension(
            str(dataset_dir), [".mp4", ".avi", ".mov", ".mkv"], recursive=True
        )

        # 3. Processing: Define Tracks & Iterate
        dfs = []

        # We merge these tracks onto the Ground Truth anchor
        # Format: (directory, column_name)
        exp_col_name = shorten_exp_name_func(exp_path.name)
        to_merge_series = [(exp_path, exp_col_name)]
        if baseline_dir:
            to_merge_series.insert(
                0, (baseline_dir, shorten_exp_name_func(baseline_dir.name))
            )

        # pprint(f"to merge series: {to_merge_series}")

        for vid_path in video_files:
            vid_name = fs.get_file_name(vid_path, split_file_ext=True)[0]
            vid_parent = Path(vid_path).parent

            # A. Load Anchor (Ground Truth)
            gt_path = vid_parent / f"{vid_name}__labels.csv"
            df_merged = _load_frame_series(gt_path, "gt_label", required=True)
            gt_len = len(df_merged)

            # B. Merge Additional Series
            for series_dir, col_name in to_merge_series:
                series_path = series_dir / f"{vid_name}_results.csv"
                df_series = _load_frame_series(series_path, col_name, required=False)

                if not df_series.empty:
                    # assert len(df_series) == gt_len, (
                    #     f"[Error][Video={vid_name}] Mismatched frame count for gt ({gt_len}) vs series data for col_name={col_name} ({len(df_series)})"
                    # )

                    if len(df_series) != gt_len:
                        console.print(
                            f"[yellow][Warning][Video={vid_name}] Mismatched frame count for gt ({gt_len}) vs series data for col_name={col_name} ({len(df_series)}). Using inner join to align frames.[/yellow]"
                        )
                    # "Inner" join ensures only frame_ids present in BOTH dataframes remain.
                    # If df_series is missing rows, df_merged will shrink to match.
                    df_merged = pd.merge(
                        df_merged,
                        df_series[["frame_id", col_name]],
                        on="frame_id",
                        how="inner",
                    )
                else:
                    df_merged[col_name] = np.nan

            df_merged["video"] = vid_name
            df_merged["video_path"] = str(os.path.abspath(vid_path))
            dfs.append(df_merged)

        # 4. Finalize
        if not dfs:
            return pd.DataFrame(), {}

        combined_df = pd.concat(dfs, ignore_index=True)

        # Reorder columns: Metadata -> GT -> Exp -> Base
        start_cols = TlReportGen.FIX_COLUMNS
        other_cols = [col for col in combined_df.columns if col not in start_cols]
        combined_df = combined_df[start_cols + other_cols]

        timeline_types_by_col = {"gt_label": "gt"}
        for col in other_cols:
            timeline_types_by_col[col] = exp_name_to_timeline_type(col)

        if do_normalize:
            combined_df = TlReportGen.norm_timeline_df(combined_df)
        return combined_df, timeline_types_by_col

    @staticmethod
    def get_unique_values(df: pd.DataFrame) -> Dict[str, list]:
        """
        Get unique values for each column in the dataframe.
        """
        assert all(col in df.columns for col in TlReportGen.FIX_COLUMNS), (
            f"This function only supports dataframes with fixed columns: {TlReportGen.FIX_COLUMNS} - for timeline dataframe"
        )
        unique_by_cols = {}
        for col in df.columns:
            if col in TlReportGen.FIX_COLUMNS[:-1]:  # skip 'video' and 'frame_id'
                continue
            unique_by_cols[col] = list(df[col].unique())
        return unique_by_cols

    @staticmethod
    def norm_timeline_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert gt_label and other columns to standard labels with respect to 'fire' and 'none' and 'skipped' (if using temp method).

        Example:
        Input:  {
                │   'gt_label': ['fire_smoke', 'none'],
                │   'no_temp_method': ['fire', 'smokeonly', 'none'],
                │   'temp_method_motion_block': ['skipped', 'fire', 'smokeonly', 'none']
                }
        Output: {
                │   'gt_label': ['fire', 'none'],
                │   'no_temp_method': ['fire', 'none'],
                │   'temp_method_motion_block': ['skipped', 'fire', 'none']
                }
        """

        def _standardize_label(col: str, label: str) -> str:
            if "fire" in label or "smoke" in label:
                return "fire"
            if col.startswith("temp_method"):
                return label  # keep 'skipped' as is
            return "none"

        df = df.copy()
        for col in df.columns:
            if col in TlReportGen.FIX_COLUMNS[:-1]:  # skip 'video' and 'frame_id'
                continue
            df[col] = df[col].apply(lambda x: _standardize_label(col, x))
        return df

    @staticmethod
    def tlreport_from_csv(
        csv_path: str,
        output_html_path: Optional[str] = None,
        title: str = "Timeline Report (Reconstructed)",
    ):
        """
        Reconstructs the HTML report from a saved CSV file (report_df).
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Load Dataframe with MultiIndex columns (2 header rows)
        df = pd.read_csv(csv_path, header=[0, 1], sep=";", encoding="utf-8")

        # Drop VIDEO_PATH if present (ignore it for reconstruction/visualization)
        # Check if any column has 'VIDEO_PATH' (case-insensitive) in either level
        cols_to_drop = [
            c
            for c in df.columns
            if "video_path" in str(c[1]).lower() or "video_path" in str(c[0]).lower()
        ]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)

        # Deduce cols_to_types
        cols_to_types = {}
        # Level 0 contains Method names. Level 1 contains Outcomes.
        # We look at unique values in Level 0
        level0_cols = df.columns.get_level_values(0).unique()

        for col in level0_cols:
            col = str(col).strip()
            # Skip empty or strict metadata parent or "Unnamed" artifacts
            if not col or col.lower() == "nan" or "unnamed" in col.lower():
                continue

            if col == "gt_label":
                cols_to_types[col] = "gt"
                continue

            try:
                t_type = TlReportGen.col_name_to_timeline_type(col)
                cols_to_types[col] = t_type
            except ValueError:
                pass  # Not a recognized method column

        # Reconstruct Styles Map
        styles_map = {}
        for col, t_type in cols_to_types.items():
            styles_map[col] = TimelineConfig.get_timeline_dict(t_type)
        if output_html_path is None:
            output_html_path = csv_path.replace(".csv", "_reconstructed.html")

        gen = TlReportGen(cols_to_types)
        gen.render_html(df, styles_map, output_html_path, title)

        with ConsoleLog("Reconstructed Report"):
            print(f"[INFO] Report generated at: ⏬")
            pprint_local_path(output_html_path, get_wins_path=True)

    @staticmethod
    def gen_TlReport_muti_exps(
        parent_dir: str = "./zout/zruns", table_mode: Literal["p", "fc", "pfc"] = "p"
    ):
        """
        Generate timeline reports for all experiment directories under the given parent directory.
        """
        parent_path = Path(parent_dir)
        exp_dirs = [p for p in parent_path.iterdir() if p.is_dir()]
        for exp_dir in exp_dirs:
            pprint(f"[INFO] Gen timeline report for exp: {exp_dir.name}")
            try:
                outfile = TlReportGen.gen_TlReport_exp(
                    exp_dir=exp_dir,
                    title=f"Timeline Report - {exp_dir.name}",
                    table_mode=table_mode,
                )
                pprint(f"[INFO] Report generated at: ⏬")
                pprint_local_path(outfile, get_wins_path=True)
            except Exception as e:
                with ConsoleLog("Error Generating Report"):
                    pprint(exp_dir.name)
                    pprint(f"   >>>[ERROR] {e}")

    @staticmethod
    def gen_TlReport_exp(
        exp_dir: Union[Path, str],
        title: Optional[str] = None,
        table_mode: Literal["p", "fc", "pfc"] = "p",
    ):
        exp_dir = Path(exp_dir)

        # from exp_dir, do something to get the dataframe and cols_to_types
        df, cols_to_types = TlReportGen.get_timeline_csv_path_df(
            str(exp_dir), do_normalize=True
        )
        # # !debug
        # csvfile.fn_display_df(df.head(5))
        # pprint(cols_to_types)
        # assert False, "stop"
        report_generator = TlReportGen(cols_to_types)
        output_path = exp_dir / "timeline_report.html"
        if title is None:
            title = f"Timeline Report - {exp_dir.name}"
        report_generator.generate(
            df, str(output_path), title=title, table_mode=table_mode
        )
        return os.path.abspath(output_path)

    def generate(
        self,
        df: pd.DataFrame,
        output_path: str,
        title: str = "Timeline Report",
        table_mode: Literal["p", "fc", "pfc"] = "p",
        sort_func_tlreport_df: Optional[
            Callable[[pd.DataFrame], pd.DataFrame]
        ] = default_sort_func_tlreport,
    ):
        """
        Main entry point: Process dataframe and generate HTML report.
        Args:
            df (pd.DataFrame): Input timeline dataframe.
            output_path (str): Path to save the HTML report.
            title (str): Title of the report.
            table_mode (Literal["p", "fc", "pfc"]): Table mode for statistics.
            p - percentages only
            fc - frame counts only
            pfc - percentages and frame counts
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
        path_list = []

        # Iterate stats_df.index to exact order (includes 'TOTAL')
        for video_name in stats_df.index:
            if video_name in video_groups:
                vid_df = video_groups[video_name]
                # !debug
                # csvfile.fn_display_df(vid_df.head(5))
                # assert False, "stop"
                frames = len(vid_df)
                viz = self.generate_timeline_html(vid_df, styles_map)
                path = str(vid_df["video_path"].iloc[0]) if "video_path" in vid_df.columns else ""
            elif video_name == "TOTAL":
                frames = sum(len(v) for v in video_groups.values())
                viz = "-"
                path = "TOTAL"
            else:
                frames = 0
                viz = "-"
                path = ""

            frames_list.append(frames)
            viz_list.append(viz)
            path_list.append(path)

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
        report_df[(" ", "VIDEO_PATH")] = path_list

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
                (" ", "VIDEO_PATH"),
                (" ", "FRAMES"),
            ]
            + method_cols
            + [(" ", "VISUALIZATION")]
        )

        # Select and reorder
        report_df = report_df[final_cols]
        # ! Sort Rows if needed (Before generating HTML, and saving CSV)
        if sort_func_tlreport_df:
            report_df = sort_func_tlreport_df(report_df)

        # 3. Render HTML
        self.render_html(report_df, styles_map, output_path, title)

        # also save report_df to csv for easier debugging
        csv_output_path = output_path.replace(".html", ".csv")
        report_df.to_csv(csv_output_path, index=False, sep=";", encoding="utf-8")
        with ConsoleLog("Saving report results:"):
            print(f"[INFO] Report generated at: ⏬")
            pprint_local_path(output_path, get_wins_path=True)
            print(f"[INFO] CSV version saved at: ⏬")
            pprint_local_path(csv_output_path, get_wins_path=True)

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

        if (" ", "VIDEO_PATH") in df_report.columns:
            styler.hide(axis="columns", subset=[(" ", "VIDEO_PATH")])

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
        # !#TODO: font-family: 'CMU Serif' can be use to show as in paper
        full_html = f"""
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{  font-family: 'Segoe UI', serif; margin: 0; padding: 0; height: 100vh; display: flex; flex-direction: column; overflow: hidden; color: #333; }}

                #top-pane {{ flex: 0 0 auto; background: #fff; padding: 15px 30px; border-bottom: 4px solid #eee; z-index: 20; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }}
                #bottom-pane {{ flex: 1 1 auto; overflow: auto; padding: 20px 30px; background: #fdfdfd; }}

                .legend-grid {{ display: flex; flex-wrap: wrap; gap: 30px; }}
                .legend-section {{ font-size: 12px; margin-bottom: 10px; }}
                .legend-title {{ font-weight: bold; border-bottom: 2px solid #eee; margin-bottom: 8px; padding-bottom: 4px; color: #444; font-size: 13px; }}
                .legend-item {{ margin-bottom: 4px; display: flex; align-items: center; }}
                .dot {{ height: 10px; width: 10px; border-radius: 2px; margin-right: 8px; display: inline-block; flex-shrink: 0; }}

                h2 {{ margin: 0 0 15px 0; color: #2c3e50; font-size: 22px; }}

                table {{ border-collapse: separate; border-spacing: 0; min-width: 100%; }}

                /* Sticky Headers Configuration */
                th {{
                    position: sticky;
                    z-index: 10;
                    border-bottom: 2px solid #ccc;
                }}

                /* Level 1 Header (Method names) */
                thead tr:nth-child(1) th {{
                    top: 0;
                    z-index: 15;
                }}

                /* Level 2 Header (Metric names) */
                /* Height estimate: 13px font + 16px padding + borders ~ 35px */
                thead tr:nth-child(2) th {{
                    top: 35px;
                    z-index: 14;
                    box-shadow: 0 2px 2px -1px rgba(0,0,0,0.1);
                }}
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
