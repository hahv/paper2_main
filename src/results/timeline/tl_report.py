import pandas as pd
import numpy as np
from typing import Dict, Literal, Callable, Optional, Union, List
from halib import *
from src.results.timeline.tl_converter import TlProcessor, TlConfig
from src.config import Config
from pathlib import Path
from halib.filetype import yamlfile
from src.common import GlobalConst
from src.metrics.loaders.base_csv_loader import BaseRawCsvLoader
from collections import OrderedDict


class TlReportGen:
    """
    Generates an HTML report for timeline visualization using TimelineProcessor.
    """

    TL_FIXED_COLUMNS = [
        GlobalConst.COL_VIDEO,
        GlobalConst.COL_VIDEO_PATH,
        GlobalConst.COL_NUM_FRAMES,
        GlobalConst.COL_FRAME_IDX,
        GlobalConst.COL_GT,
    ]

    def __init__(self, cols_to_types: Dict[str, str]):
        self.cols_to_types = cols_to_types

    @staticmethod
    def shorten_exp_name(exp_name: str) -> str:
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
        """
        Sorts the report dataframe by specific error columns defined in timeline_cfg.yaml.
        """
        # 1. Separate TOTAL row (Always keep at top)
        total_col = (" ", "VIDEO NAME")
        if total_col in df.columns:
            is_total = df[total_col] == "TOTAL"
            df_total = df[is_total]
            df_rest = df[~is_total].copy()
        else:
            df_total = pd.DataFrame()
            df_rest = df.copy()

        # 2. Collect Sort Instructions from Config
        sort_instructions = []

        # Iterate over top-level columns (Methods)
        method_cols = df.columns.get_level_values(0).unique()

        for method_col in method_cols:
            if str(method_col).strip() == "" or method_col == " ":
                continue  # Skip metadata

            try:
                # Resolve timeline type (e.g. 'no_temp_method' -> 'no_skip')
                t_type = TlReportGen.col_name_to_tl_type(method_col)
                cfg = TlConfig.get_tl_dict(t_type)

                sort_cfg = cfg.get("table", {}).get("sort_by", {})

                if not sort_cfg:
                    continue

                # sort_cfg is e.g. {'Miss (FN)': {'direction': 'desc', 'order': 1}, ...}
                for outcome_key, rule in sort_cfg.items():
                    col_tuple = (method_col, outcome_key)
                    if col_tuple in df.columns:
                        order = rule.get("order", 999)
                        direction = rule.get("direction", "asc")
                        ascending = direction.lower() == "asc"
                        sort_instructions.append((order, col_tuple, ascending))

            except (ValueError, KeyError):
                # skip columns that don't match known timeline types
                continue

        if not sort_instructions:
            return df

        # Sort by 'order' (primary key)
        sort_instructions.sort(key=lambda x: x[0])

        sort_cols = [x[1] for x in sort_instructions]
        ascending_vals = [x[2] for x in sort_instructions]

        # 3. Create a Numeric Shadow DataFrame for Sorting
        df_numeric = pd.DataFrame(index=df_rest.index)
        extract_regex = r"^([\d\.]+)"

        for col in sort_cols:
            try:
                s_str = df_rest[col].astype(str).str.strip()
                s_nums = s_str.str.extract(extract_regex, expand=False)
                df_numeric[col] = pd.to_numeric(s_nums, errors="coerce").fillna(0)
            except Exception:
                df_numeric[col] = 0.0

        # 4. Perform Sort
        sorted_index = df_numeric.sort_values(
            by=sort_cols, ascending=ascending_vals
        ).index

        # 5. Reassemble
        df_sorted = df_rest.loc[sorted_index]

        if not df_total.empty:
            return pd.concat([df_total, df_sorted])

        return df_sorted

    @staticmethod
    def _truncate_video_name(name: str, limit: int) -> str:
        """Truncate a video file name in the middle to `limit` chars.
        Example (limit=22): 'Very_long_video_name_cam1.mp4' -> 'Very_lon..._cam1.mp4'
        """
        if limit <= 0 or len(name) <= limit:
            return name

        dot_idx = name.rfind(".")
        if dot_idx > 0:
            stem = name[:dot_idx]
            ext = name[dot_idx:]
        else:
            stem = name
            ext = ""

        chars_to_keep = limit - 3 - len(ext)

        if chars_to_keep <= 0:
            return name[: limit - 3] + "..."

        keep_front = chars_to_keep // 2 + (chars_to_keep % 2)
        keep_back = chars_to_keep // 2

        if keep_back == 0:
            return stem[:keep_front] + "..." + ext

        return stem[:keep_front] + "..." + stem[-keep_back:] + ext

    @staticmethod
    def col_name_to_tl_type(col_name: str) -> str:
        # first find the part that starts with 'mt_'
        if col_name.startswith("no_temp_method"):
            return "no_skip"
        elif col_name.startswith("temp_method"):
            return "skip"
        else:
            raise ValueError(f"Cannot determine timeline type for col_name={col_name}")

    @staticmethod
    def _find_baseline_dir(exp_path: Path, ds_name: str) -> Path | None:
        """Heuristic to find the latest 'no_temp' sibling directory."""
        if GlobalConst.NOTEMP_MT_PATTERN in exp_path.name:
            return None  # Current is already no_temp_method

        candidates = []
        if exp_path.parent.exists():
            candidates = [
                p
                for p in exp_path.parent.iterdir()
                if p.is_dir()
                and p != exp_path
                and GlobalConst.NOTEMP_MT_PATTERN in p.name
                and f"ds_{ds_name}" in p.name
            ]
        return max(candidates, key=lambda p: p.name) if candidates else None

    @staticmethod
    def get_tl_df_by_exp_dir(
        exp_dir: str,
        include_baseline: bool = False,
        exp_name_shorten_func: Callable[[str], str] = shorten_exp_name,
    ) -> tuple[Dict[str, Dict], Path]:
        """
        Prepares CSV info and dataset path for a SINGLE experiment.
        Returns a dictionary of CSV load info and the dataset directory path.
        """
        exp_path = Path(exp_dir)
        if not exp_path.exists():
            raise FileNotFoundError(f"Experiment directory not found: {exp_path}")

        # 1. Load Config & Dataset Dir
        config_file = exp_path / "__config.yaml"
        if not config_file.exists():
            raise FileNotFoundError(f"Config file missing for exp: {config_file}")

        cfg_data = yamlfile.load_yaml(str(config_file), to_dict=True)
        exp_cfg = Config.from_custom_yaml_file_or_str(cfg_data.get("original-yaml-str"))
        dataset_dir = Path(exp_cfg.dbsetCfg.dir_path)  # type: ignore

        # 2. Check for Baseline if requested
        dirs_to_process = [exp_path]
        if include_baseline:
            dataset_name = exp_cfg.dbsetCfg.name
            bl_dir = TlReportGen._find_baseline_dir(
                exp_path,
                dataset_name,  # type: ignore
            )
            if bl_dir:
                dirs_to_process.insert(0, bl_dir)

        # 3. Build Info Dict for this experiment (and optional baseline)
        tl_info = {}

        for d_path in dirs_to_process:
            short_name = exp_name_shorten_func(d_path.name)

            # Determine Timeline Type
            if GlobalConst.NOTEMP_MT_PATTERN in d_path.name:
                t_type = GlobalConst.TL_TYPE_NO_SKIP
            else:
                t_type = GlobalConst.TL_TYPE_SKIP

            tl_info[short_name] = {
                "csv_pattern": GlobalConst.INFER_FILE_PATTERN,
                "is_gt": False,
                "csv_dir": str(d_path),
                "tl_type": t_type,
            }

        return tl_info, dataset_dir

    @staticmethod
    def get_df_by_exp_dirs(
        exp_dirs: List[str],
        exp_name_shorten_func: Callable[[str], str] = shorten_exp_name,
        exp_name_to_tltype: Callable[[str], str] = col_name_to_tl_type,
    ) -> tuple[pd.DataFrame, Dict[str, str]]:
        """
        Loads GT, and multiple Experiment data into a single frame-level DataFrame.
        Supports passing multiple experiment directories.

        Args:
            exp_dirs: List of experiment directory paths.
        """

        if not exp_dirs:
            raise ValueError("exp_dirs list cannot be empty.")

        all_exp_info = OrderedDict()
        dataset_dir_ref = None
        seen_short_names = set()
        # pprint(f"[INFO] Loading timeline data for experiments:")
        # pprint(f"   {exp_dirs}")

        # 1. Collect Info for ALL experiments
        for exp_dir in exp_dirs:
            try:
                # We don't auto-include baseline here because the user passes a list explicitly.
                curr_info, curr_ds_dir = TlReportGen.get_tl_df_by_exp_dir(
                    exp_dir,
                    include_baseline=False,
                    exp_name_shorten_func=exp_name_shorten_func,
                )

                # Verify Dataset Consistency
                if dataset_dir_ref is None:
                    dataset_dir_ref = curr_ds_dir
                elif curr_ds_dir != dataset_dir_ref:
                    assert False, (
                        f"All experiments must use the same dataset, but found mismatch: {dataset_dir_ref} vs {curr_ds_dir} for exp {exp_dir}"
                    )

                # Merge Info, handling duplicate names
                for name, info in curr_info.items():
                    final_name = name
                    idx = 1
                    while final_name in seen_short_names:
                        final_name = f"{name}_{idx}"
                        idx += 1
                    seen_short_names.add(final_name)
                    all_exp_info[final_name] = info

            except Exception as e:
                print(f"[Warning] Skipping {exp_dir}: {e}")
                continue

        if not all_exp_info or dataset_dir_ref is None:
            raise ValueError(
                f"No valid experiments found or dataset directory could not be determined. {all_exp_info=} - {dataset_dir_ref=}"
            )

        # 2. GT Info (Always present)
        gt_info = {
            "csv_pattern": GlobalConst.GT_FILE_PATTERN,
            "is_gt": True,
            "csv_dir": None,
            "tl_type": GlobalConst.TL_TYPE_GT,
        }

        # Prepare Metadata for return
        timeline_types_by_col = {GlobalConst.COL_GT: GlobalConst.TL_TYPE_GT}
        for k, v in all_exp_info.items():
            timeline_types_by_col[k] = v["tl_type"]

        # 3. Video files lookup
        video_files = fs.filter_files_by_extension(
            str(dataset_dir_ref), [".mp4", ".avi", ".mov", ".mkv"], recursive=True
        )

        # 4. Merging Logic
        def combine_df_single_video(vid_path: str) -> pd.DataFrame:
            # Load GT
            gt_df = BaseRawCsvLoader.load_csv_by_pattern(
                video_path=vid_path,
                csv_pattern=gt_info["csv_pattern"],  # ty:ignore[invalid-argument-type]
                is_gt=gt_info["is_gt"],  # ty:ignore[invalid-argument-type]
                csv_dir=gt_info["csv_dir"],  # ty:ignore[invalid-argument-type]
            )

            # Load and Merge Each Experiment
            for load_info_key, load_info in all_exp_info.items():
                pred_df = BaseRawCsvLoader.load_csv_by_pattern(
                    video_path=vid_path,
                    csv_pattern=load_info["csv_pattern"],
                    is_gt=load_info["is_gt"],
                    csv_dir=load_info["csv_dir"],
                )

                # Check if pred_df is empty (e.g. inference failed for this video)
                if pred_df.empty:
                    # Add column with NaNs if missing
                    gt_df[load_info_key] = np.nan
                    continue

                # Rename pred col to experiment name
                pred_df[load_info_key] = pred_df[GlobalConst.COL_PRED]
                pred_df = pred_df[BaseRawCsvLoader.RAW_FIXED_COLS + [load_info_key]]

                # Merge onto GT
                gt_df = BaseRawCsvLoader._merge_gt_pred_dfs(
                    gt_df, pred_df, vid_path, do_verify=False
                )

            gt_df[GlobalConst.COL_NUM_FRAMES] = len(gt_df)
            return gt_df

        # 5. Process All Videos
        ls_dfs = []
        for vid_path in video_files:
            single_video_df = combine_df_single_video(vid_path)
            ls_dfs.append(single_video_df)

        if not ls_dfs:
            return pd.DataFrame(), {}

        all_video_df = pd.concat(ls_dfs, ignore_index=True)

        # Re-order columns: Fixed -> Others
        all_video_df = all_video_df[
            TlReportGen.TL_FIXED_COLUMNS
            + [
                col
                for col in all_video_df.columns
                if col not in TlReportGen.TL_FIXED_COLUMNS
            ]
        ]
        return all_video_df, timeline_types_by_col

    @staticmethod
    def get_unique_values(df: pd.DataFrame) -> Dict[str, list]:
        """
        Get unique values for each column in the dataframe.
        """
        assert all(col in df.columns for col in TlReportGen.TL_FIXED_COLUMNS), (
            f"This function only supports dataframes with fixed columns: {TlReportGen.TL_FIXED_COLUMNS} - for timeline dataframe"
        )
        unique_by_cols = {}
        for col in df.columns:
            if col in TlReportGen.TL_FIXED_COLUMNS[:-1]:  # skip 'video' and 'frame_id'
                continue
            unique_by_cols[col] = list(df[col].unique())
        return unique_by_cols

    @staticmethod
    def tlReport_from_csv(
        csv_path: str,
        output_html_path: Optional[str] = None,
        title: str = "Timeline Report (Reconstructed)",
    ):
        """
        Reconstructs the HTML report from a saved CSV file (report_df).
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path, header=[0, 1], sep=";", encoding="utf-8")

        cols_to_drop = [
            c
            for c in df.columns
            if "video_path" in str(c[1]).lower() or "video_path" in str(c[0]).lower()
        ]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)

        cols_to_types = {}
        level0_cols = df.columns.get_level_values(0).unique()

        for col in level0_cols:
            col = str(col).strip()
            if not col or col.lower() == "nan" or "unnamed" in col.lower():
                continue

            if col == "gt_label":
                cols_to_types[col] = "gt"
                continue

            try:
                t_type = TlReportGen.col_name_to_tl_type(col)
                cols_to_types[col] = t_type
            except ValueError:
                pass

        styles_map = {}
        for col, t_type in cols_to_types.items():
            styles_map[col] = TlConfig.get_tl_dict(t_type)
        if output_html_path is None:
            output_html_path = csv_path.replace(".csv", "_reconstructed.html")

        gen = TlReportGen(cols_to_types)
        gen.render_html(df, styles_map, output_html_path, title)

        with ConsoleLog("Reconstructed Report"):
            print(f"[INFO] Report generated at: ⏬")
            pprint_local_path(output_html_path, get_wins_path=True)

    @staticmethod
    def gen_TlReport_muti_exps(
        parent_dir: str = "./zout/zruns",
        table_mode: Literal["p", "fc", "pfc"] = "p",
        table_decimals: int = 2,
        video_name_limit: int = 0,
    ):
        """
        Generate timeline reports for all experiment directories under the given parent directory.
        """
        parent_path = Path(parent_dir)
        exp_dirs = [p for p in parent_path.iterdir() if p.is_dir()]
        for exp_dir in exp_dirs:
            config_file = exp_dir / "__config.yaml"
            if not config_file.exists():
                print(f"[Warning] Skipping {exp_dir.name}: Invalid exp dir - Missing __config.yaml")
                continue
            pprint(f"[INFO] Gen timeline report for exp: {exp_dir.name}")
            try:
                TlReportGen.gen_TlReport_exp(
                    exp_dir=exp_dir,
                    title=f"Timeline Report - {exp_dir.name}",
                    table_mode=table_mode,
                    table_decimals=table_decimals,
                    video_name_limit=video_name_limit,
                )
            except Exception as e:
                with ConsoleLog("Error Generating Report"):
                    pprint(exp_dir.name)
                    pprint(f"   >>>[ERROR] {e}")

    @staticmethod
    def gen_TlReport_exp(
        exp_dir: Union[Path, str],
        title: Optional[str] = None,
        table_mode: Literal["p", "fc", "pfc"] = "p",
        table_decimals: int = 2,
        include_baseline: bool = True,
        video_name_limit: int = 0,
    ):
        """
        Generates a report for a SINGLE experiment (backward compatibility wrapper).
        """
        exp_dir = Path(exp_dir)
        exp_dirs = [str(exp_dir)]

        if include_baseline:
            try:
                config_file = exp_dir / "__config.yaml"
                if config_file.exists():
                    cfg_data = yamlfile.load_yaml(str(config_file), to_dict=True)
                    exp_cfg = Config.from_custom_yaml_file_or_str(
                        cfg_data.get("original-yaml-str")
                    )
                    dataset_name = exp_cfg.dbsetCfg.name

                    bl_noskip_dir = TlReportGen._find_baseline_dir(
                        exp_dir,
                        dataset_name,  # ty:ignore[invalid-argument-type]
                    )
                    if bl_noskip_dir and str(bl_noskip_dir) not in exp_dirs:
                        exp_dirs.insert(0, str(bl_noskip_dir))
            except Exception as e:
                print(f"[Warning] Failed to auto-discover baseline: {e}")

        # Use the new multi-exp loader with a single item list
        df, cols_to_types = TlReportGen.get_df_by_exp_dirs(exp_dirs)
        # !debug
        # # save df, cols_to_types for debugging
        # df.to_csv("./zout/debug.csv", sep=";", index=False, encoding="utf-8")
        # print(10*"-")
        # pprint(cols_to_types)

        report_generator = TlReportGen(cols_to_types)
        output_path = exp_dir / f"{GlobalConst.PERF_FILE_PREFIX}timeline_report.html"
        if title is None:
            title = f"Timeline Report - {exp_dir.name}"

        report_generator._generate(
            df,
            str(output_path),
            title=title,
            table_mode=table_mode,
            table_decimals=table_decimals,
            video_name_limit=video_name_limit,
        )
        return os.path.abspath(output_path)

    @staticmethod
    def gen_TlReport_compare(
        exp_dirs: List[str],
        output_path: str,
        title: str = "Comparison Timeline Report",
        table_mode: Literal["p", "fc", "pfc"] = "p",
        table_decimals: int = 2,
        video_name_limit: int = 0,
    ):
        """
        Generates a comparison report for MULTIPLE experiments.
        """
        df, cols_to_types = TlReportGen.get_df_by_exp_dirs(exp_dirs)

        report_generator = TlReportGen(cols_to_types)
        report_generator._generate(
            df,
            output_path,
            title=title,
            table_mode=table_mode,
            table_decimals=table_decimals,
            video_name_limit=video_name_limit,
        )
        return os.path.abspath(output_path)

    def _generate(
        self,
        df: pd.DataFrame,
        output_path: str,
        title: str = "Timeline Report",
        table_mode: Literal["p", "fc", "pfc"] = "p",
        table_decimals: int = 2,
        sort_func_tlreport_df: Optional[
            Callable[[pd.DataFrame], pd.DataFrame]
        ] = default_sort_func_tlreport,
        video_name_limit: int = 0,
        debug=True
    ):
        """
        Main entry point: Process dataframe and generate HTML report.
        raw -> proc -> final report df -> HTML
        """
        # 1. Process Data using the Logic Core
        raw_df = df.copy()
        proc_tl_df, stats_df, styles_map = TlProcessor.proc_dataframe(
            df, self.cols_to_types, table_mode=table_mode, table_decimals=table_decimals
        )
        # !debug
        # console.rule("Debug Info: Stats DataFrame and Styles Map")
        # stats_df.to_csv(os.path.join(os.path.dirname(output_path), "stats_debug.csv"), index=True, sep=";", encoding="utf-8")
        # pprint(styles_map)

        # assert False, "Debug"
        # !end debug

        # 2. Add "FRAMES" and "VISUALIZATION"
        final_df_reset = proc_tl_df.reset_index()
        video_groups = {
            vid: grp.sort_values(by=GlobalConst.COL_FRAME_IDX)
            for vid, grp in final_df_reset.groupby(GlobalConst.COL_VIDEO)
        }

        frames_list = []
        viz_list = []
        path_list = []

        for video_name in stats_df.index:
            if video_name in video_groups:
                vid_df = video_groups[video_name]
                frames = len(vid_df)
                viz = self.generate_timeline_html(vid_df, styles_map)
                path = (
                    str(vid_df[GlobalConst.COL_VIDEO_PATH].iloc[0])
                    if GlobalConst.COL_VIDEO_PATH in vid_df.columns
                    else ""
                )
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
        cols_to_keep = []
        for col in stats_df.columns:
            method_key = col[0]
            if styles_map.get(method_key, {}).get("table", {}).get("include", True):
                cols_to_keep.append(col)

        report_df = stats_df[cols_to_keep].copy()

        report_df[(" ", "FRAMES")] = frames_list
        report_df[(" ", "VISUALIZATION")] = viz_list
        report_df[(" ", "VIDEO NAME")] = [
            TlReportGen._truncate_video_name(str(v), video_name_limit)
            for v in report_df.index
        ]
        report_df[(" ", "VIDEO_PATH")] = path_list

        report_df.reset_index(drop=True, inplace=True)

        if not isinstance(report_df.columns, pd.MultiIndex):
            report_df.columns = pd.MultiIndex.from_tuples(report_df.columns)

        # 4. Reorder Columns
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

        report_df = report_df[final_cols]

        if sort_func_tlreport_df:
            report_df = sort_func_tlreport_df(report_df)

        # 3. Render HTML
        self.render_html(report_df, styles_map, output_path, title)
        if debug:
            # save raw timeline csv and report csv
            raw_timeline_csv_path = output_path.replace(".html", "_raw.csv")
            raw_df.to_csv(raw_timeline_csv_path, index=False, sep=";", encoding="utf-8")
            proc_timeline_csv_path = output_path.replace(".html", "_proc.csv")
            proc_tl_df.to_csv(
                proc_timeline_csv_path, index=False, sep=";", encoding="utf-8"
            )
            report_csv_output_path = output_path.replace(".html", ".csv")
            report_df.to_csv(report_csv_output_path, index=False, sep=";", encoding="utf-8")

            with ConsoleLog("Saving report results:"):
                print(f"[INFO] Report generated at: ⏬")
                pprint_local_path(output_path, get_wins_path=True)

                print(f"[INFO] Raw timeline data saved at: ⏬")
                pprint_local_path(raw_timeline_csv_path, get_wins_path=True)

                print(f"[INFO] Processed timeline data saved at: ⏬")
                pprint_local_path(proc_timeline_csv_path, get_wins_path=True)

                print(f"[INFO] CSV version saved at: ⏬")
                pprint_local_path(report_csv_output_path, get_wins_path=True)

    def generate_timeline_html(self, vid_df: pd.DataFrame, styles_map: Dict) -> str:
        """Generates the stacked bar HTML for a single video."""
        rows_html = ""

        # Maintain order from config
        for col_name, type_key in self.cols_to_types.items():
            if col_name not in vid_df.columns:
                continue

            labels = vid_df[col_name].values

            color_map = TlConfig.get_labels_color_map(type_key)

            def get_color(label, _color_map=color_map):
                assert label in _color_map, f"'{label=}' not found in {_color_map=} for column '{col_name}'"
                return _color_map[label]

            colors = [get_color(lb) for lb in labels]
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
                            ),
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

        def highlight_cells(s):
            if not isinstance(s.name, tuple):
                return ["" for _ in s]

            method_key, outcome = s.name

            if method_key.strip() == "":
                return ["" for _ in s]

            method_cfg = styles_map.get(method_key)
            if not method_cfg:
                return ["" for _ in s]

            rules = method_cfg.get("table", {}).get("highlight_rules", {})
            rule = rules.get(outcome)

            if not rule:
                return ["" for _ in s]

            condition = rule.get("condition", "")
            color = rule.get("color", "red")

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

        for col_name, type_key in self.cols_to_types.items():
            style_cfg = styles_map.get(col_name, {})
            title = style_cfg.get("meta", {}).get("legend_title") or style_cfg.get(
                "legend_title", col_name
            )
            labels = TlConfig.get_labels(type_key)

            html += (
                f'<div class="legend-section"><div class="legend-title">{title}</div>'
            )

            for label, val in labels.items():
                if isinstance(val, dict):
                    color = val.get("color", "#ccc")
                    note = val.get("additional_note")
                else:
                    # legacy bare hex string
                    color = val
                    note = None

                note_html = ""
                if note:
                    note_html = f' <span style="color:#888; font-style:italic;">— {note}</span>'

                html += f'<div class="legend-item"><span class="dot" style="background:{color}"></span><span><b>{label}</b>{note_html}</span></div>'
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
