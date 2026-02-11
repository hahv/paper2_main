import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Type, Optional, Literal, Any
from dataclasses import dataclass
from halib import *
from halib.filetype import yamlfile
from common import GlobalConst


# ======================================================
# 1. Timeline Configuration Management
# ======================================================
@dataclass
class TlConfig:
    """Singleton-like config manager."""

    _config: Optional[Dict] = None
    DEFAULT_TIMELINE_CFG = f"{GlobalConst.proj_root()}/config/mics/timeline_cfg.yaml"

    @classmethod
    def load(cls, path: Optional[str] = None) -> Dict:
        if cls._config is None:
            load_path = path or cls.DEFAULT_TIMELINE_CFG
            cls._config = yamlfile.load_yaml(load_path, to_dict=True)
        return cls._config

    @classmethod
    def get_supported_tltypes(cls) -> List[str]:
        # Returns keys like ['gt', 'no_skip', 'skip']
        return list(cls.load().keys())

    @classmethod
    def get_tl_dict(cls, tl_type: str) -> Dict[str, Any]:
        cfg = cls.load()
        assert tl_type in cfg, f"Timeline type '{tl_type}' not found in config."
        return cfg[tl_type]


# ======================================================
# 2. Parsing Logic (Decoupled)
# ======================================================
class TLConverter(ABC):
    """
    Base class for parsing logic.
    'tl_type' determines which color schema to validate against.
    """

    def __init__(self, tl_type: str):
        self.tl_type = tl_type

    @property
    def supported_labels(self) -> List[str]:
        tl_cfg_dict = TlConfig.get_tl_dict(self.tl_type)
        # Search for labels_colors in "timeline" subsection (new format) or root (old format)
        if (
            "timeline" in tl_cfg_dict
            and "labels_colors" in tl_cfg_dict["timeline"]
        ):
            return list(tl_cfg_dict["timeline"]["labels_colors"].keys())

        assert "labels_colors" in tl_cfg_dict, (
            f"Config for '{self.tl_type}' missing 'labels_colors' key (checked root and 'timeline')."
        )
        return list(tl_cfg_dict["labels_colors"].keys())

    def validate_output(self, labels: np.ndarray):
        """Ensures generated labels exist in the config for this type."""
        unique_labels = np.unique(labels)
        valid_set = set(self.supported_labels)

        if not valid_set:
            return

        for label in unique_labels:
            if label not in valid_set:
                raise ValueError(
                    f"[{self.tl_type}] Parser produced label '{label}' "
                    f"which is not in config. Allowed: {valid_set}"
                )

    @abstractmethod
    def parse_logic(self, df: pd.DataFrame, method_col: str) -> np.ndarray:
        """Pure logic implementation."""
        pass

    def run(self, df: pd.DataFrame, method_col: str) -> pd.Series:
        """
        Orchestrates parsing:
        1. Checks column existence
        2. runs logic
        3. validates output against config
        """
        if method_col not in df.columns:
            raise ValueError(f"Column '{method_col}' not found in DataFrame.")

        raw_labels = self.parse_logic(df, method_col)
        self.validate_output(raw_labels)

        # Return Series with original index
        return pd.Series(raw_labels, index=df.index, name=method_col)


# --- IMPLEMENATIONS ---
# CSV label Input (gt or pred):
#         {
#         │   'gt_label': ['fire_smoke', 'none'],
#         │   'no_temp_method': ['fire', 'smokeonly', 'none'],
#         │   'temp_method_motion_block': ['skipped', 'fire', 'smokeonly', 'none']
#         }
# !label Output (normalized):
#         {
#         │   'gt_label': ['fire', 'none'],
#         │   'no_temp_method': ['fire', 'none'],
#         │   'temp_method_motion_block': ['skipped', 'fire', 'none']
#         }


class TLGtConverter(TLConverter):
    def parse_logic(self, df: pd.DataFrame, method_col: str) -> np.ndarray:
        return np.where(df["gt_label"] == "fire", "FireSmoke", "None")


class NoSkipConverter(TLConverter):
    def parse_logic(self, df: pd.DataFrame, method_col: str) -> np.ndarray:
        is_gt_fire = df["gt_label"] == "fire"
        is_pred_fire = df[method_col] == "fire"
        return np.select(
            [(~is_gt_fire) & (is_pred_fire), (is_gt_fire) & (~is_pred_fire)],
            ["False Alarm (FP)", "Miss (FN)"],
            default="Correct",
        )


class SkipConverter(TLConverter):
    def parse_logic(self, df: pd.DataFrame, method_col: str) -> np.ndarray:
        is_gt_fire = df["gt_label"] == "fire"
        is_skipped = df[method_col] == "skipped"

        # Logic for Temporal Skipping Efficiency:
        # We evaluate whether the decision to SKIP or PROCESS was correct relative to the GT.
        # - Miss (FN): Dangerous. Fire existed but we skipped the frame.
        # - Waste (FP): Inefficient. No fire existed but we wasted resources processing it.
        # - Correct Proc.: Good. Fire existed and we correctly decided to process it.
        # - Correct Skip: Good. No fire existed and we correctly skipped it.

        return np.select(
            [
                is_gt_fire & is_skipped,  # GT=Fire, Action=Skip
                (~is_gt_fire) & (~is_skipped),  # GT=None, Action=Process
                is_gt_fire & (~is_skipped),  # GT=Fire, Action=Process
                (~is_gt_fire) & is_skipped,  # GT=None, Action=Skip
            ],
            ["Miss (FN)", "Waste (FP)", "Correct Proc.", "Correct Skip"],
            default="Unknown",
        )


# ======================================================
# 3. Factory
# ======================================================
class TLParserFactory:
    _REGISTRY: Dict[str, Type[TLConverter]] = {
        "gt": TLGtConverter,
        "no_skip": NoSkipConverter,
        "skip": SkipConverter,
    }

    @classmethod
    def create(cls, parser_type: str) -> TLConverter:
        parser_cls = cls._REGISTRY.get(parser_type)
        if not parser_cls:
            raise NotImplementedError(
                f"Logic type '{parser_type}' not found in registry. "
                f"Available: {list(cls._REGISTRY.keys())}"
            )
        return parser_cls(timeline_type=parser_type)


# ======================================================
# 4. Processor (The Driver)
# ======================================================
class TlProcessor:
    FIXED_COLS = ["video", "video_path", "frame_id", "gt_label"]

    @classmethod
    def proc_dataframe(
        cls,
        df: pd.DataFrame,
        cols_to_timeline_types: Dict[str, str],
        table_mode: Literal["p", "fc", "pfc"] = "pfc",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Generates the frame-level timeline data.
        """
        # 1. Validate Metadata
        missing_fixed = [c for c in cls.FIXED_COLS if c not in df.columns]
        if missing_fixed:
            raise ValueError(
                f"Input DataFrame is missing required fixed columns: {missing_fixed}"
            )

        parsed_series_list = []
        styles_map = {}

        for col_name, timeline_type in cols_to_timeline_types.items():
            if col_name not in df.columns:
                print(f"[Error] Configured column '{col_name}' not found. Skipping.")
                continue

            try:
                parser = TLParserFactory.create(timeline_type)
                # Parse and force name to match column
                parsed_series = parser.run(df, col_name)
                parsed_series.name = col_name

                parsed_series_list.append(parsed_series)
                styles_map[col_name] = TlConfig.get_tl_dict(timeline_type)

            except Exception as e:
                print(f"[Exception] Failed processing column '{col_name}': {e}")
                continue

        # 2. Merge Results (Fixing Duplicate Columns Issue)
        base_df = df[cls.FIXED_COLS].copy()

        if parsed_series_list:
            parsed_results = pd.concat(parsed_series_list, axis=1)

            # Identify columns in parsed_results that are ALSO in base_df (e.g., 'gt_label')
            # We must drop them from base_df to avoid duplicates (overwriting raw with parsed)
            cols_to_overwrite = [
                c for c in parsed_results.columns if c in base_df.columns
            ]
            if cols_to_overwrite:
                base_df = base_df.drop(columns=cols_to_overwrite)

            final_df = pd.concat([base_df, parsed_results], axis=1)
        else:
            final_df = base_df

        # 3. Set Index (Safe now that columns are unique)
        final_df.set_index(cls.FIXED_COLS, inplace=True)

        # 4. Compute Stats
        stats_df = cls.compute_stats_df(final_df.copy(), styles_map, mode=table_mode)
        return final_df, stats_df, styles_map

    @classmethod
    def compute_stats_df(
        cls,
        processed_df: pd.DataFrame,
        styles_map: Dict[str, Dict],
        mode: Literal["p", "fc", "pfc"] = "p",
    ) -> pd.DataFrame:
        """
        Generates the Summary Pivot Table with TOTAL row at the top.
        """
        # Ensure we are working with a flat dataframe for easy groupby
        df_flat = processed_df.reset_index()

        summary_tables = []

        # Iterate through each processed method column
        for method_col, style_cfg in styles_map.items():
            # 1. Get ordered list of expected labels from config
            # Handle new nested config structure
            labels_source = style_cfg.get("timeline", {}).get(
                "labels_colors"
            ) or style_cfg.get("labels_colors", {})
            expected_labels = list(labels_source.keys())

            # 2. Calculate Counts per Video
            counts_df = pd.crosstab(index=df_flat["video"], columns=df_flat[method_col])

            # 3. Add "TOTAL" Row at the TOP
            total_row = counts_df.sum(axis=0)
            total_row.name = "TOTAL"

            # Concatenate TOTAL first, then the rest
            counts_df = pd.concat([total_row.to_frame().T, counts_df])

            # 4. Reindex columns to ensure ALL configured labels exist (fill 0 if missing)
            counts_df = counts_df.reindex(columns=expected_labels, fill_value=0)

            # Force integer type for clean display (avoids '5.0' counts)
            counts_df = counts_df.astype(int)

            # 5. Calculate Percentages
            row_sums = counts_df.sum(axis=1)
            pct_df = counts_df.div(row_sums.replace(0, 1), axis=0) * 100

            # 6. Format Output Strings based on Mode
            formatted_df = counts_df.copy().astype(object)

            for col in counts_df.columns:
                if mode == "p":
                    formatted_df[col] = pct_df[col].map("{:.1f}%".format)
                elif mode == "fc":
                    formatted_df[col] = counts_df[col].astype(str)
                elif mode == "pfc":
                    formatted_df[col] = (
                        pct_df[col].map("{:.1f}%".format)
                        + " ("
                        + counts_df[col].astype(str)
                        + ")"
                    )

            # 7. Add Top-Level MultiIndex for the Method Name
            formatted_df.columns = pd.MultiIndex.from_product(
                [[method_col], formatted_df.columns], names=["Method", "Outcome"]
            )

            summary_tables.append(formatted_df)

        if not summary_tables:
            return pd.DataFrame()

        # Concatenate all method tables horizontally
        final_stats_df = pd.concat(summary_tables, axis=1)

        return final_stats_df
