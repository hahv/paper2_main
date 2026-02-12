import csv
from halib import *
from halib.filetype import yamlfile
from typing import Dict, List, Tuple, Type, Optional, Literal, Any
from dataclasses import dataclass

from src.common import GlobalConst
from src.metrics.base_csv_converter import *


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
class TLConverter(BaseCSVConverter):
    """
    Base class for parsing logic.
    'tl_type' determines which color schema to validate against.
    """

    def __init__(self, tl_type: str):
        self.tl_type = tl_type

    @property
    def valid_out_lbs(self) -> Optional[List[str]]:
        return self.tl_supported_labels

    @property
    def tl_supported_labels(self) -> List[str]:
        tl_cfg_dict = TlConfig.get_tl_dict(self.tl_type)
        # Search for labels_colors in "timeline" subsection (new format) or root (old format)
        if "timeline" in tl_cfg_dict and "labels_colors" in tl_cfg_dict["timeline"]:
            return list(tl_cfg_dict["timeline"]["labels_colors"].keys())

        assert "labels_colors" in tl_cfg_dict, (
            f"Config for '{self.tl_type}' missing 'labels_colors' key (checked root and 'timeline')."
        )
        return list(tl_cfg_dict["labels_colors"].keys())


class TLGtConverter(TLConverter):
    @property
    def valid_in_lbs(self) -> Optional[List[str]]:
        """Validate input labels before conversion."""
        return [GlobalConst.FIRESMOKE_LABEL, GlobalConst.NONE_LABEL]

    def convert_col(
        self, df: pd.DataFrame, target_col: str, extra_dict: Optional[dict] = None
    ) -> np.ndarray:
        rs = np.where(
            df[GlobalConst.COL_GT] == GlobalConst.FIRESMOKE_LABEL, "FireSmoke", "None"
        )
        return rs


class NoSkipConverter(TLGtConverter):
    def convert_col(
        self, df: pd.DataFrame, target_col: str, extra_dict: Optional[dict] = None
    ) -> np.ndarray:
        is_gt_fire = df[GlobalConst.COL_GT] == GlobalConst.FIRESMOKE_LABEL
        is_pred_fire = df[target_col] == GlobalConst.FIRESMOKE_LABEL
        rs = np.select(
            [(~is_gt_fire) & (is_pred_fire), (is_gt_fire) & (~is_pred_fire)],
            ["False Alarm (FP)", "Miss (FN)"],
            default="Correct",
        )
        return rs


class SkipConverter(TLConverter):
    @property
    def valid_in_lbs(self) -> Optional[List[str]]:
        """Validate input labels before conversion."""
        return [
            GlobalConst.FIRESMOKE_LABEL,
            GlobalConst.NONE_LABEL,
            GlobalConst.SKIP_LABEL,
        ]

    def convert_col(
        self, df: pd.DataFrame, target_col: str, extra_dict: Optional[dict] = None
    ) -> np.ndarray:
        is_gt_fire = df[GlobalConst.COL_GT] == GlobalConst.FIRESMOKE_LABEL
        is_skipped = df[target_col] == GlobalConst.SKIP_LABEL

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
class TLConverterFactory:
    _REGISTRY: Dict[str, Type[TLConverter]] = {
        "gt": TLGtConverter,
        "no_skip": NoSkipConverter,
        "skip": SkipConverter,
    }

    @classmethod
    def create(cls, tl_type: str) -> TLConverter:
        converter_cls = cls._REGISTRY.get(tl_type)
        if not converter_cls:
            raise NotImplementedError(
                f"Logic type '{tl_type}' not found in registry. "
                f"Available: {list(cls._REGISTRY.keys())}"
            )
        return converter_cls(tl_type=tl_type)


class TlProcessor:
    FIXED_COLS = [
        GlobalConst.COL_VIDEO,
        GlobalConst.COL_VIDEO_PATH,
        GlobalConst.COL_FRAME_IDX,
        GlobalConst.COL_GT,
    ]

    # ======================================================
    # 4. Processor (The Driver)
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
    # ======================================================
    @classmethod
    def proc_dataframe(
        cls,
        df: pd.DataFrame,
        cols_to_tl_types: Dict[str, str],
        table_mode: Literal["p", "fc", "pfc"] = "pfc",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Generates the frame-level timeline data. The input df header is expected to have at least the following columns:
        ! video | video_path| num_frames |frame_idx|gt_label| <<no_temp_method| temp_method_motion_block| ...>>

        """
        # 1. Validate Metadata
        missing_fixed = [c for c in cls.FIXED_COLS if c not in df.columns]
        if missing_fixed:
            raise ValueError(
                f"Input DataFrame is missing required fixed columns: {missing_fixed}"
            )

        styles_map = {}
        # timeline cfg per column (i.e timeline type: gt, no_skip, skip)
        for col_name, tl_type in cols_to_tl_types.items():
            styles_map[col_name] = TlConfig.get_tl_dict(tl_type)

        normed_df = df.copy()

        def __normalize_col(final_df, col_list: List[str]) -> pd.DataFrame:
            firemoske_converter = FireSmokeLabelConverter()
            BaseCSVConverter.do_convert_chain(
                final_df,
                [(col_name, firemoske_converter) for col_name in col_list],
                inplace=True,
            )
            return final_df

        # ! First normalize all specified columns to standard GlobalConst labels = [FIRESMOKE_LABEL, NONE_LABEL, SKIP_LABEL]
        normed_df = __normalize_col(normed_df, list(cols_to_tl_types.keys()))
        final_df = normed_df.copy()  # to avoid modifying during iteration
        for idx, (col_name, tl_type) in enumerate(cols_to_tl_types.items()):
            temp_df = normed_df.copy()
            converter = TLConverterFactory.create(tl_type)
            temp_df = BaseCSVConverter.do_convert_chain(
                temp_df,
                [(col_name, converter)],
                inplace=True,
                # context=f"TL Conversion: col='{col_name}', tl='{tl_type}'",
            )
            final_df[col_name] = temp_df[col_name]
        # ! debug
        # csvfile.fn_display_df(final_df.head(3))
        # final_df.to_csv("./zout/debug_timeline_converted.csv", index=False, sep=";")
        # 4. Compute Stats
        stats_df = cls.compute_stats_df(final_df.copy(), styles_map, mode=table_mode)
        return normed_df, stats_df, styles_map

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
