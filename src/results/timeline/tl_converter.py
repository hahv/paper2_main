from sympy.physics.quantum.tests.test_qapply import po
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
    DEFAULT_TIMELINE_CFG = f"{GlobalConst.proj_root()}/config/misc/timeline_cfg.yaml"

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

    @classmethod
    def get_labels(cls, tl_type: str) -> Dict[str, Any]:
        """Returns the labels dict: {label_name: {color, additional_note, ...}}"""
        tl_cfg = cls.get_tl_dict(tl_type)
        tl_section = tl_cfg.get("timeline", {})
        if "labels" in tl_section:
            return tl_section["labels"]
        # Fallback: old flat labels_colors format (bare hex strings)
        if "labels_colors" in tl_section:
            return {k: {"color": v} for k, v in tl_section["labels_colors"].items()}
        if "labels_colors" in tl_cfg:
            return {k: {"color": v} for k, v in tl_cfg["labels_colors"].items()}
        raise KeyError(
            f"Config for '{tl_type}' has no 'labels' (or legacy 'labels_colors') key."
        )

    @classmethod
    def get_labels_color_map(cls, tl_type: str) -> Dict[str, str]:
        """Returns a flat {label_name: hex_color} dict, for bar rendering."""
        return {k: v["color"] for k, v in cls.get_labels(tl_type).items()}


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
        return list(TlConfig.get_labels(self.tl_type).keys())

    @property
    def pos_labels(self) -> List[str]:
        """Defines which labels are considered 'positive' for metrics calculations."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement 'pos_labels' property."
        )

    @property
    def neg_labels(self) -> List[str]:
        assert self.valid_out_lbs is not None, (
            "valid_out_lbs must be defined to determine negative labels."
        )
        assert self.pos_labels is not None, (
            "positive_labels must be defined to determine negative labels."
        )

        return [lb for lb in self.valid_out_lbs if lb not in self.pos_labels]


class TLGtConverter(TLConverter):
    @property
    def valid_in_lbs(self) -> Optional[List[str]]:
        """Validate input labels before conversion."""
        return [GlobalConst.FIRESMOKE_LABEL, GlobalConst.NONE_LABEL]

    @property
    def pos_labels(self) -> List[str]:
        return [GlobalConst.TL_GT_FIRESMOKE]

    def convert_col(
        self, df: pd.DataFrame, target_col: str, extra_dict: Optional[dict] = None
    ) -> np.ndarray:
        rs = np.where(
            df[GlobalConst.COL_GT] == GlobalConst.FIRESMOKE_LABEL,
            GlobalConst.TL_GT_FIRESMOKE,
            GlobalConst.TL_GT_NONE,
        )
        return rs


class NoSkipConverter(TLGtConverter):
    @property
    def valid_out_lbs(self) -> Optional[List[str]]:
        return [
            GlobalConst.TL_NOSKIP_CORRECT_POS,
            GlobalConst.TL_NOSKIP_CORRECT_NEG,
            GlobalConst.TL_NOSKIP_FALSE_ALARM_FP,
            GlobalConst.TL_NOSKIP_MISS_FN,
        ]

    @property
    def pos_labels(self) -> List[str]:
        return [GlobalConst.TL_NOSKIP_CORRECT_POS, GlobalConst.TL_NOSKIP_MISS_FN]

    def convert_col(
        self, df: pd.DataFrame, target_col: str, extra_dict: Optional[dict] = None
    ) -> np.ndarray:
        is_gt_fire = df[GlobalConst.COL_GT] == GlobalConst.FIRESMOKE_LABEL
        is_pred_fire = df[target_col] == GlobalConst.FIRESMOKE_LABEL
        rs = np.select(
            [
                (is_gt_fire) & (is_pred_fire),  # GT=Fire, Pred=Fire
                (~is_gt_fire) & (~is_pred_fire),  # GT=None, Pred=None
                (~is_gt_fire) & (is_pred_fire),  # GT=None, Pred=Fire
                (is_gt_fire) & (~is_pred_fire),  # GT=Fire, Pred=None
            ],
            [
                GlobalConst.TL_NOSKIP_CORRECT_POS,  # TP
                GlobalConst.TL_NOSKIP_CORRECT_NEG,  # TN
                GlobalConst.TL_NOSKIP_FALSE_ALARM_FP,  # FP
                GlobalConst.TL_NOSKIP_MISS_FN,  # FN
            ],
            default="Unknown",
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

    @property
    def valid_out_lbs(self) -> Optional[List[str]]:
        return [
            GlobalConst.TL_SKIP_CORRECT_INFER,
            GlobalConst.TL_SKIP_CORRECT_SKIP,
            GlobalConst.TL_SKIP_FALSE_SKIP,
            GlobalConst.TL_SKIP_WASTED_INFER,
        ]

    @property
    def pos_labels(self) -> List[str]:
        return [
            GlobalConst.TL_SKIP_CORRECT_INFER,
            GlobalConst.TL_SKIP_FALSE_SKIP,
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
                (is_gt_fire) & (~is_skipped),  # GT=Fire, Processed
                (~is_gt_fire) & (is_skipped),  # GT=None, Skipped
                (is_gt_fire) & (is_skipped),  # GT=Fire, Skipped
                (~is_gt_fire) & (~is_skipped),  # GT=None, Processed
            ],
            [
                GlobalConst.TL_SKIP_CORRECT_INFER,
                GlobalConst.TL_SKIP_CORRECT_SKIP,
                GlobalConst.TL_SKIP_FALSE_SKIP,
                GlobalConst.TL_SKIP_WASTED_INFER,
            ],
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
    #         │   'gt_label':
    #                   ['fire_smoke', 'none', 'fire_smoke'],
    #         │   'no_temp_method':
    #                   ['fire', 'smokeonly', 'none'],
    #         │   'temp_method_motion_block':
    #                   ['skipped', 'fire', 'smokeonly']
    #         }
    # ! first input will be normalized to standard GlobalConst labels = [FIRESMOKE_LABEL, NONE_LABEL, SKIP_LABEL] (if applicable), then converted to timeline labels based on config mapping
    # !Example of label Output:
    #         {
    #         │   'gt_label':
    #                   ['FireSmoke', 'None', 'FireSmoke'],
    #         │   'no_temp_method':
    #                   ['Correct', 'False Alarm (FP)', 'Miss (FN)'],
    #         │   'temp_method_motion_block':
    #                   ['Miss (FN)', 'Waste (FP)', 'Correct Proc.']
    #         }
    # ======================================================
    @classmethod
    def proc_dataframe(
        cls,
        df: pd.DataFrame,
        cols_to_tl_types: Dict[str, str],
        table_mode: Literal["p", "fc", "pfc"] = "pfc",
        table_decimals: int = 2,
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
        styles_tuple_map = {}

        # timeline cfg per column (i.e timeline type: gt, no_skip, skip)
        for col_name, tl_type in cols_to_tl_types.items():
            style = TlConfig.get_tl_dict(tl_type)
            styles_map[col_name] = style
            styles_tuple_map[col_name] = (tl_type, style)

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

        # 4. Compute Stats; also returns processed_df with combine_rules applied to frame-level labels
        stats_df, final_df = cls.compute_stats_df(
            final_df.copy(),
            styles_tuple_map,
            mode=table_mode,
            table_decimals=table_decimals,
        )
        return final_df, stats_df, styles_map

    @classmethod
    def compute_stats_df(
        cls,
        processed_df: pd.DataFrame,
        styles_tuple_map: Dict[str, tuple[str, Dict]],
        mode: Literal["p", "fc", "pfc"] = "p",
        table_decimals: int = 2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generates the Summary Pivot Table with smart denominators for error rates.
        """
        df_flat = processed_df.reset_index()
        summary_tables = []

        def _get_row_total(counts_df, method_col, label, tl_type):
            # !debug start
            # console.rule(f"_get_row_total {method_col=}, {label=},{tl_type=}")
            # !debug end

            if tl_type == GlobalConst.TL_TYPE_GT or "Correct" in label:
                # ! In this case, we want to calculate the total frames for the entire row (video) as the denominator, since GT distribution is what matters for both Fire and None, and for "Correct" we want overall accuracy.
                row_total = counts_df.sum(axis=1).replace(0, 1)
            else:
                converter = TLConverterFactory.create(tl_type)
                total_labels = (
                    converter.pos_labels
                    if label in converter.pos_labels
                    else converter.neg_labels
                )
                row_total = (
                    counts_df[counts_df.columns.intersection(total_labels)]
                    .sum(axis=1)
                    .replace(0, 1)
                )
                # !debug start
                # tl_type_pos_labels = converter.pos_labels
                # tl_type_neg_labels = converter.neg_labels
                # pprint(f"tl_type_pos_labels for {method_col}: {tl_type_pos_labels}")
                # pprint(f"tl_type_neg_labels for {method_col}: {tl_type_neg_labels}")
                # pprint(f"row_total for label '{label}' in method_col '{method_col}':")
                # pprint(row_total)
                # !debug end

            return row_total

        for method_col, style_cfg_tuple in styles_tuple_map.items():
            # 1. Get ordered list of labels
            tl_type = style_cfg_tuple[0]  # extract tl_type from the tuple
            style_cfg = style_cfg_tuple[1]
            converter = TLConverterFactory.create(tl_type)
            expected_labels = converter.valid_out_lbs
            assert expected_labels is not None, (
                f"Expected labels cannot be None for method_col '{method_col}' with tl_type '{tl_type}'"
            )

            # 2. Calculate Counts per Video and add TOTAL row
            counts_df = pd.crosstab(index=df_flat["video"], columns=df_flat[method_col])
            total_row = counts_df.sum(axis=0)
            total_row.name = "TOTAL"

            counts_df = pd.concat([total_row.to_frame().T, counts_df])
            counts_df = counts_df.reindex(columns=expected_labels, fill_value=0).astype(
                int
            )

            # 3. Apply combine_rules
            combine_rules = style_cfg.get("timeline", {}).get("combine_rules", {})
            orig_counts_df = counts_df.copy()

            if combine_rules:
                label_map: Dict[str, str] = {}
                for new_col, cols_to_combine in combine_rules.items():
                    valid_cols = [c for c in cols_to_combine if c in counts_df.columns]
                    if valid_cols:
                        counts_df[new_col] = counts_df[valid_cols].sum(axis=1)
                        counts_df.drop(columns=valid_cols, inplace=True)
                        # just move the new combined column as first column for better visibility
                        cols = [new_col] + [
                            c for c in counts_df.columns if c != new_col
                        ]
                        counts_df = counts_df[cols]
                    # Map all listed labels → new_col in the frame-level data
                    for old_label in cols_to_combine:
                        label_map[old_label] = new_col
                if label_map:
                    processed_df[method_col] = processed_df[method_col].replace(
                        label_map
                    )

            # !debug start
            # console.rule(f"count_df for method_col: {method_col}")
            # csvfile.fn_display_df(counts_df.head(5))
            # counts_df.to_csv(
            # f"./zout/test/counts_df_{method_col}.csv", sep=";", index=True
            # )
            # !debug end

            # 4. Calculate Smart Percentages and Format Output Strings
            percent_df = pd.DataFrame(index=counts_df.index, columns=counts_df.columns)
            formatted_df = pd.DataFrame(
                index=counts_df.index, columns=counts_df.columns
            )

            for col in counts_df.columns:
                row_total = _get_row_total(
                    counts_df=orig_counts_df,
                    method_col=method_col,
                    label=col,
                    tl_type=tl_type,
                )
                percent_df[col] = (counts_df[col] / row_total) * 100

                if mode == "p":
                    formatted_df[col] = percent_df[col].map(
                        f"{{:.{table_decimals}f}}%".format
                    )
                elif mode == "fc":
                    formatted_df[col] = counts_df[col].astype(str)
                elif mode == "pfc":
                    formatted_df[col] = (
                        percent_df[col].map(f"{{:.{table_decimals}f}}%".format)
                        + " ("
                        + counts_df[col].astype(str)
                        + ")"
                    )

            # !debug start
            # console.rule(f"Percentages for {method_col} with tl_type={tl_type}")
            # percent_df.to_csv(
            # f"./zout/test/percent_df_{method_col}.csv", sep=";", index=True
            # )
            # !debug end

            # 5. Add MultiIndex Header
            formatted_df.columns = pd.MultiIndex.from_product(
                [[method_col], formatted_df.columns], names=["Method", "Outcome"]
            )
            summary_tables.append(formatted_df)

        if not summary_tables:
            return pd.DataFrame(), processed_df

        return pd.concat(summary_tables, axis=1), processed_df
