import pandas as pd
from halib import *
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from src.common import GlobalConst
from collections import OrderedDict


class ParamSelect(ABC):
    def __init__(self, df: pd.DataFrame, context: Optional[Dict[str, Any]] = None):
        self.df = df
        self.context = context
        assert self.validate_dataframe(), (
            "DataFrame validation failed. Please ensure it contains the required columns and data types."
        )

    def validate_dataframe(self):
        return True

    @abstractmethod
    def choose_params(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Implement the parameter selection logic based on the DataFrame and context.
        Return a tuple of two DataFrames (df_full, df_filtered) with the chosen parameters
        and their corresponding metrics Score."""
        pass


class WeightedSelect(ParamSelect):
    # ! Context bundles scoring weights plus baseline-derived metrics/precomputes needed by the chooser.
    def __init__(
        self,
        df: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(df, context)
        self.parse_context()
        self.get_baseline_metrics()

    def parse_context(self):
        assert self.context is not None, "Context must be provided."
        self.w_s = self.context.get("w_s", 0.70)
        self.w_r = self.context.get("w_r", 0.30)
        self.delta_r = self.context.get("delta_r", 0.01)

    def get_baseline_metrics(self):
        assert (
            GlobalConst.COL_PARAM_RECALL in self.df.columns
            and GlobalConst.COL_PARAM_FAR in self.df.columns
        ), (
            f"Baseline metrics columns not found in DataFrame. Expected '{GlobalConst.COL_PARAM_RECALL}' and '{GlobalConst.COL_PARAM_FAR}'."
        )
        assert len(self.df) > 0, "DataFrame is empty. Cannot extract baseline metrics."
        assert "mt_no_temp_method" in self.df["experiment"].iloc[0], (
            "First row of DataFrame does not contain expected baseline experiment identifier 'mt_no_temp_method'."
        )
        self.r_base = self.df[GlobalConst.COL_PARAM_RECALL].iloc[0]
        self.far_base = self.df[GlobalConst.COL_PARAM_FAR].iloc[0]

    def validate_dataframe(self):
        required_cols = {
            GlobalConst.COL_PARAM_SKIP_RATE,
            GlobalConst.COL_PARAM_RECALL,
            GlobalConst.COL_PARAM_FAR,  # Keep FAR in validation to ensure it's logged, even if not optimized
        }
        if not required_cols.issubset(self.df.columns):
            missing = required_cols - set(self.df.columns)
            print(f"Error: Missing required columns in DataFrame: {missing}")
            return False
        return True

    @staticmethod
    def _range_normalize(series: pd.Series) -> pd.Series:
        """Min-max normalize a series over its own range. Returns 0 if range is zero (all equal)."""
        lo, hi = series.min(), series.max()
        if abs(hi - lo) < 1e-12:
            return pd.Series(0.0, index=series.index)
        return (series - lo) / (hi - lo)

    def choose_params(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Implement Algorithm 1 with two-pass range-normalized scoring over Theta_feasible."""

        # Ensure weights sum to 1
        if abs(self.w_s + self.w_r - 1.0) > 1e-9:
            raise ValueError("Weights w_s and w_r must sum to 1.")

        df_baseline = self.df.iloc[[0]].copy()
        df_rest = self.df.iloc[1:].copy()

        # ==========================================
        # 1. Calculate rho for ALL Configurations
        # ==========================================
        df_rest[GlobalConst.COL_PARAM_RECALL_DROP] = (
            self.r_base - df_rest[GlobalConst.COL_PARAM_RECALL]
        )
        # rho(theta) = 1 - (R_base - R(theta)) / delta_r, clipped at 1.0
        df_rest[GlobalConst.COL_PARAM_RECALL_RET] = (
            1.0 - df_rest[GlobalConst.COL_PARAM_RECALL_DROP] / self.delta_r
        ).clip(upper=1.0)

        # ==========================================
        # 2. Pass 1 — Apply Hard Feasibility Constraint
        # ==========================================
        min_acceptable_recall = self.r_base - self.delta_r
        df_feasible = df_rest[
            df_rest[GlobalConst.COL_PARAM_RECALL] >= (min_acceptable_recall - 1e-9)
        ].copy()

        # ==========================================
        # 3. Pass 2 — Range-normalize OVER Theta_feasible, then score
        # ==========================================
        combine_col = GlobalConst.COL_PARAM_COMBINED_SCORE
        assert len(df_feasible) > 0, (
            "No feasible configurations found. Ensure that the DataFrame contains configurations that meet the feasibility criteria."
        )
        # Normalize rho and S_r over the feasible set only
        df_feasible[GlobalConst.COL_PARAM_RECALL_RET_NORM] = self._range_normalize(
            df_feasible[GlobalConst.COL_PARAM_RECALL_RET]
        )
        df_feasible[GlobalConst.COL_PARAM_SKIP_RATE_NORM] = self._range_normalize(
            df_feasible[GlobalConst.COL_PARAM_SKIP_RATE]
        )
        df_feasible[combine_col] = (
            self.w_r * df_feasible[GlobalConst.COL_PARAM_RECALL_RET_NORM]
            + self.w_s * df_feasible[GlobalConst.COL_PARAM_SKIP_RATE_NORM]
        )
        # Also compute raw (un-normalized) score on full df_rest for logging in df_full
        df_rest[GlobalConst.COL_PARAM_RECALL_RET_NORM] = float("nan")
        df_rest[GlobalConst.COL_PARAM_SKIP_RATE_NORM] = float("nan")
        df_rest[combine_col] = float("nan")
        # with ConsoleLog("df_rest"):
        #     csvfile.fn_display_df(df_rest)
        
        # with ConsoleLog("df_feasible"):
        #     csvfile.fn_display_df(df_feasible)
        
        # df_rest.update(df_feasible)
        # Backfill normalized values and scores into df_rest where feasible
        # ! fixed for df_rest.update(df_feasible) not working as expected, using
        # loc instead (error with pandas > 3.0)
        cols_to_merge = [
            GlobalConst.COL_PARAM_RECALL_RET_NORM,
            GlobalConst.COL_PARAM_SKIP_RATE_NORM,
            combine_col,
        ]
        for col in cols_to_merge:
            df_rest[col] = df_rest[col].astype("float64")
            df_feasible[col] = df_feasible[col].astype("float64")
            df_rest.loc[df_feasible.index, col] = df_feasible[col].to_numpy()

        # ==========================================
        # 4. Finalize DataFrames
        # ==========================================
        df_baseline[GlobalConst.COL_PARAM_RECALL_DROP] = 0.0
        df_baseline[GlobalConst.COL_PARAM_RECALL_RET] = 1.0
        df_baseline[GlobalConst.COL_PARAM_RECALL_RET_NORM] = float("nan")
        df_baseline[GlobalConst.COL_PARAM_SKIP_RATE_NORM] = float("nan")
        df_baseline[combine_col] = float("nan")  # Dummy — baseline is not ranked

        # df_full: all configurations, feasible ones have scores; infeasible have NaN scores
        df_full = pd.concat(
            [
                df_baseline,
                df_rest.sort_values(
                    by=combine_col, ascending=False, na_position="last"
                ),
            ],
            ignore_index=True,
        )
        # df_chosen: baseline + feasible only, sorted by score
        df_chosen = pd.concat(
            [df_baseline, df_feasible.sort_values(by=combine_col, ascending=False)],
            ignore_index=True,
        )

        # ==========================================
        # 5. Format Columns
        # ==========================================
        def format_df(target_df):
            const_dict_values = OrderedDict(
                {
                    GlobalConst.COL_PARAM_COMBINED_SCORE: None,
                    GlobalConst.COL_PARAM_RECALL: None,
                    GlobalConst.COL_PARAM_FAR: None,
                    GlobalConst.COL_PARAM_SKIP_RATE: None,
                    GlobalConst.COL_PARAM_RECALL_DROP: None,
                    GlobalConst.COL_PARAM_RECALL_RET: None,
                    GlobalConst.COL_PARAM_RECALL_RET_NORM: None,  # new
                    GlobalConst.COL_PARAM_SKIP_RATE_NORM: None,  # new
                    GlobalConst.COL_PARAM_W_S: self.w_s,
                    GlobalConst.COL_PARAM_W_R: self.w_r,
                    GlobalConst.COL_PARAM_DELTA_R: self.delta_r,
                }
            )
            for col, val in const_dict_values.items():
                if val is not None:
                    target_df[col] = val
                    target_df.loc[0, col] = pd.NA  # differentiate baseline
                else:
                    assert col in target_df.columns, (
                        f"Expected column '{col}' not found in DataFrame."
                    )
            front_cols = ["experiment"] + list(const_dict_values.keys())
            front_cols = [col for col in front_cols if col in target_df.columns]
            other_cols = [col for col in target_df.columns if col not in front_cols]
            return target_df[front_cols + other_cols]

        return format_df(df_chosen), format_df(df_full)
