from halib import *
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
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
    def choose_params(self) -> pd.DataFrame:
        """Implement the parameter selection logic based on the DataFrame and context.
        Return a DataFrame with the chosen parameters and their corresponding metrics Score"""
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
        # Updated defaults based on the removal of FAR optimization
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

    from scipy.constants import G


from pygments.unistring import combine
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
            GlobalConst.COL_PARAM_FAR,
        }
        if not required_cols.issubset(self.df.columns):
            missing = required_cols - set(self.df.columns)
            print(f"Error: Missing required columns in DataFrame: {missing}")
            return False
        return True

    def choose_params(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Implement Algorithm 1 using weighted scores."""

        # Ensure weights sum to 1
        total_weight = self.w_s + self.w_r
        if abs(total_weight - 1.0) > 1e-9:
            raise ValueError("Weights w_s and w_r must sum to 1.")

        df_baseline = self.df.iloc[[0]].copy()
        df_rest = self.df.iloc[1:].copy()

        # ==========================================
        # 1. Calculate Metrics for ALL Configurations
        # ==========================================
        # Calculate Recall Drop (Absolute drop from baseline)
        df_rest[GlobalConst.COL_PARAM_RECALL_DROP] = self.r_base - df_rest[GlobalConst.COL_PARAM_RECALL]

        # Compute Recall Retention (Bounded [0, 1] on the top end)
        df_rest[GlobalConst.COL_PARAM_RECALL_RET] = (
            1.0 - df_rest[GlobalConst.COL_PARAM_RECALL_DROP] / self.delta_r
        ).clip(upper=1.0)

        # Calculate Combined Score
        combine_col = GlobalConst.COL_PARAM_COMBINED_SCORE
        df_rest[combine_col] = (
            self.w_r * df_rest[GlobalConst.COL_PARAM_RECALL_RET]
            + self.w_s * df_rest[GlobalConst.COL_PARAM_SKIP_RATE]
        )

        # ==========================================
        # 2. Apply Hard Feasibility Constraint
        # ==========================================
        min_acceptable_recall = self.r_base - self.delta_r
        df_filtered = df_rest[
            df_rest[GlobalConst.COL_PARAM_RECALL] >= (min_acceptable_recall - 1e-9)
        ].copy()

        # ==========================================
        # 3. Finalize DataFrames
        # ==========================================
        # Assign dummy values to the baseline row
        df_baseline[GlobalConst.COL_PARAM_RECALL_DROP] = 0.0
        df_baseline[GlobalConst.COL_PARAM_RECALL_RET] = 1.0
        df_baseline[combine_col] = -1.0  # Dummy score for first row

        # Concat baseline with FULL results
        df_full = pd.concat(
            [df_baseline, df_rest.sort_values(by=combine_col, ascending=False)],
            ignore_index=True,
        )

        # Concat baseline with FILTERED results
        df_chosen = pd.concat(
            [df_baseline, df_filtered.sort_values(by=combine_col, ascending=False)],
            ignore_index=True,
        )

        # ==========================================
        # 4. Helper Function to Format Columns
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
                    GlobalConst.COL_PARAM_W_S: self.w_s,
                    GlobalConst.COL_PARAM_W_R: self.w_r,
                    GlobalConst.COL_PARAM_DELTA_R: self.delta_r,
                }
            )

            for col, val in const_dict_values.items():
                if val is not None:
                    target_df[col] = val  # Set the entire column to the constant value
                    target_df.loc[0, col] = pd.NA  # differentiate baseline
                else:
                    assert col in target_df.columns, (
                        f"Expected column '{col}' not found in DataFrame."
                    )

            # Reorder columns
            front_cols = ["experiment"] + list(const_dict_values.keys())
            front_cols = [col for col in front_cols if col in target_df.columns]
            other_cols = [col for col in target_df.columns if col not in front_cols]

            return target_df[front_cols + other_cols]

        # Apply formatting to both DataFrames and return
        return format_df(df_chosen), format_df(df_full)