from scipy.constants import G
from pygments.unistring import combine
from halib import *
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from src.common import GlobalConst


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
        Return a DataFrame with the chosen parameters and their corresponding metrics Score (e.g: Combined Score like in `docs/param_search.tex`)"""
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
        self.w_s = self.context.get("w_s", 0.60)
        self.w_f = self.context.get("w_f", 0.20)
        self.w_r = self.context.get("w_r", 0.20)
        self.delta_r = self.context.get("delta_r", 0.01)

    def get_baseline_metrics(self):
        # RECALL_COL = "metric_recall (tpr)"
        # FAR_COL = "metric_fpr (false alarm rate)"
        # firstrow['experiment'] value contains "mt_no_temp_method" (assert), no need exactly match
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

    def choose_params(self) -> pd.DataFrame:
        """Implement Algorithm 1 (docs/param_search.tex) using weighted scores."""

        # Mapping to Algorithm 1 symbols:
        # +--------------------+-----------------------------------------------------+
        # | Symbol (Alg. 1)    | Code reference                                      |
        # +--------------------+-----------------------------------------------------+
        # | w_S, w_F, w_R      | self.w_s, self.w_f, self.w_r                        |
        # | R_base, FAR_base   | self.r_base, self.far_base                          |
        # | δ_R                | self.delta_r                                        |
        # | S(θ), R(θ), FAR(θ) | df[COL_PARAM_SKIP_RATE_COL,                         |
        # |                    |     COL_PARAM_RECAL,                                |
        # |                    |     COL_PARAM_FAR]                                  |
        # |                    | (in common.GlobalConst)                             |
        # +--------------------+-----------------------------------------------------+

        # Ensure weights sum to 1
        total_weight = self.w_s + self.w_f + self.w_r
        if abs(total_weight - 1.0) > 1e-9:
            raise ValueError("Weights must sum to 1.")

        df_baseline = self.df.iloc[[0]].copy()
        df_rest = self.df.iloc[1:].copy()

        # Compute derived metrics (exact equations from docs/param_search.tex)
        # R̃(θ) = 1 − (R_base − R(θ)) / δ_R
        df_rest["recall_retention"] = (
            1.0 - (self.r_base - df_rest[GlobalConst.COL_PARAM_RECALL]) / self.delta_r
        )

        # ΔFAR̃(θ) = max(0, (FAR_base − FAR(θ)) / FAR_base)
        if self.far_base > 0:
            df_rest["far_reduction_norm"] = (
                self.far_base - df_rest[GlobalConst.COL_PARAM_FAR]
            ).clip(lower=0.0) / self.far_base
        else:
            df_rest["far_reduction_norm"] = 0.0

        # Hard recall constraint: R(θ) ≥ R_base − δ_R  ⟺  R̃(θ) ≥ 0
        df_filtered = df_rest[df_rest[GlobalConst.COL_PARAM_RECALL_RET] >= -1e-9].copy()

        # score = w_S·S(θ) + w_F·ΔFAR̃(θ) + w_R·R̃(θ)
        combine_col = GlobalConst.COL_PARAM_COMBINED_SCORE
        df_filtered[combine_col] = (
            self.w_r * df_filtered[GlobalConst.COL_PARAM_RECALL_RET]  # w_R · R̃(θ)
            + self.w_s * df_filtered[GlobalConst.COL_PARAM_SKIP_RATE]  # w_S · S(θ)
            + self.w_f
            * df_filtered[GlobalConst.COL_PARAM_FAR_REDUC_NORM]  # w_F · ΔFAR̃(θ)
        )

        # Assign dummy values to the baseline row
        df_baseline[GlobalConst.COL_PARAM_RECALL_RET] = 1.0
        df_baseline[GlobalConst.COL_PARAM_FAR_REDUC_NORM] = 0.0
        df_baseline[combine_col] = -1.0  # Dummy score for first row

        # Sort by Combined Score and return the top configurations (baseline remains first)
        chosen_params_df = pd.concat(
            [df_baseline, df_filtered.sort_values(by=combine_col, ascending=False)],
            ignore_index=True,
        )
        from collections import OrderedDict

        const_dict_values = OrderedDict(
            {
                GlobalConst.COL_PARAM_COMBINED_SCORE: None,
                GlobalConst.COL_PARAM_RECALL: None,
                GlobalConst.COL_PARAM_FAR: None,
                GlobalConst.COL_PARAM_SKIP_RATE: None,
                GlobalConst.COL_PARAM_RECALL_RET: None,
                GlobalConst.COL_PARAM_FAR_REDUC_NORM: None,
                GlobalConst.COL_PARAM_W_S: self.w_s,
                GlobalConst.COL_PARAM_W_F: self.w_f,
                GlobalConst.COL_PARAM_W_R: self.w_r,
                GlobalConst.COL_PARAM_DELTA_R: self.delta_r,
            }
        )
        for col, val in const_dict_values.items():
            if val is not None:
                chosen_params_df[col] = (
                    val  # Set the entire column to the constant value from context
                )
                # set the first row to NA to differentiate from actual evaluations (since baseline row is not a real config)
                chosen_params_df.loc[0, col] = pd.NA
            else:
                assert col in chosen_params_df.columns, (
                    f"Expected column '{col}' not found in DataFrame."
                )

        # Reorder columns: [experiment, combined_score, Recall, FAR, skip_rate, w_S, w_F, w_R, delta_R] then others
        front_cols = [
            "experiment",
        ] + list(const_dict_values.keys())

        # Ensure we only pick columns that actually exist, then append the rest
        front_cols = [col for col in front_cols if col in chosen_params_df.columns]
        other_cols = [col for col in chosen_params_df.columns if col not in front_cols]
        chosen_params_df = chosen_params_df[front_cols + other_cols]

        return chosen_params_df
