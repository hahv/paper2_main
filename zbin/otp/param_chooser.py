from halib import *
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class ParamSelect(ABC):

    def __init__(
        self, df: pd.DataFrame, context: Optional[Dict[str, Any]] = None
    ):
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

    def parse_context(self):
        assert self.context is not None, (
            "Context must be provided."
        )
        self.w_s = self.context.get("w_s", 0.60)
        self.w_f = self.context.get("w_f", 0.20)
        self.w_r = self.context.get("w_r", 0.20)
        self.delta_r = self.context.get("delta_r", 0.01)
        baseline = self.context.get("baseline", {})
        self.r_base = baseline["recall"]
        self.far_base = baseline["far"]

    def validate_dataframe(self):
        required_cols = {"skip_ratio", "recall", "far"}
        if not required_cols.issubset(self.df.columns):
            missing = required_cols - set(self.df.columns)
            print(f"Error: Missing required columns in DataFrame: {missing}")
            return False
        return True

    def choose_params(self) -> pd.DataFrame:
        """Implement Algorithm 1 (docs/param_search.tex) using weighted scores."""

        # Mapping to Algorithm 1 symbols:
        # +----------------------+-----------------------------------+
        # | Symbol (Alg. 1)      | Code reference                    |
        # +----------------------+-----------------------------------+
        # | w_S, w_F, w_R        | self.w_s, self.w_f, self.w_r      |
        # | R_base, FAR_base     | self.r_base, self.far_base        |
        # | δ_R                  | self.delta_r                      |
        # | S(θ), R(θ), FAR(θ)   | df["skip_ratio", "recall", "far"] |
        # +----------------------+-----------------------------------+

        # Ensure weights sum to 1
        total_weight = self.w_s + self.w_f + self.w_r
        if abs(total_weight - 1.0) > 1e-9:
            raise ValueError("Weights must sum to 1.")

        df = self.df.copy()

        # Compute derived metrics (exact equations from docs/param_search.tex)
        # R̃(θ) = 1 − (R_base − R(θ)) / δ_R
        df["recall_retention"] = 1.0 - (self.r_base - df["recall"]) / self.delta_r

        # ΔFAR̃(θ) = max(0, (FAR_base − FAR(θ)) / FAR_base)
        if self.far_base > 0:
            df["far_reduction_norm"] = (self.far_base - df["far"]).clip(
                lower=0.0
            ) / self.far_base
        else:
            df["far_reduction_norm"] = 0.0

        # Hard recall constraint: R(θ) ≥ R_base − δ_R  ⟺  R̃(θ) ≥ 0
        df_filtered = df[df["recall_retention"] >= -1e-9]

        # score = w_S·S(θ) + w_F·ΔFAR̃(θ) + w_R·R̃(θ)
        df_filtered = df_filtered.copy()
        df_filtered["Combined_Score"] = (
            self.w_r * df_filtered["recall_retention"]   # w_R · R̃(θ)
            + self.w_s * df_filtered["skip_ratio"]       # w_S · S(θ)
            + self.w_f * df_filtered["far_reduction_norm"]  # w_F · ΔFAR̃(θ)
        )

        # Sort by Combined Score and return the top configurations
        chosen_params_df = df_filtered.sort_values(
            by="Combined_Score", ascending=False
        ).reset_index(drop=True)

        return chosen_params_df
