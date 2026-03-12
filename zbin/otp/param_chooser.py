from halib import *
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class ParamChooser(ABC):
    def __init__(
        self, df: pd.DataFrame, predefined_params: Optional[Dict[str, Any]] = None
    ):
        self.df = df
        self.predefined_params = predefined_params
        assert self.validate_dataframe(), (
            "DataFrame validation failed. Please ensure it contains the required columns and data types."
        )

    def validate_dataframe(self):
        return True

    @abstractmethod
    def choose_params(self) -> pd.DataFrame:
        """Implement the parameter selection logic based on the DataFrame and predefined parameters.
        Return a DataFrame with the chosen parameters and their corresponding metrics Score (e.g: Combined Score like in `docs/param_search.tex`)"""
        pass


class WeightedScoreParamChooser(ParamChooser):
    def __init__(
        self,
        df: pd.DataFrame,
        predefined_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(df, predefined_params)
        self.parse_predefined_params()

    def parse_predefined_params(self):
        assert self.predefined_params is not None, (
            "Predefined parameters must be provided."
        )
        self.w_s = self.predefined_params.get("w_s", 0.60)
        self.w_f = self.predefined_params.get("w_f", 0.20)
        self.w_r = self.predefined_params.get("w_r", 0.20)
        self.delta_r = self.predefined_params.get("delta_r", 0.01)
        baseline = self.predefined_params.get("baseline", {})
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
        # Ensure weights sum to 1
        total_weight = self.w_s + self.w_f + self.w_r
        if abs(total_weight - 1.0) > 1e-9:
            raise ValueError("Weights must sum to 1.")

        df = self.df.copy()

        # Compute derived metrics from raw recall/far
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
