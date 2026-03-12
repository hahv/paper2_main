import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

import pytest
from halib import *
from zbin.otp.param_chooser import WeightedScoreParamChooser


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def make_predefined_params(
    r_base=0.90,
    far_base=0.10,
    delta_r=0.01,
    w_s=0.60,
    w_f=0.20,
    w_r=0.20,
):
    return {
        "baseline": {"recall": r_base, "far": far_base},
        "delta_r": delta_r,
        "w_s": w_s,
        "w_f": w_f,
        "w_r": w_r,
    }


def make_df(rows):
    """rows: list of (exp, recall, far, skip_ratio) tuples."""
    return pd.DataFrame(rows, columns=["exp", "recall", "far", "skip_ratio"])  # ty:ignore[invalid-argument-type]


# ---------------------------------------------------------------------------
# Dummy dataset
#
#  theta_A: recall drop = 0.004  (<= delta_r=0.01)  → feasible, best skip ratio
#  theta_B: recall drop = 0.008  (<= delta_r=0.01)  → feasible
#  theta_C: recall drop = 0.015  (>  delta_r=0.01)  → infeasible, must be excluded
# ---------------------------------------------------------------------------

R_BASE = 0.90
FAR_BASE = 0.10
DELTA_R = 0.01

DUMMY_ROWS = [
    # (exp,       recall,          far,   skip_ratio)
    ("theta_A", R_BASE - 0.004, 0.07, 0.873),  # feasible, highest skip ratio
    ("theta_B", R_BASE - 0.008, 0.06, 0.841),  # feasible
    ("theta_C", R_BASE - 0.015, 0.05, 0.825),  # infeasible (recall drop > delta_r)
]


@pytest.fixture
def default_params():
    return make_predefined_params(r_base=R_BASE, far_base=FAR_BASE, delta_r=DELTA_R)


@pytest.fixture
def dummy_df():
    return make_df(DUMMY_ROWS)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWeightedScoreParamChooserBasic:
    def test_returns_dataframe(self, dummy_df, default_params):
        chooser = WeightedScoreParamChooser(dummy_df, default_params)
        result = chooser.choose_params()
        assert isinstance(result, pd.DataFrame)

    def test_infeasible_row_excluded(self, dummy_df, default_params):
        """theta_C violates recall constraint and must not appear in results."""
        chooser = WeightedScoreParamChooser(dummy_df, default_params)
        result = chooser.choose_params()
        # theta_C has recall = R_BASE - 0.015; all rows in result must satisfy constraint
        assert (result["recall"] >= R_BASE - DELTA_R - 1e-9).all()
        assert len(result) == 2  # only theta_A and theta_B

    def test_combined_score_column_exists(self, dummy_df, default_params):
        chooser = WeightedScoreParamChooser(dummy_df, default_params)
        result = chooser.choose_params()
        assert "Combined_Score" in result.columns

    def test_sorted_descending_by_combined_score(self, dummy_df, default_params):
        chooser = WeightedScoreParamChooser(dummy_df, default_params)
        result = chooser.choose_params()
        scores = result["Combined_Score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_best_candidate_is_theta_a(self, dummy_df, default_params):
        """theta_A has the highest skip_ratio among feasible candidates."""
        chooser = WeightedScoreParamChooser(dummy_df, default_params)
        result = chooser.choose_params()
        assert result.iloc[0]["skip_ratio"] == pytest.approx(0.873)

    def test_does_not_mutate_input_df(self, dummy_df, default_params):
        original_cols = set(dummy_df.columns)
        chooser = WeightedScoreParamChooser(dummy_df, default_params)
        chooser.choose_params()
        assert set(dummy_df.columns) == original_cols

    def test_derived_recall_retention_formula(self, dummy_df, default_params):
        """R̃(θ) = 1 − (R_base − R(θ)) / δ_R"""
        chooser = WeightedScoreParamChooser(dummy_df, default_params)
        result = chooser.choose_params()
        for _, row in result.iterrows():
            expected = 1.0 - (R_BASE - row["recall"]) / DELTA_R
            assert row["recall_retention"] == pytest.approx(expected, abs=1e-9)

    def test_derived_far_reduction_norm_formula(self, dummy_df, default_params):
        """ΔFAR̃(θ) = max(0, (FAR_base − FAR(θ)) / FAR_base)"""
        chooser = WeightedScoreParamChooser(dummy_df, default_params)
        result = chooser.choose_params()
        for _, row in result.iterrows():
            expected = max(0.0, (FAR_BASE - row["far"]) / FAR_BASE)
            assert row["far_reduction_norm"] == pytest.approx(expected, abs=1e-9)

    def test_combined_score_formula(self, dummy_df, default_params):
        """score = w_R·R̃ + w_S·S + w_F·ΔFAR̃"""
        params = default_params
        chooser = WeightedScoreParamChooser(dummy_df, params)
        result = chooser.choose_params()
        for _, row in result.iterrows():
            expected = (
                params["w_r"] * row["recall_retention"]
                + params["w_s"] * row["skip_ratio"]
                + params["w_f"] * row["far_reduction_norm"]
            )
            assert row["Combined_Score"] == pytest.approx(expected, abs=1e-9)


class TestWeightedScoreParamChooserEdgeCases:
    def test_all_infeasible_returns_empty(self, default_params):
        """All rows violate recall constraint → empty result."""
        df = make_df(
            [
                ("theta_D", R_BASE - 0.02, 0.05, 0.90),
                ("theta_E", R_BASE - 0.05, 0.04, 0.85),
            ]
        )
        chooser = WeightedScoreParamChooser(df, default_params)
        result = chooser.choose_params()
        assert len(result) == 0

    def test_far_base_zero_sets_far_reduction_to_zero(self):
        """When FAR_base=0, far_reduction_norm must be 0.0 for all rows."""
        params = make_predefined_params(far_base=0.0)
        df = make_df([("theta_A", R_BASE - 0.005, 0.0, 0.80)])
        chooser = WeightedScoreParamChooser(df, params)
        result = chooser.choose_params()
        assert result.iloc[0]["far_reduction_norm"] == pytest.approx(0.0)

    def test_exact_recall_boundary_is_feasible(self, default_params):
        """A candidate at exactly R_base − δ_R should pass the constraint (R̃=0)."""
        df = make_df([("theta_A", R_BASE - DELTA_R, 0.08, 0.70)])
        chooser = WeightedScoreParamChooser(df, default_params)
        result = chooser.choose_params()
        assert len(result) == 1
        assert result.iloc[0]["recall_retention"] == pytest.approx(0.0, abs=1e-9)

    def test_weights_not_summing_to_one_raises(self, dummy_df):
        params = make_predefined_params(w_s=0.50, w_f=0.30, w_r=0.30)  # sum = 1.10
        chooser = WeightedScoreParamChooser(dummy_df, params)
        with pytest.raises(ValueError, match="Weights must sum to 1"):
            chooser.choose_params()

    def test_missing_column_fails_validation(self):
        df = pd.DataFrame({"exp": ["theta_A"], "recall": [0.88], "skip_ratio": [0.80]})  # missing "far"
        params = make_predefined_params()
        with pytest.raises(AssertionError):
            WeightedScoreParamChooser(df, params)

    def test_no_predefined_params_raises(self, dummy_df):
        with pytest.raises(
            AssertionError, match="Predefined parameters must be provided"
        ):
            WeightedScoreParamChooser(dummy_df, predefined_params=None)

    def test_single_feasible_row(self, default_params):
        df = make_df([("theta_A", R_BASE - 0.005, 0.08, 0.75)])
        chooser = WeightedScoreParamChooser(df, default_params)
        result = chooser.choose_params()
        assert len(result) == 1

    def test_far_worse_than_baseline_clipped_to_zero(self, default_params):
        """FAR > FAR_base should yield far_reduction_norm=0, not negative."""
        df = make_df([("theta_A", R_BASE - 0.005, FAR_BASE + 0.05, 0.70)])
        chooser = WeightedScoreParamChooser(df, default_params)
        result = chooser.choose_params()
        assert result.iloc[0]["far_reduction_norm"] == pytest.approx(0.0, abs=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
