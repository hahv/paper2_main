import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)  # Add parent directory to sys.path

from halib import *
import unittest
import tempfile

from src.common import GlobalConst


class TestReportOptim(unittest.TestCase):
    def test_report_optim(self):
        # Fallback dummy data if actual output CSV does not exist
        mock_df = pd.DataFrame(
            {
                "experiment": ["mt_no_temp_method_baseline", "exp_1", "exp_2", "exp_3"],
                GlobalConst.COL_PARAM_RECALL: [0.95, 0.94, 0.90, 0.96],
                GlobalConst.COL_PARAM_FAR: [0.10, 0.05, 0.02, 0.15],
                GlobalConst.COL_PARAM_SKIP_RATE: [0.0, 0.5, 0.8, 0.2],
            }
        )

        # Run inside a clean temporary directory so we don't pollute the actual zout workspace
        with tempfile.TemporaryDirectory() as temp_dir:
            # Provide a dummy param selection config
            param_select_cfg = os.path.join(temp_dir, "__param_select.yaml")
            with open(param_select_cfg, "w", encoding="utf-8") as f:
                f.write("w_s: 0.60\n")
                f.write("w_f: 0.40\n")
                f.write("w_r: 0.0\n")
                f.write("delta_r: 0.05\n")

            from zbin.rp.run_report import report_optim_by_csv
            out_df = report_optim_by_csv(
                optim_csv_path=mock_df,
                param_select_cfg=param_select_cfg,
                shorten=True,
            )

            console.rule("[bold green]Output DataFrame from report_optim_by_csv")
            csvfile.fn_display_df(out_df)

            # Ensure the combined score metric generated during choosing params is now present
            self.assertIn(
                GlobalConst.COL_PARAM_COMBINED_SCORE,
                out_df.columns,
                "Combined_Score column is missing",
            )

            # Ensure output dataframe has matching rows
            self.assertEqual(
                len(out_df),
                len(mock_df),
                "Mismatch between output and input frame dimensions",
            )

            # Spot check the baseline row padding features (index 0)
            self.assertAlmostEqual(
                out_df.iloc[0][GlobalConst.COL_PARAM_COMBINED_SCORE], -1.0
            )


if __name__ == "__main__":
    unittest.main()
