import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))  # Add parent directory to sys.path

import unittest
from src.methods.precomputed import PrecomputedRsProc
from src.exp import MyExp

from halib import *


class TestPrecomputedRsProc(unittest.TestCase):
    def setUp(self):
        # Use real standard experiment path
        self.exp_dir_path = "./zout/zoptim/MainPC__ds_UFireIndoor2__mt_no_temp_method__af4b0d32a3d2__20260316.172122"

        self.experiment: MyExp = MyExp.from_standard_exp(self.exp_dir_path)
        self.cfg = self.experiment.full_cfg

        # Inject the precomputed dir config dynamically as it's testing this specific feature
        self.cfg.inferCfg.pre_computed_no_skip_dir = self.exp_dir_path

        self.proc = PrecomputedRsProc(self.cfg)

        # Determine the target evaluation file from the dataset configuration
        video_list = self.cfg.dbsetCfg.get_video_list()
        self.video_name = "aihub__lb_none__0175"  # "aihub__lb_fire__0182"
        self.video_path = next(v for v in video_list if self.video_name in v)

    def test_load_video_data_success(self):
        self.proc.load_video_data(self.video_path)
        self.assertIsNotNone(self.proc.precomputes)
        self.assertGreaterEqual(len(self.proc.precomputes), 1)  # ty:ignore[invalid-argument-type]
        # Verify index is established properly (e.g. frames 1 and 2 usually exist in video outputs)
        self.assertIn(1, self.proc.precomputes.index)  # ty:ignore[unresolved-attribute]
        self.assertIn(2, self.proc.precomputes.index)  # ty:ignore[unresolved-attribute]

    def test_load_video_data_not_found(self):
        # We need an actual video to pass video_path_to_csv checks
        # Point the config to an empty precomputed dir
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            self.cfg.inferCfg.pre_computed_no_skip_dir = temp_dir
            proc = PrecomputedRsProc(self.cfg)
            with self.assertRaises(FileNotFoundError):
                proc.load_video_data(self.video_path)

    def test_get_frame_data(self):
        # Load data first
        self.proc.load_video_data(self.video_path)

        # frame 1 retrieval
        data_f1 = self.proc.get_frame_data(1)
        pprint_box(data_f1, title="Frame 1 Precomputed Data")

        # frame 2 retrieval
        data_f2 = self.proc.get_frame_data(300)
        pprint_box(data_f2, title="Frame 2 Precomputed Data")

        # Validate frame 1 data against expected values from the CSV
        self.assertIsNotNone(data_f1)
        self.assertEqual(data_f1["predLabel"], "SmokeOnly")  # ty:ignore[not-subscriptable]
        self.assertEqual(data_f1["predLabelIdx"], 2)  # ty:ignore[not-subscriptable]
        # Checking values approximately from the read CSV
        self.assertAlmostEqual(data_f1["probs"][0], 0.0714, places=3)  # ty:ignore[not-subscriptable]
        self.assertAlmostEqual(data_f1["probs"][2], 0.9013, places=3)  # ty:ignore[not-subscriptable]
        self.assertTrue(data_f1["is_precomputed"])  # ty:ignore[not-subscriptable]

        # Validate frame 2 data against expected values from the CSV
        self.assertIsNotNone(data_f2)
        self.assertEqual(data_f2["predLabel"], "SmokeOnly")  # ty:ignore[not-subscriptable]
        self.assertEqual(data_f2["predLabelIdx"], 2)  # ty:ignore[not-subscriptable]

        # Test nonexistent frame retrieval
        data_f9999 = self.proc.get_frame_data(999999)
        self.assertIsNone(data_f9999)

    def test_get_frame_data_without_loading(self):
        # Ensure it safely returns None if data isn't loaded

        # Intentionally create a fresh proc pointing nowhere
        self.cfg.inferCfg.pre_computed_no_skip_dir = None
        empty_proc = PrecomputedRsProc(self.cfg)
        empty_proc.load_video_data(
            self.video_path
        )  # Won't load anything due to None dir
        data = empty_proc.get_frame_data(1)
        self.assertIsNone(data)


if __name__ == "__main__":
    unittest.main()
