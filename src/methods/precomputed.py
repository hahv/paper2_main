import os
import ast
import pandas as pd
from typing import Dict, Any, Optional

from src.config import Config
from src.common import GlobalConst
from src.metrics.loaders.base_csv_loader import BaseRawCsvLoader


class PrecomputedRsProc:
    """
    Loads pre-computed inference results from a CSV file to bypass actual model inference.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.precomputes: Optional[pd.DataFrame] = None
        self.precomputed_dir = self.cfg.inferCfg.pre_computed_no_skip_dir

    def load_video_data(self, video_path: str):
        """Loads the precomputed CSV for the given video path."""
        self.precomputes = None
        if not self.precomputed_dir:
            return

        expected_csv_path = BaseRawCsvLoader.video_path_to_csv(
            video_path=video_path,
            csv_pattern=self.cfg.inferCfg.csv_infer_pattern,  # ty:ignore[invalid-argument-type]
            is_gt=False,
            csv_dir=self.precomputed_dir,
        )

        if not expected_csv_path or not os.path.exists(expected_csv_path):
            raise FileNotFoundError(
                f"Precomputed file not found for video: {video_path}, expected at: {expected_csv_path}"
            )

        # Load CSV using standard loader handler
        self.precomputes = BaseRawCsvLoader._read_raw_csv(
            csv_path=expected_csv_path, video_path=video_path, is_gt=False
        )

        # Ensure frame_idx is the index for fast lookup
        if GlobalConst.COL_FRAME_IDX in self.precomputes.columns:
            self.precomputes.set_index(GlobalConst.COL_FRAME_IDX, inplace=True)

    def get_frame_data(self, frame_idx: int) -> Optional[Dict[str, Any]]:
        """Returns the pre-computed results for a given frame index, if
        available."""
        # print(f"Looking up precomputed results for frame_idx: {frame_idx}")
        if self.precomputes is None:
            return None

        if frame_idx not in self.precomputes.index:
            return None

        row = self.precomputes.loc[frame_idx]

        # Parse lists
        logits = (
            ast.literal_eval(row["logits"])
            if isinstance(row["logits"], str)
            else row["logits"]
        )
        probs = (
            ast.literal_eval(row["probs"])
            if isinstance(row["probs"], str)
            else row["probs"]
        )

        # Parse scalar values correctly handling instances where pandas might return a series or scalar
        if isinstance(probs, pd.Series):
            logits = (
                ast.literal_eval(logits.iloc[0])
                if isinstance(logits.iloc[0], str)
                else logits.iloc[0]
            )
            probs = (
                ast.literal_eval(probs.iloc[0])
                if isinstance(probs.iloc[0], str)
                else probs.iloc[0]
            )
            pred_label = str(row[GlobalConst.COL_PRED].iloc[0])
            pred_label_idx = int(row["pred_label_idx"].iloc[0])
        else:
            pred_label = str(row[GlobalConst.COL_PRED])
            pred_label_idx = int(row["pred_label_idx"])

        elapsed_time = float(row.get("elapsed_time", 0.0))

        # for correct timing when using pre-computed results, we must include the elapsed time in the CSV and return it here so that BaseMethod can use it instead of wall-clock time
        return {
            "logits": logits,
            "probs": probs,
            "predLabelIdx": pred_label_idx,
            "predLabel": pred_label,
            "elapsed_time": elapsed_time,
            "is_precomputed": True,
        }
