import os
import pandas as pd
from pathlib import Path
from typing import Optional

from src.common import GlobalConst
from src.metrics.loaders.base_csv_loader import BaseRawCsvLoader


class ClsModelExternalLoader:
    """
    Loads cls model _pred.csv (like firenet or mobilenet) from an external pre-existing experiment directory
    and merges with real GT from the dataset directory.

    cls model CSV format (one row per frame):
        video;num_frames;frame_idx;gt;label;prob;elapsed_time
        - 'label' → the prediction (renamed to pred_label)
        - 'gt'    → unreliable ("Unknown"), IGNORED — real GT is loaded from the dataset

    Normalized output:
        video;video_path;frame_idx;gt_label;pred_label;elapsed_time
    """

    PRED_SUFFIX = "_pred"

    def __init__(self, exp_dir: str, gt_pattern: str = GlobalConst.GT_FILE_PATTERN):
        self.exp_dir = Path(exp_dir)
        self.gt_pattern = gt_pattern

    def _find_pred_csv(self, video_path: str) -> Optional[str]:
        stem = Path(video_path).stem
        pred_csv = self.exp_dir / f"{stem}{self.PRED_SUFFIX}.csv"
        return str(pred_csv) if pred_csv.exists() else None

    def load_video_gt_pred_df(self, video_path: str) -> pd.DataFrame:
        """
        Returns a standard merged DataFrame with columns:
            [video, video_path, frame_idx, gt_label, pred_label, elapsed_time]
        """
        # 1. Load real GT from dataset (ignore the 'gt' column inside cls model CSV)
        gt_df = BaseRawCsvLoader.load_csv_by_pattern(
            video_path=video_path,
            csv_pattern=self.gt_pattern,
            is_gt=True,
        )

        # 2. Load cls model pred CSV
        pred_csv = self._find_pred_csv(video_path)
        if pred_csv is None:
            raise FileNotFoundError(
                f"No '{self.PRED_SUFFIX}.csv' for '{Path(video_path).stem}' in {self.exp_dir}"
            )

        pred_df = pd.read_csv(
            pred_csv,
            sep=";",
            encoding="utf-8",
            keep_default_na=False,
            dtype={"label": str, GlobalConst.COL_ELAPSED_TIME: float},
        )

        # 3. Rename 'label' → pred_label (cls model uses 'label', pipeline expects 'pred_label')
        pred_df.rename(columns={"label": GlobalConst.COL_PRED}, inplace=True)

        # 4. Normalize video name and path using the actual video file as source of truth
        #    (guarantees correct extension regardless of what's stored in the CSV)
        video_name = Path(video_path).name
        pred_df[GlobalConst.COL_VIDEO] = video_name
        pred_df[GlobalConst.COL_VIDEO_PATH] = str(os.path.abspath(video_path))

        # 5. Keep only required columns for merge
        needed_cols = [
            GlobalConst.COL_VIDEO,
            GlobalConst.COL_VIDEO_PATH,
            GlobalConst.COL_FRAME_IDX,
            GlobalConst.COL_PRED,
            GlobalConst.COL_ELAPSED_TIME,
        ]
        pred_df = pred_df[[c for c in needed_cols if c in pred_df.columns]]

        # 6. Merge GT + Pred on [video, video_path, frame_idx]
        return BaseRawCsvLoader._merge_gt_pred_dfs(gt_df, pred_df, video_path)
