from halib import *
from loguru import logger
from abc import ABC
import pandas as pd
from typing import List, Optional
from src.config import Config
import os
from pathlib import Path
from src.common import GlobalConst


class BaseVideoRawCsvLoader(ABC):
    """
    Base class for dataset-specific logic to load predictions and ground truth from CSV files.
    Load the RAW (gt_label, pred_label) without any processing.
    Raw label can be further processed later (e.g., mapping to binary labels, etc).
    """

    RAW_FIXED_COLS = [
        GlobalConst.COL_VIDEO,
        GlobalConst.COL_VIDEO_PATH,
        GlobalConst.COL_FRAME_IDX,
    ]

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.gt_file_pattern = self.cfg.dbsetCfg.get_gt_file_pattern()
        self.infer_file_pattern = self.cfg.inferCfg.csv_infer_pattern

    @staticmethod
    def video_path_to_csv(
        video_path: str,
        csv_pattern: str,
        is_gt: bool = True,
        csv_dir: Optional[str] = None,
    ) -> str:
        """
        Given a video path, returns the corresponding CSV file path for predictions or ground truth (is_gt=True for GT).
        """
        assert os.path.exists(video_path), f"Video path {video_path} does not exist"
        if is_gt and csv_dir is not None:
            raise ValueError(
                "csv_dir should be None when is_gt is True, i.e. GT csv is in the same dir as video"
            )
        if is_gt:
            csv_dir = os.path.dirname(video_path)

        video_name = Path(video_path).stem
        csv_file = os.path.join(str(csv_dir), f"{video_name}{csv_pattern}.csv")
        if not os.path.exists(csv_file):
            return ""
        return csv_file

    @staticmethod
    def verify_gt_pred(df: pd.DataFrame, required_cols: List[str]) -> bool:
        """
        Verifies that the GT and Prediction DataFrame contains the required columns.
        """
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame")
        return True

    def load_video_gt_pred_df(
        self, video_path: str, extra_data: Optional[dict] = None
    ) -> pd.DataFrame:
        """
        Loads both GT and Prediction for a given video, merges them into a single DataFrame, and returns it.
        The returned DataFrame will have columns: [video, video_path, frame_idx] + any GT and Prediction columns defined by the subclass.
        """
        gt_df = self.get_gt_df(video_path, extra_data)
        pred_df = self.load_pred_df(video_path, extra_data)
        # Verify required columns
        self.verify_gt_pred(gt_df, self.RAW_FIXED_COLS + [GlobalConst.COL_GT])
        self.verify_gt_pred(
            pred_df,
            self.RAW_FIXED_COLS + [GlobalConst.COL_PRED, GlobalConst.COL_ELAPSED_TIME],
        )
        # ! do WARNING if lengths do not match?
        if len(gt_df) != len(pred_df):
            logger.warning(
                f"WARNING: GT and Prediction lengths do not match for video {video_path} ({len(gt_df)} vs {len(pred_df)})"
            )
        # Merge on FIXED_COLS
        merged_df = pd.merge(gt_df, pred_df, on=self.RAW_FIXED_COLS, how="inner")
        return merged_df

    def get_gt_df(
        self, video_path: str, extra_data: Optional[dict] = None
    ) -> pd.DataFrame:
        csv_file = self.video_path_to_csv(
            video_path=video_path,
            csv_pattern=self.gt_file_pattern,
            is_gt=True,
            csv_dir=None,
        )
        assert os.path.exists(csv_file), f"GT CSV file {csv_file} does not exist"
        # 3. Read GT CSV
        gt_df = pd.read_csv(
            csv_file,
            sep=";",
            encoding="utf-8",
            dtype={GlobalConst.COL_GT: str},
            keep_default_na=False,
        )
        gt_df.rename(columns={"label": GlobalConst.COL_GT}, inplace=True)
        # Add fixed columns if not present
        video_name = Path(video_path).name
        if GlobalConst.COL_VIDEO not in gt_df.columns:
            gt_df[GlobalConst.COL_VIDEO] = video_name
        if GlobalConst.COL_VIDEO_PATH not in gt_df.columns:
            gt_df[GlobalConst.COL_VIDEO_PATH] = video_path
        if GlobalConst.COL_FRAME_IDX not in gt_df.columns:
            gt_df[GlobalConst.COL_FRAME_IDX] = (
                gt_df.index
            )  # Assuming frame_idx is the row index
        return gt_df

    def load_pred_df(
        self, video_path: str, extra_data: Optional[dict] = None
    ) -> pd.DataFrame:
        csv_file = self.video_path_to_csv(
            video_path=video_path,
            csv_pattern=self.infer_file_pattern,  # ty:ignore[invalid-argument-type]
            is_gt=False,
            csv_dir=self.cfg.get_outdir(),
        )
        assert os.path.exists(csv_file), (
            f"Prediction CSV file {csv_file} does not exist"
        )
        pred_df = pd.read_csv(
            csv_file,
            sep=";",
            encoding="utf-8",
            dtype={GlobalConst.COL_PRED: str, GlobalConst.COL_ELAPSED_TIME: float},
            keep_default_na=False,
        )
        # Add fixed columns if not present
        video_name = Path(video_path).name
        if GlobalConst.COL_VIDEO not in pred_df.columns:
            pred_df[GlobalConst.COL_VIDEO] = video_name
        if GlobalConst.COL_VIDEO_PATH not in pred_df.columns:
            pred_df[GlobalConst.COL_VIDEO_PATH] = video_path
        if GlobalConst.COL_FRAME_IDX not in pred_df.columns:
            pred_df[GlobalConst.COL_FRAME_IDX] = (
                pred_df.index
            )  # Assuming frame_idx is the row index
        return pred_df
