from halib import *
from loguru import logger
from abc import ABC
import pandas as pd
from typing import List, Optional
from src.config import Config
import os
from pathlib import Path
from src.common import GlobalConst


class BaseRawCsvLoader(ABC):
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
    def check_required_cols(df: pd.DataFrame, required_cols: List[str]) -> bool:
        """
        Verifies that the GT and Prediction DataFrame contains the required columns.
        """
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in DataFrame")
        return True

    @staticmethod
    def _read_raw_csv(csv_path: str, video_path: str, is_gt: bool) -> pd.DataFrame:
        """
        Helper to read a CSV file, standardizes columns (renaming label -> gt_label),
        and adds metadata columns (video, video_path, frame_idx).
        """
        assert os.path.exists(csv_path), f"CSV file {csv_path} does not exist"

        # Prepare read_csv options
        if is_gt:
            dtype_map = {GlobalConst.COL_GT: str}
            # Also handle 'label' if it exists in the file before renaming
            dtype_map["label"] = str
        else:
            dtype_map = {
                GlobalConst.COL_PRED: str,
                GlobalConst.COL_ELAPSED_TIME: float,
            }

        df = pd.read_csv(
            csv_path,
            sep=";",
            encoding="utf-8",
            dtype=dtype_map,
            keep_default_na=False,
        )

        # Standardize GT label column
        if is_gt and "label" in df.columns and GlobalConst.COL_GT not in df.columns:
            df.rename(columns={"label": GlobalConst.COL_GT}, inplace=True)

        # Add fixed columns if not present
        video_name = Path(video_path).name
        if GlobalConst.COL_VIDEO not in df.columns:
            df[GlobalConst.COL_VIDEO] = video_name
        # ! Add video_path column (event if already exists, to ensure consistency)
        df[GlobalConst.COL_VIDEO_PATH] = str(os.path.abspath(video_path))
        if GlobalConst.COL_FRAME_IDX not in df.columns:
            df[GlobalConst.COL_FRAME_IDX] = df.index

        return df

    @staticmethod
    def _merge_gt_pred_dfs(
        gt_df: pd.DataFrame,
        pred_df: pd.DataFrame,
        video_path: str,
        do_verify: bool = True,
    ) -> pd.DataFrame:
        """
        Verifies and merges GT and Prediction DataFrames on fixed columns.
        """
        if do_verify:
            # Verify required columns
            BaseRawCsvLoader.check_required_cols(
                gt_df, BaseRawCsvLoader.RAW_FIXED_COLS + [GlobalConst.COL_GT]
            )
            BaseRawCsvLoader.check_required_cols(
                pred_df,
                BaseRawCsvLoader.RAW_FIXED_COLS
                + [GlobalConst.COL_PRED, GlobalConst.COL_ELAPSED_TIME],
            )

        if len(gt_df) != len(pred_df):
            logger.warning(
                f"WARNING: GT and Prediction lengths do not match for video {video_path} ({len(gt_df)} vs {len(pred_df)})"
            )

        merged_df = pd.merge(
            gt_df, pred_df, on=BaseRawCsvLoader.RAW_FIXED_COLS, how="inner"
        )
        return merged_df

    @staticmethod
    def load_gt_pred_df_from_files(
        video_path: str, gt_csv_path: str, pred_csv_path: str
    ) -> pd.DataFrame:
        """
        Loads and merges GT and Pred DataFrames from explicit file paths.
        """
        gt_df = BaseRawCsvLoader._read_raw_csv(gt_csv_path, video_path, is_gt=True)
        pred_df = BaseRawCsvLoader._read_raw_csv(pred_csv_path, video_path, is_gt=False)
        return BaseRawCsvLoader._merge_gt_pred_dfs(gt_df, pred_df, video_path)

    @staticmethod
    def load_csv_by_pattern(
        video_path: str,
        csv_pattern: str,
        is_gt: bool = True,
        csv_dir: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Resolves the CSV path and loads it with standard columns.
        """
        csv_file = BaseRawCsvLoader.video_path_to_csv(
            video_path=video_path,
            csv_pattern=csv_pattern,
            is_gt=is_gt,
            csv_dir=csv_dir,
        )
        return BaseRawCsvLoader._read_raw_csv(csv_file, video_path, is_gt=is_gt)

    @staticmethod
    def load_gt_pred_df_by_paths(
        video_path: str,
        gt_csv_dir: Optional[str] = None,
        pred_csv_dir: Optional[str] = None,
        gt_pattern: str = GlobalConst.GT_FILE_PATTERN,
        pred_pattern: str = GlobalConst.INFER_FILE_PATTERN,
    ) -> pd.DataFrame:
        """
        Resolves CSV paths from directories and patterns, then loads and merges them.
        """
        gt_df = BaseRawCsvLoader.load_csv_by_pattern(
            video_path, gt_pattern, is_gt=True, csv_dir=gt_csv_dir
        )
        pred_df = BaseRawCsvLoader.load_csv_by_pattern(
            video_path, pred_pattern, is_gt=False, csv_dir=pred_csv_dir
        )
        return BaseRawCsvLoader._merge_gt_pred_dfs(gt_df, pred_df, video_path)

    def load_video_gt_pred_df(
        self, video_path: str, extra_data: Optional[dict] = None
    ) -> pd.DataFrame:
        """
        Loads both GT and Prediction for a given video, merges them into a single DataFrame.
        """
        return self.load_gt_pred_df_by_paths(
            video_path=video_path,
            gt_csv_dir=None,
            pred_csv_dir=self.cfg.get_outdir(),
            gt_pattern=self.gt_file_pattern,
            pred_pattern=self.infer_file_pattern,  # ty:ignore[invalid-argument-type]
        )

    def get_gt_df(
        self, video_path: str, extra_data: Optional[dict] = None
    ) -> pd.DataFrame:
        return self.load_csv_by_pattern(
            video_path, self.gt_file_pattern, is_gt=True, csv_dir=None
        )

    def load_pred_df(
        self, video_path: str, extra_data: Optional[dict] = None
    ) -> pd.DataFrame:
        return self.load_csv_by_pattern(
            video_path,
            self.infer_file_pattern,  # ty:ignore[invalid-argument-type]
            is_gt=False,
            csv_dir=self.cfg.get_outdir(),
        )
