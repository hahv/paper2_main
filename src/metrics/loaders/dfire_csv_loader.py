import os
import pandas as pd
from halib import *  # Assuming fs and other utils are here
from src.metrics.loaders.base_csv_loader import BaseCsvLoader


class DFireCsvLoader(BaseCsvLoader):
    """
    Adapter for the DFire dataset where Ground Truth is inferred
    heuristically from the video filename (e.g., if "FP" is in name).
    """

    def load_pred_df(self, csv_file: str) -> pd.DataFrame:
        # 1. Read Raw CSV
        df = pd.read_csv(
            csv_file,
            sep=";",
            encoding="utf-8",
            dtype={"pred_label": str, "elapsed_time": float},
            keep_default_na=False,
        )

        # 2. Normalize Video Name
        # Logic: Extract filename, remove _results extension
        video_name = fs.get_file_name(csv_file, split_file_ext=True)[0]
        video_name = video_name.replace("_results", "")
        df["video"] = video_name

        # 3. Normalize Prediction Labels
        # Logic: If 'fire' or 'smoke' in text -> Positive
        df["pred"] = (
            df["pred_label"]
            .str.lower()
            .apply(
                lambda x: self.POS_LABEL
                if ("fire" in x or "smoke" in x)
                else self.NEG_LABEL
            )
        )
        return df

    def get_gt(self, video_name: str, num_frames: int, pred_df: pd.DataFrame) -> list:
        # Logic: Heuristic based on filename
        gt_label = self.POS_LABEL
        if "FP" in video_name:
            gt_label = self.NEG_LABEL

        return [gt_label] * num_frames

