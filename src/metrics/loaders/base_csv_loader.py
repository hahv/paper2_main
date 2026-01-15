from abc import ABC, abstractmethod
import pandas as pd
from typing import List
from src.config import Config


class BaseCsvLoader(ABC):
    """
    Abstract base class for dataset-specific logic.
    Handles how to parse predictions and how to retrieve ground truth.
    """

    # Default labels (can be overridden by subclasses)
    POS_LABEL = "O_FireSmoke"
    NEG_LABEL = "X_None"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @abstractmethod
    def load_pred_df(self, csv_file: str) -> pd.DataFrame:
        """
        Reads a prediction CSV file and returns a standardized DataFrame.

        The returned DataFrame MUST have:
        - "video": (str) The video name
        - "pred": (str) The normalized prediction label (POS_LABEL or NEG_LABEL)
        - "elapsed_time": (float) Inference time per frame
        """
        pass

    @abstractmethod
    def get_gt(
        self, video_name: str, num_frames: int, pred_df: pd.DataFrame
    ) -> List[str]:
        """
        Returns a list of ground truth labels for the given video.
        Length of list must match num_frames.
        """
        pass
