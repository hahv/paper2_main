from halib import *
from typing import Optional
from src.metrics.loaders.base_csv_loader import BaseRawVideoCsvLoader
from pathlib import Path
from src.common import GlobalConstants


class DFireCsvLoader(BaseRawVideoCsvLoader):
    """
    Adapter for the DFire dataset where Ground Truth is inferred
    heuristically from the video filename (e.g., if "FP" is in name).
    """

    def get_gt_df(
        self, video_path: str, extra_data: Optional[dict] = None
    ) -> pd.DataFrame:
        num_frames: int = extra_data.get("num_frames")  # ty:ignore[possibly-missing-attribute, invalid-assignment]
        video_name = Path(video_path).name
        data_dict = {
            GlobalConstants.COL_VIDEO: [video_name] * num_frames,
            GlobalConstants.COL_VIDEO_PATH: [video_path] * num_frames,
            GlobalConstants.COL_FRAME_IDX: list(range(num_frames)),
        }
        video_label = (
            GlobalConstants.NO_SMOKE_LABEL
            if "FP" in video_name
            else GlobalConstants.FIRESMOKE_LABEL
        )
        data_dict[GlobalConstants.COL_GT] = [video_label] * num_frames
        gt_df = pd.DataFrame(data_dict)
        return gt_df
