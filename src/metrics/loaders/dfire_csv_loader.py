from halib import *
from typing import Optional
from src.metrics.loaders.base_csv_loader import BaseRawVideoCsvLoader
from pathlib import Path


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
            "video": [video_name] * num_frames,
            "video_path": [video_path] * num_frames,
            "frame_idx": list(range(num_frames)),
        }
        video_label = (
            BaseRawVideoCsvLoader.NONE_LABEL
            if "FP" in video_name
            else BaseRawVideoCsvLoader.FIRESMOKE_LABEL
        )
        data_dict[self.COL_GT] = [video_label] * num_frames
        gt_df = pd.DataFrame(data_dict)
        return gt_df
