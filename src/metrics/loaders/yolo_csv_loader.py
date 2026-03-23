import os
import pandas as pd
from pathlib import Path
from typing import Optional

from src.common import GlobalConst
from src.metrics.loaders.base_csv_loader import BaseRawCsvLoader

# Class names in YOLO output that map to the "firesmoke" label
_FIRE_SMOKE_KEYWORDS = ("fire", "smoke")


def _any_fire_smoke(cls_series: pd.Series) -> bool:
    return cls_series.apply(
        lambda x: any(kw in str(x).lower() for kw in _FIRE_SMOKE_KEYWORDS)
    ).any()  # ty:ignore[invalid-return-type]


class YoloExternalLoader:
    """
    Loads YOLO object-detection _od.csv from an external experiment directory and
    merges with real GT from the dataset directory.

    YOLO OD CSV format (sparse — only frames WITH detections have rows; can be all-header/empty):
        video;frame_id;pred_id;cls;cls_conf;x1;y1;x2;y2;elapsed_time
        - 'frame_id'  → frame index (renamed to frame_idx)
        - 'cls'       → detected class name (e.g. "Fire", "Smoke")
        - Multiple rows per frame (one per detected bounding box)
        - video name has NO file extension (e.g. "aihub__lb_fire__0016")

    Expansion logic:
        - Frames present in OD CSV: any detection with fire/smoke → "firesmoke", else "none"
        - Frames absent from OD CSV (no detection): "none"
        - Full frame range is derived from the GT CSV (not from video file reading)

    Normalized output:
        video;video_path;frame_idx;gt_label;pred_label;elapsed_time
    """

    OD_SUFFIX = "_od"

    def __init__(self, exp_dir: str, gt_pattern: str = GlobalConst.GT_FILE_PATTERN):
        self.exp_dir = Path(exp_dir)
        self.gt_pattern = gt_pattern

    def _find_od_csv(self, video_path: str) -> Optional[str]:
        stem = Path(video_path).stem
        od_csv = self.exp_dir / f"{stem}{self.OD_SUFFIX}.csv"
        return str(od_csv) if od_csv.exists() else None

    def _expand_od_to_per_frame(
        self, od_df: pd.DataFrame, full_frame_idx: pd.Series
    ) -> pd.DataFrame:
        """
        Converts a sparse OD DataFrame into a dense per-frame prediction DataFrame.

        - Multiple detections per frame are aggregated: any fire/smoke class → "firesmoke"
        - Frames missing from OD CSV (no detection) → "none"
        - elapsed_time per frame: max across all detections for that frame (0.0 if absent)
        """
        if od_df.empty:
            return pd.DataFrame(
                {
                    GlobalConst.COL_FRAME_IDX: full_frame_idx,
                    GlobalConst.COL_PRED: GlobalConst.NONE_LABEL,
                    GlobalConst.COL_ELAPSED_TIME: 0.0,
                }
            )

        # Rename frame_id → frame_idx to match the pipeline's column name convention
        od_df = od_df.rename(columns={"frame_id": GlobalConst.COL_FRAME_IDX})

        # Aggregate per frame: label and elapsed_time
        agg_rows = []
        for frame_idx, group in od_df.groupby(GlobalConst.COL_FRAME_IDX):
            label = (
                GlobalConst.FIRESMOKE_LABEL
                if _any_fire_smoke(group["cls"])
                else GlobalConst.NONE_LABEL
            )
            elapsed = float(group[GlobalConst.COL_ELAPSED_TIME].max())
            agg_rows.append(
                {
                    GlobalConst.COL_FRAME_IDX: frame_idx,
                    GlobalConst.COL_PRED: label,
                    GlobalConst.COL_ELAPSED_TIME: elapsed,
                }
            )
        frame_agg = pd.DataFrame(agg_rows)

        # Left-join onto the full GT-derived frame range;
        # frames with no detections get "none" and elapsed_time 0.0
        full_df = pd.DataFrame({GlobalConst.COL_FRAME_IDX: full_frame_idx})
        merged = full_df.merge(frame_agg, on=GlobalConst.COL_FRAME_IDX, how="left")
        merged[GlobalConst.COL_PRED] = merged[GlobalConst.COL_PRED].fillna(
            GlobalConst.NONE_LABEL
        )
        merged[GlobalConst.COL_ELAPSED_TIME] = merged[
            GlobalConst.COL_ELAPSED_TIME
        ].fillna(0.0)
        return merged

    def load_video_gt_pred_df(self, video_path: str) -> pd.DataFrame:
        """
        Returns a standard merged DataFrame with columns:
            [video, video_path, frame_idx, gt_label, pred_label, elapsed_time]
        """
        # 1. Load real GT — also gives us the canonical full frame index range
        gt_df = BaseRawCsvLoader.load_csv_by_pattern(
            video_path=video_path,
            csv_pattern=self.gt_pattern,
            is_gt=True,
        )

        # 2. Load YOLO OD CSV (may have only a header line = empty detections)
        od_csv = self._find_od_csv(video_path)
        if od_csv is None:
            raise FileNotFoundError(
                f"No '{self.OD_SUFFIX}.csv' for '{Path(video_path).stem}' in {self.exp_dir}"
            )

        od_df = pd.read_csv(od_csv, sep=";", encoding="utf-8", keep_default_na=False)

        # 3. Expand sparse OD detections → dense per-frame predictions
        full_frame_idx = gt_df[GlobalConst.COL_FRAME_IDX].reset_index(drop=True)
        pred_df = self._expand_od_to_per_frame(od_df, full_frame_idx)

        # 4. Normalize video name/path using the actual video file as source of truth
        #    (YOLO CSVs omit the extension; this ensures a clean match with GT)
        video_name = Path(video_path).name
        pred_df[GlobalConst.COL_VIDEO] = video_name
        pred_df[GlobalConst.COL_VIDEO_PATH] = str(os.path.abspath(video_path))

        # 5. Merge GT + Pred on [video, video_path, frame_idx]
        return BaseRawCsvLoader._merge_gt_pred_dfs(gt_df, pred_df, video_path)
