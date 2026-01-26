from halib import *
import cv2
from src.config import Config
from typing import Dict, Any
from src.results.video_block_skip_rs_proc import VideoBlockSkipRsProc


class FgmaskBlockSkipRsProc(VideoBlockSkipRsProc):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.out_video_postfix = "fgmask_out"

    def get_block_extra_dict(self) -> Dict[str, Any]:
        return {"viz_in_fgmask": True}

    def get_vis_frame(self, frame_bgr, frame_rs_dict: dict):
        mt_proc = frame_rs_dict["infer_rs"]["mt_proc"]
        fg_mask = mt_proc.get("motion_mask_frame")
        # If the mask is single channel (H, W), convert to (H, W, 3)
        if len(fg_mask.shape) == 2:
            fg_mask_bgr = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        else:
            fg_mask_bgr = fg_mask
        return fg_mask_bgr
