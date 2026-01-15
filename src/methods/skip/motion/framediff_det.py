import cv2
import numpy as np
from typing import Optional, Dict, Any
from src.methods.skip.motion.base_motion_det import BaseMotionDet


class FrameDiffDet(BaseMotionDet):
    """
    Standard Frame Difference implementation.
    """

    def __init__(self, name: str, params: Dict[str, Any]):
        super().__init__(name, params)
        self.diff_threshold = params.get("diff_thresh", 30)
        self.prev_frame = None

    def apply(
        self,
        frame_bgr: np.ndarray,
        extra_dict: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.prev_frame is None:
            self.prev_frame = gray
            return np.zeros_like(gray)
        # print(f"Prev: {self.prev_frame.shape}, Curr: {gray.shape}")
        diff = cv2.absdiff(self.prev_frame, gray)
        _, fgmask = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        self.prev_frame = gray
        return fgmask

    def reset(self):
        self.prev_frame = None
