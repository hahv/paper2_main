import cv2
import numpy as np
from src.methods.skip.motion.base_motion_det import BaseMotionDet
class SimpleFrameDifference(BaseMotionDet):
    """
    Standard Frame Difference implementation.
    """

    def __init__(self, params):
        super().__init__(params)
        self.diff_threshold = params.get("diff_threshold", 30)
        self.prev_frame = None

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.prev_frame is None:
            self.prev_frame = gray
            return np.zeros_like(gray)

        diff = cv2.absdiff(self.prev_frame, gray)
        _, fgmask = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        self.prev_frame = gray
        return fgmask