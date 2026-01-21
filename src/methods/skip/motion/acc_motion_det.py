import cv2
import numpy as np
from typing import Optional, Dict, Any
from src.methods.skip.motion.base_motion_det import BaseMotionDet


# ! see @doc AccMotionDet in src/methods/skip/motion/acc_motion_det.md
class AccMotionDet(BaseMotionDet):
    """
    Accumulation Motion Detector (formerly TemporalMotionDetector).
    Port of C++ FireDetector::temporalStabilization logic.
    Implements Accumulation + Decay for robust motion detection.
    """

    def __init__(self, name: str, params: Dict[str, Any]):
        super().__init__(name, params)

        # Configuration (Defaults matched to original C++ logic)
        self.diff_frame_th = params.get(
            "diff_frame_th", 1
        )  # Sensitivity to pixel change
        self.impact_plus_one = params.get(
            "impact_plus_one", 5
        )  # Weight added per change
        self.mask_th = params.get("mask_th", 10)  # Threshold to activate motion
        self.max_val = params.get("max_val", 25)  # Cap for accumulation
        self.decay = params.get("decay", 1)  # Decay per frame

        # State
        self.prev_frame = None
        self.delta_mask = None

    def apply(
        self,
        frame_bgr: np.ndarray,
        extra_dict: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ) -> np.ndarray:
        # 1. Convert to Gray
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Lazy Initialization of state buffers
        if self.prev_frame is None or self.delta_mask is None:
            self.prev_frame = gray.copy()
            self.delta_mask = np.zeros_like(gray, dtype=np.uint8)
            return np.zeros_like(gray)

        # 2. Frame Difference (absdiff)
        delta = cv2.absdiff(self.prev_frame, gray)

        # 3. Threshold Difference
        # C++: threshold(delta, delta, DIFF_FRAME_TH, IMPACK_PLUS_ONE, THRESH_BINARY);
        # Result: Pixels > 1 become 5, others 0
        _, binary_delta = cv2.threshold(
            delta, self.diff_frame_th, self.impact_plus_one, cv2.THRESH_BINARY
        )

        # 4. Accumulation (deltaMask = deltaMask + delta)
        # We use cv2.add to ensure saturation logic (though max is small here)
        self.delta_mask = cv2.add(self.delta_mask, binary_delta)

        # 5. Cap values (cv::min(deltaMask, 25, deltaMask))
        # cv2.threshold with THRESH_TRUNC caps values at MAX_VAL
        _, self.delta_mask = cv2.threshold(
            self.delta_mask, self.max_val, self.max_val, cv2.THRESH_TRUNC
        )

        # 6. Decay (subtract(deltaMask, 1, deltaMask))
        # Using cv2.subtract handles underflow (0 - 1 = 0) automatically for uint8
        self.delta_mask = cv2.subtract(self.delta_mask, self.decay)  # ty:ignore[no-matching-overload]

        # 7. Generate Current Mask (compare(deltaMask, MASK_TH, curMask, cv::CMP_GE))
        _, cur_mask = cv2.threshold(
            self.delta_mask, self.mask_th, 255, cv2.THRESH_BINARY
        )

        # Update previous frame
        self.prev_frame = gray.copy()

        return cur_mask

    def reset(self):
        """Clears the accumulation buffer and previous frame."""
        self.prev_frame = None
        self.delta_mask = None
