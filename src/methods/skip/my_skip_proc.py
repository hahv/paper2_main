from halib import *  # noqa: F403
from typing import Tuple, Dict, Any
from src.methods.skip.base_skip_proc import BaseSkipProc
from src.config import Config
import cv2
import pywt


class SimpleFrameDifference:
    """
    A simple Frame Difference implementation.
    Logic: |Current_Frame - Previous_Frame| > Threshold
    """

    def __init__(self, diff_threshold):
        self.diff_threshold = diff_threshold
        self.prev_frame = None

    def apply(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.prev_frame is None:
            self.prev_frame = gray
            return np.zeros_like(gray)

        diff = cv2.absdiff(self.prev_frame, gray)
        _, fgmask = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        self.prev_frame = gray
        return fgmask


class BlockAnalyzer:
    """
    Analyzes specific blocks for Fire/Smoke signatures using Wavelets and Color.
    Ported from zrun_idea2.py.
    """

    def __init__(self, history_len=30):
        self.spatial_energy_history = {}
        self.background_energy_history = {}
        self.history_len = history_len

    def get_spatial_wavelet_energy(self, roi, channel="r"):
        try:
            if channel == "r":
                c_idx = 2
            elif channel == "g":
                c_idx = 1
            else:
                c_idx = 0

            gray = roi[:, :, c_idx]
            # Safety check for very small blocks
            if gray.shape[0] < 2 or gray.shape[1] < 2:
                return 0.0

            coeffs = pywt.dwt2(gray, "haar")
            cA, (cH, cV, cD) = coeffs
            energy = np.mean(np.square(cH) + np.square(cV) + np.square(cD))
            return energy
        except Exception:
            return 0.0


    def check_fire_candidate(self, roi, block_id, global_means):
        """
        Uses YCbCr Method 3 with Global Means.
        global_means: tuple (mean_Y, mean_Cr, mean_Cb) calculated from the whole frame.
        """
        # Paper 9:  T. W. Hsu, S. Pare, M. S. Meena, D. K. Jain, D. L. Li, A. Saxena, M. Prasad, and C. T. Lin, "An early flame detection system based on image block threshold selection using knowledge of local and global feature analysis" Sustainability, vol. 12, no. 21, p. 8899, 2020.
        # Unpack global means
        mean_Y, mean_Cr, mean_Cb = global_means

        # Convert Block to YCrCb
        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(ycrcb)

        # Parameters
        tau = 40  # Constant threshold for rule r9

        # --- Rules from Method 2 & 3 ---

        # r6: Y > Cb
        r6 = Y > Cb

        # r7: Cr > Cb
        r7 = Cr > Cb

        # r9: |Cb - Cr| >= tau
        diff_cb_cr = cv2.absdiff(Cb, Cr)
        r9 = diff_cb_cr >= tau

        # --- Split Logic (Method 3) ---

        # r10: (Y < 220) & (Y > Cb) & (Cr > Cb)
        # Note: (Y > Cb) & (Cr > Cb) is effectively (r6 & r7)
        r10 = (Y < 220) & r6 & r7

        # r11: (Y >= 220) & (Y > mean_Y) & (Cr > Cb)
        # Here we use the GLOBAL mean_Y passed in from the full frame
        r11 = (Y >= 220) & (Y > mean_Y) & r7

        # Final Combination: F(3) = r6 & r7 & r9 & (r10 | r11)
        fire_mask = r6 & r7 & r9 & (r10 | r11)

        # --- Verification ---
        fire_pixels = cv2.countNonZero(fire_mask.astype(int))
        pixel_count = roi.shape[0] * roi.shape[1]

        if pixel_count == 0:
            return False

        color_prob = fire_pixels / pixel_count

        if color_prob < 0.1:
            return False
        return True

        # energy = self.get_spatial_wavelet_energy(roi, channel="r")
        # if energy > 50.0:
        #     return True
        # return False

    def check_smoke_candidate(self, roi, block_id):
        # Paper: SMOKE DETECTION USING SPATIO-TEMPORAL ANALYSIS
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]
        color_mask = (s_channel < 0.28) & (v_channel > 108)
        smoke_pixels = cv2.countNonZero(color_mask.astype(int))
        pixel_count = roi.shape[0] * roi.shape[1]
        if pixel_count == 0:
            return False
        color_prob = smoke_pixels / pixel_count
        if color_prob < 0.1:
            return False
        return True


class MySkipProc(BaseSkipProc):
    def __init__(self, cfg: Config, params: dict):
        super().__init__(cfg, params)
        self.cfg = cfg
        self.params = params

        # Params
        self.block_size = params.get("block_size", 32)
        self.scale_factor = params.get("scale_factor", 1.0)
        self.diff_threshold = params.get("diff_threshold", 5)
        self.base_active_threshold = params.get("base_active_threshold", 0.05)

        # Components
        self.diff_engine = SimpleFrameDifference(diff_threshold=self.diff_threshold)
        self.analyzer = BlockAnalyzer()

    def _resize_and_pad(self, frame: np.ndarray) -> np.ndarray:
        if self.scale_factor != 1.0:
            scaled_frame = cv2.resize(
                frame,
                None,
                fx=self.scale_factor,
                fy=self.scale_factor,
                interpolation=cv2.INTER_AREA,
            )
        else:
            scaled_frame = frame

        H, W = scaled_frame.shape[:2]
        pad_h = (self.block_size - (H % self.block_size)) % self.block_size
        pad_w = (self.block_size - (W % self.block_size)) % self.block_size

        if pad_h > 0 or pad_w > 0:
            scaled_frame = cv2.copyMakeBorder(
                scaled_frame, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
        return scaled_frame

    def _get_active_blocks(self, fg_mask: np.ndarray) -> Tuple[np.ndarray, float]:
        H, W = fg_mask.shape
        blk_h, blk_w = H // self.block_size, W // self.block_size

        blocks = fg_mask.reshape(
            blk_h, self.block_size, blk_w, self.block_size
        ).swapaxes(1, 2)
        counts = (blocks > 0).sum(axis=(2, 3))
        percentages = counts / (self.block_size * self.block_size)

        avg_percentage = np.mean(percentages) if percentages.size > 0 else 0
        adapted_threshold = max(
            self.base_active_threshold,
            self.base_active_threshold + (avg_percentage * 0.1),
        )

        active_indices_2d = np.argwhere(percentages > adapted_threshold)
        return active_indices_2d, adapted_threshold

    def should_skip(
        self, frame_idx: int, frame: np.ndarray
    ) -> Tuple[bool, Dict[str, Any]]:
        # 1. Preprocessing & Motion
        scaled_frame = self._resize_and_pad(frame)
        fgmask = self.diff_engine.apply(scaled_frame)
        active_indices, adapt_thres = self._get_active_blocks(fgmask)

        # 2. Global Mean Calculation (Method 3 requirement)
        # Calculate means on the ENTIRE frame ("For a given image")
        # Convert to YCrCb once for mean calculation
        full_ycrcb = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2YCrCb)

        # cv2.mean returns (mean_ch1, mean_ch2, mean_ch3, 0)
        # OpenCV YCrCb: Ch1=Y, Ch2=Cr, Ch3=Cb
        means = cv2.mean(full_ycrcb)
        global_means = (means[0], means[1], means[2])  # (Y, Cr, Cb)

        # 2. Block Analysis (The logic you requested)
        has_fire_or_smoke = False

        # If no motion, we definitely skip
        if len(active_indices) == 0:
            should_skip = True
        else:
            # If there is motion, check if it looks like fire or smoke
            for r, c in active_indices:
                y1 = r * self.block_size
                y2 = y1 + self.block_size
                x1 = c * self.block_size
                x2 = x1 + self.block_size

                block_roi = scaled_frame[y1:y2, x1:x2]
                block_id = (r, c)

                # Check candidates
                is_fire = self.analyzer.check_fire_candidate(block_roi, block_id, global_means)
                is_smoke = self.analyzer.check_smoke_candidate(block_roi, block_id)
                # is_smoke = False  # Disable smoke detection for now

                if is_fire or is_smoke:
                    block_id_integer = (int(r), int(c))
                    pprint(f"Frame {frame_idx}: Detected {'Fire' if is_fire else 'Smoke'} in block {block_id_integer}")
                    has_fire_or_smoke = True
                    # Optimization: If we found one confirmed candidate, we can stop checking others
                    # and decide to PROCESS this frame.
                    break

            # If we found candidates, DO NOT skip (should_skip = False)
            # If we checked all blocks and found only random motion (no fire/smoke), SKIP (should_skip = True)
            should_skip = not has_fire_or_smoke

        meta_data = {
            "scaled_frame": scaled_frame,
            "motion_mask": fgmask,
            "active_blocks_indices": active_indices,
            "block_size": self.block_size,
            "scale_factor": self.scale_factor,
        }

        return should_skip, meta_data