import cv2
import numpy as np
from typing import Any, Dict, Optional
from .base_rule import *
import pywt
import cv2


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
