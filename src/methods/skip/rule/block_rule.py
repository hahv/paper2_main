import cv2
import numpy as np
from .base_rule import *
from typing import Any, Dict, Optional
import pywt


class FireBlockYCbCrRule(BlockBasedRule):
    """
    Advanced Fire Detection using YCbCr Method 3 with Global Means.
    Ported from BlockAnalyzer.check_fire_candidate.

    Requires 'global_means' (tuple: mean_Y, mean_Cr, mean_Cb) in extra_dict.
    """

    def __init__(self, name: str = "", params: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, params=params)
        self.tau = self.params["rule_params"].get("tau", 40)
        self.block_active_thresh = self.params["rule_params"].get(
            "block_active_thres", 0.1
        )

    def check(
        self,
        frame_or_roi: np.ndarray,
        extra_dict: Optional[Dict[str, Any]] = None,
    ) -> RuleResult:
        details = self.prepare(extra_dict)
        # 1. Retrieve Global Means from context (Calculated from full frame)
        if extra_dict is None or "global_means" not in extra_dict:
            return RuleResult(
                self.name,
                RuleStatus.FAIL,
                {"msg": "Missing global_means in extra_dict"},
            )

        global_means = extra_dict[
            "global_means"
        ]  # Expecting (mean_Y, mean_Cr, mean_Cb)
        mean_Y, mean_Cr, mean_Cb = global_means

        # 2. Convert Block to YCbCr
        ycrcb = cv2.cvtColor(frame_or_roi, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(ycrcb)

        # 4. Apply Rules (Vectorized)

        # r6: Y > Cb
        r6 = Y > Cb

        # r7: Cr > Cb
        r7 = Cr > Cb

        # r9: |Cb - Cr| >= tau
        diff_cb_cr = cv2.absdiff(Cb, Cr)
        r9 = diff_cb_cr >= self.tau

        # Split Logic (Method 3)
        # r10: (Y < 220) & r6 & r7
        r10 = (Y < 220) & r6 & r7

        # r11: (Y >= 220) & (Y > GLOBAL_MEAN_Y) & r7
        r11 = (Y >= 220) & (Y > mean_Y) & r7

        # Final Combination: F(3) = r6 & r7 & r9 & (r10 | r11)
        fire_mask = r6 & r7 & r9 & (r10 | r11)

        # 5. Verification
        total_pixels = frame_or_roi.shape[0] * frame_or_roi.shape[1]
        if total_pixels == 0:
            return RuleResult(self.name, RuleStatus.FAIL, {"percent": 0.0})

        fire_pixel_count = cv2.countNonZero(fire_mask.astype(int))
        percent_fire = fire_pixel_count / total_pixels

        status = (
            RuleStatus.PASS
            if percent_fire > self.block_active_thresh
            else RuleStatus.FAIL
        )

        details.update(
            {
                "percent_fire": percent_fire,
                "threshold": self.block_active_thresh,
                "global_mean_Y": float(mean_Y),
                "msg": f"Method3 Fire Ratio {percent_fire:.2f} > {self.block_active_thresh}",
            }
        )

        return RuleResult(rule_name=self.name, status=status, details=details)


class FireBlockWaveletRule(BlockBasedRule):
    """
    Analyzes block texture/frequency using Haar Wavelets.
    Ported from BlockAnalyzer.get_spatial_wavelet_energy.
    """

    def __init__(self, name: str = "", params: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, params=params)
        self.channel_name = self.params.get("rule_params", {}).get(
            "wavelet_channel", "r"
        )
        self.energy_thres = self.params.get("rule_params", {}).get(
            "wavelet_energy_thres", 50.0
        )

    def check(
        self,
        frame_or_roi: np.ndarray,
        extra_dict: Optional[Dict[str, Any]] = None,
    ) -> RuleResult:
        details = self.prepare(extra_dict)
        # Parameters

        # Determine channel index (BGR format)
        if self.channel_name == "r":
            c_idx = 2
        elif self.channel_name == "g":
            c_idx = 1
        else:
            c_idx = 0

        try:
            # Extract single channel for analysis
            gray = frame_or_roi[:, :, c_idx]

            # Safety check for very small blocks (DWT requires minimum size)
            if gray.shape[0] < 2 or gray.shape[1] < 2:
                return RuleResult(
                    self.name, RuleStatus.FAIL, {"msg": "Block too small"}
                )

            # Perform 2D Discrete Wavelet Transform
            coeffs = pywt.dwt2(gray, "haar")
            cA, (cH, cV, cD) = coeffs

            # Calculate High-Frequency Energy
            # Energy = Mean of squares of Detail coefficients (Horizontal, Vertical, Diagonal)
            energy = np.mean(np.square(cH) + np.square(cV) + np.square(cD))

            # Decide status based on threshold
            # Note: High energy usually implies complex texture (like fire edges).
            # Adjust logic if you want to detect 'smooth' core (Low energy).
            status = RuleStatus.PASS if energy > self.energy_thres else RuleStatus.FAIL

            details.update(
                {
                    "energy": float(energy),
                    "threshold": self.energy_thres,
                    "msg": f"Wavelet Energy {energy:.2f} > {self.energy_thres}",
                }
            )

            return RuleResult(
                rule_name=self.name,
                status=status,
                details=details,
            )

        except Exception as e:
            # Fallback for mathematical/shape errors
            return RuleResult(
                self.name, RuleStatus.FAIL, {"msg": f"Wavelet Error: {str(e)}"}
            )


class SmokeBlockSpatioTemporalRule(BlockBasedRule):
    """
    Smoke detection logic based on HSV Saturation and Value.
    Ported from BlockAnalyzer.check_smoke_candidate.
    """

    def check(
        self,
        frame_or_roi: np.ndarray,
        extra_dict: Optional[Dict[str, Any]] = None,
    ) -> RuleResult:
        details = self.prepare(extra_dict)
        # Convert to HSV
        hsv = cv2.cvtColor(frame_or_roi, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]

        # Parameters
        # Note: Original code used s < 0.28.
        # In OpenCV uint8 (0-255), 0.28 maps to approx 71.4 (0.28 * 255).
        # We default to 72 to match standard integer processing.
        s_thres_max = self.params.get("smoke_sat_max", 72)
        v_thres_min = self.params.get("smoke_val_min", 108)
        block_active_thres = self.params.get("block_active_thres", 0.1)

        # Rule: Low Saturation (Grayish) AND High Value (Bright-ish)
        color_mask = (s_channel < s_thres_max) & (v_channel > v_thres_min)

        # Verification
        total_pixels = frame_or_roi.shape[0] * frame_or_roi.shape[1]
        if total_pixels == 0:
            return RuleResult(self.name, RuleStatus.FAIL, {"percent": 0.0})

        smoke_pixels = cv2.countNonZero(color_mask.astype(int))
        percent_smoke = smoke_pixels / total_pixels

        status = (
            RuleStatus.PASS if percent_smoke > block_active_thres else RuleStatus.FAIL
        )
        details.update(
            {
                "percent_smoke": percent_smoke,
                "threshold": block_active_thres,
                "msg": f"Smoke HSV Ratio {percent_smoke:.2f} > {block_active_thres}",
            }
        )

        return RuleResult(
            rule_name=self.name,
            status=status,
            details=details,
        )
