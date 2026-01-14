import cv2
import numpy as np
from .base_rule import *
from typing import Any, Dict, Optional


class FireBlockYCbCrRule(BaseRule):
    def check(
        self,
        frame_or_roi: np.ndarray,
        extra_dict: Optional[Dict[str, Any]] = None,
    ) -> RuleResult:
        # Simplified Method 3 Check
        ycrcb = cv2.cvtColor(frame_or_roi, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(ycrcb)

        # Fire Rule: Y > Cb AND Cr > Cb AND High Luminance
        fire_mask = (Y > Cb) & (Cr > Cb) & (Y > 200)

        total_pixels = fire_mask.size
        if total_pixels == 0:
            return RuleResult(self.name, RuleStatus.FAIL, {"percent": 0.0})

        percent_fire = cv2.countNonZero(fire_mask.astype(int)) / total_pixels
        block_active_thres = self.params.get("block_active_thres", 0.05)

        status = (
            RuleStatus.PASS if percent_fire > block_active_thres else RuleStatus.FAIL
        )

        return RuleResult(
            rule_name=self.name,
            status=status,
            details={
                "percent_fire": percent_fire,
                "threshold": block_active_thres,
                "msg": f"Fire pixel ratio {percent_fire:.2f} > {block_active_thres}",
            },
        )


class FireBlockLowEnergyRule(BaseRule):
    """Checks if the block has Low Spatial Energy (Blurry)."""

    def check(
        self,
        frame_or_roi: np.ndarray,
        extra_dict: Optional[Dict[str, Any]] = None,
    ) -> RuleResult:
        gray = cv2.cvtColor(frame_or_roi, cv2.COLOR_BGR2GRAY)

        # Calculate Laplacian Variance (Measure of texture/edges)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        var_thres = self.params.get("fire_var_thres", 100)

        # Pass if variance is LOW (Blurry)
        status = RuleStatus.PASS if laplacian_var < var_thres else RuleStatus.FAIL

        return RuleResult(
            rule_name=self.name,
            status=status,
            details={
                "laplacian_var": laplacian_var,
                "threshold": var_thres,
                "msg": f"Texture Energy {laplacian_var:.1f} < {var_thres}",
            },
        )


class SmokeBlockHSVRule(BaseRule):
    def check(
        self,
        frame_or_roi: np.ndarray,
        extra_dict: Optional[Dict[str, Any]] = None,
    ) -> RuleResult:
        hsv = cv2.cvtColor(frame_or_roi, cv2.COLOR_BGR2HSV)

        # Smoke Rule: Low Saturation (<72) AND High Value (>108)
        mask = (hsv[:, :, 1] < 72) & (hsv[:, :, 2] > 108)

        total_pixels = mask.size
        if total_pixels == 0:
            return RuleResult(self.name, RuleStatus.FAIL, {"percent": 0.0})

        percent_smoke = cv2.countNonZero(mask.astype(int)) / total_pixels
        block_active_thres = self.params.get("block_active_thres", 0.1)

        status = (
            RuleStatus.PASS if percent_smoke > block_active_thres else RuleStatus.FAIL
        )

        return RuleResult(
            rule_name=self.name,
            status=status,
            details={
                "percent_smoke": percent_smoke,
                "threshold": block_active_thres,
                "msg": f"Smoke pixel ratio {percent_smoke:.2f} > {block_active_thres}",
            },
        )


class SmokeBlockHighVarRule(BaseRule):
    """Checks if the block has High Variance (Turbulence/Contrast)."""

    def check(
        self,
        frame_or_roi: np.ndarray,
        extra_dict: Optional[Dict[str, Any]] = None,
    ) -> RuleResult:
        # Calculate Standard Deviation (Contrast/Variance)
        mean, stddev = cv2.meanStdDev(frame_or_roi)
        std_val = stddev[0][0]  # Take first channel or simplify

        var_thres = self.params.get("smoke_var_thres", 40)

        # Pass if Variance is HIGH
        status = RuleStatus.PASS if std_val > var_thres else RuleStatus.FAIL

        return RuleResult(
            rule_name=self.name,
            status=status,
            details={
                "std_dev": std_val,
                "threshold": var_thres,
                "msg": f"StdDev {std_val:.1f} > {var_thres}",
            },
        )
