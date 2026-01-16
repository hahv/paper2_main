from src.viz.renderers.base_renderer import BaseRenderer
import cv2
from typing import Any, Dict
import numpy as np


class MotionGridRenderer(BaseRenderer):
    """Draws the Grid, Yellow Motion Blocks, and Fire/Smoke Classification."""

    def render(self, frame_bgr: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        fg_mask_dict = context.get("fg_mask_dict")
        if not fg_mask_dict:
            return frame_bgr  # Skip if no mask data available

        vis = frame_bgr.copy()

        block_size = fg_mask_dict.get("block_size", 32)
        active_blocks = fg_mask_dict.get("active_motion_blocks_info", [])
        firesmoke_info = fg_mask_dict.get("firesmoke_blocks_cls_info", {})
        roi_rect = fg_mask_dict.get("ROI_rect", None)

        H, W = vis.shape[:2]

        # 1. Grid
        for y in range(0, H, block_size):
            cv2.line(vis, (0, y), (W, y), (50, 50, 50), 1)
        for x in range(0, W, block_size):
            cv2.line(vis, (x, 0), (x, H), (50, 50, 50), 1)

        # 2. Active Blocks (Yellow)
        for idx, percent in active_blocks:
            by, bx = divmod(idx, W // block_size)
            x0, y0 = bx * block_size, by * block_size
            cv2.rectangle(
                vis, (x0, y0), (x0 + block_size, y0 + block_size), (0, 255, 255), 2
            )
            cv2.putText(
                vis,
                f"{percent:.1f}",
                (x0 + 2, y0 + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

        # 3. Fire/Smoke (Red/Green)
        all_active = firesmoke_info.get("all_active_blocks", [])
        firesmoke_active = set(firesmoke_info.get("firesmoke_active_blocks", []))

        for idx, prob in all_active:
            by, bx = divmod(idx, W // block_size)
            x0, y0 = bx * block_size, by * block_size
            color = (0, 0, 255) if idx in firesmoke_active else (0, 255, 0)
            cv2.rectangle(vis, (x0, y0), (x0 + block_size, y0 + block_size), color, 2)

        # 4. ROI
        if roi_rect:
            x, y, w, h = roi_rect  # Check if rect is xyxy or xywh format in your logic
            cv2.rectangle(vis, (x, y), (w, h), (255, 0, 0), 2)

        return vis
