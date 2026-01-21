from halib import *

import cv2
import numpy as np
from typing import Any, Dict
from src.results.viz.base_renderer import BaseRenderer


class BlockProfRenderer(BaseRenderer):
    """Draws the Grid, Yellow Motion Blocks, and Fire/Smoke Classification."""

    MOTION_BLOCK_COLOR = (0, 255, 255)  # yellow
    BLOCK_THICKNESS = 2

    ROI_COLOR = (0, 0, 255)  # red
    ROI_THICKNESS = 4

    def global_ctx_to_render_ctx(
        self, global_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Converts global context to renderer-specific context.
        global_context: contains all inference results (fps, fg_mask_dict, etc.)
        """
        mt_cfg = global_context["infer_rs"]["mt_cfg"]["params"]
        mt_proc = global_context["infer_rs"]["mt_proc"]
        return {"mt_cfg": mt_cfg, "mt_proc": mt_proc}

    def render(self, frame_bgr: np.ndarray, renderer_ctx: Dict[str, Any]) -> np.ndarray:
        # Basic validation
        if not renderer_ctx:
            return frame_bgr

        mt_cfg = renderer_ctx.get("mt_cfg", {})
        mt_proc = renderer_ctx.get("mt_proc", {})

        block_size = mt_cfg.get("block_size", 32)
        block_info = mt_proc.get("block_info", [])
        crop_roi = mt_proc.get("crop_roi", None)

        H, W = frame_bgr.shape[:2]

        step = block_size
        # pprint(f"BlockProfRenderer: Drawing blocks with size {block_size} on frame {W}x{H}")
        # pprint(f'BlockProfRenderer: Number of active blocks: {len(block_info)}')
        # pprint(f'BlockProfRenderer: Crop ROI: {crop_roi}')

        # 3. Draw Active Blocks
        for block_item in block_info:
            block_id = block_item["block_id"]
            y_idx, x_idx = block_id  # (row, col)
            num_active_pixels = block_item["active_pixels"]
            # Map grid index to pixel coordinates
            x1 = x_idx * step
            y1 = y_idx * step
            x2 = x1 + step
            y2 = y1 + step

            # Clamp to frame boundaries
            x2 = min(W, x2)
            y2 = min(H, y2)

            # Draw Yellow Box
            cv2.rectangle(
                frame_bgr,
                (x1, y1),
                (x2, y2),
                self.MOTION_BLOCK_COLOR,
                self.BLOCK_THICKNESS,
            )

            # Draw Pixel Count Text
            # Position text slightly inside the top-left corner of the block
            text_pos = (x1 + 4, y1 + 14)
            cv2.putText(
                frame_bgr,
                str(num_active_pixels),
                text_pos,
                cv2.FONT_HERSHEY_PLAIN,
                0.8,  # Small font scale
                self.MOTION_BLOCK_COLOR,
                1,
                cv2.LINE_AA,
            )

        # 4. Draw Crop ROI (Aggregated Bounding Box)
        if crop_roi:
            rx, ry, rw, rh = crop_roi

            # Draw Blue Box
            cv2.rectangle(frame_bgr, (rx, ry), (rx + rw, ry + rh), self.ROI_COLOR, 2)

            # Draw Label
            cv2.putText(
                frame_bgr,
                f"ROI {rw}x{rh}",
                (rx, ry - 5),  # Slightly above the box
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.ROI_COLOR,
                1,
                cv2.LINE_AA,
            )

        return frame_bgr
