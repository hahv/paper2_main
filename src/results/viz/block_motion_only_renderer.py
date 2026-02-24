from halib import *

import cv2
import numpy as np
from typing import Any, Dict
from src.results.viz.grid_renderer import GridRenderer
from src.results.viz.renderer_utils import RenderUtils
from line_profiler import profile


class BlockMontionOnlyRenderer(GridRenderer):
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

        render_ctx = {
            "block_size_orig": mt_cfg["block_size_orig"],
            "scale_factor": mt_cfg["scale_factor"],
            "mt_cfg": mt_cfg,
            "mt_proc": mt_proc,
        }
        if self.extra_cfg:
            render_ctx.update(self.extra_cfg)
        return render_ctx

    # @log_func(log_time=True)
    # @profile
    def render(self, frame_bgr: np.ndarray, renderer_ctx: Dict[str, Any]) -> np.ndarray:
        # Basic validation
        if not renderer_ctx:
            return frame_bgr
        mt_cfg = renderer_ctx["mt_cfg"]
        scale_factor = mt_cfg["scale_factor"]
        mt_proc = renderer_ctx.get("mt_proc", {})

        block_size = self.get_render_block_size(renderer_ctx)
        block_info = mt_proc.get("block_info", [])
        crop_roi = mt_proc.get("crop_roi", None)
        if crop_roi is not None and self.context == "resized_frame":
            # Adjust crop_roi to resized frame space
            rx, ry, rw, rh = crop_roi
            rx = int(rx * scale_factor)
            ry = int(ry * scale_factor)
            rw = int(rw * scale_factor)
            rh = int(rh * scale_factor)
            crop_roi = (rx, ry, rw, rh)

        H, W = frame_bgr.shape[:2]
        step = block_size
        # 3. Draw Active Blocks
        for block_item in block_info:
            block_id = block_item["block_id"]
            y_idx, x_idx = block_id  # (row, col)
            percent_active_pixels = block_item["percent_active_pixels"]
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

            # Direction 2: draw block percentage text with cv2.putText (no PIL roundtrip per block)
            text = f"p:{percent_active_pixels:.1%}"
            cv2.putText(
                frame_bgr,
                text,
                (x1 + 4, y1 + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                self.MOTION_BLOCK_COLOR,
                1,
                cv2.LINE_AA,
            )

        # 4. Draw Crop ROI (Aggregated Bounding Box)
        if crop_roi:
            rx, ry, rw, rh = crop_roi
            # Draw Blue Box
            cv2.rectangle(frame_bgr, (rx, ry), (rx + rw, ry + rh), self.ROI_COLOR, 2)

            # Direction 1: use draw_osd (OpenCV) instead of draw_osd_pil for ROI label
            frame_bgr = RenderUtils.draw_osd(
                frame=frame_bgr,
                data={"ROI": f"{rw} x {rh}"},
                config={"ROI": {"color": self.ROI_COLOR}},
                pos=(rx, ry + 20),
                bg_opacity=0.5,
            )

        return frame_bgr
