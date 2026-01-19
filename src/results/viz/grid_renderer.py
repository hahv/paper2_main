from halib import *

import cv2
import numpy as np
from typing import Any, Dict
from src.utils import filter_dict_by_keys
from src.results.viz.base_renderer import BaseRenderer
from src.results.viz.renderer_utils import RenderUtils
# ! Note that this only draws the grid, motion blocks will be draw in different renderer
class GridRenderer(BaseRenderer):
    """Draws the Grid, Yellow Motion Blocks, and Fire/Smoke Classification."""

    def global_ctx_to_render_ctx(
        self, global_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Converts global context to renderer-specific context.
        global_context: contains all inference results (fps, fg_mask_dict, etc.)
        """
        # with ConsoleLog("global ctx"):
        #     pprint(global_context)
        # assert False, "Debug stop"
        mt_cfg =  global_context["infer_rs"]["mt_cfg"]["params"]
        render_ctx = filter_dict_by_keys(mt_cfg, ["scale_factor", "block_size"])
        return render_ctx

    def render(self, frame_bgr: np.ndarray, renderer_ctx: Dict[str, Any]) -> np.ndarray:
        assert renderer_ctx is not None and len(renderer_ctx) > 0, "Renderer context is empty!"
        H, W = frame_bgr.shape[:2]

        block_size = renderer_ctx['block_size']
        # ! render_ctx: scale_factor, block_size
        box_h, box_w = RenderUtils.calculate_osd_box(
            renderer_ctx,
        )
        START_Y = 30
        START_X = W - box_w - 20
        frame_bgr = RenderUtils.draw_osd(
            frame_bgr, renderer_ctx, pos=(START_X, START_Y)
        )
        for y in range(0, H, block_size):
            cv2.line(frame_bgr, (0, y), (W, y), (50, 50, 50), 1)
        for x in range(0, W, block_size):
            cv2.line(frame_bgr, (x, 0), (x, H), (50, 50, 50), 1)

        return frame_bgr
