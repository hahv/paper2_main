from halib import *

import cv2
import numpy as np
from typing import Any, Dict
from src.methods.skip.rule.base_rule import RuleResult
from src.results.viz.renderer_utils import RenderUtils
from src.results.viz.block_motion_only_renderer import BlockMontionOnlyRenderer

from line_profiler import profile

# ! Note that this only draws the grid, motion blocks will be draw in different renderer
class BlockRuleBasedRenderer(BlockMontionOnlyRenderer):
    """Draws the Grid, Yellow Motion Blocks, and Fire/Smoke Classification."""

    # COLOR for fire block: red, smoke block
    FIRE_BLOCK_COLOR = (0, 0, 255)  # red
    SMOKE_BLOCK_COLOR = (255, 0, 0)  # blue
    MOTION_BLOCK_COLOR = (0, 255, 255)  # yellow
    BLOCK_THICKNESS = 2

    def global_ctx_to_render_ctx(
        self, global_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Converts global context to renderer-specific context.
        global_context: contains all inference results (fps, fg_mask_dict, etc.)
        """
        return super().global_ctx_to_render_ctx(global_context)

    # @log_func(log_time=True)
    # @profile
    def render(self, frame_bgr: np.ndarray, renderer_ctx: Dict[str, Any]) -> np.ndarray:
        assert renderer_ctx is not None and len(renderer_ctx) > 0, (
            "Renderer context is empty!"
        )
        mt_proc = renderer_ctx["mt_proc"]
        block_info = mt_proc.get("block_info", [])

        # ! Adjust block size based on context (original vs resized)
        block_size = self.get_render_block_size(renderer_ctx)

        for block_item in block_info:
            block_id = block_item["block_id"]
            y_idx = block_id[0]
            x_idx = block_id[1]
            rule_dict: Dict[str, RuleResult] = block_item["rule_dict"]

            # FIX 1: Define x1, y1... even if scale_factor is 1.0
            # (Original code would crash here if scale_factor == 1.0)
            x1 = int(x_idx * block_size)
            x2 = int((x_idx + 1) * block_size)
            y1 = int(y_idx * block_size)
            y2 = int((y_idx + 1) * block_size)

            full_osd_dict = {}
            full_osd_cfg_dict = {}

            for rule_path, rule_rs in rule_dict.items():
                # FIX 2: Python strings use .lower(), not .to_lower()
                # ! rule_rs: RuleResult cls
                rule_rs_name = rule_rs.rule_name.lower()

                if "fire" in rule_rs_name:
                    color = self.FIRE_BLOCK_COLOR
                    cv2.rectangle(
                        frame_bgr,
                        (x1, y1),
                        (x2, y2),
                        color,
                        thickness=self.BLOCK_THICKNESS,
                    )
                elif "smoke" in rule_rs_name:
                    color = self.SMOKE_BLOCK_COLOR
                    cv2.rectangle(
                        frame_bgr,
                        (x1, y1),
                        (x2, y2),
                        color,
                        thickness=self.BLOCK_THICKNESS,
                    )
                else:
                    raise ValueError(
                        f"Unknown rule name in block visualization: {rule_rs_name}"
                    )

                rule_abbr = rule_rs.abbr_name()
                osd_dict = {rule_rs_name: rule_abbr}

                rule_rs_details = rule_rs.details
                # append rule_abbr to details keys to avoid collision (e.g., both fire and smoke rules have "threshold" key)
                rule_rs_details = {
                    f"{rule_abbr}_{k}": v for k, v in rule_rs_details.items()
                }

                osd_dict.update(rule_rs_details)
                full_osd_dict.update(osd_dict)

                # (alias, fmt, color, scale, thickness)
                # cv2.FONT_HERSHEY_COMPLEX_SMALL, cv2.FONT_HERSHEY_PLAIN, FONT_HERSHEY_SIMPLEX
                osd_cfg_dict = {
                    "font": cv2.FONT_HERSHEY_PLAIN,
                    "color": color,
                    "scale": 0.5,
                    "thickness": 1,
                }
                osd_cfg_dict = {key: osd_cfg_dict.copy() for key in osd_dict.keys()}
                assert rule_abbr not in full_osd_dict, (
                    "Duplicate rule abbreviation in OSD!"
                )
                full_osd_cfg_dict.update(osd_cfg_dict)

            if full_osd_dict:
                frame_bgr = RenderUtils.draw_osd_pil(
                    frame_bgr,
                    full_osd_dict,
                    config=full_osd_cfg_dict,
                    pos=(x1, y1),
                    bg_opacity=-1,  # no bg
                    padding=3,
                    line_spacing=2,
                    base_font_size=20,
                )

        return frame_bgr
