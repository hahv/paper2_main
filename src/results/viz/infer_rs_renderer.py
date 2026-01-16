import numpy as np
from typing import Any, Dict

from src.results.viz.base_renderer import BaseRenderer
from src.results.viz.renderer_utils import RenderUtils
from src.utils import filter_dict_by_keys


class InferRsRenderer(BaseRenderer):
    def global_ctx_to_render_ctx(
        self, global_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        renderer_dict = filter_dict_by_keys(global_context, keys=["fps"])
        renderer_dict["frame_idx"] = (
            f"{global_context['frame_idx']}/{global_context['num_frames']}"
        )
        infer_rs = global_context["infer_rs"]
        probs = infer_rs["probs"]
        labelIdx = infer_rs["predLabelIdx"]
        predLabel = infer_rs["predLabel"]
        pred_str = f"{predLabel} ({probs[labelIdx] * 100:.1f}%)"
        renderer_dict["-Pred: "] = pred_str
        return renderer_dict

    """Draws the semi-transparent black box with Text Info (FPS, Frame #)."""

    def render(self, frame_bgr: np.ndarray, renderer_ctx: Dict[str, Any]) -> np.ndarray:
        return RenderUtils.draw_osd(frame=frame_bgr, data=renderer_ctx)
