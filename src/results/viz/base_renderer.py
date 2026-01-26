from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class BaseRenderer(ABC):
    """Abstract base class for any visualization step."""

    def __init__(self, context = "original_frame", extra_cfg: Optional[Dict[str, Any]] = None):
        self.context = context
        assert self.context in ["original_frame", "resized_frame"], (
            f"Invalid context: {self.context}. "
            "Must be 'original_frame' or 'resized_frame'."
        )
        self.extra_cfg = extra_cfg if extra_cfg is not None else {}

    @abstractmethod
    def global_ctx_to_render_ctx(
        self, global_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Converts global context to renderer-specific context.
        global_context: contains all inference results (fps, fg_mask_dict, etc.)
        """
        pass

    @abstractmethod
    def render(self, frame_bgr: np.ndarray, renderer_ctx: Dict[str, Any]) -> np.ndarray:
        """
        Draws on the frame and returns it.
        context: contains all inference results (fps, fg_mask_dict, etc.)
        """
        pass
