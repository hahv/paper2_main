from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np


class BaseRenderer(ABC):
    """Abstract base class for any visualization step."""

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
