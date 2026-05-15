from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np
from src.utils import get_cls_in_pkg


class MotionDetFactory:
    @staticmethod
    def create_method(name: str, params: Dict[str, Any]) -> "BaseMotionDet":
        cls = get_cls_in_pkg(
            pkg_name=f"src.methods.skip.motion",
            fileName_ClsName=name,
        )
        kwargs = {"name": name, "params": params}
        return cls(**kwargs)


class BaseMotionDet(ABC):
    """
    Interface for any background subtraction / motion detection logic.
    """

    def __init__(self, name: str, params: Dict[str, Any]):
        self.name = name
        self.params = params

    @abstractmethod
    def apply(
        self,
        frame_bgr: np.ndarray,
        extra_dict: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ) -> np.ndarray:
        """Returns a binary foreground mask (255=motion, 0=static)."""
        pass

    def peek(self, frame_bgr: np.ndarray) -> None:
        """
        Advance internal state for a frame that is NOT being evaluated
        for motion (e.g., EAGER mode). Default: no-op.
        Override in detectors that maintain temporal state.
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset any internal state."""
        pass
