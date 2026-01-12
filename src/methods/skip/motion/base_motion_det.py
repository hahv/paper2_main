from abc import ABC, abstractmethod
import numpy as np
class BaseMotionDet(ABC):
    """
    Interface for any background subtraction / motion detection logic.
    """
    def __init__(self, params):
        self.params = params

    @abstractmethod
    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Returns a binary foreground mask (255=motion, 0=static)."""
        pass
