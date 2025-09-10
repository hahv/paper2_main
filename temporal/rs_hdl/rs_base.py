from abc import ABC, abstractmethod
import os
import cv2
from temporal.config import *
from halib.system import filesys as fs
from halib.filetype import csvfile
import torch
import torch.nn.functional as F
from collections import OrderedDict


class BaseRSHandler(ABC):
    """Abstract base class for handling inference results."""

    def before_video(
        self,
        video_path: str,
        **kwargs,
    ):
        """Called once before processing a video."""
        pass

    @abstractmethod
    def handle_frame_results(self, frame_bgr, frame_rs_dict: dict) -> Any:
        """Called for each frame to handle its results."""
        pass

    def after_video(self):
        """Called once after processing a video is complete."""
        pass
