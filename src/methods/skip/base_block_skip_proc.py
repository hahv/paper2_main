import cv2
from halib import *  # noqa: F403
from src.config import Config
from src.methods.skip.base_skip_proc import BaseSkipProc


class BaseBlockSkipProc(BaseSkipProc):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.scale_factor: float = self.params.get("scale_factor", 1.0)
        # Original/effective block size before any scaling
        self.block_size_orig: int = self.params.get("block_size_orig")
        # Block size using in the scaled frame (after padding_and_resizing)
        self.block_size = int(self.block_size_orig * self.scale_factor)

    # input frames will be first resized based on scale_factor, then padded to be divisible by block_size
    def resize_and_pad(self, frame: np.ndarray) -> np.ndarray:
        if self.scale_factor != 1.0:
            scaled_frame = cv2.resize(
                frame,
                None,
                fx=self.scale_factor,
                fy=self.scale_factor,
                interpolation=cv2.INTER_AREA,
            )
        else:
            scaled_frame = frame

        H, W = scaled_frame.shape[:2]
        # padding calculation (using the block_size_scaled)
        pad_h = (self.block_size - (H % self.block_size)) % self.block_size
        pad_w = (self.block_size - (W % self.block_size)) % self.block_size

        if pad_h > 0 or pad_w > 0:
            scaled_frame = cv2.copyMakeBorder(
                scaled_frame, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
        return scaled_frame
