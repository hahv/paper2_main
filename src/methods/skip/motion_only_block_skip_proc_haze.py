from halib import *  # noqa: F403
from typing import Tuple, Dict, Any
import cv2
from src.config import Config
from src.methods.skip.base_block_skip_proc import BaseBlockSkipProc


# ! @Also see: src/methods/skip/__prof_skip_meta.md for further details.
class MotionOnlyBlockSkipProcHaze(BaseBlockSkipProc):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.block_ratio_th = self.params.get("block_ratio_th")
        self.haze_size_width = self.params.get("haze_size_width", 120)
        self.tau_dark = self.params.get("tau_dark", 0.5)
        # self.min_roi_ratio = self.params.get("min_roi_ratio")

    # ---------------------------------------------------------------------- #
    # Haze Detection (Dark Channel Prior)
    # Copied from test_dark.py: detect_haze_traditional_dcp
    # Returns a score in [0.0, 1.0]. Higher = more haze/whitish content.
    # ---------------------------------------------------------------------- #
    @staticmethod
    def _detect_haze_dcp(
        img: np.ndarray, small_width: int = 120
    ) -> float:
        """
        Traditional Dark Channel Prior haze score.
          1. Downsample to small_size for speed (~0.3-0.5 ms)
          2. Pixel-wise min across B, G, R channels
          3. Patch-wise min via 5x5 erosion
          4. Return mean score normalised to [0.0, 1.0]
        """
        aspect_ratio = img.shape[1] / img.shape[0]
        small_height = int(small_width / aspect_ratio)
        small_size = (small_width, small_height)
        small = cv2.resize(img, small_size, interpolation=cv2.INTER_NEAREST)
        pixel_min = np.min(small, axis=2)
        kernel = np.ones((5, 5), np.uint8)
        dark_channel = cv2.erode(pixel_min, kernel)
        return float(np.mean(dark_channel) / 255.0)

    # ! Update gaze with haze
    def should_skip(
        self, frame_idx: int, frame: np.ndarray
    ) -> Tuple[bool, Dict[str, Any]]:
        original_h, original_w = frame.shape[:2]

        # Resize and pad (returns frame in SCALED space)
        scaled_padded_frame = self.resize_and_pad(frame)

        # 1. Motion Detection (Performed in SCALED space)
        fgmask = self.motion_det.apply(scaled_padded_frame)

        # 2. Handle Non-Divisible Dimensions
        H_scaled, W_scaled = fgmask.shape
        B = self.block_size

        blk_h = H_scaled // B
        blk_w = W_scaled // B

        try:
            # Reshape to (GridRows, BlockHeight, GridCols, BlockWidth) -> Swap -> Sum
            blocks = fgmask.reshape(blk_h, B, blk_w, B).swapaxes(1, 2)
            counts = (blocks > 0).sum(axis=(2, 3))
        except ValueError:
            counts = np.zeros((blk_h, blk_w))

        # 4. Filter Active Blocks
        total_pixels_per_block = self.block_size * self.block_size
        active_mask = counts / total_pixels_per_block >= self.block_ratio_th
        active_indices = np.argwhere(active_mask)

        # --- COLLECT BLOCK INFO ---
        block_info = []
        for r, c in active_indices:
            # pixel_count = int(counts[r, c])
            percent_pixels = counts[r, c] / total_pixels_per_block
            block_info.append(
                {"block_id": (int(r), int(c)), "percent_active_pixels": percent_pixels}
            )
        # --------------------------

        has_motion = len(active_indices) > 0
        should_skip = not has_motion

        # 5. Haze Gate (only runs when motion gate says skip)
        #    If tau_dark is set and the frame looks hazy -> override skip to False
        haze_score = None
        haze_triggered = False

        if should_skip and self.tau_dark is not None:
            haze_score = self._detect_haze_dcp(frame)
            haze_triggered = haze_score >= self.tau_dark
            if haze_triggered:
                should_skip = False  # rescue frame: force DL inference

        meta_data = {
            "mt_proc": {
                "resized_frame": scaled_padded_frame,  # Scaled frame
                "fgmask_frame": fgmask,  # Scaled mask
                "block_info": block_info,
                "haze_score": haze_score,
                "haze_triggered": haze_triggered,
            }
        }
        return should_skip, meta_data