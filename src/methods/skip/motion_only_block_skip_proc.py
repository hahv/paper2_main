from halib import *  # noqa: F403
from typing import Tuple, Dict, Any

from src.config import Config
from src.methods.skip.base_block_skip_proc import BaseBlockSkipProc


# ! @Also see: src/methods/skip/__prof_skip_meta.md for further details.
class MotionOnlyBlockSkipProc(BaseBlockSkipProc):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.block_ratio_th = self.params.get("block_ratio_th")
        # self.min_roi_ratio = self.params.get("min_roi_ratio")

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

        meta_data = {
            "mt_proc": {
                "resized_frame": scaled_padded_frame,  # Scaled frame
                "fgmask_frame": fgmask,  # Scaled mask
                "block_info": block_info,
                # "crop_roi": crop_roi,  # Now in original coords
            }
        }
        return should_skip, meta_data