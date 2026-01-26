from halib import *  # noqa: F403
from typing import Tuple, Dict, Any

from src.config import Config
from src.methods.skip.base_block_skip import BaseBlockSkipProc

# ! @Also see: src/methods/skip/__prof_skip_meta.md for further details.
class MotionOnlyBlockSkipProc(BaseBlockSkipProc):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.block_ratio_th = self.params.get("block_ratio_th")
        self.min_roi_ratio = self.params.get("min_roi_ratio")

    def should_skip(
        self, frame_idx: int, frame: np.ndarray
    ) -> Tuple[bool, Dict[str, Any]]:
        original_h, original_w = frame.shape[:2]

        # Resize and pad (returns frame in SCALED space)
        scaled_padded_frame = self.resize_and_pad(frame)

        # 1. Motion Detection (Performed in SCALED space)
        scaled_padded_fgmask = self.motion_det.apply(scaled_padded_frame)

        # 2. Handle Non-Divisible Dimensions
        H_scaled, W_scaled = scaled_padded_fgmask.shape
        B = self.block_size

        blk_h = H_scaled // B
        blk_w = W_scaled // B

        try:
            # Reshape to (GridRows, BlockHeight, GridCols, BlockWidth) -> Swap -> Sum
            blocks = scaled_padded_fgmask.reshape(blk_h, B, blk_w, B).swapaxes(1, 2)
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
            pixel_count = int(counts[r, c])
            block_info.append(
                {"block_id": (int(r), int(c)), "active_pixels": pixel_count}
            )
        # --------------------------

        has_motion = len(active_indices) > 0
        crop_roi = None

        if has_motion:
            # 5. Find Bounding Box (in Grid/Scaled Space)
            y_min_grid, x_min_grid = active_indices.min(axis=0)
            y_max_grid, x_max_grid = active_indices.max(axis=0)

            # 5a. Convert Grid -> Pixel Coordinates (SCALED Space)
            x1_p = x_min_grid * B
            y1_p = y_min_grid * B
            x2_p = (x_max_grid + 1) * B
            y2_p = (y_max_grid + 1) * B

            # 5b. Transform to ORIGINAL Frame Space
            # Formula: Original = Scaled / scale_factor
            x1 = int(x1_p / self.scale_factor)
            y1 = int(y1_p / self.scale_factor)
            x2 = int(x2_p / self.scale_factor)
            y2 = int(y2_p / self.scale_factor)

            # Recalculate raw width/height in original space
            curr_w = x2 - x1
            curr_h = y2 - y1

            # 6. Enforce Minimum ROI Size (Based on ORIGINAL Frame)
            # Use self.min_roi_ratio instead of hardcoded 0.75
            req_w = int(original_w * self.min_roi_ratio)
            req_h = int(original_h * self.min_roi_ratio)

            # Center Expansion Logic
            if curr_w < req_w:
                diff = req_w - curr_w
                x1 = max(0, x1 - diff // 2)
                x2 = x1 + req_w

            if curr_h < req_h:
                diff = req_h - curr_h
                y1 = max(0, y1 - diff // 2)
                y2 = y1 + req_h

            # 7. Clamp to Original Frame Size
            # Ensure we don't return coordinates outside the real image
            x1 = max(0, min(x1, original_w))
            y1 = max(0, min(y1, original_h))
            x2 = max(0, min(x2, original_w))
            y2 = max(0, min(y2, original_h))

            # Recalculate Final Width/Height
            final_w = x2 - x1
            final_h = y2 - y1

            if final_w > 0 and final_h > 0:
                crop_roi = (int(x1), int(y1), int(final_w), int(final_h))  # xywh
            else:
                has_motion = False

        should_skip = not has_motion

        meta_data = {
            "mt_proc": {
                "vis_frame": frame,  # Original frame for viz
                "block_size_in_original_space": int(B / self.scale_factor),
                "motion_mask_frame": scaled_padded_fgmask,  # Scaled mask
                "block_info": block_info,
                "crop_roi": crop_roi,  # Now in original coords
            }
        }
        return should_skip, meta_data

    def prepare_infer_input(
        self, frame: np.ndarray, meta_data: Dict[str, Any]
    ) -> np.ndarray:
        """
        Crops the frame based on the ROI calculated in should_skip.
        Returns the cropped numpy array.
        """
        # 1. Get ROI from metadata (format: xywh)
        roi = meta_data.get("crop_roi")

        if roi:
            x, y, w, h = roi
            H, W = frame.shape[:2]

            # 2. Safety Clamping (Crucial step)
            # Ensure start coordinates are within bounds
            x = max(0, min(x, W - 1))
            y = max(0, min(y, H - 1))

            # Ensure width/height don't overflow the frame
            # Example: if x=10, w=100, W=100 -> valid w is 90
            w = max(1, min(w, W - x))
            h = max(1, min(h, H - y))

            # 3. Perform Crop
            # NumPy slicing format: frame[y_start : y_end, x_start : x_end]
            crop = frame[y : y + h, x : x + w]

            # Final sanity check: if crop is empty, return original
            if crop.size == 0:
                return frame

            return crop

        # Fallback: return full frame if no ROI (e.g. should_skip=True or initialization)
        return frame
