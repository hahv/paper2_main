from halib import *  # noqa: F403
from typing import Tuple, Dict, Any

import cv2
from src.config import Config
from src.methods.skip.base_skip_proc import BaseSkipProc


class ProfSkipProc(BaseSkipProc):
    def __init__(self, cfg: Config):
        super().__init__(cfg)  # call BaseSkipProc init
        self.scale_factor: float = self.params.get("scale_factor", 1.0)
        self.block_size: int = self.params.get("block_size", 32)
        self.block_roi_th = self.params.get("block_roi_th", 200)  # ROI_TH

    def should_skip(
        self, frame_idx: int, frame: np.ndarray
    ) -> Tuple[bool, Dict[str, Any]]:
        # 1. Motion Detection
        fgmask = self.motion_det.apply(frame)

        # 2. Handle Non-Divisible Dimensions (Padding Strategy)
        H, W = fgmask.shape
        B = self.block_size

        # Calculate padding needed
        pad_h = (B - (H % B)) % B
        pad_w = (B - (W % B)) % B

        if pad_h > 0 or pad_w > 0:
            # Pad Right and Bottom with 0 (Black/No Motion)
            padded_mask = cv2.copyMakeBorder(
                fgmask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
            )  # ty:ignore[no-matching-overload]
        else:
            padded_mask = fgmask

        # 3. Vectorized Block Analysis
        # Now dimensions are perfectly divisible by B
        H_pad, W_pad = padded_mask.shape
        blk_h = H_pad // B
        blk_w = W_pad // B

        try:
            # Reshape to (GridRows, BlockHeight, GridCols, BlockWidth) -> Swap -> Sum
            blocks = padded_mask.reshape(blk_h, B, blk_w, B).swapaxes(1, 2)
            counts = (blocks > 0).sum(axis=(2, 3))
        except ValueError:
            counts = np.zeros((blk_h, blk_w))

        # 4. Filter Active Blocks
        active_mask = counts > self.block_roi_th
        active_indices = np.argwhere(active_mask)

        # --- COLLECT BLOCK INFO ---
        block_info = []
        for r, c in active_indices:
            # Get the exact number of active pixels for this block
            pixel_count = int(counts[r, c])
            block_info.append(
                {"block_id": (int(r), int(c)), "active_pixels": pixel_count}
            )
        # --------------------------

        has_motion = len(active_indices) > 0
        crop_roi = None

        if has_motion:
            # 5. Find Bounding Box (Grid Coords)
            y_min_grid, x_min_grid = active_indices.min(axis=0)
            y_max_grid, x_max_grid = active_indices.max(axis=0)

            # Convert Grid -> Pixel Coordinates (Padded Frame Space)
            x1 = x_min_grid * B
            y1 = y_min_grid * B
            x2 = (x_max_grid + 1) * B
            y2 = (y_max_grid + 1) * B

            curr_w = x2 - x1
            curr_h = y2 - y1

            # 6. Enforce Minimum ROI Size (C++ Logic)
            req_w = int(W * 0.75)
            req_h = int(H * 0.75)

            # Center Expansion Logic
            if curr_w < req_w:
                diff = req_w - curr_w
                x1 = max(0, x1 - diff // 2)
                x2 = x1 + req_w

            if curr_h < req_h:
                diff = req_h - curr_h
                y1 = max(0, y1 - diff // 2)
                y2 = y1 + req_h

            # 7. Clamp to Original Frame Size (Ignore Padding)
            # Ensure we don't return coordinates outside the real image
            x1 = max(0, min(x1, W))
            y1 = max(0, min(y1, H))
            x2 = max(0, min(x2, W))
            y2 = max(0, min(y2, H))

            # Recalculate Width/Height after clamping
            final_w = x2 - x1
            final_h = y2 - y1

            # Only valid if we have a positive area
            if final_w > 0 and final_h > 0:
                crop_roi = (int(x1), int(y1), int(final_w), int(final_h))  # xywh
            else:
                has_motion = False  # Should be rare, but safe

        should_skip = not has_motion
        # pprint(f'{frame.shape=}, fgmask={fgmask.shape}, pad_h={pad_h}, pad_w={pad_w}, active_blocks={len(active_indices)}, should_skip={should_skip}')
        meta_data = {
            "mt_proc": {
                "vis_frame": frame,
                "motion_mask_frame": fgmask,
                "block_info": block_info,
                "crop_roi": crop_roi,
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
