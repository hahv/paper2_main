import cv2
from halib import *  # noqa: F403
from typing import Tuple, Dict, Any

from src.config import Config
from src.methods.skip.rule.base_rule import *
from src.methods.skip.rule.block_rule import *
from src.methods.skip.base_skip_proc import BaseSkipProc


class BlockSkipProc(BaseSkipProc):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.scale_factor: float = self.params.get("scale_factor", 1.0)
        self.block_size: int = self.params.get("block_size", 32)
        self.block_active_thresh: float = self.params.get("block_active_thresh", 0.1)
        self.update_rules()

    def update_rules(self):
        fire_complex = AllRule(
            [
                FireBlockYCbCrRule(params=self.params),
                FireBlockWaveletRule(params=self.params),
            ],
            name="FireCheck",
        )
        smoke_rule = SmokeBlockSpatioTemporalRule(params=self.params, name="SmokeCheck")
        self.rules = AnyRule([fire_complex, smoke_rule], name="FireOrSmokeCheck")

    @staticmethod
    def get_skip_proc_frame_size(
        frame_w_h: tuple, scale_factor: float, block_size: int
    ) -> tuple:
        """
        Calculates the final shape (H, W) after resizing and padding to block_size.
        mimics logic of _resize_and_pad without processing the image.
        """
        assert len(frame_w_h) == 2, "frame_w_h must be a tuple of (width, height)"

        w, h = frame_w_h[:2]

        # 1. Calculate Scaled Dimensions (OpenCV uses round() for fx/fy)
        if scale_factor != 1.0:
            new_h = int(round(h * scale_factor))
            new_w = int(round(w * scale_factor))
        else:
            new_h, new_w = h, w

        # 2. Calculate Padding (matches your formula)
        # (block - (dim % block)) % block ensures 0 padding if already divisible
        pad_h = (block_size - (new_h % block_size)) % block_size
        pad_w = (block_size - (new_w % block_size)) % block_size

        # 3. Add Padding
        final_h = new_h + pad_h
        final_w = new_w + pad_w
        return (final_w, final_h)

    # input frames will be first resized based on scale_factor, then padded to be divisible by block_size
    def _resize_and_pad(self, frame: np.ndarray) -> np.ndarray:
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
        pad_h = (self.block_size - (H % self.block_size)) % self.block_size
        pad_w = (self.block_size - (W % self.block_size)) % self.block_size

        if pad_h > 0 or pad_w > 0:
            scaled_frame = cv2.copyMakeBorder(
                scaled_frame, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
        return scaled_frame

    def _get_active_blocks(self, fg_mask: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Vectorized calculation of active blocks using NumPy reshaping.
        Assumes fg_mask is binary (0 or 255).
        """
        # 1. Get Dimensions
        # Input Shape: (768, 1024)
        H, W = fg_mask.shape
        B = self.block_size  # 32

        # 2. Calculate Grid Dimensions (Number of blocks)
        # blk_h = 768 // 32 = 24
        # blk_w = 1024 // 32 = 32
        blk_h, blk_w = H // B, W // B

        # 3. Create 4D Block View (Vectorization)
        # Read: https://sparrow.dev/numpy-reshape/
        # ---------------------------------------------------------
        # Step A: .reshape(blk_h, B, blk_w, B)
        #   Splits image into strips.
        #   Intermediate Shape: (24, 32, 32, 32) -> (Grid_Rows, Block_Height, Grid_Cols, Block_Width)
        #
        # Step B: .swapaxes(1, 2)
        #   Groups the block width/height dimensions together at the end.
        #   Final Shape: (24, 32, 32, 32) -> (Grid_Rows, Grid_Cols, Block_Height, Block_Width)
        # ---------------------------------------------------------
        blocks = fg_mask.reshape(blk_h, B, blk_w, B).swapaxes(1, 2)

        # 4. Count Active Pixels
        # We sum over axis 2 and 3 (the 32x32 pixels inside each block).
        # (blocks > 0) creates a massive boolean mask of shape (24, 32, 32, 32).
        # .sum(axis=(2,3)) collapses the last two dimensions.
        # Result Shape: (24, 32) -> Grid of integer counts (0 to 1024).
        counts = (blocks > 0).sum(axis=(2, 3))

        # 5. Calculate Motion Percentage
        # Divide counts by total pixels in a block (32*32 = 1024).
        # Result Shape: (24, 32) -> Grid of floats (0.0 to 1.0).
        percentages = counts / (B * B)

        # 6. Adaptive Threshold Logic
        # Calculate the average motion of the ENTIRE frame.
        # Value: Scalar Float (e.g., 0.02 if 2% of the frame is moving).
        avg_percentage = np.mean(percentages) if percentages.size > 0 else 0.0

        # Calculate dynamic threshold to ignore environmental noise (wind/rain).
        # e.g., max(0.05, 0.05 + (0.02 * 0.1)) = 0.052
        adapted_thresh = max(
            self.block_active_thresh,
            self.block_active_thresh + (avg_percentage * 0.1),
        )

        # 7. Filter Active Blocks
        # Find coordinates where the percentage exceeds the threshold.
        # np.argwhere returns a list of [row, col] indices.
        # Result Shape: (N, 2) where N is the number of active blocks.
        active_indices_2d = np.argwhere(percentages > adapted_thresh)

        return active_indices_2d, adapted_thresh

    def should_skip(
        self, frame_idx: int, frame: np.ndarray
    ) -> Tuple[bool, Dict[str, Any]]:
        # 1. Preprocessing & Motion
        scaled_frame = self._resize_and_pad(frame)
        fgmask = self.motion_det.apply(scaled_frame)
        active_indices, adapt_thres = self._get_active_blocks(fgmask)

        # 2. Global Mean Calculation (Method 3 requirement)
        # Calculate means on the ENTIRE frame ("For a given image")
        # Convert to YCrCb once for mean calculation
        full_ycrcb = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2YCrCb)

        # cv2.mean returns (mean_ch1, mean_ch2, mean_ch3, 0)
        # OpenCV YCrCb: Ch1=Y, Ch2=Cr, Ch3=Cb
        means = cv2.mean(full_ycrcb)
        global_means = (means[0], means[1], means[2])  # (Y, Cr, Cb)

        # 2. Block Analysis (The logic you requested)
        has_fire_or_smoke = False

        # We will collect detailed logs for debugging
        block_info = []
        if len(active_indices) > 0:
            for r, c in active_indices:
                y1 = r * self.block_size
                y2 = y1 + self.block_size
                x1 = c * self.block_size
                x2 = x1 + self.block_size
                block_roi = scaled_frame[y1:y2, x1:x2]
                block_id = (int(r), int(c))
                console.print(f"[red] Analyzing Block {block_id}... [/red]", end="\r")

                # --- RUN CHECK ---
                result = self.rules.check(
                    block_roi, extra_dict={"global_means": global_means}
                )

                # Store result for visualization/debugging if needed
                if result.is_pass():
                    has_fire_or_smoke = True
                    rule_dict = BaseRule.collect_leaf_results(result)
                    block_info.append({"block_id": block_id, "rule_dict": rule_dict})

        should_skip = not has_fire_or_smoke

        meta_data = {
            "mt_proc": {
                "vis_frame": scaled_frame,
                "motion_mask_frame": fgmask,
                "block_info": block_info,
            }
        }
        return should_skip, meta_data

