from halib import *  # noqa: F403
from typing import Tuple, Dict, Any

from src.config import Config
from src.methods.skip.base_block_skip_proc import BaseBlockSkipProc


# ! @Also see: src/methods/skip/__prof_skip_meta.md for further details.
class MotionOnlyBlockSkipProcEager(BaseBlockSkipProc):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.block_ratio_th = self.params.get("block_ratio_th")
        # Eager mode params
        self.n_chk: int = self.params.get("n_chk", 30)  # forced check interval
        self.w_clr: int = self.params.get("w_clr", 10)  # eager exit window

        # Eager mode state — start True (skipping must be earned)
        self.eager_mode: bool = True
        self.non_firemmoke_streak: int = 0
        self.skip_streak: int = 0
        # self.min_roi_ratio = self.params.get("min_roi_ratio")

    def should_skip(
        self, frame_idx: int, frame: np.ndarray
    ) -> Tuple[bool, Dict[str, Any]]:
        original_h, original_w = frame.shape[:2]
        scaled_padded_frame = self.resize_and_pad(frame)

        # ── Step 2: Eager Mode Override ───────────────────────────────────────
        if self.eager_mode:
            meta_data = {
                "mt_proc": {
                    "resized_frame": scaled_padded_frame,
                    "fgmask_frame": None,
                    "block_info": [],
                    "is_forced_check": False,
                    "eager_mode": True,
                }
            }
            return False, meta_data  # never skip in eager mode

        # ── Step 3: Periodic Forced Check ─────────────────────────────────────
        if self.skip_streak >= self.n_chk:
            self.skip_streak = 0
            meta_data = {
                "mt_proc": {
                    "resized_frame": scaled_padded_frame,
                    "fgmask_frame": None,
                    "block_info": [],
                    "is_forced_check": True,  # flag for TempMethod
                    "eager_mode": False,
                }
            }
            return False, meta_data  # force DL, motion gate overridden

        # ── Step 4: Compute Foreground Mask (normal motion gate) ──────────────
        fgmask = self.motion_det.apply(scaled_padded_frame)

        H_scaled, W_scaled = fgmask.shape
        B = self.block_size
        blk_h = H_scaled // B
        blk_w = W_scaled // B

        try:
            blocks = fgmask.reshape(blk_h, B, blk_w, B).swapaxes(1, 2)
            counts = (blocks > 0).sum(axis=(2, 3))
        except ValueError:
            counts = np.zeros((blk_h, blk_w))

        total_pixels_per_block = self.block_size * self.block_size
        active_mask = counts / total_pixels_per_block >= self.block_ratio_th
        active_indices = np.argwhere(active_mask)

        block_info = []
        for r, c in active_indices:
            percent_pixels = counts[r, c] / total_pixels_per_block
            block_info.append(
                {"block_id": (int(r), int(c)), "percent_active_pixels": percent_pixels}
            )

        has_motion = len(active_indices) > 0

        # ── Step 5: Gate Decision ─────────────────────────────────────────────
        if has_motion:
            self.skip_streak = 0
        else:
            self.skip_streak += 1

        should_skip = not has_motion
        meta_data = {
            "mt_proc": {
                "resized_frame": scaled_padded_frame,
                "fgmask_frame": fgmask,
                "block_info": block_info,
                "is_forced_check": False,
                "eager_mode": False,
            }
        }
        return should_skip, meta_data

    def update_eager_state(self, pred_label: str) -> None:
        """Called by TempMethod after every DL inference result is known."""
        fire_smoke_label = ["Fire", "SmokeOnly"]  # match your cfg class name
        fire_smoke_label = [
            lbl.lower() for lbl in fire_smoke_label
        ]  # case-insensitive match

        if pred_label.lower() in fire_smoke_label:
            # Fire detected — enter/stay in eager mode, reset clear counter
            self.eager_mode = True
            self.no_fire_streak = 0
        else:
            # No fire — count toward eager exit
            if self.eager_mode:
                self.no_fire_streak += 1
                if self.no_fire_streak >= self.w_clr:
                    self.eager_mode = False
                    self.no_fire_streak = 0
                    self.skip_streak = 0  # fresh start after eager exit

    def reset(self):
        """Reset all state at video boundary."""
        super().reset()  # resets motion_det
        self.eager_mode = True  # start each video in eager mode
        self.no_fire_streak = 0
        self.skip_streak = 0
