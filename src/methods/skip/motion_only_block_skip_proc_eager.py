# ===============src/methods/skip/motion_only_block_skip_proc_eager.py===============#
from halib import *  # noqa: F403
from typing import Tuple, Dict, Any

from src.config import Config
from src.methods.skip.base_block_skip_proc import BaseBlockSkipProc


class MotionOnlyBlockSkipProcEager(BaseBlockSkipProc):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.block_ratio_th = self.params.get("block_ratio_th")

        # --- FSM Parameters ---
        # N_chk: max skipped frames before burst trigger
        self.n_chk: int = self.params.get("n_chk", 30)
        # W_clr: consecutive safe frames needed to exit EAGER
        self.w_clr: int = self.params.get("w_clr", 10)
        # W_fire: consecutive positives needed to enter EAGER;
        #         ALSO the burst window size (c_burst ← W_fire)
        self.w_fire: int = self.params.get("fire_confirm_k", 2)

        # --- FSM State (start in EAGER — skipping must be earned) ---
        self.eager_mode: bool = True

        # c_fire: consecutive positive detections (NORMAL mode)
        self.c_fire: int = 0
        # c_clear: consecutive safe frames (EAGER mode)
        self.c_clear: int = 0
        # c_skip: consecutive skipped frames (NORMAL mode)
        self.c_skip: int = 0
        # c_burst: remaining forced-inference frames (NORMAL mode)
        self.c_burst: int = 0

    def should_skip(
        self, frame_idx: int, frame: np.ndarray
    ) -> Tuple[bool, Dict[str, Any]]:
        scaled_padded_frame = self.resize_and_pad(frame)

        # ── EAGER mode: bypass skip module entirely ───────────────────────────
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
            return False, meta_data  # never skip in EAGER

        # ── NORMAL mode ───────────────────────────────────────────────────────

        # ── Burst trigger: c_skip >= N_chk → set c_burst ← W_fire ───────────
        # (zero-inference step per FSM: handled before computing motion)
        if self.c_skip >= self.n_chk:
            self.c_skip = 0
            self.c_burst = self.w_fire  # arm the burst window

        # ── Burst active OR motion check ──────────────────────────────────────
        if self.c_burst > 0:
            # Force inference — motion gate overridden
            # c_burst decremented in update_eager_state after DL result
            meta_data = {
                "mt_proc": {
                    "resized_frame": scaled_padded_frame,
                    "fgmask_frame": None,
                    "block_info": [],
                    "is_forced_check": True,
                    "eager_mode": False,
                }
            }
            return False, meta_data

        # ── Normal motion gate ────────────────────────────────────────────────
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

        if has_motion:
            # Motion detected (s_t = 1): run inference, reset c_skip
            # c_fire updated in update_eager_state after DL result
            self.c_skip = 0
        else:
            # No motion (s_t = 0): skip, increment c_skip, RESET c_fire
            self.c_skip += 1
            self.c_fire = 0  # ← FSM: skip loop resets c_fire

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
        """
        Called by TempMethod after every DL inference result is known.
        Implements the FSM update for both EAGER and NORMAL modes.
        """
        fire_smoke_labels = {"fire", "smokeonly"}
        is_fire = pred_label.lower() in fire_smoke_labels  # ŷ_t = 1 or 0

        if self.eager_mode:
            # ── EAGER mode updates ────────────────────────────────────────────
            if is_fire:
                # ŷ_t = 1: reset c_clear
                self.c_clear = 0
            else:
                # ŷ_t = 0: increment c_clear
                if self.c_clear < self.w_clr:
                    self.c_clear += 1
                # T1: EAGER → NORMAL when c_clear >= W_clr
                if self.c_clear >= self.w_clr:
                    self.eager_mode = False
                    self.c_fire = 0  # reset per FSM T1
                    self.c_skip = 0  # reset per FSM T1
                    self.c_clear = 0  # clean slate

        else:
            # ── NORMAL mode updates (inference ran: s_t=1 or burst active) ───
            # Decrement burst countdown
            if self.c_burst > 0:
                self.c_burst = max(0, self.c_burst - 1)

            # Update c_fire
            if is_fire:
                self.c_fire += 1
            else:
                self.c_fire = 0  # single non-fire resets streak

            # T2: NORMAL → EAGER when c_fire >= W_fire
            if self.c_fire >= self.w_fire:
                self.eager_mode = True
                self.c_clear = 0  # reset per FSM T2

    def reset(self):
        """Reset all FSM state at video boundary."""
        super().reset()  # resets motion_det
        self.eager_mode = True  # start each video in EAGER
        self.c_fire = 0
        self.c_clear = 0
        self.c_skip = 0
        self.c_burst = 0
