# ===============src/methods/skip/motion_only_block_skip_proc_eager.py===============#
from halib import *  # noqa: F403
from typing import Tuple, Dict, Any

from src.config import Config
from src.methods.skip.base_block_skip_proc import BaseBlockSkipProc


class MotionOnlyBlockSkipProcEager(BaseBlockSkipProc):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.block_ratio_th = self.params.get("block_ratio_th")

        self.eager_mode: bool = True
        # ! generator.cpp - line 491:
        # cfg.fdWindowSize = 16; //32
        # cfg.fdPeriod = 16;  // check period in the idle mode

        self.fd_period: int = self.params.get("fd_period", 16) # fixed at 16
        self.window_size: int = self.params.get("window_size", 16) # fixed 16
        self.reliable_th = self.params.get("reliable_th", 0.7)

        self.fd_cnt: int = 0
        self.fire_hist = [0] * self.window_size
        self.smoke_hist = [0] * self.window_size

    def should_skip(
        self, frame_idx: int, frame: np.ndarray
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        C++-style eager mode + temporal (motion) gating.

        - If eager_mode == False:
            * Run inference only once every fd_period frames (pure frame-count gate).
        - If eager_mode == True:
            * Frame is eligible every time, but still passes through motion-only block skip.
        """

        # 1) Resize + pad input frame (same as before)
        scaled_padded_frame = self.resize_and_pad(frame)

        # 2) C++-style eager gate: when eager_mode is OFF, use a pure frame-period gate
        if not self.eager_mode:
            if self.motion_det is not None:
                self.motion_det.peek(scaled_padded_frame)  # ← ADD THIS
            # increment frame counter
            self.fd_cnt += 1

            # if we haven't reached fd_period yet, skip entirely (no motion check)
            if self.fd_cnt < self.fd_period:
                meta_data = {
                    "mt_proc": {
                        "resized_frame": scaled_padded_frame,
                        "fgmask_frame": None,
                        "block_info": [],
                        "is_forced_check": False,
                        "eager_mode": False,
                    }
                }
                return True, meta_data  # skip DL inference

            # when we hit fd_period, reset counter and allow one inference
            self.fd_cnt = 0

        # 3) Motion gate (temporal stabilization analogue)
        #    If eager_mode is True we always come here.
        #    If eager_mode is False we come here only once per fd_period.
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
                {
                    "block_id": (int(r), int(c)),
                    "percent_active_pixels": float(percent_pixels),
                }
            )

        has_motion = len(active_indices) > 0

        # 4) Final decision: if no motion → skip; if motion → run DL
        should_skip_frame = not has_motion

        meta_data = {
            "mt_proc": {
                "resized_frame": scaled_padded_frame,
                "fgmask_frame": fgmask,
                "block_info": block_info,
                "eager_mode": bool(self.eager_mode),
            }
        }

        return should_skip_frame, meta_data

    def update_eager_state(self, pred_info: dict, meta_data: dict) -> None:

        """
        Mirror C++ runModelSingle eager logic:

        - Use per-frame softmax score + reliable_th to decide final label.
        - Convert label into fire_flag / smoke_flag.
        - Update sliding histories and compute fireProb / smokeProb.
        - If fireProb > 0 or smokeProb > 0: eager_mode = True, else False.
        """

        # 1) Extract label and probability info
        pred_label = pred_info.get("predLabel", "")
        pred_probs = pred_info.get("predProbs", [])
        pred_label_idx = pred_info.get("predLabelIdx", -1)

        # Softmax score for the chosen label (like C++ `score`) [file:1]
        if (
            pred_probs
            and pred_label_idx is not None
            and 0 <= pred_label_idx < len(pred_probs)
        ):
            score = float(pred_probs[pred_label_idx])
        else:
            score = 0.0

        # 2) Apply reliability threshold: score < reliable_th → label = "None" [file:1]
        if score < self.reliable_th:
            pred_label = "None"

        # 3) Map final label to fire/smoke flags (C++ behavior) [file:1]
        label_lower = pred_label.lower()
        if label_lower == "none":
            fire_flag = 0
            smoke_flag = 0
        elif label_lower == "fire":
            fire_flag = 1
            smoke_flag = 1  # fire implies smoke
        elif label_lower == "smokeonly":
            fire_flag = 0
            smoke_flag = 1
        else:
            # Any unknown label treated as none
            fire_flag = 0
            smoke_flag = 0

        # 4) Update sliding history windows (length = self.window_size) [file:1]
        # fireHistories[vchID]
        self.fire_hist.pop(0)
        self.fire_hist.append(fire_flag)

        # smokeHistories[vchID]
        self.smoke_hist.pop(0)
        self.smoke_hist.append(smoke_flag)

        # 5) Compute fireProb and smokeProb over window [file:1]
        fire_prob = sum(self.fire_hist) / float(self.window_size)
        smoke_prob = sum(self.smoke_hist) / float(self.window_size)

        if fire_prob > 0.0 or smoke_prob > 0.0:
            self.eager_mode = True
        else:
            self.eager_mode = False



    def reset(self):
        """Reset all state at video boundary (C++-style)."""
        super().reset()  # resets motion_det

        # Start in eager mode (like eagerEnables[vchID] default true) [file:1]
        self.eager_mode = True

        # Reset frame counter (fdCnts[vchID]) [file:1]
        self.fd_cnt = 0

        # Reset history windows (fireHistories / smokeHistories) [file:1]
        self.fire_hist = [0] * self.window_size
        self.smoke_hist = [0] * self.window_size
