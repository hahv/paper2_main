from halib import *  # noqa: F403
from typing import Tuple, Dict, Any

from src.config import Config
from src.methods.skip.base_skip_proc import BaseSkipProc

# ! This skip method is solely based on Eager/Non-Eager logic, implemented by
# ! Prof.Park (in iNet framework)
class WindowVoteEagerProc(BaseSkipProc):
    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.eager_mode: bool = True

        # ! generator.cpp - line 491:
        # cfg.fdWindowSize = 16; //32
        # cfg.fdPeriod = 16;  // check period in the idle mode
        self.fd_period: int = self.params.get("fd_period", 16)  # fixed at 16
        self.window_size: int = self.params.get("window_size", 16)  # fixed at 16
        self.fd_cnt: int = 0
        self.fire_hist = [0] * self.window_size
        self.smoke_hist = [0] * self.window_size

    def should_skip(
        self, frame_idx: int, frame: np.ndarray
    ) -> Tuple[bool, Dict[str, Any]]:

        # ── EAGER mode: infer every frame ────────────────────────────────────
        if self.eager_mode:
            meta_data = {
                "mt_proc": {
                    "fgmask_frame": None,
                    "block_info": [],
                    "is_forced_check": False,
                    "eager_mode": True,
                }
            }
            return False, meta_data

        # ── NON-EAGER mode: periodic wake-up only ────────────────────────────
        self.fd_cnt += 1
        is_wakeup = self.fd_cnt >= self.fd_period

        if is_wakeup:
            self.fd_cnt = 0

        meta_data = {
            "mt_proc": {
                "fgmask_frame": None,
                "block_info": [],
                "is_forced_check": is_wakeup,
                "eager_mode": False,
            }
        }
        return not is_wakeup, meta_data

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
        assert len(pred_label) > 0, "Predicted label is empty."
        # 3) Map final label to fire/smoke flags (C++ behavior)
        label_lower = pred_label.lower()
        if label_lower == "fire":
            fire_flag, smoke_flag = 1, 1  # fire implies smoke
        elif label_lower == "smokeonly":
            fire_flag, smoke_flag = 0, 1
        else:
            fire_flag, smoke_flag = 0, 0  # "none" or unknown

        # 4) Update sliding history windows (length = self.window_size)
        self.fire_hist.pop(0)
        self.fire_hist.append(fire_flag)

        self.smoke_hist.pop(0)
        self.smoke_hist.append(smoke_flag)

        # 5) Compute fireProb / smokeProb and update eager_mode
        fire_prob = sum(self.fire_hist) / float(self.window_size)
        smoke_prob = sum(self.smoke_hist) / float(self.window_size)

        self.eager_mode = fire_prob > 0.0 or smoke_prob > 0.0

        if "mt_proc" in meta_data:
            meta_data["mt_proc"]["eager_mode"] = self.eager_mode

    def reset(self):
        """Reset all state at video boundary (C++-style)."""
        super().reset()

        self.eager_mode = True
        self.fd_cnt = 0
        self.fire_hist = [0] * self.window_size
        self.smoke_hist = [0] * self.window_size