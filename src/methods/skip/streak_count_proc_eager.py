from halib import *  # noqa: F403
from typing import Tuple, Dict, Any
from src.config import Config
from src.methods.skip.base_block_skip_proc import BaseSkipProc

# ! This skip method is solely based on Eager/Non-Eager FSM logic, and does not
# ! use motion detection at all.
class StreakCountEagerProc(BaseSkipProc):
    def __init__(self, cfg: Config):
        super().__init__(cfg)

        # --- FSM Parameters ---
        # n_chk: periodic wake-up interval (frames to skip before forced infer)
        self.n_chk: int = self.params.get("n_chk", 50)
        # w_clr: consecutive non-fire frames needed to exit EAGER → NON-EAGER
        self.w_clr: int = self.params.get("w_clr", 7)
        # w_fire: consecutive fire/smoke infers needed to exit NON-EAGER → EAGER
        self.w_fire: int = self.params.get("fire_confirm_k", 1)

        with ConsoleLog("DEBUG"):
            pprint(f"{self.n_chk=}, {self.w_clr=}, {self.w_fire=}")

        # --- FSM State (start in EAGER — skipping must be earned) ---
        self.eager_mode: bool = True

        # c_fire: consecutive fire/smoke detections (NON-EAGER mode, wake-up checks only)
        self.c_fire: int = 0
        # c_clear: consecutive non-fire frames (EAGER mode)
        self.c_clear: int = 0
        # c_skip: frames skipped since last wake-up infer (NON-EAGER mode)
        self.c_skip: int = 0

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
            return False, meta_data  # never skip in EAGER

        # ── NON-EAGER mode: skip unless periodic wake-up ──────────────────────
        self.c_skip += 1
        is_wakeup = self.c_skip >= self.n_chk

        if is_wakeup:
            self.c_skip = 0  # reset counter after wake-up

        meta_data = {
            "mt_proc": {
                "fgmask_frame": None,
                "block_info": [],
                "is_forced_check": is_wakeup,
                "eager_mode": False,
            }
        }
        return not is_wakeup, meta_data  # skip=True unless it's a wake-up frame

    def update_eager_state(self, pred_info: dict, meta_data: dict) -> None:
        """
        Called by TempMethod after every DL inference result is known.
        FSM transitions:
          EAGER     → NON-EAGER : c_clear >= w_clr  (enough consecutive non-fire)
          NON-EAGER → EAGER     : c_fire  >= w_fire  (enough confirmed fire/smoke at wake-up)
        """
        pred_label = pred_info.get("predLabel", "")
        is_fire = pred_label.lower() in {"fire", "smokeonly"}

        if self.eager_mode:
            # ── EAGER updates ─────────────────────────────────────────────────
            if is_fire:
                self.c_clear = 0
            else:
                self.c_clear += 1

            # T1: EAGER → NON-EAGER
            if self.c_clear >= self.w_clr:
                self.eager_mode = False
                self.c_fire = 0
                self.c_skip = 0
                self.c_clear = 0

        else:
            # ── NON-EAGER updates (only called on wake-up infer frames) ───────
            if is_fire:
                self.c_fire += 1
            else:
                self.c_fire = 0  # single non-fire resets streak

            # T2: NON-EAGER → EAGER
            if self.c_fire >= self.w_fire:
                self.eager_mode = True
                self.c_clear = 0

        if "mt_proc" in meta_data:
            meta_data["mt_proc"]["eager_mode"] = self.eager_mode

    def reset(self):
        """Reset all FSM state at video boundary."""
        super().reset()  # resets motion_det
        self.eager_mode = True  # start each video in EAGER
        self.c_fire = 0
        self.c_clear = 0
        self.c_skip = 0