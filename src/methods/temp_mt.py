from halib import *

from abc import abstractmethod
from typing import Optional, Tuple, Dict
from src.methods.noTemp_mt import NoTempMethod

class TemporalMethod(NoTempMethod):
    @abstractmethod
    def skip_module(self, frame_idx, cv2_bgr_frame) -> Tuple[bool, Optional[Dict]]:
        """
        Analyzes a frame to decide whether to skip it or process a specific ROI.
        Args:
            frame_idx (int): Index of the current frame.
            cv2_bgr_frame (np.ndarray): The input frame in BGR format.
        Returns:
            should_skip (bool): Whether to skip processing this frame.
            skip_info (dict): Additional information such as ROI coordinates and motion mask.
        """
        pass

    @abstractmethod
    def prep_frame_skip(
        self, original_frame, skip_info: Optional[Dict] = None
    ):
        pass
    # ! not abstract, but should override in child class
    def update_infer_results(
        self, infer_res: dict, should_skip: bool, skip_info: Optional[Dict] = None
    ):
        return infer_res

    def _get_skipped_result(self) -> dict:
        """Returns a standardized dummy result for skipped frames."""
        num_classes = len(self.cfg.modelCfg.class_names)
        return {
            "logits": [0.0] * num_classes,
            "probs": [0.0] * num_classes,
            "predLabelIdx": -1,
            "predLabel": "skipped",
        }

    def infer_frame(self, frame, frame_idx: int) -> dict:
        # Capture the context scope as 'ctx'
        with self.profiler.measure("infer_frame") as ctx:
            infer_res = None
            # Now just use ctx.step() - no need to type "infer_frame"
            with ctx.step("skip_module"):
                should_skip, skip_info = self.skip_module(frame_idx, frame)

            if should_skip:
                infer_res = self._get_skipped_result()
            else:
                with ctx.step("big_infer"):
                    cropped_frame = self.prep_frame_skip(frame, skip_info)
                    infer_res = super().infer_frame(cropped_frame, frame_idx)
            res = self.update_infer_results(infer_res, should_skip, skip_info)
            return res
