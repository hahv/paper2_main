from halib import *
import sys
import torch
from torch.nn import functional as F
from src.methods.noTemp_mt import NoTempMethod


class TempBaselineTPTMethod(NoTempMethod):

    def before_infer_video(self, video_path: str):
        # ! do validation: method name in cfg matches class name
        super()._validate_method_name()
        self.window_size = self.cfg.methodCfg.extra_cfgs["window_size"]  # ty:ignore[not-subscriptable]
        self.persist_thres = self.cfg.methodCfg.extra_cfgs["persist_thres"]  # ty:ignore[not-subscriptable]
        self.temporal_buffer = np.zeros(self.window_size, dtype=bool)
        self.pos = 0

    def infer_frame(self, frame, frame_idx: int) -> dict:
        """Perform inference on the pre-processed frame."""
        assert self.model is not None, "Model is not loaded."
        with torch.no_grad():
            frame = self._pre_process_frame(frame)
            # 1. Get raw scores (logits) from the model
            logits: torch.Tensor = self.model(frame)

            # 2. Calculate probabilities using the softmax function
            probs = F.softmax(logits, dim=1)

        # 3. Get the index of the most likely class
        labelIdx = int(torch.argmax(probs, dim=1).item())

        # 4. Convert tensors to lists for easier handling
        logits = logits.cpu().squeeze().tolist()
        probs = probs.cpu().squeeze().tolist()

        # 5. Get the predicted class name
        classNames: list[str] = self.cfg.modelCfg.class_names
        assert labelIdx < len(classNames), "Class index out of range."
        pred_label: str = classNames[labelIdx]
        # implement Temporal Persistence Thresholding (TPT)
        if pred_label.lower() != "none":
            self.temporal_buffer[self.pos] = True
        self.pos = (self.pos + 1) % self.window_size  # circular buffer
        if pred_label != "none":
            num_det_frames = np.sum(self.temporal_buffer)
            if num_det_frames <= self.persist_thres * self.window_size:
                console.print(
                    f"Suppressing `fire/smoke` detection at frame [bold cyan]{frame_idx}[/] by TPT.",
                    end="\r"
                )
                sys.stdout.flush()  # Force the flush manually
                pred_label = "none"
        return {
            "logits": logits,
            "probs": probs,
            "predLabelIdx": labelIdx,
            "predLabel": pred_label,
        }
