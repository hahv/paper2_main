from temporal.methods.base_method import *
from torchvision import transforms
from PIL import Image
from temporal.config import Config
from temporal.metric_src.test_metric_src import TestDSMetricSrc
from temporal.methods.no_temp import NoTempMethod
import copy
import glob
from halib.system import filesys as fs
from halib import *


class TempTptMethod(NoTempMethod):
    def before_infer_video(self, video_path: str):
        assert self.cfg.method_cfg.method_used.name == "temp_tpt", (
            f"Method {self.cfg.method_cfg.method_used.name} is not supported for this operation"
        )
        self.window_size = self.cfg.method_cfg.method_used.extra_cfgs["window_size"]
        self.persist_thres = self.cfg.method_cfg.method_used.extra_cfgs["persist_thres"]
        self.temporal_buffer = np.zeros(self.window_size, dtype=bool)
        self.pos = 0

    def infer_frame(self, frame, frame_idx: int) -> dict:
        """Perform inference on the pre-processed frame."""
        assert self.model is not None, "Model is not loaded."
        with torch.no_grad():
            frame = self._pre_process_frame(frame)
            # 1. Get raw scores (logits) from the model
            logits = self.model(frame)

            # 2. Calculate probabilities using the softmax function
            probs = F.softmax(logits, dim=1)

        # 3. Get the index of the most likely class
        labelIdx = torch.argmax(probs, dim=1).item()

        # 4. Convert tensors to lists for easier handling
        logits = logits.cpu().squeeze().tolist()
        probs = probs.cpu().squeeze().tolist()

        # 5. Get the predicted class name
        classNames = self.cfg.model_cfg.class_names
        assert labelIdx < len(classNames), "Class index out of range."
        pred_label = classNames[labelIdx]
        # implement Temporal Persistence Thresholding (TPT)
        if pred_label.lower() != "none":
            self.temporal_buffer[self.pos] = True
        self.pos = (self.pos + 1) % self.window_size  # circular buffer
        if pred_label != "none":
            num_det_frames = np.sum(self.temporal_buffer)
            if num_det_frames <= self.persist_thres * self.window_size:
                pprint(
                    f"Suppressing `fire/smoke` detection at frame {frame_idx} by TPT."
                )
                pred_label = "none"
        return {
            "logits": logits,
            "probs": probs,
            "predLabelIdx": labelIdx,
            "predLabel": pred_label,
        }
