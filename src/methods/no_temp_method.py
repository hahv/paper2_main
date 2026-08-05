from halib import *
import torch
import cv2
from PIL import Image
import torch.nn.functional as F

from src.config import Config
from src.methods.base_method import BaseMethod
from src.utils import get_transform, default_fileName_to_clsName


class NoTempMethod(BaseMethod):
    USED_CACHED_PRINTED = (
        False  # Class variable to track if the warning has been printed
    )

    def _validate_method_name(self):
        method_name: str = default_fileName_to_clsName(self.cfg.methodCfg.name)  # ty:ignore[invalid-argument-type]
        current_class_name = self.__class__.__name__
        assert method_name.lower() in current_class_name.lower(), (
            f"Config method '{method_name}' does not match class '{current_class_name}'"
        )

    def _pre_process_frame(self, frame):
        """Pre-process the frame before inference.
        if roi is provided, it will crop the frame to the ROI.
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        assert isinstance(self.cfg, Config), (
            "current method Cfg is not an instance of temporal.Config"
        )
        model_name: str = fs.get_file_name(
            self.cfg.modelCfg.model_path, split_file_ext=True
        )[0]
        pil_img = Image.fromarray(frame_rgb)
        # global LOG_TRANSFORM
        val_transform = get_transform(model_name, self.cfg.modelCfg.input_size)
        # with ConsoleLog("Infer transform"):
        #     pprint(val_transform)
        # Apply the transformation
        frame_batch = val_transform(pil_img).unsqueeze(0)  # Add batch dimension
        # Move the frame batch to the appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        frame_batch = frame_batch.to(device)
        return frame_batch

    def infer_frame(self, frame, frame_idx: int) -> dict:
        """Perform inference on the pre-processed frame."""
        # 0. Check if we have precomputed results for this frame to bypass the
        #    heavy model
        if (
            hasattr(self, "precomputed_rs_proc")
            and self.precomputed_rs_proc is not None
        ):
            precomputed_rs = self.precomputed_rs_proc.get_frame_data(frame_idx)
            if precomputed_rs is not None:
                if not NoTempMethod.USED_CACHED_PRINTED:
                    with ConsoleLog("Important", characters="🐸"):
                        pprint("Using precomputed results")
                    NoTempMethod.USED_CACHED_PRINTED = True
                return precomputed_rs

        assert self.model is not None, "Model is not loaded."
        with torch.no_grad():
            frame = self._pre_process_frame(frame)
            # 1. Get raw scores (logits) from the model
            logits: torch.Tensor = self.model(frame)

            # 2. Calculate probabilities using the softmax function
            probs = F.softmax(logits, dim=1)
        
        logits_ls = logits.cpu().squeeze().tolist()
        probs_ls = probs.cpu().squeeze().tolist()
        classNames: list[str] = self.cfg.modelCfg.class_names
        return self.infer_proc.proc_infer_results(logits_ls, probs_ls, classNames)