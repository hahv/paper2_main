from sympy.physics.units import l
from halib import *
import sys
import torch
from torch.nn import functional as F
from src.methods.no_temp_method import NoTempMethod


class TempBaselineTptMethod(NoTempMethod):
    PRINT_USING_PRECOMPUTED = False  # Class variable to track if precomputed results are used
    def before_infer_video(self, video_path: str):
        # ! do validation: method name in cfg matches class name
        super()._validate_method_name()
        self.window_size = self.cfg.methodCfg.extra_cfgs["window_size"]  # ty:ignore[not-subscriptable]
        self.persist_thres = self.cfg.methodCfg.extra_cfgs["persist_thres"]  # ty:ignore[not-subscriptable]
        self.temporal_buffer = np.zeros(self.window_size, dtype=bool)
        self.pos = 0

    def infer_frame(self, frame, frame_idx: int) -> dict:
        """Perform inference on the pre-processed frame."""
        pre_rs = None
        if (
            hasattr(self, "precomputed_rs_proc")
            and self.precomputed_rs_proc is not None
        ):
            pre_rs = self.precomputed_rs_proc.get_frame_data(frame_idx)
        if pre_rs is not None:
            # print('A precomputed result is used for frame_idx', frame_idx)
            logits = pre_rs["logits"]
            probs = pre_rs["probs"]
            labelIdx = pre_rs["predLabelIdx"]
            if not TempBaselineTptMethod.PRINT_USING_PRECOMPUTED:
                with ConsoleLog("Important", characters="🐸"):
                    pprint("Using precomputed results")
                TempBaselineTptMethod.PRINT_USING_PRECOMPUTED = True
        else:
            #print('No precomputed result for frame_idx', frame_idx, '- running full inference')
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
        time_for_tpt = time.perf_counter()  # Start timer for TPT processing
        classNames: list[str] = self.cfg.modelCfg.class_names
        assert labelIdx < len(classNames), "Class index out of range."
        pred_label: str = classNames[labelIdx]
        # implement Temporal Persistence Thresholding (TPT)
        if pred_label.lower() != "None":
            self.temporal_buffer[self.pos] = True
        self.pos = (self.pos + 1) % self.window_size  # circular buffer
        if pred_label != "None":
            num_det_frames = np.sum(self.temporal_buffer)
            if num_det_frames <= self.persist_thres * self.window_size:
                console.print(
                    f"Suppressing `fire/smoke` detection at frame [bold cyan]{frame_idx}[/] by TPT.",
                    end="\r",
                )
                sys.stdout.flush()  # Force the flush manually
                pred_label = "None"
                # update predix_idx to match the new pred_label
                labelIdx = classNames.index(pred_label)
        tpt_time = time.perf_counter() - time_for_tpt  # End timer for TPT processing
        ret_dict = {
            "logits": logits,
            "probs": probs,
            "predLabelIdx": labelIdx,
            "predLabel": pred_label,
        }

        if pre_rs is not None and "elapsed_time" in pre_rs:
            ret_dict["elapsed_time"] = (
                pre_rs["elapsed_time"] + tpt_time
            )  # Add TPT time to precomputed elapsed time

        return ret_dict
