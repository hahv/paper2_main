from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

import numpy as np

from src.config import Config
from src.utils import get_cls_in_pkg
from collections import deque
from typing import Dict, Any, Optional

class InferProcFactory:
    @staticmethod
    def create_infer_proc(config: Config) -> "BaseInferProc":
        infer_proc_cfg = config.inferCfg.infer_proc
        if not infer_proc_cfg:
            # If no infer_proc config is provided, use the default NormalInferProc
            return NormalInferProc(params=None)

        cls = get_cls_in_pkg(
            pkg_name="src.methods.infer_proc",
            fileName_ClsName=infer_proc_cfg.name,
        )
        params = infer_proc_cfg.extra or {}
        return cls(params=params)

class BaseInferProc(ABC):
    """
    Interface for model inference post-processing logic.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}

    @abstractmethod
    def proc_infer_results(self, logits_ls, probs_ls, classNames) -> Dict[str, Any]:
        """
        Process the inference results and return a dictionary of processed results.

        Args:
            logits_ls: List of logits from model inference.
            probs_ls: List of probabilities from model inference.
            classNames: List of class names corresponding to model outputs.

        Returns:
            A dictionary containing processed inference results.
        """
        raise NotImplementedError


class NormalInferProc(BaseInferProc):
    """
    Simple inference post-processor that returns argmax result directly.
    """

    def proc_infer_results(self, logits_ls, probs_ls, classNames) -> Dict[str, Any]:
        label_idx = int(np.argmax(probs_ls))
        assert label_idx < len(classNames), "Class index out of range."

        pred_label = classNames[label_idx]
        return {
            "logits": logits_ls,
            "probs": probs_ls,
            "predLabelIdx": label_idx,
            "predLabel": pred_label,
        }


class ProfInferProc(BaseInferProc):
    """
    C++-style inference post-processor (from Professor's code) that applies a reliability threshold to determine the final predicted label.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.reliable_th = self.params.get("reliable_th", 0.7)

    def proc_infer_results(self, logits_ls, probs_ls, classNames) -> Dict[str, Any]:
        probs = np.asarray(probs_ls, dtype=float)
        logits = np.asarray(logits_ls, dtype=float) if logits_ls is not None else None
        # class_name = [Fire, None, SmokeOnly]
        NONE_LB_IDX = classNames.index("None")
        label_idx = int(np.argmax(probs))
        original_prob = probs[label_idx]
        if original_prob < self.reliable_th:
            if label_idx != NONE_LB_IDX:
                print(f"Unreliable prediction: original label '{classNames[label_idx]}' with prob {original_prob:.4f} below threshold {self.reliable_th}. Setting to 'None'.")
            label_idx = NONE_LB_IDX
            predLabel = classNames[NONE_LB_IDX]
        else:
            predLabel = classNames[label_idx]
        
        return {
            "logits": logits,
            "probs": probs,
            "predLabelIdx": label_idx,
            "predLabel": predLabel,
        }