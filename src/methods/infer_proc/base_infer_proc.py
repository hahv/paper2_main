from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

import numpy as np

from src.config import Config
from src.utils import get_cls_in_pkg


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
    Inference post-processor with access to runtime config for more advanced logic.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.cfg = self.params["cfg"]

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
