from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import numpy as np
from src.config import Config
from src.utils import *

class SkipFactory:
    @staticmethod
    def create_skip_proc(config: Config, *args, **kwargs):
        assert config.methodCfg.name == "temp_method", "SkipFactory only works with temp_method"
        temp_method_cfg: dict = config.methodCfg.extra_cfgs["skip_proc"]  # ty:ignore[not-subscriptable]
        skip_proc_name: str = temp_method_cfg["name"]
        skip_proc_params = temp_method_cfg.get("params", {})
        cls = get_cls_in_pkg(pkg_name="src.methods.skip", fileName_ClsName=skip_proc_name)

        kwargs = {"cfg": config, "params": skip_proc_params}
        return cls(**kwargs)


class BaseSkipProc(ABC):
    """
    Strategy Interface for frame skipping logic.
    """
    def __init__(self, cfg: Config, params: dict):
        self.cfg = cfg
        self.params = params

    @abstractmethod
    def should_skip(
        self, frame_idx: int, frame: np.ndarray
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine if the frame should be skipped.
        Returns:
            should_skip (bool)
            meta_data (dict): Data needed for preprocessing (e.g., ROI coords, motion mask)
        """
        pass

    def prepare_input(self, frame: np.ndarray, meta_data: Dict[str, Any]) -> np.ndarray:
        """
        Optional: Transform the frame before inference (e.g., crop to ROI).
        Defaults to passing the original frame.
        """
        return frame

    def post_process_result(self, result: dict, meta_data: Dict[str, Any]) -> dict:
        """
        Optional: Modify the inference result (e.g., offset bbox coordinates back to original).
        Defaults to returning the result as-is.
        """
        return result

    def get_dummy_result(self, class_names) -> dict:
        """Returns a standardized dummy result for skipped frames."""
        num_classes = len(class_names)
        return {
            "logits": [0.0] * num_classes,
            "probs": [0.0] * num_classes,
            "predLabelIdx": -1,
            "predLabel": "skipped",
        }
