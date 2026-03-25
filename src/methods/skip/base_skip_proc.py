from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import numpy as np
from src.config import Config
from src.utils import *
from src.methods.skip.motion.base_motion_det import *


class SkipProcFactory:
    @staticmethod
    def create_skip_proc(config: Config, *args, **kwargs):
        assert config.methodCfg.name.startswith("temp_method"), (  # ty:ignore[unresolved-attribute]
            "SkipProcFactory only works with temp_method"
        )
        temp_method_cfg: dict = config.methodCfg.extra_cfgs.get("skip_proc", {})  # ty:ignore[unresolved-attribute]
        skip_proc_name: str = temp_method_cfg.get("name")
        assert skip_proc_name is not None and len(skip_proc_name) > 0, (
            "Skip proc name must be specified"
        )
        cls = get_cls_in_pkg(
            pkg_name="src.methods.skip", fileName_ClsName=skip_proc_name
        )

        kwargs = {"cfg": config}
        return cls(**kwargs)


class BaseSkipProc(ABC):
    """
    Strategy Interface for frame skipping logic.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

        skip_proc_dict: dict = self.cfg.methodCfg.extra_cfgs.get("skip_proc", {})  # ty:ignore[possibly-missing-attribute, unused-ignore-comment, unresolved-attribute]
        self.name = skip_proc_dict.get("name", "no_skip_proc")
        self.params = skip_proc_dict.get("params")
        self.motion_det: BaseMotionDet = None  # ty:ignore[invalid-assignment]

        if "motion" in self.params:
            motion_cfg = self.params.get("motion", {})
            assert "name" in motion_cfg, "motion detector config must have a name"
            assert "params" in motion_cfg, "motion detector config must have params"
            self.motion_det: BaseMotionDet = MotionDetFactory.create_method(
                name=motion_cfg["name"], params=motion_cfg["params"]
            )

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

    def prepare_infer_input(
        self, frame: np.ndarray, meta_data: Dict[str, Any]
    ) -> np.ndarray:
        """
        Optional: Transform the frame before inference (e.g., crop to ROI).
        Defaults to passing the original frame.
        """
        return frame

    def get_dummy_result(self, class_names) -> dict:
        """Returns a standardized dummy result for skipped frames."""
        num_classes = len(class_names)
        return {
            "logits": [0.0] * num_classes,
            "probs": [0.0] * num_classes,
            "predLabelIdx": -1,
            "predLabel": "skipped",
        }
