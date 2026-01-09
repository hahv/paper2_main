from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional, Any
import numpy as np
from src.config import Config

class SkipFactory:
    @staticmethod
    def create_skip_proc(config: Config, *args, **kwargs):
        def method_name_to_cls_name(name: str, suffix: str = "Proc") -> str:
            """
            Convert snake_case string to PascalCase and append suffix.
            Example: "no_temp" -> "NoTempMethod"
            """
            parts = name.split("_")[:-1]  # remove the 'mt' postfix
            # Capitalize the first letter fo each part, but keep the rest as is
            for i in range(len(parts)):
                word = parts[i]
                word = word[0].upper() + word[1:]
                parts[i] = word
            pascal = "".join(parts)
            return pascal + suffix

        pkg_name = "src.methods.skip"
        # ! method_name == module_name
        method_postfix = "mt"
        module_name = f"{config.methodCfg.name}_{method_postfix}"
        cls_name = method_name_to_cls_name(module_name)
        pprint(f"Creating method class: {pkg_name}.{module_name}.{cls_name}")
        cls = get_cls(f"{pkg_name}.{module_name}.{cls_name}")
        assert cls is not None, f"Class '{cls_name}' not found in module '{pkg_name}'."

        rs_handler_list: list[BaseRsProc] = []
        if config.inferCfg.save_csv_results:
            rs_handler_list.append(CsvRsProc(config))
        if config.inferCfg.save_video_results:
            pkg_name = "src.results"
            chosen_video_handler = config.methodCfg.extra_cfgs.get(  # ty:ignore[possibly-missing-attribute]
                "video_rs_proc", "VideoRSProc"
            )
            rs_handler_list.append(
                get_cls(f"{pkg_name}.{chosen_video_handler}")(cfg=config)
            )

        kwargs = {"cfg": config, "rs_handlers": rs_handler_list}
        return cls(**kwargs)


class BaseSkipProc(ABC):
    """
    Strategy Interface for frame skipping logic.
    """

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
