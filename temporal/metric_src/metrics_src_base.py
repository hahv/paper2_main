from abc import ABC, abstractmethod

from duckdb import torch
from temporal.config import *
from halib.filetype import csvfile

from abc import ABC, abstractmethod
from typing import Dict, Any, Callable


from temporal.utils import get_cls
class MetricSrcFactory:
    @staticmethod
    def create_metric_source(config: Config, *args, **kwargs):

        def ds_name_to_metric_source(
            dsname: str, suffix: str = "DSMetricSrc"
        ) -> str:
            return dsname + suffix

        dataset_name = config.dataset_cfg.dataset_used.name
        pkg_name = "temporal.metric_src"  # package name (folder)
        module_name = f"{dataset_name.lower()}_metric_src"  # py file name
        cls_name = ds_name_to_metric_source(dataset_name)  # class name
        cls = get_cls(f"{pkg_name}.{module_name}.{cls_name}")  # e.g.,
        assert cls is not None, f"Class '{cls_name}' not found in module '{pkg_name}'."
        kwargs = {"cfg": config}
        return cls(**kwargs)


class BaseMetricSrc(ABC):
    """
    Abstract base class for metric data sources. Each concrete subclass represents
    a specific dataset and handles data retrieval for various metrics and modes.
    """

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

        # metric_name => func_to_get_data for that metric
        self.metric_data_getters_dict: Dict[str, Callable[..., Dict[str, Any]]] = {}
        # mode_name (per-video, per-frame, etc) => func_to_process_data for that mode (data processor use the metric_data_getters to get data for each metric)
        self.mode_processors_dict: Dict[str, Callable[..., Dict[str, Any]]] = {}
        self._register_handlers()
        if len(self.mode_processors_dict) == 0:
            self.mode_processors_dict["default"] = self.default_mode_processor

    def default_mode_processor(self, metric: str, mode: str, metric_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Default processor for handling raw metric data.
        Can be overridden by specific modes if needed.
        """
        return metric_data

    @abstractmethod
    def _register_handlers(self):
        """
        Abstract method where subclasses register their metric data getters and mode processors.

        def metric_data_getter(metric: str, **kwargs) -> Dict[str, Any]:
            # Implementation for fetching metric-specific data
            pass

        def mode_processor(metric: str, metric_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            # Implementation for processing data according to the mode
            return metric_data

        """
        pass

    def get_data_metrics(self, **kwargs) -> Dict[str, Any]:
        """
        Retrieves data for the specified metric and mode.
        return data format:
        {"mode": {"metric1": metric1_data, "metric2": metric2_data, ...}}
        """
        assert len(self.metric_data_getters_dict) > 0, "No metric data getters registered"
        metrics = self.metric_data_getters_dict.keys()

        final_data = {}
        for mode in self.mode_processors_dict:
            mode_proc = self.mode_processors_dict[mode]
            mode_proc_dict = {}
            for metric in metrics:
                metric_data_getter = self.metric_data_getters_dict.get(metric)
                metric_data = metric_data_getter(metric=metric,**kwargs)
                proc_data = mode_proc(metric=metric, mode=mode, metric_data=metric_data, **kwargs)
                mode_proc_dict[metric] = proc_data
            final_data[mode] = mode_proc_dict
        return final_data
