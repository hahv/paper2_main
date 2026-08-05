from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, List, Optional

from src.config import Config
from src.utils import get_cls_in_pkg


class MetricSrcFactory:
    @staticmethod
    def create_metric_src(config: Config, *args, **kwargs):
        # ! instead of create each metric src for each dataset, we specify the metric src class in the dataset config => mulitple datasets can share the same metric src class
        ds_metric_src: str = config.dbsetCfg.extra_cfgs.get("ds_metric_src", None)  # ty:ignore[unresolved-attribute, invalid-assignment]
        cls = get_cls_in_pkg(
            pkg_name="src.metrics",
            fileName_ClsName=ds_metric_src,
        )
        kwargs = {"cfg": config}
        return cls(**kwargs)


class BaseMetricSrc(ABC):
    """
    Abstract base class for metric data sources. Each concrete subclass represents
    a specific dataset and handles data retrieval for various metrics and modes.
    """

    DEFAULT_METRIC_MODE = "DEFAULT"

    def __init__(self, dataset_name: str, modes: Optional[List[str]] = None):
        self.dataset_name = dataset_name
        if modes is None:
            self.modes = [self.DEFAULT_METRIC_MODE]
        else:
            self.modes = modes

        # ! metric_name => func_to_get_data for that metric
        self.metric_data_getters_dict: Dict[str, Callable[..., Dict[str, Any]]] = {}
        self._register_handlers()

    def default_mode_processor(
        self, metric: str, mode: str, metric_data: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
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

        """
        pass

    def get_data_metrics(self, **kwargs) -> Dict[str, Any]:
        """
        Retrieves data for the specified metric and mode.
        return data format:
        {"mode": {"metric1": metric1_data, "metric2": metric2_data, ...}}
        """
        assert len(self.metric_data_getters_dict) > 0, (
            "No metric data getters registered"
        )
        metrics = self.metric_data_getters_dict.keys()

        final_data = {}
        for mode in self.modes:
            mode_proc_dict = {}
            for metric in metrics:
                metric_data_getter = self.metric_data_getters_dict.get(metric)
                metric_data_by_mode = metric_data_getter(
                    metric=metric, mode=mode, **kwargs
                )  # ty:ignore[call-non-callable]
                mode_proc_dict[metric] = metric_data_by_mode
            final_data[mode] = mode_proc_dict
        return final_data
