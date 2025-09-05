import sys

sys.path.append("E:/NextCloud/base_exp")

from halib import *
from halib.common import seed_everything
from halib.research.base_exp import BaseExperiment
from halib.research.metrics import MetricsBackend, TorchMetricsBackend

from temporal.config import *
from typing import Dict, Any, Union, List
import torchmetrics
from temporal.metrics import FPS, FPR
import importlib
import importlib
from temporal.rs_handler import *
from temporal.methods.base_method import BaseMethod


def get_cls(class_path: str, *args, **kwargs):
    """
    Dynamically import class and create instance.
    class_path format: 'mypkg.shapes.circle.Circle'
    """
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls


class MethodFactory:
    @staticmethod
    def create_method(config: Config, *args, **kwargs):
        def method_name_to_cls_name(name: str, suffix: str = "Method") -> str:
            """
            Convert snake_case string to PascalCase and append suffix.
            Example: "no_temp" -> "NoTempMethod"
            """
            parts = name.split("_")
            pascal = "".join(word.capitalize() for word in parts)
            return pascal + suffix

        pkg_name = "temporal.methods"
        # ! method_name == module_name
        module_name = config.method_cfg.method_used.name
        cls_name = method_name_to_cls_name(module_name)
        cls = get_cls(f"{pkg_name}.{module_name}.{cls_name}")
        assert cls is not None, f"Class '{cls_name}' not found in module '{pkg_name}'."

        rs_handler_list: list[RSHandlerBase] = []
        if config.infer_cfg.save_csv_results:
            rs_handler_list.append(CsvRSHandler(config))
        if config.infer_cfg.save_video_results:
            pkg_name = "temporal.rs_handler"
            chosen_video_handler = config.method_cfg.method_used.extra_cfgs.get("video_rs_handler", "BaseVideoRSHandler")
            rs_handler_list.append(get_cls(f"{pkg_name}.{chosen_video_handler}")(cfg=config))

        kwargs = {"cfg": config, "rs_handlers": rs_handler_list}
        return cls(**kwargs)


class MetricSourceFactory:
    @staticmethod
    def create_metric_source(config: Config, *args, **kwargs):

        def ds_name_to_metric_source(
            dsname: str, suffix: str = "DSMetricSource"
        ) -> str:
            return dsname + suffix

        dataset_name = config.dataset_cfg.dataset_used.name
        pkg_name = "temporal.metrics_src" # package name (folder)
        module_name = f'{dataset_name.lower()}_metric_src' # py file name
        cls_name = ds_name_to_metric_source(dataset_name) # class name
        cls = get_cls(f"{pkg_name}.{module_name}.{cls_name}") # e.g.,
        assert cls is not None, f"Class '{cls_name}' not found in module '{pkg_name}'."
        kwargs = {"cfg": config}
        return cls(**kwargs)


class OurExp(BaseExperiment):
    """
    Custom experiment class that extends BaseExperiment.
    This class should implement the specific logic for the custom experiment.
    """

    def __init__(self, config: Config):
        """
        Initialize the custom experiment with a configuration object.
        :param config: An instance of Config or its subclass.
        """
        super().__init__(config)
        self.full_cfg = config
        self.metric_results = {}
        self.metric_backend = None
        self.video_dir_path = None

    def init_general(self, general_cfg: GeneralConfig):
        """
        Setup general settings for the experiment like SEED, logging, etc.
        This method should be implemented in subclasses.
        """
        console.rule("General initialization")
        seed_everything(general_cfg.seed)
        # setup log here

    def prepare_dataset(self, dataset_cfg: DatasetInfo):
        """
        Prepare the dataset for the experiment.
        This method should be implemented in subclasses.
        """
        dataset_name = dataset_cfg.get_name()
        console.rule(f"Preparing dataset - {dataset_name}")
        self.video_dir_path = dataset_cfg.dir_path

    def prepare_metrics(self, metric_cfg: MetricSet) -> MetricsBackend:
        """
        Prepare the metrics for the experiment.
        This method should be implemented in subclasses.
        """
        num_classes = len(self.full_cfg.model_cfg.class_names)
        num_classes = 2 # force binary classification (fire_smoke OR none)
        if num_classes > 2:
            task = "multiclass"
        else:
            task = "binary"

        metric_names = self.full_cfg.metric_cfg.metricSet_used.metric_names
        name_and_tmetric = {}
        if "accuracy" in metric_names:
            acc = torchmetrics.Accuracy(task=task, num_classes=num_classes)
            name_and_tmetric["accuracy"] = acc
        if "f1_score" in metric_names:
            f1 = torchmetrics.F1Score(task=task, num_classes=num_classes)
            name_and_tmetric["f1_score"] = f1
        if "precision" in metric_names:
            p = torchmetrics.Precision(task=task, num_classes=num_classes)
            name_and_tmetric["precision"] = p
        if "recall (TPR)" in metric_names:
            r = torchmetrics.Recall(task=task, num_classes=num_classes)
            name_and_tmetric["recall (TPR)"] = r
        if "FPR" in metric_names:
            fpr = FPR()
            name_and_tmetric["FPR"] = fpr
        if "FPS" in metric_names:
            fps = FPS()
            name_and_tmetric["FPS"] = fps

        # make sure all metrics are initialized
        for metric in metric_names:
            assert metric in name_and_tmetric, f"Metric '{metric}' is not initialized."
        self.metric_backend = TorchMetricsBackend(name_and_tmetric)
        return self.metric_backend

    def exec_exp(self, *args, **kwargs):
        """
        Run the experiment (training, can be INCLUDE evaluation).
        This method should be implemented in subclasses.
        """
        console.rule(f"Exec Experiment")
        console.print(f"[red]{self.config.get_cfg_name()}[/red]")

        # metric_infos = self.metric_backend.metric_info # {key: torch_metrics}
        method_instance = MethodFactory.create_method(self.config)
        assert isinstance(method_instance, BaseMethod), "Method instance is not of type BaseMethod"
        method: BaseMethod = method_instance
        method.infer_video_dir(self.video_dir_path)
        eval_data_dict = method.eval() # {metric: <value for compute metrics>}
        extra_data = None
        exp_rs = eval_data_dict, extra_data
        return exp_rs

    def run_exp(self, do_calc_metrics=True, *args, **kwargs):
        self.init_general(self.config.get_general_cfg())
        self.prepare_dataset(self.config.get_dataset_cfg())
        self.prepare_metrics(self.config.get_metric_cfg())

        # Save config before running
        self.config.save_to_outdir()

        # Execute experiment
        results = self.exec_exp(*args, **kwargs)
        if do_calc_metrics:
            mode_metrics_data_dict, _ = results
            for mode in mode_metrics_data_dict:
                console.rule(f"Calculating metrics for mode: {mode}")
                metrics_data = mode_metrics_data_dict[mode]
                CSV_FILE_POSTFIX = "__perf"
                outfile = self.full_cfg.get_outdir() + f"/{self.full_cfg.get_cfg_name()}__{mode}{CSV_FILE_POSTFIX}.csv"
                perf_results, outfile = self.calc_and_save_exp_perfs(
                    raw_metrics_data=metrics_data, extra_data=None,
                    outfile=outfile, return_df=True,
                    *args, **kwargs
                )
                csvfile.fn_display_df(perf_results)
                pprint_local_path(outfile)
