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

    def prepare_metrics(self, metric_cfg: MetricSet) -> MetricsBackend:
        """
        Prepare the metrics for the experiment.
        This method should be implemented in subclasses.
        """
        num_classes = self.full_cfg.model_cfg.class_names
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
        return TorchMetricsBackend(name_and_tmetric)

    def exec_exp(self, *args, **kwargs):
        """
        Run the experiment (training, can be INCLUDE evaluation).
        This method should be implemented in subclasses.
        """
        console.rule(f"Exec Experiment")
        console.print(f"[red]{self.config.get_cfg_name()}[/red]")
        epch = 3  # Example epoch count

        proc_metric_ls = []
        proc_extra_data = []
        for epoch in range(epch):
            proc_extra_data.append({"epoch": epoch + 1})
            pprint(f"Running epoch {epoch + 1}/{epch}")
            import time

            time.sleep(1)  # Simulate some processing time
            sample_metric_dict = {
                k: np.random.rand() for k in self.get_metric_backend().metric_names
            }
            proc_metric_ls.append(sample_metric_dict)
        console.rule("Experiment run completed.")
        return proc_metric_ls, proc_extra_data


def main():
    config = Config.from_custom_yaml_file(r"config/base.yaml")
    experiment = OurExp(config)
    results, outfile = experiment.run_exp(
        do_calc_metrics=True, outdir=config.get_outdir(), return_df=True
    )
    console.rule(f"Experiment Results")
    csvfile.fn_display_df(results)
    pprint_local_path(outfile)


if __name__ == "__main__":
    main()
