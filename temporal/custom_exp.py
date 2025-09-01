import sys

sys.path.append("E:/NextCloud/base_exp")

from halib import *
from halib.common import seed_everything
from halib.research.base_exp import BaseExperiment
from halib.research.metrics import MetricsBackend

from pipeline.config import *
from typing import Dict, Any, Union, List


class IdentityMetricsBackend(MetricsBackend):
    """
    A simple metrics backend that does not perform any calculations.
    It simply returns the input data as is.
    This can be useful for debugging or when no metrics are needed.
    """

    def compute_metrics(
        self,
        metrics_info: Union[List[str], Dict[str, Any]],
        metrics_data_dict: Dict[str, Any],
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        return metrics_data_dict


class CustomExperiment(BaseExperiment):
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
        return IdentityMetricsBackend(metric_cfg.metric_names)

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
    experiment = CustomExperiment(config)
    results, outfile = experiment.run_exp(
        do_calc_metrics=True, outdir=config.get_outdir(), return_df=True
    )
    console.rule(f"Experiment Results")
    csvfile.fn_display_df(results)
    pprint_local_path(outfile)


if __name__ == "__main__":
    main()
