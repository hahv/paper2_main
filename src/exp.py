import sys

from halib import *
from halib.common.common import seed_everything
from halib.exp.core.base_exp import BaseExp
from halib.exp.perf.perfmetrics import MetricsBackend, TorchMetricsBackend

from src.config import *
from src.metrics.custom_metrics import MetricFactory
from src.methods.base_mt import *


class Paper2Exp(BaseExp):
    """
    Custom experiment class that extends BaseExperiment.
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self.full_cfg: Config = config
        self.metric_results = {}
        self.metric_backend = None
        self.video_dir_path = None

    def init_general(self, general_cfg: GeneralConfig):
        console.rule("General initialization")
        seed_everything(general_cfg.seed)
        # setup log here

    def prepare_dataset(self, dataset_cfg: DatasetCfg):
        dataset_name = dataset_cfg.get_name()
        console.rule(f"Preparing dataset - {dataset_name}")
        self.video_dir_path = dataset_cfg.dir_path

    def prepare_metrics(self, metric_cfg: MetricSetCfg) -> MetricsBackend:
        """
        Prepare the metrics for the experiment.
        This method should be implemented in subclasses.
        """
        num_classes = len(self.full_cfg.modelCfg.class_names)
        num_classes = 2  # force binary classification (fire_smoke OR none)
        console.print(f"[bold red] FORCED Number of classes: {num_classes} [/bold red]")
        name_and_tmetric = MetricFactory.create_metrics(
            metric_cfg.metric_names, num_classes
        )
        self.metric_backend = TorchMetricsBackend(name_and_tmetric)
        return self.metric_backend

    def exec_exp(self, *args, **kwargs):
        """
        Run the experiment (training, can be INCLUDE evaluation).
        This method should be implemented in subclasses.
        """
        console.rule(f"Exec Experiment")
        console.print(f"[red]{self.config.get_cfg_name()}[/red]")

        method_instance = MethodFactory.create_method(self.config)
        assert isinstance(method_instance, BaseMethod), (
            "Method instance is not of type BaseMethod"
        )
        method: BaseMethod = method_instance
        method.infer_video_dir(self.video_dir_path)
        eval_data_dict = (
            method.prepare_metric_src()
        )  # {metric: <value for compute metrics>}
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
                outfile = (
                    self.full_cfg.get_outdir()
                    + f"/{self.full_cfg.get_cfg_name()}__{mode}.csv"
                )
                perf_results, outfile = self.calc_perfs(
                    raw_metrics_data=metrics_data,
                    extra_data=None,
                    outfile=outfile,
                    return_df=True,
                    *args,
                    **kwargs,
                )
                df = pd.read_csv(outfile, sep=";", encoding="utf-8")
                # get row 0
                df.at[0, "experiment"] = f"{self.full_cfg.get_cfg_name()}_{mode}"
                df.to_csv(outfile, sep=";", encoding="utf-8", index=False)
                csvfile.fn_display_df(df)
                pprint_local_path(outfile, get_wins_path=True)
