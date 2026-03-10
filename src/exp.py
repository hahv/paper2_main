from dill.tests.test_registered import p
from halib import *
from halib.common.common import seed_everything
from halib.exp.core.base_exp import BaseExp
from halib.exp.perf.perfmetrics import MetricsBackend, TorchMetricsBackend

from src.config import *
from src.metrics.custom_metrics import MetricFactory
from src.methods.base_method import *
from halib.utils.dict import DictUtils
from collections import OrderedDict
from src.common import GlobalConst
from typing import Callable


class Paper2Exp(BaseExp):
    """
    Custom experiment class that extends BaseExperiment.
    """

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(config)
        self.full_cfg: Config = config
        self.metric_results = {}
        self.metric_backend = None
        self.video_dir_path = None
        self.wandb_logger = None
        if "wandb_logger" in kwargs:
            self.wandb_logger = kwargs["wandb_logger"]

    @staticmethod
    # This func to get the config file path based on the exp dir path, e.g: exp can be a classification or detection exp, and the config file can be different
    def default_expDir_to_cfgFile_fn(exp_dir_path: str) -> str:
        return f"./config/zruns/run_base.yaml"

    # ! add a function to allow dynmmically load with baseline method (not Paper2Exp class) by creating a custom Config
    @classmethod
    def from_custom_exp(
        cls,
        exp_dir_path: str,
        find_cfgFile_func: Callable[[str], str] = default_expDir_to_cfgFile_fn,
    ):  # noqa: F811
        # Load the configuration from the provided file path
        placeholder_cfg_file = find_cfgFile_func(exp_dir_path)
        console.print(
            f"[bold green]Loading config from file: {placeholder_cfg_file}[/bold green]"
        )
        config = Config.from_custom_yaml_file(placeholder_cfg_file)

        # Update the experiment directory in the config
        config.update_custom_exp(exp_dir_path)

        # Create and return a new instance of Paper2Exp with the updated config
        return cls(config, from_custom_exp=True)

    def init_general(self, general_cfg: GeneralCfg):
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

        method_instance = MethodFactory.create_method(self.config)  # ty:ignore[invalid-argument-type]
        assert isinstance(method_instance, BaseMethod), (
            "Method instance is not of type BaseMethod"
        )
        method: BaseMethod = method_instance
        method.infer_video_dir(
            self.video_dir_path, max_workers=self.full_cfg.inferCfg.num_infer_workers
        )
        eval_data_dict = (
            method.prepare_metric_src()
        )  # {metric: <value for compute metrics>}
        extra_data = None
        exp_rs = eval_data_dict, extra_data
        return exp_rs

    def reset_metric_backend(self):
        if self.metric_backend is not None:
            for metric_instance in self.metric_backend.metric_info.values():  # ty:ignore[possibly-missing-attribute]
                metric_instance.reset()

    def run_exp(self, should_calc_metrics=True, reload_env=False, *args, **kwargs):
        if self.full_cfg.shouldSkipExp:
            return

        self.init_general(self.config.get_general_cfg())  # ty:ignore[invalid-argument-type]
        self.prepare_dataset(self.config.get_dataset_cfg())  # ty:ignore[invalid-argument-type]
        self.prepare_metrics(self.config.get_metric_cfg())  # ty:ignore[invalid-argument-type]

        # ! creates output directory and save config before running exp
        self.config.save_to_outdir()
        exp_start = time.time()
        # Execute experiment
        results = self.exec_exp(*args, **kwargs)
        exp_end = time.time()
        with ConsoleLog("Experiment Summary"):
            console.print(f"Exp time: {exp_end - exp_start:.2f} seconds")
        if should_calc_metrics:
            mode_metrics_data_dict, _ = results
            for mode in mode_metrics_data_dict:
                console.rule(f"Calculating metrics for mode: {mode}")
                # !Fix bug: [Important] Make sure to reset the metric backend before calculating metrics for each mode (per-frame or per-video), otherwise the metric values will accumulate and be incorrect across modes.
                self.reset_metric_backend()
                metrics_data = mode_metrics_data_dict[mode]
                outfile = (
                    self.full_cfg.get_outdir()
                    + f"/{GlobalConst.PERF_FILE_PREFIX}{self.full_cfg.get_cfg_name()}__{mode}.csv"
                )
                extra_data = None
                if self.full_cfg.methodCfg.extra_cfgs is not None:
                    extra_data_orig = self.full_cfg.methodCfg.extra_cfgs.copy()
                    if "result_proc" in extra_data_orig:
                        del extra_data_orig["result_proc"]
                    extra_data = DictUtils.flatten(extra_data_orig)
                    # with ConsoleLog("Extra data"):
                    #     pprint(extra_data)

                perf_results, outfile = self.calc_perfs(
                    raw_metrics_data=metrics_data,
                    extra_data=extra_data,
                    outfile=outfile,
                    return_df=False,
                    *args,  # ty:ignore[parameter-already-assigned]
                    **kwargs,
                )
                # with ConsoleLog(f"Performance Results-{mode}"):
                #     pprint(perf_results)
                #     assert False, "Stop here for debugging"
                # ! perf_results example: a list with an OrderDict inside
                # {
                #     "experiment": "MainPC__ds_UFireIndoor2__mt_temp_method_motion_block__a7b567955502__20260202.131451",
                #     "dataset": "UFireIndoor2",
                #     "skip_proc.name": "motion_only_block_skip_proc.MotionOnlyBlockSkipProc",
                #     "skip_proc.params.motion.name": "acc_motion_det.AccMotionDet",
                #     "skip_proc.params.motion.params.diff_frame_th": 2,
                #     "skip_proc.params.motion.params.impact_plus_one": 5,
                #     "skip_proc.params.motion.params.mask_th": 10,
                #     "skip_proc.params.motion.params.max_val": 25,
                #     "skip_proc.params.motion.params.decay": 1,
                #     "skip_proc.params.scale_factor": 1.0,
                #     "skip_proc.params.block_size_orig": 32,
                #     "skip_proc.params.block_ratio_th": 0.05,
                #     "skip_proc.params.min_roi_ratio": 0.75,
                #     "metric_accuracy": 1.0,
                #     "metric_f1_score": 1.0,
                #     "metric_precision": 1.0,
                #     "metric_recall (TPR)": 1.0,
                #     "metric_FPR (False Alarm Rate)": 0.0,
                #     "metric_FPS": 25.530609130859375,
                # }
                if self.wandb_logger is not None:
                    metric_rs_dict = {}
                    perf_dict: OrderedDict = perf_results[0]
                    for key in perf_dict:
                        if key.startswith("metric_"):
                            metric_name = key.replace("metric_", f"{mode}_metric_")
                            metric_rs_dict[metric_name] = perf_dict[key]
                    self.wandb_logger.log_metrics(metric_rs_dict)

                df = pd.read_csv(outfile, sep=";", encoding="utf-8")  # ty:ignore[no-matching-overload]
                df.at[0, "experiment"] = f"{self.full_cfg.get_cfg_name()}_{mode}"
                df.to_csv(outfile, sep=";", encoding="utf-8", index=False)
                csvfile.fn_display_df(df)
                pprint_local_path(outfile, get_wins_path=True)  # ty:ignore[invalid-argument-type]
