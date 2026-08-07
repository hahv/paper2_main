from IPython.testing.decorators import skip
from dill.tests.test_registered import p
from halib import *
from halib.common.common import seed_everything
from halib.exp.core.base_exp import BaseExp
from halib.exp.perf.perfmetrics import MetricsBackend, TorchMetricsBackend

from src.config import *
from src.metrics.custom_metrics import MetricFactory
from src.methods.base_method import *
from halib.utils.dict import DictUtils
from src.common import GlobalConst

from pathlib import Path
from src.metrics.loaders.base_csv_loader import BaseRawCsvLoader


class MyExp(BaseExp):
    """
    Custom experiment class that extends BaseExperiment.
    """

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(config)
        self.full_cfg: Config = config
        self.metric_results = {}
        self.metric_backend = None
        self.video_dir_path = None
        self.exp_time_start = 0

    # ! add a function to allow dynmmically load with baseline method (not Paper2Exp class) by creating a custom Config
    @classmethod
    def from_standard_exp(cls, exp_dir_path: str) -> "MyExp":
        """
        Reload a previously-completed standard experiment from its output directory.
        Reconstructs the Config from the saved __config.yaml (original-yaml-str),
        skips video inference (reuses existing CSVs), and forces metric recalculation.
        """
        exp_dir = Path(exp_dir_path)
        assert exp_dir.is_dir(), f"Provided path is not a directory: {exp_dir_path}"
        config_file = exp_dir / "__config.yaml"
        assert config_file.exists(), (
            f"Config file not found in experiment directory: {config_file}\n"
            f"Only standard experiments (produced by Paper2Exp) are supported."
        )
        cfg_data = yamlfile.load_yaml(str(config_file), to_dict=True)
        original_yaml_str = cfg_data.get("original-yaml-str")
        assert original_yaml_str, (
            f"'original-yaml-str' key missing in {config_file}. "
            f"The experiment may have been created without saving the original YAML string."
        )
        exp_cfg = Config.from_custom_yaml_file_or_str(original_yaml_str)

        # ! Allow run_exp to proceed — we want metrics recalculated, not skipped
        exp_cfg.general.skip_exp_if_exists = False

        # update the experiment directory so that it correctly points to the new path (even if moved)
        exp_cfg.update_custom_exp(str(exp_dir))

        # Skip per-video inference if result CSVs already exist — reuse them
        exp_cfg.inferCfg.skip_if_exists = True
        return cls(exp_cfg)

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
        # ! Force binary classification for metrics calculation (fire_smoke OR none)
        num_classes = 2
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
            for metric_instance in self.metric_backend.metric_info.values():  # ty:ignore[unresolved-attribute]
                metric_instance.reset()
        
    def calc_skip_rate(self) -> float:
        """
        Calculates the skip rate for temporal methods with skip processing enabled.

        The skip rate is defined as the ratio of correctly skipped frames to the
        total number of safe frames (i.e., frames without fire or smoke). If the
        current method does not utilize a skip process, this returns 0.0.

        Returns:
            float: The calculated skip rate.
        """
        method_cfg = self.full_cfg.methodCfg

        is_temp_method = "temp_method" in method_cfg.name  # ty:ignore[unsupported-operator]
        has_skip_proc = method_cfg.extra_cfgs and "skip_proc" in method_cfg.extra_cfgs

        # Early exit if the method does not support skip processing
        if not (is_temp_method and has_skip_proc):
            return -1

        csv_loader_name = self.full_cfg.dbsetCfg.extra_cfgs.get("csv_loader_cls")  # ty:ignore[unresolved-attribute]

        # ! create a dynamic loader
        csv_loader_cls = get_cls_in_pkg(
            pkg_name="src.metrics.loaders",
            fileName_ClsName=csv_loader_name,  # ty:ignore[invalid-argument-type]
        )
        # Initialize loader
        self.csv_loader: BaseRawCsvLoader = csv_loader_cls(self.full_cfg)
        video_list = self.full_cfg.dbsetCfg.get_video_list()
        # ! global cache of video_name => raw_gt_pred_df
        gt_pred_df_list = []
        for vpath in video_list:
            raw_gt_pred_df = self.csv_loader.load_video_gt_pred_df(video_path=vpath)
            assert raw_gt_pred_df is not None, (
                f"Failed to load GT/Pred for video {vpath}"
            )
            gt_pred_df_list.append(raw_gt_pred_df)
        # Concatenate all video DataFrames into a single DataFrame
        all_videos_df = pd.concat(gt_pred_df_list, ignore_index=True)        
        pprint_box(all_videos_df.columns.tolist(), title="Columns in GT/Pred DataFrame")
        # columns = ['frame_idx', 'video_path', 'gt_label', 'video',
        # 'num_frames', 'elapsed_time', 'class_names', 'logits', 'probs',
        # 'pred_label_idx', 'pred_label']
        
        outdir = self.full_cfg.get_outdir()
        # save all df to CSV for debugging
        all_videos_df.to_csv(f"{outdir}/_all_videos_gt_pred_skip_rate_calc.csv", sep=";", encoding="utf-8", index=False)
        
        safe_frames_df = all_videos_df[all_videos_df["gt_label"] == "none"]
        # count correctly skipped frames (pred_label == "none" and gt_label == "none")
        correctly_skipped_frames_df = safe_frames_df[safe_frames_df["pred_label"] == "skipped"]

        # debug:
        console.print(f"Total safe frames: {len(safe_frames_df)}")
        console.print(f"Correctly skipped frames: {len(correctly_skipped_frames_df)}")
                
        skip_rate = len(correctly_skipped_frames_df) / len(safe_frames_df) if len(safe_frames_df) > 0 else 0.0
        return skip_rate
    
    def run_exp(self, should_calc_metrics=True, reload_env=False, *args, **kwargs):
        with ConsoleLog("Exp Info", characters="🔻"):
            exp_info = {
                "dataset": self.full_cfg.dbsetCfg.get_name(),
                "method": self.full_cfg.methodCfg.name,
                "is_optim_mode": self.full_cfg.general.is_optim_mode,
            }
            pprint_box(
                exp_info, title=f"Running Experiment: {self.full_cfg.get_cfg_name()}"
            )

        if self.full_cfg.shouldSkipExp:
            return
        self.exp_time_start = time.perf_counter()

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
                
                # ! add skip rate as extra data item
                if mode == "per_frame":
                    skip_rate = self.calc_skip_rate()
                    if skip_rate >= 0.0:
                        if extra_data is None:
                            extra_data = {}
                        extra_data["skip_rate"] = skip_rate

                perf_results, outfile = self.calc_perfs(
                    raw_metrics_data=metrics_data,
                    extra_data=extra_data,
                    outfile=outfile,
                    return_df=False,
                    *args,  # ty:ignore[parameter-already-assigned]
                    **kwargs,
                )
                df = pd.read_csv(outfile, sep=";", encoding="utf-8")  # ty:ignore[no-matching-overload]
                df.at[0, "experiment"] = f"{self.full_cfg.get_cfg_name()}_{mode}"
                df.to_csv(outfile, sep=";", encoding="utf-8", index=False)
                csvfile.fn_display_df(df)
                pprint_local_path(outfile, get_wins_path=True)  # ty:ignore[invalid-argument-type]

        # ! Final experiment summary with total time
        with ConsoleLog("Exp end summary", characters="🔺"):
            exp_time = time.perf_counter() - self.exp_time_start
            console.print(f"Exp time (Secs): {exp_time:.2f} seconds")
            # convert to hours, minutes, seconds
            exp_time_hms = time.strftime("%H:%M:%S", time.gmtime(exp_time))
            console.print(f"Exp time (h:m:s): {exp_time_hms}")
            # write to file in the output directory
            exp_end_summary_file = self.full_cfg.get_outdir() + "/__exp_end_summary.txt"
            with open(exp_end_summary_file, "w") as f:
                f.write(f"Exp time (Secs): {exp_time:.2f} seconds\n")
                f.write(f"Exp time (h:m:s): {exp_time_hms}\n")