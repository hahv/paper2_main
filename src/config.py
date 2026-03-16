from sqlalchemy.sql.operators import from_
from halib import *
import yaml

from dataclasses import dataclass, field
from dataclass_wizard import YAMLWizard
from typing import List, Optional, Dict, Any
from halib.utils.dict import DictUtils

from halib.filetype import yamlfile
from halib.research.mics import *
from halib.system.path import *
from halib.exp.core.base_config import (
    AutoNamedCfg,
    BaseSelectorCfg,
    ExpBaseCfg,
    NamedCfg,
)
from src.common import GlobalConst

import wandb
from lightning.pytorch.loggers.wandb import WandbLogger
# -----------------------------------------------------------------------------
# 4. GENERAL CONFIGS
# -----------------------------------------------------------------------------


@dataclass
class WanDBCfg(YAMLWizard):
    project: str
    mode: str
    wandb_key: str

    def get_logger(self, name: Optional[str] = None) -> WandbLogger:
        # 1. Authenticate using the key from config
        # relogin=True ensures it overwrites any previously cached local keys
        if self.wandb_key:
            wandb.login(key=self.wandb_key, relogin=True)
        wandb_logger = WandbLogger(
            project=self.project,
            mode=self.mode,
            name=name,
        )
        return wandb_logger

    def get_hash(self):
        cfg_dict = yaml.safe_load(self.to_yaml())
        return DictUtils.get_unique_hash(cfg_dict)


@dataclass
class LogCfg(YAMLWizard):
    wandb_cfg: Optional[WanDBCfg] = None


@dataclass
class GeneralCfg(NamedCfg, YAMLWizard):
    seed: int
    skip_exp_if_exists: bool
    is_optim_mode: bool
    project_dir: str
    outdir: str
    log_cfg: LogCfg
    computer_name: Optional[str] = None
    time_stamp: Optional[str] = None

    def get_name(self):
        return self.computer_name

    def __post_init__(self):
        if self.time_stamp is None:
            self.time_stamp = now_str()
        if self.computer_name is None or len(str(self.computer_name)) == 0:
            self.computer_name = get_PC_abbr_name()
        self.project_dir = normalize_paths(self.project_dir)  # ty:ignore[missing-argument]


@dataclass
class InferConfig(YAMLWizard):
    do_infer: bool
    num_infer_workers: int
    skip_if_exists: bool
    limit: int
    save_video_results: bool
    save_csv_results: bool
    csv_columns: List[str]
    calc_metrics: bool
    log_transforms: bool
    use_profiler: bool
    verbose: bool
    save_timeline_vis: Optional[bool] = True
    timeline_table_mode: Optional[str] = (
        "p"  # options: p (percent), fc (frame count), both (pfc)
    )
    csv_infer_pattern: Optional[str] = GlobalConst.INFER_FILE_PATTERN
    timeline_video_name_limit: Optional[int] = 40
    timeline_table_decimals: Optional[int] = 4

@dataclass
class ModelConfig(YAMLWizard):
    base_model: str
    model_path: str
    class_names: List[str]
    input_size: Optional[List[int]] = None


# -----------------------------------------------------------------------------
# 3. SELECTION CONFIGS
# -----------------------------------------------------------------------------
@dataclass
class DatasetCfg(AutoNamedCfg):
    dir_path: Optional[str] = None
    extra_cfgs: Optional[Dict[str, Any]] = None
    vname2path: Optional[Dict[str, str]] = None

    def get_video_list(self):
        recursive = True
        if self.extra_cfgs:
            recursive = self.extra_cfgs.get("ds_recursive", recursive)
        video_files = fs.filter_files_by_extension(
            self.dir_path, [".mp4", ".avi", ".mov"], recursive=recursive
        )
        assert len(video_files) > 0, f"No video files found in {self.dir_path}"
        return video_files

    def get_num_videos(self):
        return len(self.get_video_list())

    def get_gt_file_pattern(self):
        pattern: str = GlobalConst.GT_FILE_PATTERN
        if self.extra_cfgs is not None:
            pattern = self.extra_cfgs.get("ds_gt_file_pattern", pattern)
        return pattern


@dataclass
class MetricSetCfg(AutoNamedCfg):
    metric_names: List[str] = field(default_factory=list)
    extra_cfgs: Optional[Dict[str, Any]] = None


@dataclass
class MethodCfg(AutoNamedCfg):
    extra_cfgs: Optional[Dict[str, Any]] = None

    def get_skip_method_name(self) -> str:
        skip_name = "Unknown"
        try:
            skip_name = self.extra_cfgs.get("skip_proc").get("name", "Unknown")  # ty:ignore[possibly-missing-attribute]
        except Exception:
            pass
        return skip_name

    def get_dict(self) -> Dict[str, Any]:
        return {
            "method_name": self.name,
            "extra_cfgs": self.extra_cfgs,
        }

    def get_wandb_dict(
        self, config_mask: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Dictionary representation suitable for logging to Weights & Biases."""
        if config_mask is None:
            return self.get_dict()
        else:
            filtered_dict = DictUtils.apply_inclusion_mask(self.extra_cfgs, config_mask)  # ty:ignore[invalid-argument-type]
            wandb_dict = {"method_name": self.name}
            wandb_dict.update(filtered_dict)
            return wandb_dict


@dataclass
class DatasetSelector(BaseSelectorCfg[DatasetCfg]):
    list_dbsets: List[DatasetCfg] = field(default_factory=list)
    selected_dbset: Optional[str] = None
    dbset_used: Optional[DatasetCfg] = None

    def post_init(self):
        self.dbset_used = self._resolve_selection(
            self.list_dbsets,
            self.selected_dbset,  # ty:ignore[invalid-argument-type]
            "dataset",
        )


@dataclass
class MetricSelector(BaseSelectorCfg[MetricSetCfg]):
    list_metrics: List[MetricSetCfg] = field(default_factory=list)
    selected_metric: Optional[str] = None
    metric_used: Optional[MetricSetCfg] = None

    def post_init(self):
        self.metric_used = self._resolve_selection(
            self.list_metrics,
            self.selected_metric,  # ty:ignore[invalid-argument-type]
            "metric set",
        )


@dataclass
class MethodSelector(BaseSelectorCfg[MethodCfg]):
    list_methods: List[MethodCfg] = field(default_factory=list)
    selected_method: Optional[str] = None
    method_used: Optional[MethodCfg] = None

    def post_init(self):
        self.method_used = self._resolve_selection(
            self.list_methods,
            self.selected_method,  # ty:ignore[invalid-argument-type]
            "method",
        )


# -----------------------------------------------------------------------------
# 5. MAIN CONFIG
# -----------------------------------------------------------------------------


@dataclass
class Config(ExpBaseCfg):
    CFG_SEP = "__"
    dbset_selector: DatasetSelector
    metric_selector: MetricSelector
    method_selector: MethodSelector
    general: GeneralCfg
    inferCfg: InferConfig
    modelCfg: ModelConfig
    original_yaml_str: Optional[str] = None
    custom_exp_dir: Optional[str] = None

    def save_to_outdir(
        self, filename: str = "__config.yaml", outdir=None, override: bool = False
    ) -> None:
        super().save_to_outdir(filename, outdir, override)
        # !we also save the method_cfg.yaml for easier reference
        method_dict = self.methodCfg.get_dict()
        outfile = os.path.join(self.get_outdir(), "__method_cfg.yaml")
        with open(outfile, "w") as f:
            yaml.dump(method_dict, f, default_flow_style=False)

    # --- Base Class Implementations ---
    def get_general_cfg(self) -> GeneralCfg:
        return self.general

    def get_dataset_cfg(self) -> DatasetCfg:
        return self.dbset_selector.dbset_used  # ty:ignore[invalid-return-type]

    def get_metric_cfg(self) -> MetricSetCfg:
        return self.metric_selector.metric_used  # ty:ignore[invalid-return-type]

    def get_method_cfg(self) -> MethodCfg:
        return self.method_selector.method_used  # ty:ignore[invalid-return-type]

    def get_cfg_name(self, sep=CFG_SEP, *args, **kwargs):
        if not self.custom_exp_dir:
            time_stamp_info = self.general.time_stamp
            mt_cfg_hash = DictUtils.get_unique_hash(
                self.method_selector.method_used.extra_cfgs  # ty:ignore[unresolved-attribute]
            )
            extra_info = f"{mt_cfg_hash}{sep}{time_stamp_info}"
            return super().get_cfg_name(sep, extra=extra_info, *args, **kwargs)
        else:
            # For custom experiments, we ignore the usual cfg_name generation and just use the directory name as cfg_name
            return os.path.basename(self.custom_exp_dir)

    def get_outdir(self):
        exp_outdir = self.expDir
        # pprint(f"Experiment output directory: {exp_outdir}")
        return exp_outdir

    # --- SHORTCUT PROPERTIES (THE REQUESTED FEATURE) ---
    @property
    def dbsetCfg(self) -> DatasetCfg:
        return self.dbset_selector.dbset_used  # ty:ignore[invalid-return-type]

    @property
    def metricCfg(self) -> MetricSetCfg:
        return self.metric_selector.metric_used  # ty:ignore[invalid-return-type]

    @property
    def methodCfg(self) -> MethodCfg:
        return self.method_selector.method_used  # ty:ignore[invalid-return-type]

    @property
    def expDir(self) -> str:
        if not self.custom_exp_dir:
            # Default behavior: construct expDir from project_dir, outdir, and cfg_name
            assert self.cfg_name is not None, "cfg_name is not set"
            return os.path.join(
                self.general.project_dir, self.general.outdir, self.cfg_name
            )
        else:
            return self.custom_exp_dir

    @property
    def expSameCfgExists(self) -> tuple[bool, str]:
        # Check if experiment with the same cfg existed.
        exp_with_same_cfg_existed = False
        time_stamp_info = self.general.time_stamp
        cfg_name_posfix_len = len(f"{self.CFG_SEP}{time_stamp_info}")
        exp_dir_with_hash = self.expDir[:-cfg_name_posfix_len]

        # Check if any directory in general.project_dir/general.outdir matches exp_dir_with_hash
        exp_outdir = os.path.join(self.general.project_dir, self.general.outdir)
        existing_dir = ""
        # find directories that start with exp_dir_with_hash
        if os.path.exists(exp_outdir):
            for name in os.listdir(exp_outdir):
                dir_path = os.path.join(exp_outdir, name)
                if os.path.isdir(dir_path) and name.startswith(
                    os.path.basename(exp_dir_with_hash)
                ):
                    exp_with_same_cfg_existed = True
                    existing_dir = dir_path
                    break
        return (exp_with_same_cfg_existed, existing_dir)

    @property
    def shouldSkipExp(self) -> bool:
        exists, existing_dir = self.expSameCfgExists
        should_skip = exists and self.general.skip_exp_if_exists
        if should_skip:
            with ConsoleLog("[red] <<Skip>> Exp existed [/red]"):
                pprint_local_path(existing_dir, get_wins_path=True)
        return should_skip

    def get_wandb_logger(self, name: Optional[str] = None) -> Optional[WandbLogger]:
        logger = None
        if self.general.log_cfg.wandb_cfg is not None:
            logger = self.general.log_cfg.wandb_cfg.get_logger(name=name)
        return logger

    def print_meta_info(self):
        with ConsoleLog("Meta Info"):
            pprint(self.dbsetCfg)
            pprint(self.methodCfg)
            pprint(self.metricCfg)
            pprint(self.modelCfg)
            pprint(self.inferCfg)

    # ---------------------------------------------------

    def finalize_config(self):
        """
        Links names (strs) to actual objects and generates the canonical Config Name.
        """
        # pprint("Finalizing configuration...")
        # 1. Resolve sub-configs
        self.dbset_selector.post_init()
        self.metric_selector.post_init()
        self.method_selector.post_init()
        # ! must called for generating cfg_name
        self.get_cfg_name()

    # !#TODO: TO_DELETE
    # def get_infer_csv_files(self, recursive=False):
    #     exp_dir = self.get_outdir()
    #     csv_files = fs.filter_files_by_extension(
    #         exp_dir, [".csv"], recursive=recursive
    #     )
    #     # only keep those that match the gt file pattern
    #     infer_pattern = self.inferCfg.csv_infer_pattern
    #     csv_files = [f for f in csv_files if infer_pattern in os.path.basename(f)]
    #     assert len(csv_files) > 0, f"No infer CSV files found in {exp_dir} with pattern {infer_pattern}"
    #     return csv_files

    def update_optim_params(self, optim_params: Dict[str, Any]) -> None:
        """
        Update the method extra_cfgs with the given optimization parameters.
        """
        if self.methodCfg.extra_cfgs is None:
            self.methodCfg.extra_cfgs = {}
        self.methodCfg.extra_cfgs = optim_params  # force replace
        # ! must update cfg_name after changing method cfgs
        self.get_cfg_name()

    @classmethod
    def from_custom_yaml_file(cls, yaml_file_or_dict: str) -> "Config":  # ty:ignore[invalid-method-override]
        """
        Wrapper for from_custom_yaml_file_or_str
        """
        return cls.from_custom_yaml_file_or_str(yaml_file_or_dict)

    @classmethod
    def from_yaml_str(cls, yaml_str: str) -> "Config":
        instance = Config.from_yaml(yaml_str)
        instance.original_yaml_str = yaml_str  # ty:ignore[invalid-assignment]
        # 2. Configuration for dynamic loading
        # Map: Attribute Name -> (Class Type, Folder Name Suffix)
        # Note: Logic assumes list attribute is "list_" + suffix + "s"
        load_map = {
            "dbset_selector": (DatasetCfg, "dbset"),
            "metric_selector": (MetricSetCfg, "metric"),
            "method_selector": (MethodCfg, "method"),
        }

        for attr_name, (cls_type, file_suffix) in load_map.items():
            cfg_obj = getattr(instance, attr_name)

            # folder: e.g., config/datasets, config/trains
            folder_name = f"{file_suffix}s"
            attr_folder = os.path.join(
                instance.general.project_dir,  # ty:ignore[unresolved-attribute]
                f"config/{folder_name}",
            )

            # list attribute: e.g., list_datasets
            list_attr_name = f"list_{folder_name}"

            # Scan folder for YAMLs
            if os.path.exists(attr_folder):
                found_files = fs.filter_files_by_extension(
                    attr_folder, ".yaml", recursive=False
                )

                target_list = getattr(cfg_obj, list_attr_name)

                for fpath in found_files:
                    # pprint(f'Loading config file: {fpath}')
                    data = yamlfile.load_yaml(fpath, to_dict=True)

                    # Inject filename as 'name'
                    fname = fs.get_file_name(fpath, split_file_ext=True)[0]
                    data["name"] = fname

                    # Convert to Object and Append
                    item_str = yaml.dump(data, default_flow_style=False)
                    item_instance = cls_type.from_yaml(item_str)
                    target_list.append(item_instance)
            else:
                print(f"Warning: Config folder not found: {attr_folder}")

        # 3. Finalize (Link strings to objects)
        instance.finalize_config()  # ty:ignore[unresolved-attribute]
        return instance  # ty:ignore[invalid-return-type]

    @classmethod
    def from_custom_yaml_file_or_str(cls, yaml_file_or_dict: str) -> "Config":
        """
        Loads the main config, then scans specific folders to populate lists
        (e.g., list_datasets) from external YAML files.
        """
        yaml_str = ""
        if isinstance(yaml_file_or_dict, str):
            if os.path.exists(yaml_file_or_dict):  # is a file path
                # 1. Load Base Config
                cfg_dict = yamlfile.load_yaml(yaml_file_or_dict, to_dict=True)
                if "__base__" in cfg_dict:
                    del cfg_dict["__base__"]
                yaml_str = yaml.dump(cfg_dict, default_flow_style=False)
            else:  # is a yaml string
                yaml_str = yaml_file_or_dict
        elif isinstance(yaml_file_or_dict, dict):
            yaml_str = yaml.dump(yaml_file_or_dict, default_flow_style=False)
        else:
            raise ValueError("Input must be a file path, YAML string, or dict.")
        return cls.from_yaml_str(yaml_str)

    def update_custom_exp(self, new_exp_dir: str):
        """
        Redirect get_outdir() to the given external experiment directory.

        expDir computes: os.path.join(general.project_dir, general.outdir, cfg_name)
        We set project_dir = parent of new_exp_dir, outdir = "", cfg_name = dir name
        so that expDir resolves directly to new_exp_dir.
        """
        assert os.path.exists(new_exp_dir) and os.path.isdir(new_exp_dir), (
            f"Not found - custom exp dir: {new_exp_dir}"
        )

        self.custom_exp_dir = new_exp_dir
        console.print(
            "[bold red] This config is updated for custom experiment. [/bold red]"
        )
        exp_path = Path(os.path.abspath(new_exp_dir)).resolve()
        self.general.project_dir = str(exp_path.parent)
        self.general.outdir = ""
        self.cfg_name = exp_path.name

    def update_for_optim_mode(self):
        """
        When in optimization mode, we want to save all runs under a common directory for easier comparison.
        This function modifies the general.project_dir and general.outdir to achieve that.
        """
        optim_dir_name = "optim_runs"
        self.general.outdir = os.path.join(self.general.outdir, optim_dir_name)