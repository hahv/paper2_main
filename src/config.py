from halib import *
import yaml

from dataclasses import dataclass, field
from dataclass_wizard import YAMLWizard
from typing import List, Optional, Dict, Any

from halib.filetype import yamlfile
from halib.research.mics import *
from halib.system.path import *
from halib.exp.core.base_config import (
    AutoNamedCfg,
    BaseSelectorCfg,
    ExpBaseCfg,
    NamedCfg,
)

# -----------------------------------------------------------------------------
# 4. GENERAL CONFIGS
# -----------------------------------------------------------------------------


@dataclass
class WanDBCfg(YAMLWizard):
    project: str
    mode: str
    wandb_key: str


@dataclass
class LogCfg(YAMLWizard):
    wandb_cfg: Optional[WanDBCfg] = None


@dataclass
class GeneralConfig(YAMLWizard):
    seed: int
    outdir: str
    log_cfg: LogCfg
    computer_name: Optional[str] = None
    time_stamp: Optional[str] = None


@dataclass
class GeneralCfg(NamedCfg, YAMLWizard):
    seed: int
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
        self.project_dir = normalize_paths(self.project_dir)


@dataclass
class InferConfig(YAMLWizard):
    do_infer: bool
    skip_if_exists: bool
    limit: int
    save_video_results: bool
    save_csv_results: bool
    csv_columns: List[str]
    calc_metrics: bool
    log_transforms: bool
    use_profiler: bool
    verbose: bool


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
    dir_path: str = None
    extra_cfgs: Optional[Dict[str, Any]] = None
    vname2path: Optional[Dict[str, str]] = None

    def get_vname2path(self, recursive=True):
        if self.vname2path is None:
            video_files = fs.filter_files_by_extension(
                self.dir_path, [".mp4", ".avi", ".mov"], recursive=recursive
            )
            self.vname2path = {
                fs.get_file_name(fpath, split_file_ext=True)[0]: fpath
                for fpath in video_files
            }
        return self.vname2path

    def get_num_videos(self, recursive=False):
        video_files = fs.filter_files_by_extension(
            self.dir_path, [".mp4", ".avi", ".mov"], recursive=recursive
        )
        return len(video_files)

    def get_csv_labels(self, recursive=False):
        csv_files = fs.filter_files_by_extension(
            self.dir_path, [".csv"], recursive=recursive
        )
        return csv_files


@dataclass
class MetricSetCfg(AutoNamedCfg):
    metric_names: List[str] = field(default_factory=list)
    extra_cfgs: Optional[Dict[str, Any]] = None


@dataclass
class MethodCfg(AutoNamedCfg):
    extra_cfgs: Optional[Dict[str, Any]] = None


@dataclass
class DatasetSelector(BaseSelectorCfg[DatasetCfg]):
    list_dbsets: List[DatasetCfg] = field(default_factory=list)
    selected_dbset: Optional[str] = None
    dbset_used: Optional[DatasetCfg] = None

    def post_init(self):
        self.dbset_used = self._resolve_selection(
            self.list_dbsets, self.selected_dbset, "dataset"
        )


@dataclass
class MetricSelector(BaseSelectorCfg[MetricSetCfg]):
    list_metrics: List[MetricSetCfg] = field(default_factory=list)
    selected_metric: Optional[str] = None
    metric_used: Optional[MetricSetCfg] = None

    def post_init(self):
        self.metric_used = self._resolve_selection(
            self.list_metrics, self.selected_metric, "metric set"
        )


@dataclass
class MethodSelector(BaseSelectorCfg[MethodCfg]):
    list_methods: List[MethodCfg] = field(default_factory=list)
    selected_method: Optional[str] = None
    method_used: Optional[MethodCfg] = None

    def post_init(self):
        self.method_used = self._resolve_selection(
            self.list_methods, self.selected_method, "method"
        )


# -----------------------------------------------------------------------------
# 5. MAIN CONFIG
# -----------------------------------------------------------------------------


@dataclass
class Config(ExpBaseCfg):
    dbset_selector: DatasetSelector
    metric_selector: MetricSelector
    method_selector: MethodSelector
    general: GeneralCfg
    inferCfg: InferConfig
    modelCfg: ModelConfig

    # --- Base Class Implementations ---
    def get_general_cfg(self) -> GeneralCfg:
        return self.general

    def get_dataset_cfg(self) -> DatasetCfg:
        return self.dbset_selector.dbset_used

    def get_metric_cfg(self) -> MetricSetCfg:
        return self.metric_selector.metric_used

    def get_method_cfg(self) -> MethodCfg:
        return self.method_selector.method_used

    def get_cfg_name(self, sep="__", *args, **kwargs):
        extra_info = self.general.time_stamp
        return super().get_cfg_name(sep, extra=extra_info, *args, **kwargs)

    def get_outdir(self):
        exp_outdir = self.expDir
        # pprint(f"Experiment output directory: {exp_outdir}")
        return exp_outdir

    # --- SHORTCUT PROPERTIES (THE REQUESTED FEATURE) ---
    @property
    def dbsetCfg(self) -> DatasetCfg:
        return self.dbset_selector.dbset_used

    @property
    def metricCfg(self) -> MetricSetCfg:
        return self.metric_selector.metric_used

    @property
    def methodCfg(self) -> MethodCfg:
        return self.method_selector.method_used

    @property
    def expDir(self) -> str:
        assert self.cfg_name is not None, "cfg_name is not set"
        return os.path.join(self.general.project_dir, self.general.outdir, self.cfg_name)

    def print_meta_info(self) -> str:
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
        pprint("Finalizing configuration...")
        # 1. Resolve sub-configs
        self.dbset_selector.post_init()
        self.metric_selector.post_init()
        self.method_selector.post_init()
        # ! must called for generating cfg_name
        self.get_cfg_name()

    @classmethod
    def from_custom_yaml_file(cls, yaml_file_or_dict: str) -> "Config":
        """
        Wrapper for from_custom_yaml_file_or_str
        """
        return cls.from_custom_yaml_file_or_str(yaml_file_or_dict)

    @classmethod
    def from_custom_yaml_file_or_str(cls, yaml_file_or_dict: str) -> "Config":
        """
        Loads the main config, then scans specific folders to populate lists
        (e.g., list_datasets) from external YAML files.
        """
        yaml_str = ""
        if isinstance(yaml_file_or_dict, str) and os.path.exists(yaml_file_or_dict):
            # 1. Load Base Config
            cfg_dict = yamlfile.load_yaml(yaml_file_or_dict, to_dict=True)
            if "__base__" in cfg_dict:
                del cfg_dict["__base__"]
            yaml_str = yaml.dump(cfg_dict, default_flow_style=False)
        elif isinstance(yaml_file_or_dict, dict):
            yaml_str = yaml.dump(yaml_file_or_dict, default_flow_style=False)
        instance = Config.from_yaml(yaml_str)
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
                instance.general.project_dir, f"config/{folder_name}"
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
        instance.finalize_config()
        return instance
