from halib import *

from dataclasses import dataclass
from halib.filetype import yamlfile
from dataclass_wizard import YAMLWizard
from typing import List, Optional, Dict, Any
from halib.research.base_config import ExpBaseConfig, NamedConfig
from halib.research.mics import *

import yaml

SEP = "__"

@dataclass
class DatasetInfo(YAMLWizard, NamedConfig):
    name: str = None
    dir_path: str = None
    extra_cfgs: Optional[Dict[str, Any]] = None

    def get_name(self):
        return self.name

    def get_num_videos(self, recursive=False):
        video_files = fs.filter_files_by_extension(self.dir_path, [".mp4", ".avi", ".mov"], recursive=recursive)
        return len(video_files)

@dataclass
class MetricSet(YAMLWizard, NamedConfig):
    name: str
    metric_names: List[str]
    extra_cfgs: Optional[Dict[str, Any]] = None

    def get_name(self):
        return self.name


@dataclass
class TrainCfgSet(YAMLWizard, NamedConfig):
    name: str
    epochs: int
    batch_size: int
    device: List[int]
    num_workers: int
    extra_cfgs: Optional[Dict[str, Any]] = None

    def get_name(self):
        return self.name


@dataclass
class Method(YAMLWizard, NamedConfig):
    name: str
    extra_cfgs: Optional[Dict[str, Any]] = None

    def get_name(self):
        return self.name


@dataclass
class WanDBConfig(YAMLWizard):
    project: str
    mode: str


@dataclass
class LogConfig(YAMLWizard):
    wandb_cfg: WanDBConfig = None


@dataclass
class GeneralConfig(YAMLWizard):
    seed: int
    outdir: str
    log_cfg: LogConfig
    computer_name: Optional[str] = None
    time_stamp: Optional[str] = None


@dataclass
class DatasetConfig(YAMLWizard):
    list_datasets: List[DatasetInfo] = None
    selected_dataset_name: str = None
    dataset_used: DatasetInfo = None

    def post_init(self):
        if self.selected_dataset_name is None:
            raise ValueError("No dataset selected in the configuration.")
        for dataset in self.list_datasets:
            if dataset.name == self.selected_dataset_name:
                self.dataset_used = dataset

        if self.dataset_used is None:
            raise ValueError(
                f"Dataset '{self.selected_dataset_name}' not found in the configuration."
            )


@dataclass
class MetricConfig(YAMLWizard):
    list_metrics: List[MetricSet] = None
    selected_metric_set_name: str = None
    metricSet_used: Optional[MetricSet] = None

    def post_init(self):
        """
        Post-initialization to set the used metric set based on the selected name.
        """
        if self.selected_metric_set_name is None:
            raise ValueError("No metric set selected in the configuration.")
        for metric_set in self.list_metrics:
            if metric_set.name == self.selected_metric_set_name:
                self.metricSet_used = metric_set
                return

        raise ValueError(
            f"Metric set '{self.selected_metric_set_name}' not found in the configuration."
        )

@dataclass
class MethodConfig(YAMLWizard):
    list_methods: List[Method]
    selected_method_name: str = None
    method_used: Optional[Method] = None

    def post_init(self):
        """
        Post-initialization to set the used method based on the selected name.
        """
        if self.selected_method_name is None:
            raise ValueError("No method selected in the configuration.")
        for method in self.list_methods:
            if method.name == self.selected_method_name:
                self.method_used = method
                return

        raise ValueError(
            f"Method '{self.selected_method_name}' not found in the configuration."
        )


@dataclass
class InferConfig(YAMLWizard):
    do_infer: bool
    skip_if_exists: bool
    limit: int
    save_video_results: bool
    save_csv_results: bool
    csv_columns: List[str]
    calc_metrics: bool
    verbose: bool

@dataclass
class ModelConfig(YAMLWizard):
    base_model: str
    model_path: str
    class_names: List[str]

@dataclass
class Config(ExpBaseConfig):
    general: GeneralConfig
    dataset_cfg: DatasetConfig
    metric_cfg: MetricConfig
    method_cfg: MethodConfig
    infer_cfg: InferConfig
    model_cfg: ModelConfig
    cfg_name: Optional[str] = None

    def post_init(self):
        selected_method = self.method_cfg.method_used
        assert selected_method is not None, "No method selected in the configuration"
        # get computer name
        if (
            isinstance(self.general.computer_name, str)
            and len(self.general.computer_name) > 0
        ):
            # pprint(f"1>.COMPUTER NAME: {self.general.computer_name=}")
            abbr = self.general.computer_name
        else:
            abbr = get_PC_abbr_name()
            # pprint(f"COMPUTER NAME: {abbr=}")
            self.general.computer_name = abbr  # set computer name in general config

        if self.general.time_stamp is None:
            # set time_stamp to current time
            self.general.time_stamp = now_str()

        name_parts = [
            abbr,
            f"ds_{self.dataset_cfg.dataset_used.name}",
            f"mt_{self.method_cfg.method_used.name}",
            self.general.time_stamp,
        ]
        self.cfg_name = SEP.join(name_parts)

    @classmethod
    def from_custom_yaml_file(cls, yaml_file: str) -> "Config":
        def load_yaml(yaml_file_path: str) -> Dict[str, Any]:
            cfg_dict = yamlfile.load_yaml(yaml_file_path, to_dict=True)
            if "__base__" in cfg_dict:
                del cfg_dict["__base__"]
            return cfg_dict

        cfg_dict = load_yaml(yaml_file)
        yaml_str = yaml.dump(cfg_dict, default_flow_style=False)
        instance = Config.from_yaml(yaml_str)

        def get_attr_name(obj, attr_value):
            for name, value in vars(obj).items():
                if value is attr_value:
                    return name
            return None

        cfg_attr_to_class = {
            "dataset_cfg": DatasetInfo,
            "metric_cfg": MetricSet,
            "method_cfg": Method,
        }

        for attr_name, cls_type in cfg_attr_to_class.items():
            attr_obj = getattr(instance, attr_name)
            # find the attr of attr_obj that is a list (must me start with 'list_')
            list_attr_post_fix = (
                attr_name.split("_")[0] + "s"
            )  # plural; e.g. 'datasets', 'metrics', 'trains', 'methods'
            list_attr_name = get_attr_name(
                attr_obj, getattr(attr_obj, "list_" + list_attr_post_fix)
            )

            attr_folder = f"config/{list_attr_post_fix}"
            # filter all yaml files in the attr_folder
            yaml_files = fs.filter_files_by_extension(
                attr_folder, ".yaml", recursive=False
            )
            for yaml_file in yaml_files:
                # load the yaml file
                yaml_data_dict = load_yaml(yaml_file)
                fname = fs.get_file_name(yaml_file, split_file_ext=True)[0]
                # dump yaml_data_dict to yaml string
                yaml_data_dict["name"] = (
                    fname  # set the name of the item to the file name
                )
                yaml_data_str = yaml.dump(yaml_data_dict, default_flow_style=False)
                # create an instance of the class
                item_instance = cls_type.from_yaml(yaml_data_str)
                # append to the list attribute of the attr_obj
                getattr(attr_obj, list_attr_name).append(item_instance)

        # post-init
        instance.dataset_cfg.post_init()
        instance.metric_cfg.post_init()
        instance.method_cfg.post_init()
        instance.post_init()
        return instance

    # implement base class methods
    def get_general_cfg(self) -> GeneralConfig:
        """
        Get the general configuration like output directory, log settings, SEED, etc.
        """
        return self.general

    def get_dataset_cfg(self) -> DatasetInfo:
        """
        Get the dataset configuration.
        """
        return self.dataset_cfg.dataset_used

    def get_metric_cfg(self) -> MetricSet:
        """
        Get the metric configuration.
        """
        return self.metric_cfg.metricSet_used

    def get_cfg_name(self):
        """
        Get the name of the configuration.
        """
        return self.cfg_name

    def get_outdir(self):
        return os.path.join(self.general.outdir, self.cfg_name)
