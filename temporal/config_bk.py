from halib import *
from dataclasses import dataclass
from typing import List, Optional
from halib.filetype import yamlfile
from dataclass_wizard import YAMLWizard
from typing import List, Optional, Dict, Any, Union

import torchmetrics
from torchmetrics import Metric
from .metrics import FPS, FPR
import yaml

SEP = "__"


@dataclass
class WandbCfg(YAMLWizard):
    project: str
    mode: str


@dataclass
class General(YAMLWizard):
    outdir: str
    device: str
    wandb: WandbCfg
    time_stamp: Optional[str] = None
    computer_name: Optional[str] = None


@dataclass
class DatasetEntry(YAMLWizard):
    dataset_name: str
    metric_set: str
    test_dir: str


@dataclass
class MetricSet(YAMLWizard):
    metric_set_name: str
    metric_names: List[str]
    metric_set_cfgs: Dict[str, Any]=None  # e.g. {'mode': 'per_video'}


@dataclass
class DatasetCfg(YAMLWizard):
    all_datasets: List[DatasetEntry]
    target_ds_name: str
    metric_sets: List[MetricSet]

    @property
    def selected_ds(self) -> DatasetEntry:
        """
        Get the dataset entry for the selected dataset name.
        """
        for ds in self.all_datasets:
            if ds.dataset_name == self.target_ds_name:
                return ds
        raise ValueError(
            f"Dataset '{self.target_ds_name}' not found in the configuration."
        )

    @property
    def selected_metricSet(self) -> MetricSet:
        """
        Get the metric set for the selected dataset.
        """
        ds_entry = self.selected_ds
        for metric_set in self.metric_sets:
            if metric_set.metric_set_name == ds_entry.metric_set:
                return metric_set
        raise ValueError(
            f"Metric set '{ds_entry.metric_set}' not found for dataset '{ds_entry.dataset_name}'."
        )

    @property
    def ds_metrics_dict(self) -> Dict[str, Metric]:
        """
        Get a dictionary of torchmetrics based on the selected metrics in the dataset configuration.
        """
        metrics_dict = {}
        mapp_dict = {
            "accuracy": torchmetrics.Accuracy(task="binary"),
            "precision": torchmetrics.Precision(task="binary"),
            "recall (TPR)": torchmetrics.Recall(task="binary"),
            "FPR": FPR(),
            "f1_score": torchmetrics.F1Score(task="binary"),
            "FPS": FPS(),
        }
        for metric_name in self.selected_metricSet.metric_names:
            if metric_name in mapp_dict:
                metrics_dict[metric_name] = mapp_dict[metric_name]
            else:
                raise ValueError(f"Metric '{metric_name}' not recognized")

        return metrics_dict


@dataclass
class ModelCfg(YAMLWizard):
    base_timm_model: str
    model_path: str
    class_names: List[str]


@dataclass
class MethodEntry(YAMLWizard):
    method_name: str = None
    params: Union[Dict[str, Any], None] = None


@dataclass
class MethodCfg(YAMLWizard):
    method_cfg_dir: str = ""
    list_methods: List[MethodEntry] = None
    selected_method: Optional[str] = None

    @property
    def methodUsed(self) -> MethodEntry:
        """
        Get the method entry for the selected method name.
        """
        if self.selected_method is None:
            raise ValueError("No method selected.")
        if self.list_methods is None:
            raise ValueError("No methods available.")
        for method in self.list_methods:
            if method.method_name == self.selected_method:
                return method
        raise ValueError(
            f"Method '{self.selected_method}' not found in the configuration."
        )

    @property
    def all_method_names(self) -> List[str]:
        """
        Get a list of all method names available in the configuration.
        """
        if self.list_methods is None:
            raise ValueError("No methods available.")
        return [method.method_name for method in self.list_methods]

    @property
    def name(self) -> str:
        """
        Get the name of the selected method.
        """
        if self.selected_method is None:
            raise ValueError("No method selected.")
        return self.selected_method

    def __repr__(self):
        return f"method(params={self.methodUsed.params}, name={self.name})"

    def __str__(self):
        list_meta = []
        for key, value in self.methodUsed.params.items():
            key_value = f"{key}_{value}"
            list_meta.append(key_value)
        global SEP
        return SEP.join(list_meta)


@dataclass
class InferCfg(YAMLWizard):
    do_infer: bool
    skip_if_exists: bool
    limit: int
    save_out_video: bool
    save_results: bool
    csv_columns: List[str]
    gpu_monitor: bool
    calc_perf_metrics: bool
    verbose: bool


@dataclass
class Config(YAMLWizard):
    general: General
    dataset: DatasetCfg
    model: ModelCfg
    method: MethodCfg
    infer: InferCfg

    @classmethod
    def from_custom_yaml_file(cls, yaml_file: str) -> "Config":
        cfg_dict = yamlfile.load_yaml(yaml_file, to_dict=True)
        if "__base__" in cfg_dict:
            del cfg_dict["__base__"]
        yaml_str = yaml.dump(cfg_dict, default_flow_style=False)
        # !hahv: debug
        # print(yaml_str)
        instance = Config.from_yaml(yaml_str)
        # only load methods if not already loaded
        if len(instance.method.list_methods) == 0:
            method_dir = instance.method.method_cfg_dir
            assert method_dir, "Method configuration directory must be specified."
            assert os.path.exists(
                method_dir
            ), f"Method configuration directory '{method_dir}' does not exist."
            method_yaml_files = fs.filter_files_by_extension(method_dir, ".yaml")
            assert (
                len(method_yaml_files) > 0
            ), f"No YAML files found in method configuration directory '{method_dir}'."
            method_list = []
            for method_yaml in method_yaml_files:
                fname = fs.get_file_name(method_yaml, split_file_ext=True)[0]
                method_cfg_dict = yamlfile.load_yaml(method_yaml, to_dict=True)
                raw_params_dict = method_cfg_dict.get("params", None)
                method_entry = MethodEntry(method_name=fname, params=raw_params_dict)
                method_list.append(method_entry)
            instance.method.list_methods = method_list
        return instance

    def get_output_dir(self) -> str:
        selected_method = self.method.methodUsed
        assert selected_method is not None, "No method selected in the configuration"
        # get computer name
        if self.general.computer_name is not None:
            abbr = self.general.computer_name
        else:
            import socket
            computer_name = socket.gethostname()
            df_computer = pd.read_csv("./config/__list_pc.csv", sep=";")
            # cols: pc_name;abbr
            # get all rows of dfcomputer and find the row for the current computer
            computer_row = df_computer[df_computer["pc_name"] == computer_name]
            if computer_row.empty:
                raise ValueError(f"Computer '{computer_name}' not found in __list_pc.csv")
            abbr = computer_row["abbr"].values[0]
            self.general.computer_name = abbr  # set computer name in general config
        if self.general.time_stamp is None:
            # set time_stamp to current time
            self.general.time_stamp = now_str()

        name_parts = [
            abbr,
            f"ds_{self.dataset.target_ds_name}",
            f"mt_{self.method.name}",
            self.general.time_stamp,
        ]
        folder_name = SEP.join(name_parts)
        return os.path.join(self.general.outdir, folder_name)


def test1():
    cfg_file = "./config/_base.yaml"
    cfg = Config.from_custom_yaml_file(cfg_file)
    pprint(cfg)
    cfg.to_yaml_file("./new_cfg.yaml")


def test2():
    cfg = Config.from_custom_yaml_file("./new_cfg.yaml")
    pprint(cfg)


if __name__ == "__main__":
    # test1()
    test2()
