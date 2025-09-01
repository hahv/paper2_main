from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from halib import *
from halib.filetype import yamlfile
from pathlib import Path

import torchmetrics
from torchmetrics import Metric
from temporal.metrics import FPS
import yaml

SEP = "__"

@dataclass
class WandbCfg:
    project: str
    mode: str


@dataclass
class General:
    outdir: str
    device: str
    wandb: WandbCfg
    time_stamp: str = (
        None  # will be set automatically using halib.now_str() if not provided
    )

@dataclass
class MetricSet:
    metric_names: List[str]
    metric_cfgs: Dict[str, Any] = None # configuration for metrics, e.g., mode: "per_video" or "per_frame"


@dataclass
class DatasetEntry:
    name: str
    metric_set: MetricSet
    test_dir: str

@dataclass
class DatasetCfg:
    all_datasets: List[DatasetEntry]
    metric_sets: Dict[str, MetricSet]
    selected_ds: DatasetEntry
    @property
    def ds_metrics_dict(self) -> Dict[str, Metric]:
        """
        Get a dictionary of torchmetrics based on the selected metrics in the dataset configuration.
        """
        metrics_dict = {}
        for metric_name in self.selected_ds.metric_set.metric_names:
            if metric_name == "accuracy":
                metric_item = torchmetrics.Accuracy()
            elif metric_name == "precision":
                metric_item = torchmetrics.Precision()
            elif metric_name == "recall (TPR)":
                metric_item = torchmetrics.Recall()
            elif metric_name == "FPR":
                metric_item = torchmetrics.FalsePositiveRate()
            elif metric_name == "f1":
                metric_item = torchmetrics.F1Score()
            elif metric_name == "fps":
                metric_item = FPS()
            else:
                raise ValueError(f"Metric '{metric_name}' not recognized")
            metrics_dict[metric_name] = metric_item
        return metrics_dict

@dataclass
class ModelConfig:
    base_timm_model: str
    model_path: str
    class_names: List[str]


@dataclass
class MethodConfig:
    method_cfg_dir: str
    params: Dict[str, Any]

    def __str__(self):
        list_meta = []
        for key, value in self.params.items():
            key_value = f"{key}_{value}"
            list_meta.append(key_value)
        global SEP
        return SEP.join(list_meta)

    @property
    def name(self) -> str:
        """
        Returns the method name based on the parameters.
        """
        return self.params.get("method_name")

    def __repr__(self):
        return f"method(params={self.params})"


@dataclass
class InferCfg:
    do_infer: bool
    skip_if_exists: bool  # skip inference if results already exist
    limit: int  # -1 for full inference, otherwise limit to this number of frames
    save_results: bool
    save_out_video: bool
    verbose: bool
    csv_columns: List[str]
    gpu_monitor: bool = False
    calc_perf_metrics: bool = True  # calculate performance metrics after inference


@dataclass
class ExpConfig:
    selected_method: str
    list_methods: Dict[str, MethodConfig]
    @property
    def methodUsed(self) -> Optional[MethodConfig]:
        assert (
            self.selected_method is not None or len(self.list_methods) > 0
        ), "No method selected in the configuration"
        assert (
            self.selected_method in self.list_methods
        ), f"Selected method '{self.selected_method}' not found in list of methods; available methods: {list(self.list_methods.keys())}"
        return self.list_methods[self.selected_method]

    def all_available_methods(self) -> List[str]:
        """
        Returns a list of all available methods.
        """
        return list(self.list_methods.keys())


@dataclass
class FullConfig:
    general: General
    dataset: DatasetCfg
    model: ModelConfig
    expConfig: ExpConfig
    infer: InferCfg
    raw_dict: dict

    def get_output_dir(self) -> str:
        selected_method = self.expConfig.methodUsed
        assert selected_method is not None, "No method selected in the configuration"
        # get computer name
        import socket

        computer_name = socket.gethostname()
        df_computer = pd.read_csv("./config/__list_pc.csv", sep=";")
        # cols: pc_name;abbr
        # get all rows of dfcomputer and find the row for the current computer
        computer_row = df_computer[df_computer["pc_name"] == computer_name]
        if computer_row.empty:
            raise ValueError(f"Computer '{computer_name}' not found in __list_pc.csv")
        abbr = computer_row["abbr"].values[0]
        if self.general.time_stamp is None:
            # set time_stamp to current time
            self.general.time_stamp = now_str()
            # update raw_dict for yaml file saving
            self.raw_dict["general"]["time_stamp"] = self.general.time_stamp

        name_parts = [
            abbr,
            self.general.time_stamp,
            f"mt_{self.expConfig.methodUsed.name}",
            f"ds_{self.dataset.selected_ds.name}",
        ]
        folder_name = SEP.join(name_parts)
        return os.path.join(self.general.outdir, folder_name)

    @classmethod
    def load_yaml(cls, file_path: str) -> dict:
        """
        Load a YAML file and return its content as a dictionary.
        """
        cfg_dict = yamlfile.load_yaml(file_path, to_dict=True)
        if "__base__" in cfg_dict:
            del cfg_dict["__base__"]  # Remove __base__ key if it exists
        return cfg_dict

    @classmethod
    def from_dict(cls, main_cfg: dict) -> "FullConfig":

        def _build_methods(method_dir) -> Dict[str, MethodConfig]:
            """
            Build a dictionary of MethodConfig objects from the method configuration directory.
            """
            method_files = Path(method_dir).glob("*.yaml")
            methods = {}
            for file in method_files:
                name = file.stem
                params = FullConfig.load_yaml(str(file))["params"]
                if params is None:
                    params = (
                        {}
                    )  # in case no params are defined in the method file, like "no_temp.yaml"
                params["method_name"] = name  # Ensure method name is included in params
                methods[name] = MethodConfig(method_cfg_dir=method_dir, params=params)
            return methods

        method_dir = main_cfg["method"]["method_cfg_dir"]
        general = General(
            outdir=main_cfg["general"]["outdir"],
            device=main_cfg["general"]["device"],
            wandb=WandbCfg(**main_cfg["general"]["wandb"]),
            time_stamp=main_cfg["general"].get(
                "time_stamp", None
            ),  # Use provided time_stamp or None
        )
        all_metric_sets = main_cfg["dataset"]["metric_sets"]
        for set_name, metric_set_values in all_metric_sets.items():
            a_metric_set = MetricSet(
                metric_names=metric_set_values["metric_names"],
                metric_cfgs=metric_set_values.get("metric_cfgs", {})
            )
            all_metric_sets[set_name] = a_metric_set

        all_datasets = []
        for ds_name in main_cfg["dataset"]["all_datasets"]:
            ds_cfg = main_cfg["dataset"]["all_datasets"][ds_name]
            if "metric_set" not in ds_cfg:
                raise ValueError(f"Dataset '{ds_name}' does not have metric_set defined")
            ds_metric = all_metric_sets.get(ds_cfg["metric_set"], None)
            assert (
                ds_metric is not None
            ), f"Metric set '{ds_cfg['metric_set']}' not found in metric sets"
            all_datasets.append(
                DatasetEntry(
                    name=ds_name, metric_set=ds_metric, test_dir=ds_cfg.get("test_dir", "")
                )
            )
        selected_ds_name = main_cfg["dataset"].get("selected_dataset", None)
        assert (
            selected_ds_name is not None
        ), "No dataset selected in the configuration"
        selected_ds = None
        for ds in all_datasets:
            if ds.name == selected_ds_name:
                selected_ds = ds
                break
        assert (
            selected_ds is not None
        ), f"Selected dataset '{selected_ds_name}' not found in all datasets"
        dataset = DatasetCfg(
            all_datasets=all_datasets,
            metric_sets=all_metric_sets,
            selected_ds=selected_ds,
        )
        model = ModelConfig(**main_cfg["model"])
        temporal = ExpConfig(
            selected_method=main_cfg["method"]["selected_method"],
            list_methods=_build_methods(method_dir),
        )
        infer = InferCfg(
            **main_cfg["infer"],
        )

        return FullConfig(
            general=general,
            dataset=dataset,
            model=model,
            expConfig=temporal,
            infer=infer,
            raw_dict=main_cfg,
        )

    @classmethod
    def from_yaml(cls, yaml_cfg_file: str) -> "FullConfig":
        # load the main configuration file
        main_cfg = FullConfig.load_yaml(yaml_cfg_file)
        return cls.from_dict(main_cfg)

    def save_to_yaml(self, output_file: str):
        """
        Save the configuration to a YAML file.
        """
        # Convert the dataclass to a dictionary
        cfg_dict = self.raw_dict
        with open(output_file, "w") as f:
            yaml.dump(cfg_dict, f, default_flow_style=False, sort_keys=False)


def main():
    test_cfg_file = "config/_base.yaml"
    cfg = FullConfig.from_yaml(test_cfg_file)
    pprint(cfg.expConfig.all_available_methods())
    pprint(cfg)
    pprint(cfg.get_output_dir())
    pprint(cfg.raw_dict)
    # Save the configuration to a new YAML file
    # output_file = "test_output.yaml"
    # cfg = FullConfig.from_yaml(output_file)
    # pprint(cfg.get_output_dir())


if __name__ == "__main__":
    main()
