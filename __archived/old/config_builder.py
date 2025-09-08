import yaml
from pathlib import Path
from .zz_config import *
from halib.filetype import yamlfile


class ConfigFactory:
    def __init__(self, main_config_path: str, methods_dir: str):
        self.main_config_path = main_config_path
        self.methods_dir = methods_dir

    def _load_yaml(self, path: str) -> dict:
        cfg_dict = yamlfile.load_yaml(path, to_dict=True)
        if "__base__" in cfg_dict:
            del cfg_dict["__base__"]  # Remove __base__ key if it exists
        return cfg_dict

    def _build_methods(self) -> Dict[str, MethodConfig]:
        method_files = Path(self.methods_dir).glob("*.yaml")
        methods = {}
        for file in method_files:
            name = file.stem
            params = self._load_yaml(str(file))["params"]
            if params is None:
                params = {}  # in case no params are defined in the method file, like "no_temp.yaml"
            params["method_name"] = name  # Ensure method name is included in params
            methods[name] = MethodConfig(params=params)
        return methods

    def build(self) -> FullConfig:
        main_cfg = self._load_yaml(self.main_config_path)

        general = General(
            outdir=main_cfg["general"]["outdir"],
            device=main_cfg["general"]["device"],
            wandb=WandbCfg(**main_cfg["general"]["wandb"]),
        )

        dataset = DatasetEntry(**main_cfg["dataset"])
        model = ModelConfig(**main_cfg["model"])
        temporal = ExpConfig(
            selected_method=main_cfg["method"]["selected_method"],
            list_methods=self._build_methods(),
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
