from halib import *
from halib.filetype import yamlfile
from temporal.methods.__temp_stabilize import ProposedDetector
from temporal.archived.config_bk import Config
import time
import itertools


def mk_exp_cfg_list(datasets, methods, base_cfg_file, timestamps=None) -> list[Config]:
    """Create a list of configurations for each dataset and method."""
    exp_cfgdict_list = []
    if timestamps is not None:
        assert len(timestamps) == len(datasets) * len(methods), (
            "Number of timestamps must match the number of dataset-method combinations."
        )

    assert len(datasets) > 0, "No datasets provided"
    assert len(methods) > 0, "No methods provided"
    mt_ds_combinations = list(itertools.product(methods, datasets))
    for idx, (method, dataset) in enumerate(mt_ds_combinations):
        # pprint(f"Creating config for dataset: {dataset}, method: {method}"  )
        cfg = Config.from_custom_yaml_file(base_cfg_file)
        cfg.method.selected_method = method
        cfg.dataset.target_ds_name = dataset
        if timestamps is not None:
            cfg.general.time_stamp = timestamps[idx]
        # cfg.infer.limit = 20
        exp_cfgdict_list.append(cfg)
    # print(len(exp_cfgdict_list), "Experiment configurations created")
    return exp_cfgdict_list


def main():
    run_cfg_file = "./config/run.yaml"
    run_dict = yamlfile.load_yaml(run_cfg_file, to_dict=True)
    datasets = run_dict.get("datasets", [])
    methods = run_dict.get("methods", [])
    time_stamps = run_dict.get("time_stamps", None)

    base_cfg_file = "./config/_base.yaml"
    exp_cfg_dict_ls = mk_exp_cfg_list(
        datasets, methods, base_cfg_file, timestamps=time_stamps
    )
    for idx, cfg in enumerate(exp_cfg_dict_ls):
        msg = f"Proc: [{idx + 1}/{len(exp_cfg_dict_ls)}]"
        with ConsoleLog(msg):
            detector = ProposedDetector(cfg)
            test_dir = cfg.dataset.selected_ds.test_dir
            detector.infer_video_dir(video_dir=test_dir, recursive=True)
        time.sleep(3)  # a short delay to avoid duplicate outdir names


if __name__ == "__main__":
    main()
