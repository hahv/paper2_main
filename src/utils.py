import re
import importlib
from typing import List, Optional, Tuple, Callable
from torchvision import transforms
from timm.data import resolve_data_config, create_transform

from halib.utils.slack_op import SlackUtils
import os
from halib.filetype import yamlfile
from halib.system.path import get_PC_abbr_name
from halib import pprint_local_path
import math
from halib import pprint

SLACK_TOKEN = None
SLACK_CHANNEL_ID = None

def split_task_by_cfg(task_cfg_yaml: str, total_exps: int, pc_name=get_PC_abbr_name()):
    """
    This function splits a task into sub-tasks based on the provided YAML configuration. The YAML file should specify the number of experiments to allocate to each sub-task. The function returns a list of sub-task configurations, each containing the allocated number of experiments and the corresponding configuration details.

    Args:
        task_cfg_yaml (str): Path to the YAML configuration file that defines the sub-tasks and their experiment allocations.
        total_exps (int): The total number of experiments that need to be allocated across the sub-tasks.

    Returns:
        List[dict]: A list of dictionaries, where each dictionary represents a sub-task with its allocated number of experiments and configuration details.
    """
    # Load the YAML configuration
    cfg_dict = yamlfile.load_yaml(task_cfg_yaml, to_dict=True)
    pprint(cfg_dict)

    result = {}
    prev = 0
    items = list(cfg_dict.items())
    for i, (machine, weight) in enumerate(items):
        end = (
            total_exps
            if i == len(items) - 1
            else prev + math.ceil(weight / 100 * total_exps)
        )
        result[machine] = (prev, end)
        prev = end
    assert pc_name in result, f"PC name '{pc_name}' not found in the configuration. Available keys: {list(result.keys())}"
    return result[pc_name], result


def clear_slack_channel(sleep_interval: float = 0.5):
    global SLACK_TOKEN, SLACK_CHANNEL_ID
    if SLACK_TOKEN is None or SLACK_CHANNEL_ID is None:
        # get current dir of this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        slack_env_yaml = os.path.join(current_dir, ".slack_env.yaml")
        slack_cfg_dict = yamlfile.load_yaml(slack_env_yaml, to_dict=True)
        SLACK_TOKEN = slack_cfg_dict["SLACK_TOKEN"]
        SLACK_CHANNEL_ID = slack_cfg_dict["SLACK_CHANNEL_ID"]

    slack_util = SlackUtils(token=SLACK_TOKEN)
    slack_util.clear_channel(channel_id=SLACK_CHANNEL_ID, sleep_interval=sleep_interval)


def filter_dict_by_keys(input_dict: dict, keys: List[str]) -> dict:
    # Iterate over 'keys' to preserve their order
    return {k: input_dict[k] for k in keys if k in input_dict}


def to_abbr(text: str) -> str:
    """
    Extracts all uppercase letters to form an abbreviation.
    Example: 'SmokeCheck' -> 'SC'
    """
    return "".join(re.findall(r"[A-Z]", text))


def get_cls(class_path: str, *args, **kwargs):
    """
    Dynamically import class and create instance.
    class_path format: 'mypkg.shapes.circle.Circle'
    """
    # print(f">>Importing class from path: {class_path}")
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    assert cls is not None, f"Class '{class_name}' not found in module '{module_name}'."
    return cls


def default_fileName_to_clsName(file_name: str) -> str:
    return "".join(p.title() for p in file_name.split("_"))


def get_cls_in_pkg(
    pkg_name: str,
    fileName_ClsName: str,
    fileName_to_clsName_func: Optional[
        Callable[[str], str]
    ] = default_fileName_to_clsName,
):
    "This function imports a class from a specified package and file name. Using a provided function, it converts the file name to the corresponding class name and retrieves the class from the module."

    # pprint(locals())
    having_cls_name = "." in fileName_ClsName
    class_path = None
    if having_cls_name:
        file_name = fileName_ClsName.split(".")[0]
        cls_name = fileName_ClsName.split(".")[1]
        class_path = f"{pkg_name}.{file_name}.{cls_name}"
    else:
        assert fileName_to_clsName_func is not None, (
            "fileName_to_clsName_func must be provided if class name is not included."
        )
        class_path = f"{pkg_name}.{fileName_ClsName}.{fileName_to_clsName_func(fileName_ClsName)}"
    return get_cls(class_path)


# ! Since fire/smoke images have important color info, we remove color jitter to preserve it.
def get_transform(model_name: str, input_size: Optional[List[int]] = None):
    def _remove_color_jitter(tfm_pipeline: transforms.Compose) -> transforms.Compose:
        """Filters out ColorJitter from an existing Compose pipeline."""
        clean_transforms = [
            t
            for t in tfm_pipeline.transforms
            if not isinstance(t, transforms.ColorJitter)
        ]
        return transforms.Compose(clean_transforms)

    def _force_deterministic_resize(
        tfm_pipeline: transforms.Compose, target_size: Tuple[int, int]
    ) -> transforms.Compose:
        """
        Replaces the first transform (usually RandomResizedCrop or CenterCrop)
        with a deterministic Resize. This 'squishes' the image to the target size.
        """
        tfms_list = list(tfm_pipeline.transforms)

        # Overwrite the first transform (standard timm practice is geometry first)
        tfms_list[0] = transforms.Resize(
            target_size,
            interpolation=transforms.InterpolationMode.BICUBIC,
        )
        return transforms.Compose(tfms_list)

    """Get the appropriate transformation based on the model name."""
    if "tinycnn" in model_name.lower():
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        # 2. Resolve default config from timm (mean, std, interpolation)
        data_cfg = resolve_data_config(model=model_name)

        # 3. Update config if custom input size is provided
        target_size = None
        if input_size is not None:
            # timm expects (channels, height, width)
            data_cfg["input_size"] = (3, input_size[0], input_size[1])
            target_size = (input_size[0], input_size[1])

        # 4. Create base transforms using timm
        val_tfm = create_transform(**data_cfg, is_training=False)

        # 5. Apply Custom Overrides (Resize Strategy & Color Removal)
        if target_size is not None:
            val_tfm = _force_deterministic_resize(val_tfm, target_size)

        val_tfm = _remove_color_jitter(val_tfm)

        return val_tfm


def copy_to_paper_raw_csv(
    infile: str, outdir: str = r"./paper/4.table/raw", add_prefix="_raw"
) -> str:
    os.makedirs(outdir, exist_ok=True)
    filename = os.path.basename(infile)

    if add_prefix and not filename.startswith(add_prefix):
        filename = f"{add_prefix}{filename}"

    base_name, ext = os.path.splitext(filename)
    outfile = os.path.join(outdir, filename)

    counter = 1
    while os.path.exists(outfile):
        filename = f"{base_name}_{counter}{ext}"
        outfile = os.path.join(outdir, filename)
        counter += 1

    os.system(f"cp {infile} {outfile}")
    pprint_local_path(outfile, get_wins_path=True, tag_or_box_title="Copied to paper raw csv at ⏬:", using_box=True)

    # Generate YAML ONLY referencing the base, un-numbered file
    original_base_filename = f"{base_name}{ext}"
    yaml_outfile = os.path.join(outdir, original_base_filename.replace(".csv", ".yaml"))
    if not os.path.exists(yaml_outfile):
        yaml_content = f'input: raw/{original_base_filename}\noutput: output/\nsep: ";"\n'
        with open(yaml_outfile, "w", encoding="utf-8") as f:
            f.write(yaml_content)

    return outfile


def test():
    pkg_cls = {
        "src.methods": ["no_temp_method", "temp_Baseline_TPT_method", "temp_method"],
        "src.metrics": ["csv_metric_src"],
        "src.results": ["csv_rs_proc", "video_base_rs_proc", "video_rs_fgmask_proc"],
    }
    from rich.pretty import pprint
    import sys

    # get current folder of this file
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    proj_dir = os.path.dirname(current_dir)
    pprint(f"Proj dir: {proj_dir}")
    sys.path.append(proj_dir)

    for pkg_name, file_names in pkg_cls.items():
        for file_name in file_names:
            cls = get_cls_in_pkg(pkg_name, fileName_ClsName=file_name)
            pprint(cls)


if __name__ == "__main__":
    test()
