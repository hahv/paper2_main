import re
import importlib
from typing import List, Optional, Tuple, Callable
from torchvision import transforms
from timm.data import resolve_data_config, create_transform

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