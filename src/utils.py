import importlib
from typing import List, Optional, Tuple
from torchvision import transforms
from timm.data import resolve_data_config, create_transform


def get_cls(class_path: str, *args, **kwargs):
    """
    Dynamically import class and create instance.
    class_path format: 'mypkg.shapes.circle.Circle'
    """
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls


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
