from temporal.methods.base_method import *
from torchvision import transforms
from PIL import Image
from temporal.config import Config
import torch.nn.functional as F
from timm.data import resolve_data_config, create_transform
from typing import List, Optional

LOG_TRANSFORM = False


class NoTempMethod(BaseMethod):
    def _get_transform(self, model_name: str, input_size: Optional[List[int]] = None):
        """Get the appropriate transformation based on the model name."""

        if "prof" in model_name.lower():
            return transforms.Compose(
                [
                    transforms.Resize((360, 640)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
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
        if ("efficientnet_b0" in model_name.lower()) or (
            "hgnetv2_b5.ssld_stage2_ft_in1k" in model_name.lower()
        ):
            assert input_size is not None, "input_size must be provided for timm models"
            assert isinstance(input_size, (list, tuple)) and len(input_size) == 2, (
                "input_size must be a list or tuple of two integers"
            )
            cfg = resolve_data_config(model=model_name)
            # ! disable color jitter since fire/smoke are sensitive to color changes
            cfg["color_jitter"] = None
            cfg["input_size"] = (3, input_size[0], input_size[1])  # C, H, W
            # val: Compose(
            #     ResizeKeepRatio(size=(411, 731), interpolation=torch.bilinear, longest=0.000, random_scale_prob=0.000, random_scale_range=(0.850, 1.110), random_aspect_prob=0.000, random_aspect_range=(0.900, 1.110))
            #     CenterCrop(size=(360, 640))
            #     MaybeToTensor()
            #     Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]))
            # )
            # ! replace ResizeKeepRatio + CenterCrop with a simple Resize
            val_tfm = create_transform(**cfg, is_training=False)
            tfms = list(val_tfm.transforms)
            tfms[0] = transforms.Resize(
                (360, 640), interpolation=transforms.InterpolationMode.BICUBIC
            )
            tfms = [
                tfm for tfm in tfms if not isinstance(tfm, transforms.CenterCrop)
            ]  # remove CenterCrop
            val_tfm = transforms.Compose(tfms)

            return val_tfm

        raise ValueError(f"Unsupported model: {model_name}")

    def _pre_process_frame(self, frame):
        """Pre-process the frame before inference.
        if roi is provided, it will crop the frame to the ROI.
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        assert isinstance(self.cfg, Config), (
            "current method Cfg is not an instance of temporal.Config"
        )
        full_cfg: Config = self.cfg
        model_name: str = fs.get_file_name(
            full_cfg.model_cfg.model_path, split_file_ext=True
        )[0]
        pil_img = Image.fromarray(frame_rgb)
        # global LOG_TRANSFORM
        transform = self._get_transform(model_name, full_cfg.model_cfg.input_size)
        # if not LOG_TRANSFORM:
        #     with ConsoleLog('Infer transform'):
        #         pprint(transform)
        # Apply the transformation
        frame_batch = transform(pil_img).unsqueeze(0)  # Add batch dimension
        # Move the frame batch to the appropriate device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        frame_batch = frame_batch.to(device)
        return frame_batch

    def infer_frame(self, frame, frame_idx: int) -> dict:
        """Perform inference on the pre-processed frame."""
        assert self.model is not None, "Model is not loaded."
        with torch.no_grad():
            frame = self._pre_process_frame(frame)
            # 1. Get raw scores (logits) from the model
            logits = self.model(frame)

            # 2. Calculate probabilities using the softmax function
            probs = F.softmax(logits, dim=1)

        # 3. Get the index of the most likely class
        labelIdx = torch.argmax(probs, dim=1).item()

        # 4. Convert tensors to lists for easier handling
        logits = logits.cpu().squeeze().tolist()
        probs = probs.cpu().squeeze().tolist()
        assert len(probs) == len(self.cfg.model_cfg.class_names), (
            "Mismatch in number of classes and probabilities."
        )

        # 5. Get the predicted class name
        classNames = self.cfg.model_cfg.class_names
        assert labelIdx < len(classNames), "Class index out of range."
        pred_label = classNames[labelIdx]
        return {
            "logits": logits,
            "probs": probs,
            "predLabelIdx": labelIdx,
            "predLabel": pred_label,
        }
