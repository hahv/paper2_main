from temporal.methods.base_method import *
from torchvision import transforms
from PIL import Image
from temporal.config import Config
from temporal.metric_src.test_metric_src import TestDSMetricSrc


class NoTempMethod(BaseMethod):
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
        if "prof" in model_name.lower():
            transform = transforms.Compose(
                [
                    transforms.Resize((360, 640)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
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
