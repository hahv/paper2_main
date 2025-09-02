from temporal.methods.eval_base import *

class EvalNoTemp(EvalBase):

    def infer_frame(self, frame, frame_idx: int) -> dict:
        """Perform inference on the pre-processed frame."""
        assert self.model is not None, "Model is not loaded."
        with torch.no_grad():
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