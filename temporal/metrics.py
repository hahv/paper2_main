import torch
from torchmetrics import Metric


class FPS(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total_time", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_frames", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, elapsed_times):
        """
        Args:
            elapsed_times (List[float] or Tensor): A list or tensor of per-frame durations (in seconds).
        """
        if not isinstance(elapsed_times, torch.Tensor):
            elapsed_times = torch.tensor(elapsed_times, dtype=torch.float32)

        if elapsed_times.numel() == 0:
            return  # Skip empty input

        self.total_time += elapsed_times.sum()
        self.num_frames += elapsed_times.numel()

    def compute(self):
        if self.total_time > 0:
            return self.num_frames.float() / self.total_time
        return torch.tensor(0.0)

    def __str__(self):
        return f"{self.compute():.2f} FPS"


class FPR(Metric):  # False Positive Rate
    def __init__(self):
        super().__init__()
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds.int()
        target = target.int()
        self.fp += ((preds == 1) & (target == 0)).sum()
        self.tn += ((preds == 0) & (target == 0)).sum()

    def compute(self):
        total = self.fp + self.tn
        return self.fp.float() / total if total > 0 else torch.tensor(0.0)


def testFPS():
    fps = FPS()
    fps.update([0.033, 0.040, 0.042, 0.037])  # Simulate frame timings
    print(f"Computed FPS: {fps.compute():.2f}")


def testFPR():
    preds = torch.tensor([1, 0, 1, 0, 1])
    target = torch.tensor([0, 0, 1, 0, 1])

    from torchmetrics.classification import ConfusionMatrix

    # Compute confusion matrix
    confmat = ConfusionMatrix(task="binary")
    cm = confmat(preds, target)  # shape: [2, 2]

    # Extract FP and TN from confusion matrix
    TN = cm[0, 0].item()
    FP = cm[0, 1].item()

    # Compute FPR
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    print(f"False Positive Rate (from CM): {fpr:.4f}")

    # Using custom FPR metric
    cfpr = FPR()
    cfpr.update(preds, target)
    print(f"Computed FPR: {cfpr.compute():.4f}")


def main():
    testFPS()
    testFPR()


if __name__ == "__main__":
    main()
