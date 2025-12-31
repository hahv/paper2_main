from halib import *
from argparse import ArgumentParser
from halib.research.perfcalc import (
    PerfCalc,
    validate_torch_metrics,
    valid_custom_fields,
)
import torchmetrics
import torch


class PerfTest(PerfCalc):
    """
    A class for performance testing, inheriting from PerfCalc.
    It can be extended with additional methods or properties specific to performance testing.
    """

    # override abstract methods if needed
    @validate_torch_metrics
    def get_exp_torch_metrics(self):
        """
        Return a dictionary of torchmetrics to be used for performance calculation.
        Example: {"accuracy": Accuracy(), "precision": Precision()}
        """

        acc = torchmetrics.Accuracy(task="binary")
        p = torchmetrics.Precision(task="binary")
        r = torchmetrics.Recall(task="binary")
        f1 = torchmetrics.F1Score(task="binary")
        return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

    def get_dataset_name(self):
        return "TestDataset"

    def get_experiment_name(self):
        return "TestExperiment"

    def prepare_exp_data_for_metrics(self, metric_names, *args, **kwargs):
        targets = torch.tensor([1, 0, 1, 0])  # Example target tensor
        preds = torch.tensor([1, 0, 1, 1])
        metric_data = {
            "accuracy": {"preds": preds, "target": targets},
            "precision": {"preds": preds, "target": targets},
            "recall": {"preds": preds, "target": targets},
            "f1": {"preds": preds, "target": targets},
        }
        return metric_data

    @valid_custom_fields
    def calc_exp_outdict_custom_fields(self, outdict, *args, **kwargs):
        outdict["custom_field"] = "custom_value"
        return outdict, ["custom_field"]


def main():
    test = PerfTest()
    resultDF, outfile = test.calculate_exp_perf_metrics(outdir=".", return_df=True)
    print("Result DataFrame:")
    csvfile.fn_display_df(resultDF)
    print("Output File:", outfile)


if __name__ == "__main__":
    main()
