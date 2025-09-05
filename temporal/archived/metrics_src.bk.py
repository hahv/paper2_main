from abc import ABC, abstractmethod

from duckdb import torch
from temporal.config import *
from halib.filetype import csvfile

class BaseMetricDataSource(ABC):

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name # we must know which dataset we're working with

    # a method that return the data for the metric calculation
    @abstractmethod
    def get_data(self, kwargs):
        pass

class CSVBinaryMetricDataSource(BaseMetricDataSource):
    POSITIVE_LABEL = "positive"
    NEGATIVE_LABEL = "negative"
    def __init__(self, cfg: Config):
        super().__init__(cfg.dataset_cfg.dataset_used.name)
        self.cfg = cfg

    def _get_data_per_video(self, csv_path, metric_mode):
        if metric_mode == "per_video":
            if self.dataset_name == "Test":
                fname = fs.get_file_name(csv_path, split_file_ext=True)[0]
                gt_label = self.POSITIVE_LABEL
                if 'nofire' in fname:
                    gt_label = self.NEGATIVE_LABEL

                pred_label = CSVBinaryMetricDataSource.POSITIVE_LABEL
                df = csvfile.read_auto_sep(csv_path)
                pred_label_list = df['pred_label'].unique().tolist()
                if len(pred_label_list) == 1 and pred_label_list[0] == "None":
                    pred_label = CSVBinaryMetricDataSource.NEGATIVE_LABEL
                return pred_label, gt_label
            else:
                raise NotImplementedError(f"Dataset {self.dataset_name} not supported yet")
        elif metric_mode == "per_frame":
            raise NotImplementedError(f"Per-frame mode not implemented for dataset {self.dataset_name}")

    def get_data(self, **kwargs):
        indir = kwargs.get("indir", None)
        assert indir is not None, "indir must be provided"
        metric_mode = self.cfg.metric_cfg.metricSet_used.extra_cfgs.get("mode")
        # per_video, per_frame
        assert metric_mode in ["per_video", "per_frame"], f"Unsupported metric mode: {metric_mode}"

        # first list all video files
        num_videos = self.cfg.dataset_cfg.dataset_used.get_num_videos(recursive=False)

        csv_files = fs.filter_files_by_extension(indir, [".csv"], recursive=False)
        assert len(csv_files) == num_videos, f"Number of CSV files ({len(csv_files)}) does not match number of video files ({num_videos})"

        if metric_mode == "per_video":
            if self.dataset_name == "Test":
                preds, gts = [], []
                for csvfile in csv_files:
                    pred, gt = self._get_data_per_video(csvfile, metric_mode)
                    preds.append(pred)
                    gts.append(gt)

                np_preds, np_gts = np.array(preds), np.array(gts)
                # convert these to torch
                preds_tensor = torch.from_numpy(
                    (
                        np.array(preds) == CSVBinaryMetricDataSource.POSITIVE_LABEL
                    ).astype(np.int32)
                )
                gts_tensor = torch.from_numpy(
                    (
                        np.array(gts) == CSVBinaryMetricDataSource.POSITIVE_LABEL
                    ).astype(np.int32)
                )

                return preds_tensor, gts_tensor

        elif metric_mode == "per_frame":
            raise NotImplementedError(f"Per-frame mode not implemented for dataset {self.dataset_name}")
        else:
            raise NotImplementedError(f"Metric mode {metric_mode} not implemented for dataset {self.dataset_name}")
