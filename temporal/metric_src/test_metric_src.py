from temporal.config import *
from halib.filetype import csvfile
from typing import Dict, Any
from temporal.metric_src.metrics_src_base import *
import torch

class TestDSMetricSource(BaseMetricSource):
    """
    Concrete data source for a hypothetical video dataset.
    Assumes data structure: videos with frames, each frame has gt_label, pred_label, timestamp, etc.
    Predictions might be loaded from a file specified in kwargs (e.g., 'prediction_path').
    """

    POS_LABEL = "positive"
    NEG_LABEL = "negative"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        super().__init__(cfg.dataset_cfg.dataset_used.name)

    def _register_handlers(self):
        metric_set_meta = self.cfg.metric_cfg.metricSet_used
        metric_names = metric_set_meta.metric_names
        modes = metric_set_meta.extra_cfgs.get("mode", ["per-video"])
        # ! set up data getters
        for metric in metric_names:
            self.metric_data_getters_dict[metric] = self._get_cls_data
        # ! setup mode proccessors
        for mode in modes:
            self.mode_processors_dict[mode] = self.proc_data_by_mode

    def _load_raw_metric_data(self, **kwargs) -> Dict[str, Any]:
        # Implementation-specific: Load ground truths and predictions.
        # For flexibility, use kwargs to specify paths, filters, etc.
        # Example: Simulate loading video data with frames.
        # In real code, this would read from files/databases.
        indir = kwargs.get("indir", None)
        if indir is None:
            indir = self.cfg.get_outdir()
        assert indir is not None, "indir must be provided"
        # first list all video files
        num_videos = self.cfg.dataset_cfg.dataset_used.get_num_videos(recursive=False)

        csv_files = fs.filter_files_by_extension(indir, [".csv"], recursive=False)
        assert (
            len(csv_files) == num_videos
        ), f"Number of CSV files ({len(csv_files)}) does not match number of video files ({num_videos})"

        # return video_name, gt, and df
        per_video_out_list = []  # gt, df for each frames in each video
        for csv_file in csv_files:
            video_name = fs.get_file_name(csv_file, split_file_ext=True)[0]
            gt = (
                TestDSMetricSource.NEG_LABEL
                if "nofire" in video_name
                else TestDSMetricSource.POS_LABEL
            )
            df = pd.read_csv(csv_file, sep=';', encoding='utf-8', dtype= {'pred_label': str, 'elapsed_time': float}, keep_default_na=False)
            # pprint(csv_file)
            # pprint(df.columns)
            # csvfile.fn_display_df(df.head(3))
            # pprint(list(df.columns))
            # ! fix this code to load correct gt (i.g. in case of per-frame gt; CURRENT is per-video), i.e all frames have the same gt as video
            gt = [gt] * len(
                df
            )
            # both are numpy
            per_video_out_list.append((gt, df))
        return per_video_out_list

    def _get_cls_data(self, metric, **kwargs) -> Dict[str, Any]:
        # Load raw data tailored for classification metrics (labels)
        per_video_out_list = self._load_raw_metric_data(**kwargs)
        # pprint(per_video_out_list)
        if metric == "FPS":
            # Compute FPS from raw_data
            per_video_preds_all = []
            for per_video_data in per_video_out_list:
                per_video_preds_all.append(per_video_data[1]['elapsed_time'].tolist()[1:])  # skip first frame, which is always two slow due to model initialization

            return per_video_preds_all  # list of list of elapse_time
        else:
            per_video_preds_all = []
            per_video_gts_all = []
            for per_video_data in per_video_out_list:
                per_video_pred_df = per_video_data[1]
                # pprint(per_video_pred_df.columns)
                preds = per_video_pred_df["pred_label"].tolist()
                preds = np.array(preds) == TestDSMetricSource.POS_LABEL
                preds = preds.astype(int).tolist()  # convert to int
                gts = per_video_data[0]  # already numpy
                gts = np.array(gts) == TestDSMetricSource.POS_LABEL
                gts = gts.astype(int).tolist()  # convert to int
                per_video_preds_all.append(preds)
                per_video_gts_all.append(gts)

            return per_video_preds_all, per_video_gts_all

    def proc_list_to_tensor(self, data_list, flatten, dtype):
        data_npy = np.array(data_list)
        if flatten:
            data_npy = data_npy.flatten()
        return torch.from_numpy(data_npy).to(dtype)

    def proc_data_by_mode(self, metric: str, mode: str, metric_data: Dict[str, Any], **kwargs):
        if metric == "FPS":
            flatten = True
            torch_data =  self.proc_list_to_tensor(data_list=metric_data, flatten=flatten, dtype=torch.float)
            return torch_data
        else:
            if mode == "per_frame":
                flatten = True
                per_video_preds, per_video_gts = metric_data
                preds_tensor = self.proc_list_to_tensor(data_list=per_video_preds, flatten=flatten, dtype=torch.int)
                gts_tensor = self.proc_list_to_tensor(data_list=per_video_gts, flatten=flatten, dtype=torch.int)
                return (preds_tensor, gts_tensor)

            elif mode == "per_video":
                video_level_preds = []
                video_level_gts = []
                flatten = False
                zip_metric_data = list(zip(metric_data[0], metric_data[1]))
                for per_video_pred, per_video_gt in zip_metric_data:
                    # if any frame is positive, the video is positive
                    video_pred = int(any(per_video_pred))
                    video_gt = int(any(per_video_gt))
                    video_level_preds.append(video_pred)
                    video_level_gts.append(video_gt)

                preds_tensor = self.proc_list_to_tensor(data_list=video_level_preds, flatten=flatten, dtype=torch.int)
                gts_tensor = self.proc_list_to_tensor(data_list=video_level_gts, flatten=flatten, dtype=torch.int)
                return (preds_tensor, gts_tensor)
            else:
                raise NotImplementedError(f"Mode {mode} not implemented yet")
