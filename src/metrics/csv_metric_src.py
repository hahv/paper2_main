import torch
from halib import *
from typing import Dict, Any
from src.config import Config
from src.metrics.base_metric_src import BaseMetricSrc


class CsvMetricSrc(BaseMetricSrc):
    """
    Concrete data source for a hypothetical video dataset.
    Assumes data structure: videos with frames, each frame has gt_label, pred_label, timestamp, etc.
    Predictions might be loaded from a file specified in kwargs (e.g., 'prediction_path').
    """

    POS_LABEL = "O_FireSmoke"
    NEG_LABEL = "X_None"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        super().__init__(cfg.dbsetCfg.name)  # ty:ignore[invalid-argument-type]
        self.per_video_out_list = None

    def _register_handlers(self):
        metric_set_meta = self.cfg.metricCfg
        metric_names = metric_set_meta.metric_names
        modes = metric_set_meta.extra_cfgs.get("mode", ["per-video"])  # ty:ignore[possibly-missing-attribute]
        self.did_save_raw_pred_and_gt = {mode: False for mode in modes}
        # ! set up data getters
        for metric in metric_names:
            self.metric_data_getters_dict[metric] = self.get_metric_data_by_mode
        # ! setup mode proccessors
        for mode in modes:
            self.mode_processors_dict[mode] = self.proc_data_by_mode

    def save_raw_pred_and_gt(self, mode: str, pervideo_pred_gt_ls: Any):
        assert mode in ["per_video", "per_frame"], f"Mode {mode} not implemented yet"
        if not self.did_save_raw_pred_and_gt[mode]:
            target_out = os.path.join(self.cfg.get_outdir(), f"[{mode}]_pred_vs_gt_csv")
            if not os.path.exists(target_out):
                os.makedirs(target_out, exist_ok=True)
            dfmk = csvfile.DFCreator()
            if mode == "per_video":
                dfmk.create_table("raw_preds", ["video_name", "gt", "pred", "correct"])
                rows = []
                for pred_df, gt in pervideo_pred_gt_ls:
                    video_name = pred_df["video"].iloc[0]
                    gt = gt[0]  # all frames have same gt
                    v_pred = CsvMetricSrc.NEG_LABEL
                    preds = pred_df["pred"].unique().tolist()
                    if CsvMetricSrc.POS_LABEL in preds:
                        v_pred = CsvMetricSrc.POS_LABEL
                    rows.append([video_name, gt, v_pred, int(v_pred == gt)])
                dfmk.insert_rows("raw_preds", rows)
                dfmk.fill_table_from_row_pool("raw_preds")
                dfmk["raw_preds"].sort_values(
                    by=["correct", "video_name"], ascending=[True, True], inplace=True
                )
                # ! save to single csv
                dfmk["raw_preds"].to_csv(
                    os.path.join(
                        target_out,
                        f"all_pred_vs_gt.csv",
                    ),
                    sep=";",
                    encoding="utf-8",
                    index=False,
                )
            else:
                for pred_df, gts in pervideo_pred_gt_ls:
                    outdf = pred_df.copy()
                    outdf["gt"] = gts
                    outdf["correct"] = outdf["gt"] == outdf["pred"]
                    # only rows if correct is False
                    outdf = outdf[outdf["correct"] == False]
                    if len(outdf) > 0:  # only save if there are some incorrect frames
                        video_name = outdf["video"].iloc[0]
                        outdf.to_csv(
                            os.path.join(
                                target_out,
                                f"{video_name}_pred_vs_gt.csv",
                            ),
                            sep=";",
                            encoding="utf-8",
                            index=False,
                        )
            self.did_save_raw_pred_and_gt[mode] = True

    def get_gt_df(self, csv_file, mode, dataset_name, num_frames, has_csv_label=False):
        video_name = fs.get_file_name(csv_file, split_file_ext=True)[0]
        video_name = video_name.replace("_results", "")
        recursive = self.cfg.dbsetCfg.extra_cfgs.get("ds_recursive", False)
        vname2path_dict = self.cfg.dbsetCfg.get_vname2path(
            recursive=recursive
        )


        if has_csv_label:
            label_gt_file_name = f"{video_name}__labels.csv"
            video_dir = os.path.dirname(vname2path_dict[video_name])
            csv_file = os.path.join(video_dir, label_gt_file_name)
            assert os.path.exists(csv_file), f"CSV label file {csv_file} does not exist"
            gt = []
            gt_df = pd.read_csv(
                csv_file,
                sep=";",
                encoding="utf-8",
                dtype={"label": str},
                keep_default_na=False,
            )
            gt_col_list = gt_df["label"].tolist()
            # ! make sure num_frames match
            gt_col_list = gt_col_list[:num_frames]
            for label in gt_col_list:
                if "fire" in label.lower() or "smoke" in label.lower():
                    gt.append(CsvMetricSrc.POS_LABEL)
                else:
                    gt.append(CsvMetricSrc.NEG_LABEL)
            return gt
        else:
            pprint(locals())
            # there is no csv labels so should be infer gt from "video_name" in csv_file
            video_name = video_name.replace("_results", "")
            gt = CsvMetricSrc.POS_LABEL
            if dataset_name == "DFire":
                if "FP" in video_name:
                    gt = CsvMetricSrc.NEG_LABEL
            else:
                raise NotImplementedError(
                    f"get gt label for whole video in Dataset {dataset_name} not implemented yet"
                )
            gt = [gt] * num_frames
            return gt

    def pred_csv_to_pred_gt_df(self, csv_file, mode, dataset_name, has_csv_label=False):
        pred_df = pd.read_csv(
            csv_file,
            sep=";",
            encoding="utf-8",
            dtype={"pred_label": str, "elapsed_time": float},
            keep_default_na=False,
        )
        # Convert labels: heuristically, if "fire" or "smoke" in label, it's positive
        pred_df["pred"] = (
            pred_df["pred_label"]
            .str.lower()
            .apply(
                lambda x: (
                    CsvMetricSrc.POS_LABEL
                    if ("fire" in x or "smoke" in x)
                    else CsvMetricSrc.NEG_LABEL
                )
            )
        )
        gt = self.get_gt_df(
            csv_file,
            mode,
            dataset_name,
            num_frames=len(pred_df),
            has_csv_label=has_csv_label,
        )
        return pred_df, gt

    def load_raw_pred_and_gt(self, mode, **kwargs):
        if self.per_video_out_list is not None:
            return self.per_video_out_list
        indir = kwargs.get("indir", None)
        if indir is None:
            indir = self.cfg.get_outdir()
        assert indir is not None, "indir must be provided"
        # first list all video files
        recursive = self.cfg.dbsetCfg.extra_cfgs.get("ds_recursive", False)
        num_videos = self.cfg.dbsetCfg.get_num_videos(recursive=recursive)
        csv_files = fs.filter_files_by_extension(indir, [".csv"], recursive=False)
        # only keep those with "_results" in the name
        # console.rule("Filtered CSV files for metric data")
        filtered_csv_files = []
        for csv_file in csv_files:
            if "_results" in os.path.basename(csv_file):
                filtered_csv_files.append(csv_file)
        csv_files = filtered_csv_files
        assert len(csv_files) == num_videos, (
            f"Number of CSV files ({len(csv_files)}) does not match number of video files ({num_videos})"
        )

        # return video_name, gt, and df
        pervideo_pred_gt_ls = []  # gt, df for each frames in each video
        csv_labels = self.cfg.dbsetCfg.get_csv_labels(recursive=False)
        has_csv_label = len(csv_labels) > 0
        for csv_file in csv_files:
            df_pred, gt = self.pred_csv_to_pred_gt_df(
                csv_file,
                mode=mode,
                dataset_name=self.cfg.dbsetCfg.name,
                has_csv_label=has_csv_label,
            )
            pervideo_pred_gt_ls.append((df_pred, gt))
        self.per_video_out_list = pervideo_pred_gt_ls
        return self.per_video_out_list

    def get_metric_data_by_mode(self, metric, mode, **kwargs) -> Dict[str, Any]:
        # Load raw data tailored for classification metrics (labels)
        pervideo_pred_gt_ls = self.load_raw_pred_and_gt(mode=mode, **kwargs)
        self.save_raw_pred_and_gt(mode=mode, pervideo_pred_gt_ls=pervideo_pred_gt_ls)
        if metric == "FPS":
            # Compute FPS from raw_data
            pervideo_preds_all = []
            for per_video_data in pervideo_pred_gt_ls:
                pred_df = per_video_data[0]
                pervideo_preds_all.append(
                    pred_df["elapsed_time"].tolist()[1:]
                )  # skip first frame, which is always two slow due to model initialization

            return pervideo_preds_all  # list of list of elapse_time
        else:
            pervideo_preds_all = []
            pervideo_gts_all = []
            for per_video_data in pervideo_pred_gt_ls:
                per_video_pred_df = per_video_data[0]
                preds = per_video_pred_df["pred"].tolist()
                preds = np.array(preds) == CsvMetricSrc.POS_LABEL
                preds = preds.astype(int).tolist()  # convert to int
                gts = per_video_data[1]  # already numpy
                gts = np.array(gts) == CsvMetricSrc.POS_LABEL
                gts = gts.astype(int).tolist()  # convert to int
                pervideo_preds_all.append(preds)
                pervideo_gts_all.append(gts)

            return pervideo_preds_all, pervideo_gts_all

    def proc_data_by_mode(
        self, metric: str, mode: str, metric_data: Dict[str, Any], **kwargs
    ):
        def proc_list_to_tensor(data_list, flatten, dtype):
            if flatten:
                # np.concatenate takes a list of lists/arrays and joins them into one.
                data_npy = np.concatenate(data_list)
            else:
                # If not flattening, create the array as before.
                data_npy = np.array(data_list)

            return torch.from_numpy(data_npy).to(dtype)

        if metric == "FPS":
            flatten = True
            torch_data = proc_list_to_tensor(
                data_list=metric_data, flatten=flatten, dtype=torch.float
            )
            return torch_data
        else:
            if mode == "per_frame":
                flatten = True
                per_video_preds, per_video_gts = metric_data
                preds_tensor = proc_list_to_tensor(
                    data_list=per_video_preds, flatten=flatten, dtype=torch.int
                )
                gts_tensor = proc_list_to_tensor(
                    data_list=per_video_gts, flatten=flatten, dtype=torch.int
                )
                return (preds_tensor, gts_tensor)

            elif mode == "per_video":
                video_level_preds = []
                video_level_gts = []
                flatten = False
                zip_metric_data = list(zip(metric_data[0], metric_data[1]))
                for pervideo_pred, pervideo_gt in zip_metric_data:
                    # if any frame is positive, the video is positive
                    video_pred = int(any(pervideo_pred))
                    video_gt = int(any(pervideo_gt))
                    video_level_preds.append(video_pred)
                    video_level_gts.append(video_gt)

                preds_tensor = proc_list_to_tensor(
                    data_list=video_level_preds, flatten=flatten, dtype=torch.int
                )
                gts_tensor = proc_list_to_tensor(
                    data_list=video_level_gts, flatten=flatten, dtype=torch.int
                )
                return (preds_tensor, gts_tensor)
            else:
                raise NotImplementedError(f"Mode {mode} not implemented yet")
