import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

from halib import *
from src.config import Config
from src.utils import get_cls_in_pkg
from src.metrics.base_metric_src import BaseMetricSrc
from src.metrics.loaders.base_csv_loader import BaseCsvLoader


class CsvMetricSrc(BaseMetricSrc):
    """
    Concrete data source that delegates dataset-specific parsing to an Adapter.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        super().__init__(cfg.dbsetCfg.name)  # ty:ignore[invalid-argument-type]

        # -----------------------------------------------------------
        # ! LOAD ADAPTER
        # We read 'adapter_cls' from config (e.g., 'DFireAdapter')
        # Defaulting to DFireAdapter to maintain backward compatibility
        # -----------------------------------------------------------
        csv_loader_name = cfg.dbsetCfg.extra_cfgs.get("csv_loader_cls")  # ty:ignore[possibly-missing-attribute]

        # Dynamically load the adapter class from src.metrics.adapters.dataset_adapters
        csv_loader_cls = get_cls_in_pkg(
            pkg_name="src.metrics.loaders",
            fileName_ClsName=csv_loader_name,  # ty:ignore[invalid-argument-type]
        )

        # Initialize adapter
        self.csv_loader: BaseCsvLoader = csv_loader_cls(self.cfg)

        # Expose labels from adapter for consistency
        self.POS_LABEL = self.csv_loader.POS_LABEL
        self.NEG_LABEL = self.csv_loader.NEG_LABEL

        self.per_video_out_list = None

    def _register_handlers(self):
        metric_set_meta = self.cfg.metricCfg
        metric_names = metric_set_meta.metric_names
        modes = metric_set_meta.extra_cfgs.get("mode", ["per-video"])  # ty:ignore[possibly-missing-attribute]

        self.did_save_raw_pred_and_gt = {mode: False for mode in modes}

        # ! set up data getters
        for metric in metric_names:
            self.metric_data_getters_dict[metric] = self.get_metric_data_by_mode

        # ! setup mode processors
        for mode in modes:
            self.mode_processors_dict[mode] = self.proc_data_by_mode

    def load_raw_pred_and_gt(self, mode, **kwargs):
        """
        Orchestrator: Finds files -> Uses Adapter to Parse -> Returns List
        """
        if self.per_video_out_list is not None:
            return self.per_video_out_list

        indir = kwargs.get("indir", None)
        if indir is None:
            indir = self.cfg.get_outdir()
        assert indir is not None, "indir must be provided"

        # 1. Filter CSV Files
        recursive = self.cfg.dbsetCfg.extra_cfgs.get("ds_recursive", False)  # ty:ignore[possibly-missing-attribute]
        csv_files = fs.filter_files_by_extension(indir, [".csv"], recursive=recursive)

        # Filter for "_results"
        result_csv_files = [f for f in csv_files if "_results" in os.path.basename(f)]

        # (Optional) Validation logic regarding number of videos can go here

        pervideo_pred_gt_ls = []

        for csv_file in result_csv_files:
            # -------------------------------------------------------
            # ! DELEGATE TO ADAPTER
            # -------------------------------------------------------
            # A. Load Preds
            pred_df = self.csv_loader.load_pred_df(csv_file)

            # B. Load GT
            video_name = pred_df["video"].iloc[0]
            gt = self.csv_loader.get_gt(
                video_name=video_name, num_frames=len(pred_df), pred_df=pred_df
            )

            pervideo_pred_gt_ls.append((pred_df, gt))

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
                # skip first frame
                pervideo_preds_all.append(pred_df["elapsed_time"].tolist()[1:])
            return pervideo_preds_all  # ty:ignore[invalid-return-type]
        else:
            pervideo_preds_all = []
            pervideo_gts_all = []

            for per_video_data in pervideo_pred_gt_ls:
                per_video_pred_df = per_video_data[0]

                # Convert string labels to 1/0 integers
                preds = per_video_pred_df["pred"].tolist()
                preds = (np.array(preds) == self.POS_LABEL).astype(int).tolist()

                gts = per_video_data[1]
                gts = (np.array(gts) == self.POS_LABEL).astype(int).tolist()

                pervideo_preds_all.append(preds)
                pervideo_gts_all.append(gts)

            return pervideo_preds_all, pervideo_gts_all  # ty:ignore[invalid-return-type]

    def proc_data_by_mode(
        self, metric: str, mode: str, metric_data: Dict[str, Any], **kwargs
    ):
        # ... (This method remains largely identical to your original code) ...
        # ... (Implementation omitted for brevity, copy paste from original) ...

        # Helper to convert list to tensor
        def proc_list_to_tensor(data_list, flatten, dtype):
            if flatten:
                data_npy = np.concatenate(data_list)
            else:
                data_npy = np.array(data_list)
            return torch.from_numpy(data_npy).to(dtype)

        if metric == "FPS":
            flatten = True
            return proc_list_to_tensor(metric_data, flatten=flatten, dtype=torch.float)
        else:
            if mode == "per_frame":
                flatten = True
                per_video_preds, per_video_gts = metric_data
                preds_tensor = proc_list_to_tensor(
                    per_video_preds, flatten=flatten, dtype=torch.int
                )
                gts_tensor = proc_list_to_tensor(
                    per_video_gts, flatten=flatten, dtype=torch.int
                )
                return (preds_tensor, gts_tensor)

            elif mode == "per_video":
                video_level_preds = []
                video_level_gts = []
                flatten = False
                zip_metric_data = list(zip(metric_data[0], metric_data[1]))  # ty:ignore[invalid-argument-type]

                for pervideo_pred, pervideo_gt in zip_metric_data:
                    # Logic: if any frame is positive, video is positive
                    video_pred = int(any(pervideo_pred))
                    video_gt = int(any(pervideo_gt))
                    video_level_preds.append(video_pred)
                    video_level_gts.append(video_gt)

                preds_tensor = proc_list_to_tensor(
                    video_level_preds, flatten=flatten, dtype=torch.int
                )
                gts_tensor = proc_list_to_tensor(
                    video_level_gts, flatten=flatten, dtype=torch.int
                )
                return (preds_tensor, gts_tensor)
            else:
                raise NotImplementedError(f"Mode {mode} not implemented yet")

    # ... (Include save_raw_pred_and_gt method from original code here) ...
    def save_raw_pred_and_gt(self, mode: str, pervideo_pred_gt_ls: Any):
        # This method is mostly identical to your original,
        # just ensure you use self.POS_LABEL and self.NEG_LABEL
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
                    gt_val = gt[0]

                    # Calculate video-level pred
                    v_pred = self.NEG_LABEL
                    preds = pred_df["pred"].unique().tolist()
                    if self.POS_LABEL in preds:
                        v_pred = self.POS_LABEL

                    rows.append([video_name, gt_val, v_pred, int(v_pred == gt_val)])

                dfmk.insert_rows("raw_preds", rows)
                dfmk.fill_table_from_row_pool("raw_preds")
                dfmk["raw_preds"].sort_values(
                    by=["correct", "video_name"], ascending=[True, True], inplace=True
                )
                dfmk["raw_preds"].to_csv(
                    os.path.join(target_out, f"all_pred_vs_gt.csv"),
                    sep=";",
                    index=False,
                )

            else:  # per_frame
                for pred_df, gts in pervideo_pred_gt_ls:
                    outdf = pred_df.copy()
                    outdf["gt"] = gts
                    outdf["correct"] = outdf["gt"] == outdf["pred"]
                    outdf = outdf[~outdf["correct"]]
                    if len(outdf) > 0:
                        video_name = outdf["video"].iloc[0]
                        outdf.to_csv(
                            os.path.join(target_out, f"{video_name}_pred_vs_gt.csv"),
                            sep=";",
                            index=False,
                        )

            self.did_save_raw_pred_and_gt[mode] = True
