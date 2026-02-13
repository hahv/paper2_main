import torch
from halib import *

from typing import Any
from pathlib import Path

from src.common import GlobalConst
from src.config import Config
from src.utils import get_cls_in_pkg

from src.metrics.base_metric_src import BaseMetricSrc
from src.metrics.loaders.base_csv_loader import BaseRawCsvLoader
from src.metrics.base_csv_converter import *


class CsvMetricSrc(BaseMetricSrc):
    """
    Concrete data source that delegates dataset-specific parsing to an Adapter.
    """

    SKIP_FRAME_FOR_FPS = 3  # skip first frame for FPS calculation
    UNIFED_CSV_FILE = "pred_vs_gt.csv"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        modes = cfg.metricCfg.extra_cfgs.get("mode", ["per-video"])  # ty:ignore[possibly-missing-attribute]
        super().__init__(cfg.dbsetCfg.name, modes=modes)  # ty:ignore[invalid-argument-type]

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
        self.csv_loader: BaseRawCsvLoader = csv_loader_cls(self.cfg)
        video_list = self.cfg.dbsetCfg.get_video_list()
        # ! global cache of video_name => raw_gt_pred_df
        self.video_gt_pred_df_dict = {}
        for vpath in video_list:
            raw_gt_pred_df = self.csv_loader.load_video_gt_pred_df(video_path=vpath)
            assert raw_gt_pred_df is not None, (
                f"Failed to load GT/Pred for video {vpath}"
            )
            self.video_gt_pred_df_dict[Path(vpath).name] = raw_gt_pred_df

        self.did_save_unified_metric_df = {mode: False for mode in modes}
        self.cache_unified_df_dict = {}

    def _register_handlers(self):
        metric_names = self.cfg.metricCfg.metric_names

        # ! set up data getters
        for metric in metric_names:
            self.metric_data_getters_dict[metric] = self.get_metric_data_by_mode

    def metric_mode_to_cache_key(self, mode, metric):
        if metric == "FPS":
            return "FPS"  # FPS is always per-frame, so we can use a unified key for caching
        else:
            return mode  # for other metrics, we cache by mode (per-frame or per-video)

    def unify_df_by_mode(self, mode, metric, **kwargs) -> tuple[pd.DataFrame, str]:
        list_of_converted_dfs = []
        for video_name, df in self.video_gt_pred_df_dict.items():
            df = df.copy()
            df = BaseCSVConverter.do_convert_chain(
                df,
                [
                    (GlobalConst.COL_GT, FireSmokeLabelConverter()),
                    (GlobalConst.COL_PRED, FireSmokeLabelConverter()),
                ],
                inplace=True,
            )
            if metric == "FPS":
                mode = GlobalConst.METRIC_PER_FRAME  # FPS is always per-frame
            converted_df = BaseCSVConverter.do_convert_chain(
                df,
                [
                    (GlobalConst.COL_GT, TorchMetricsConverter()),
                    (GlobalConst.COL_PRED, TorchMetricsConverter()),
                ],
                inplace=True,
                extra_dict={"metric_mode": mode},
            )
            csvfile.fn_display_df(converted_df.head(10))
            if metric == "FPS":
                # we need to skip first frame for FPS calculation
                converted_df = converted_df.iloc[self.SKIP_FRAME_FOR_FPS :].reset_index(
                    drop=True
                )
            # assert False, "Debugging unify_df_by_mode"
            list_of_converted_dfs.append(converted_df)
        all_videos_df = pd.concat(list_of_converted_dfs, ignore_index=True)
        cache_key = self.metric_mode_to_cache_key(mode, metric)
        all_videos_df.to_csv(
            f"{self.cfg.get_outdir()}/{cache_key}.csv", index=False, sep=";"
        )
        return all_videos_df, cache_key

    def get_metric_data_by_mode(self, metric, mode, **kwargs) -> Any:
        metric = metric.strip()
        cache_key = self.metric_mode_to_cache_key(mode, metric)
        if cache_key in self.cache_unified_df_dict:
            print("Using cached unified_df for key:", cache_key)
            unified_df = self.cache_unified_df_dict[cache_key]
        else:
            print("Generating unified_df for key:", cache_key)
            unified_df, cache_key = self.unify_df_by_mode(mode, metric, **kwargs)
            self.cache_unified_df_dict[cache_key] = unified_df

        if metric == "FPS":
            elapsed_times = unified_df[GlobalConst.COL_ELAPSED_TIME].to_numpy()
            return torch.from_numpy(elapsed_times).to(torch.float)
        else:
            preds = unified_df[GlobalConst.COL_PRED].to_numpy()
            gts = unified_df[GlobalConst.COL_GT].to_numpy()
            return (
                torch.from_numpy(preds).to(torch.int),
                torch.from_numpy(gts).to(torch.int),
            )
        if self.did_save_unified_metric_df[mode] is False:
            # Save unified CSV for reference
            outfile = os.path.join(
                self.cfg.get_outdir(),
                f"{GlobalConst.PERF_FILE_PREFIX}[{mode}]__{UNIFIED_CSV_FILE}",
            )
            report_df = unified_df.copy()
            report_df["correct"] = (
                report_df[GlobalConst.COL_PRED] == report_df[GlobalConst.COL_GT]
            )
            # sort by video, frame_idx, correctness
            if mode == GlobalConst.METRIC_PER_FRAME:
                report_df = report_df.sort_values(
                    by=[
                        GlobalConst.COL_VIDEO,
                        "correct",
                        GlobalConst.COL_FRAME_IDX,
                    ],
                    ascending=[True, False, True],
                )
            else:
                report_df = report_df.sort_values(
                    by=[
                        "correct",
                    ],
                    ascending=[False],
                )
            report_df.to_csv(outfile, sep=";", index=False, encoding="utf-8")
            pprint(f"Saved unified metric CSV for mode {mode} at ⏬:")
            pprint_local_path(outfile, get_wins_path=True)
            self.did_save_unified_metric_df[mode] = True
