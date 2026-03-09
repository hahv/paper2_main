"""
ExternalMethod — a BaseMethod subclass that replaces model inference
with loading pre-existing CSV results from the experiment directory.

Supported CSV formats (auto-detected from the experiment directory name):
  - "yolo" in dir name  → YoloExternalLoader  (sparse _od.csv, OD format)
  - otherwise           → ClsModelExternalLoader (_pred.csv, classifier format)

After loading and normalizing each video's CSV, it writes the standard
<stem>_results.csv to cfg.get_outdir(), then the rest of the Paper2Exp
pipeline (calc_perfs, TlReportGen) proceeds unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from halib import *

from src.common import GlobalConst
from src.config import Config
from src.metrics.base_csv_converter import FireSmokeLabelConverter
from src.methods.base_method import BaseMethod
from src.results.base_rs_proc import BaseRsProc


class ExternalMethod(BaseMethod):
    """
    Reads pre-existing inference CSVs instead of running a model.
    Drop-in replacement for NoTempMethod / TempMethod in from_custom_exp flows.
    """

    def __init__(self, cfg: Config, rs_handlers: Optional[list[BaseRsProc]] = None, **kwargs):
        super().__init__(cfg, rs_handlers)

    # ------------------------------------------------------------------
    # No model — override to do nothing
    # ------------------------------------------------------------------

    def load_model(self):
        return None

    def prepare_model(self):
        return None

    # ------------------------------------------------------------------
    # infer_frame is never called in this method
    # ------------------------------------------------------------------

    def infer_frame(self, frame, frame_idx: int) -> dict:
        raise NotImplementedError(
            "ExternalMethod does not run frame-level inference. "
            "It reads pre-existing CSV files instead."
        )

    # ------------------------------------------------------------------
    # Core override: replace video-dir inference with CSV loading
    # ------------------------------------------------------------------

    def _get_loader(self):
        """Auto-detect loader from the experiment directory name."""
        dir_name = Path(self.cfg.get_outdir()).name.lower()
        if "yolo" in dir_name:
            from src.metrics.loaders.yolo_csv_loader import YoloExternalLoader
            return YoloExternalLoader(self.cfg.get_outdir())
        else:
            from src.metrics.loaders.cls_csv_loader import ClsModelExternalLoader
            return ClsModelExternalLoader(self.cfg.get_outdir())

    def infer_video_dir(self, video_dir: str, recursive: bool = True, max_workers: int = 0):
        """
        For every video in video_dir:
          1. Load pre-existing CSV via the auto-detected loader
          2. Normalize GT + pred labels to "firesmoke" / "none"
          3. Write <stem>_results.csv to cfg.get_outdir()
        Then trigger the normal after_infer_video_dir hook (timeline report, profiler).
        """
        loader = self._get_loader()
        label_converter = FireSmokeLabelConverter()
        outdir = Path(self.cfg.get_outdir())

        video_files = fs.filter_files_by_extension(
            video_dir, [".mp4", ".avi", ".mov", ".mkv"], recursive=recursive
        )
        assert len(video_files) > 0, f"No video files found in {video_dir}"

        self.before_infer_video_dir(video_dir)

        for video_path in video_files:
            stem = Path(video_path).stem
            try:
                df = loader.load_video_gt_pred_df(video_path)
            except FileNotFoundError as e:
                console.print(f"[yellow][Warning] Skipping '{stem}': {e}[/yellow]")
                continue

            # Normalize labels to "firesmoke" / "none"
            label_converter.do_convert(
                df, [GlobalConst.COL_GT, GlobalConst.COL_PRED], inplace=True
            )

            # Write standard _results.csv — always overwrite
            cols_to_write = [
                GlobalConst.COL_VIDEO,
                GlobalConst.COL_VIDEO_PATH,
                GlobalConst.COL_FRAME_IDX,
                GlobalConst.COL_PRED,
                GlobalConst.COL_ELAPSED_TIME,
            ]
            out_csv = outdir / f"{stem}{GlobalConst.INFER_FILE_PATTERN}.csv"
            df[[c for c in cols_to_write if c in df.columns]].to_csv(
                str(out_csv), sep=";", index=False, encoding="utf-8"
            )
            print(f"Written <*_results.csv>: {out_csv.name}", end="\r")

        self.after_infer_video_dir(video_dir)
