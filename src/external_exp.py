"""
ExternalExpRunner — perf calculation and timeline report for pre-existing
custom experiment directories (firenet, YOLO, etc.) without a Paper2Exp config.

Pipeline (mirrors Paper2Exp.run_exp):
  Step A  Load each video's CSV via the loader, normalize labels,
          write <stem>_results.csv to exp_dir (feeds TlReportGen).
  Step B  Compute per_frame + per_video metrics, write __perf*.csv.
  Step C  Generate _timeline_report.html via TlReportGen.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from halib import *

from src.common import GlobalConst
from src.metrics.base_csv_converter import FireSmokeLabelConverter

# Number of leading frames to skip per video when computing FPS
# (consistent with CsvMetricSrc's SKIP_N_FRAMES convention)
_SKIP_N_FRAMES_FPS = 3


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_binary(series: pd.Series) -> np.ndarray:
    """Convert a normalized label series (firesmoke / none) to 1 / 0."""
    return (series == GlobalConst.FIRESMOKE_LABEL).astype(int).values


def _compute_binary_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Compute accuracy, F1, precision, recall (TPR) and FPR from binary arrays."""
    tp = int(((gt == 1) & (pred == 1)).sum())
    tn = int(((gt == 0) & (pred == 0)).sum())
    fp = int(((gt == 0) & (pred == 1)).sum())
    fn = int(((gt == 1) & (pred == 0)).sum())
    total = tp + tn + fp + fn
    return {
        "metric_accuracy": (tp + tn) / total if total > 0 else 0.0,
        "metric_f1_score": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0,
        "metric_precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "metric_recall (TPR)": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "metric_FPR (False Alarm Rate)": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
    }


def _compute_fps(all_dfs: Dict[str, pd.DataFrame]) -> float:
    """Frame-level FPS, skipping the first _SKIP_N_FRAMES_FPS frames of each video."""
    total_elapsed = 0.0
    total_frames = 0
    for df in all_dfs.values():
        elapsed = df[GlobalConst.COL_ELAPSED_TIME].values
        if len(elapsed) > _SKIP_N_FRAMES_FPS:
            total_elapsed += float(elapsed[_SKIP_N_FRAMES_FPS:].sum())
            total_frames += len(elapsed) - _SKIP_N_FRAMES_FPS
    return total_frames / total_elapsed if total_elapsed > 0 else 0.0


# ---------------------------------------------------------------------------
# ExternalExpRunner
# ---------------------------------------------------------------------------

class ExternalExpRunner:
    """
    Runs the full evaluation pipeline for a pre-existing external experiment
    directory (firenet, YOLO OD, etc.) that was NOT produced by Paper2Exp.

    Instantiate via the class methods:
        ExternalExpRunner.from_firenet_dir(...)
        ExternalExpRunner.from_yolo_dir(...)

    Or directly by passing a loader instance:
        loader = FirenetExternalLoader(exp_dir)
        runner = ExternalExpRunner(exp_dir, dataset_dir, loader)
        runner.run()
    """

    def __init__(
        self,
        exp_dir: Union[str, Path],
        dataset_dir: Union[str, Path],
        loader,
        exp_name: Optional[str] = None,
        tl_type: str = GlobalConst.TL_TYPE_NO_SKIP,
        dataset_name: Optional[str] = None,
    ):
        self.exp_dir = Path(exp_dir)
        self.dataset_dir = Path(dataset_dir)
        self.loader = loader
        self.exp_name = exp_name or self.exp_dir.name
        self.tl_type = tl_type
        self.dataset_name = dataset_name or self.dataset_dir.name

    # ------------------------------------------------------------------
    # Step A — normalize CSVs and write *_results.csv
    # ------------------------------------------------------------------

    def _load_and_write_normalized_csvs(self) -> Dict[str, pd.DataFrame]:
        """
        For every video in dataset_dir:
          1. Load GT + pred via self.loader → merged DataFrame
          2. Normalize labels with FireSmokeLabelConverter
          3. Write <stem>_results.csv to exp_dir (standard pred format for TlReportGen)

        Returns:
            {video_stem: normalized_merged_df}
        """
        video_files = fs.filter_files_by_extension(
            str(self.dataset_dir), [".mp4", ".avi", ".mov"], recursive=True
        )
        assert len(video_files) > 0, f"No video files found in {self.dataset_dir}"

        label_converter = FireSmokeLabelConverter()
        all_dfs: Dict[str, pd.DataFrame] = {}

        for video_path in video_files:
            stem = Path(video_path).stem
            try:
                df = self.loader.load_video_gt_pred_df(video_path)
            except FileNotFoundError as e:
                print(f"[Warning] Skipping '{stem}': {e}")
                continue

            # Normalize GT and pred labels to "firesmoke" / "none"
            label_converter.do_convert(
                df, [GlobalConst.COL_GT, GlobalConst.COL_PRED], inplace=True
            )

            # Write normalized *_results.csv for downstream TlReportGen consumption.
            # Always overwrite.
            out_csv = self.exp_dir / f"{stem}{GlobalConst.INFER_FILE_PATTERN}.csv"
            cols_to_write = [
                GlobalConst.COL_VIDEO,
                GlobalConst.COL_VIDEO_PATH,
                GlobalConst.COL_FRAME_IDX,
                GlobalConst.COL_PRED,
                GlobalConst.COL_ELAPSED_TIME,
            ]
            df[[c for c in cols_to_write if c in df.columns]].to_csv(
                str(out_csv), sep=";", index=False, encoding="utf-8"
            )

            all_dfs[stem] = df

        return all_dfs

    # ------------------------------------------------------------------
    # Step B — compute metrics and write __perf*.csv
    # ------------------------------------------------------------------

    def _compute_per_frame_metrics(
        self, all_dfs: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        gt = np.concatenate(
            [_to_binary(df[GlobalConst.COL_GT]) for df in all_dfs.values()]
        )
        pred = np.concatenate(
            [_to_binary(df[GlobalConst.COL_PRED]) for df in all_dfs.values()]
        )
        metrics = _compute_binary_metrics(gt, pred)
        metrics["metric_FPS"] = _compute_fps(all_dfs)
        return metrics

    def _compute_per_video_metrics(
        self, all_dfs: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """
        Per-video aggregation: a video is positive if ANY of its frames is
        "firesmoke" (consistent with TorchMetricsConverter METRIC_PER_VIDEO logic).
        """
        video_gts: List[int] = []
        video_preds: List[int] = []
        for df in all_dfs.values():
            video_gts.append(
                int((df[GlobalConst.COL_GT] == GlobalConst.FIRESMOKE_LABEL).any())
            )
            video_preds.append(
                int((df[GlobalConst.COL_PRED] == GlobalConst.FIRESMOKE_LABEL).any())
            )
        metrics = _compute_binary_metrics(
            np.array(video_gts), np.array(video_preds)
        )
        # FPS is always frame-level (same for both modes)
        metrics["metric_FPS"] = _compute_fps(all_dfs)
        return metrics

    def _write_perf_csv(self, mode: str, metrics: Dict[str, float]) -> str:
        row: OrderedDict = OrderedDict()
        row["experiment"] = self.exp_name
        row["dataset"] = self.dataset_name
        row.update(metrics)

        df = pd.DataFrame([row])
        out_path = str(
            self.exp_dir
            / f"{GlobalConst.PERF_FILE_PREFIX}{self.exp_name}__{mode}.csv"
        )
        df.to_csv(out_path, sep=";", index=False, encoding="utf-8")
        pprint_local_path(out_path, get_wins_path=True)
        return out_path

    # ------------------------------------------------------------------
    # Step C — timeline report
    # ------------------------------------------------------------------

    def _gen_timeline_report(
        self,
        table_mode: Literal["p", "fc", "pfc"],
        table_decimals: int,
        video_name_limit: int,
    ) -> str:
        from src.results.timeline.tl_report import TlReportGen

        return TlReportGen.gen_TlReport_external_exp(
            exp_dir=str(self.exp_dir),
            dataset_dir=str(self.dataset_dir),
            exp_col_name=self.exp_name,
            tl_type=self.tl_type,
            table_mode=table_mode,
            table_decimals=table_decimals,
            video_name_limit=video_name_limit,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        table_mode: Literal["p", "fc", "pfc"] = "pfc",
        table_decimals: int = 2,
        video_name_limit: int = 40,
    ) -> None:
        console.rule(f"[bold green]ExternalExpRunner: {self.exp_name}[/bold green]")

        # --- Step A ---
        console.rule("Step A: Load & normalize CSVs → write *_results.csv")
        all_dfs = self._load_and_write_normalized_csvs()
        if not all_dfs:
            raise RuntimeError(
                f"No valid videos processed for experiment '{self.exp_name}'. "
                f"Check that pred CSV files exist in {self.exp_dir}."
            )
        console.print(f"[green]Processed {len(all_dfs)} video(s).[/green]")

        # --- Step B ---
        console.rule("Step B: Compute metrics → write __perf*.csv")
        pf_metrics = self._compute_per_frame_metrics(all_dfs)
        self._write_perf_csv(GlobalConst.METRIC_PER_FRAME, pf_metrics)
        pv_metrics = self._compute_per_video_metrics(all_dfs)
        self._write_perf_csv(GlobalConst.METRIC_PER_VIDEO, pv_metrics)

        # --- Step C ---
        console.rule("Step C: Generate timeline report")
        report_path = self._gen_timeline_report(
            table_mode=table_mode,
            table_decimals=table_decimals,
            video_name_limit=video_name_limit,
        )
        console.print(f"[green]Timeline report: {report_path}[/green]")

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_cls_model_dir(
        cls,
        exp_dir: Union[str, Path],
        dataset_dir: Union[str, Path],
        exp_name: Optional[str] = None,
        **kwargs,
    ) -> "ExternalExpRunner":
        """Create a runner for a cls model-style _pred.csv experiment directory."""
        from src.metrics.loaders.cls_csv_loader import ClsModelExternalLoader

        loader = ClsModelExternalLoader(str(exp_dir))
        return cls(exp_dir, dataset_dir, loader, exp_name=exp_name, **kwargs)

    @classmethod
    def from_yolo_dir(
        cls,
        exp_dir: Union[str, Path],
        dataset_dir: Union[str, Path],
        exp_name: Optional[str] = None,
        **kwargs,
    ) -> "ExternalExpRunner":
        """Create a runner for a YOLO OD _od.csv experiment directory."""
        from src.metrics.loaders.yolo_csv_loader import YoloExternalLoader

        loader = YoloExternalLoader(str(exp_dir))
        return cls(exp_dir, dataset_dir, loader, exp_name=exp_name, **kwargs)
