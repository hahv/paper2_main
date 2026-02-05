from halib import *

import cv2
import time
import timm
import torch
from abc import ABC, abstractmethod
from typing import Optional
from halib.exp.perf.profiler import zProfiler

from src.results import *
from src.config import Config
from src.metrics.base_metric_src import *
from src.results.base_rs_proc import BaseRsProc
from src.results.csv_rs_proc import CsvRsProc
import sys
from src.utils import get_cls_in_pkg

import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# Constants for package paths (avoids magic strings scattered in code)
PKG_METHODS = "src.methods"
PKG_RESULTS = "src.results"
DEFAULT_VIDEO_PROC = "video_infer_rs_proc"


class MethodFactory:
    TEMP_METHOD_NAME_PREFIX = "temp_method"

    @staticmethod
    def create_method(config: Config, *args, **kwargs):
        # Constants
        METHOD_PKG = "src.methods"
        RESULTS_PKG = "src.results"
        DEFAULT_VIDEO_PROC = "video_infer_rs_proc"

        # 1. Load the Method Class
        method_name = config.methodCfg.name
        assert method_name is not None, (
            "Method name must be specified in config.methodCfg.name"
        )
        if method_name.startswith(MethodFactory.TEMP_METHOD_NAME_PREFIX):
            method_name = MethodFactory.TEMP_METHOD_NAME_PREFIX
        # pprint(f"Loading method class: {method_name}")
        method_cls = get_cls_in_pkg(
            pkg_name=METHOD_PKG,
            fileName_ClsName=str(method_name),
        )

        # 2. Assemble Result Handlers
        rs_handlers: list[BaseRsProc] = []

        # --- CSV Handler ---
        if config.inferCfg.save_csv_results:
            rs_handlers.append(CsvRsProc(config))

        # --- Video Handlers ---
        if config.inferCfg.save_video_results:
            # Extract video config safely (defaults to [DEFAULT_VIDEO_PROC] if missing)
            extra_cfgs = getattr(config.methodCfg, "extra_cfgs", {})
            proc_names = extra_cfgs.get("result_proc", {}).get(
                "video", [DEFAULT_VIDEO_PROC]
            )

            # Ensure it is a list
            if isinstance(proc_names, str):
                proc_names = [proc_names]

            # Load and instantiate processors
            for name in proc_names:
                proc_cls = get_cls_in_pkg(pkg_name=RESULTS_PKG, fileName_ClsName=name)
                rs_handlers.append(proc_cls(cfg=config))

        # 3. Instantiate Method
        return method_cls(cfg=config, rs_handlers=rs_handlers, *args, **kwargs)


class BaseMethod(ABC):
    """
    An abstract base class for video inference that decouples inference logic
    from output handling (e.g., saving CSVs or videos) via a handler system.
    """

    REQUIRED_INFER_RS = ["logits", "probs", "predLabelIdx", "predLabel"]

    def __init__(self, cfg: Config, rs_handlers: Optional[list[BaseRsProc]] = None):
        """
        Initializes the detector.

        Args:
            cfg (Config): The configuration object.
            rs_handlers (Optional[list[BaseRsProc]], optional): A list of handlers to process the inference results. Defaults to None.
        """
        self.cfg: Config = cfg
        self.model = None
        self.gpu_monitor = None
        self.outdir = os.path.abspath(cfg.get_outdir())
        os.makedirs(self.outdir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.profiler: zProfiler = zProfiler(enabled=self.cfg.inferCfg.use_profiler)

        # Store the list of handlers that will process the results
        self.result_handlers = rs_handlers if rs_handlers is not None else []
        self.num_infer_workers = self.cfg.inferCfg.num_infer_workers

    @abstractmethod
    def infer_frame(self, frame, frame_idx: int) -> dict:
        """
        Handles detection for a single frame.

        Returns:
            a dict contains:
                logits
                probs
                labelIdx
                predLabel
                extra: if needed
        """
        pass

    # ! override if needed
    def prepare_metric_src(self, **kwargs):
        """
        Prepares the metric source and retrieves metric data.
        """
        perf_dir = self.cfg.get_outdir()
        metric_source = MetricSrcFactory.create_metric_src(self.cfg)
        base_metric_src: BaseMetricSrc = metric_source
        return base_metric_src.get_data_metrics(in_dir=perf_dir, **kwargs)

    # -------------------------------------------------------------------------
    # Core Methods
    # -------------------------------------------------------------------------

    def load_model(self):
        """Custom method to load the model, can be overridden if needed."""
        return timm.create_model(
            self.cfg.modelCfg.base_model,
            pretrained=False,
            num_classes=len(self.cfg.modelCfg.class_names),
            checkpoint_path=self.cfg.modelCfg.model_path,
        )

    def prepare_model(self):
        """Loads the model onto the appropriate device if it hasn't been loaded."""
        if self.model is None:
            self.model = self.load_model()
            self.model.eval()
            self.model = self.model.to(self.device)
            print(f"Model loaded on {self.device}.")
        return self.model

    def _log_progress(self, frame_idx: int, total_frames: int):
        """Logs the processing progress to the console."""
        percentage = (frame_idx / total_frames) * 100
        console.print(
            f"Infer frame {frame_idx}/{total_frames} ({percentage:.2f}%)...",
            end="\r",
            highlight=False,
        )
        sys.stdout.flush()  # Force the flush manually

    # ! ----HOOK METHODS for process VIDEO DIR or SINGLE VIDEO-----------
    def before_infer_video_dir(self, video_dir: str):
        """Hook method called before starting inference on a video directory."""
        pass

    def after_infer_video_dir(self, video_dir: str):
        """Hook method called after completing inference on a video directory."""
        if self.profiler:
            self.profiler.report_and_plot(outdir=self.outdir)

    def before_infer_video(self, video_path: str):
        """Hook method called before starting inference on a video."""
        pass

    def after_infer_video(self, video_path: str):
        """Hook method called after completing inference on a video."""
        pass

    # !------End HOOKS------------------------------------------------------

    def infer_video_dir_no_parallel(self, video_dir: str, recursive: bool = True):
        """Processes all videos in a specified directory."""
        assert os.path.exists(video_dir), f"Video directory {video_dir} does not exist."
        video_files = fs.filter_files_by_extension(
            video_dir, [".mp4", ".avi", ".mov", ".mkv"], recursive=recursive
        )
        assert len(video_files) > 0, f"No video files found in {video_dir}."

        self.before_infer_video_dir(video_dir)

        for i, video_path in enumerate(video_files):
            self.infer_video(video_path, video_idx=i, total_videos=len(video_files))

        self.after_infer_video_dir(video_dir)

    def infer_video_dir(
        self, video_dir: str, recursive: bool = True, max_workers: int = 0
    ):
        """
        Processes all videos in a specified directory in parallel.
        max_workers: Number of parallel processes (videos) to run at once.
                     CAUTION: Each worker loads a copy of the model.
                     If you have 1 GPU and a big model, set this to 1 or 2.
        """
        if max_workers <= 1:
            self.infer_video_dir_no_parallel(video_dir, recursive=recursive)
            return
        else:
            assert os.path.exists(video_dir), (
                f"Video directory {video_dir} does not exist."
            )

            # 1. Filter files
            video_files = fs.filter_files_by_extension(
                video_dir, [".mp4", ".avi", ".mov", ".mkv"], recursive=recursive
            )
            assert len(video_files) > 0, f"No video files found in {video_dir}."

            self.before_infer_video_dir(video_dir)
            total_videos = len(video_files)

            # 2. Parallel Execution
            # We use 'spawn' to avoid CUDA initialization errors in forked processes
            try:
                mp.set_start_method("spawn", force=True)
            except RuntimeError:
                pass

            print(f"Starting parallel inference with {max_workers} workers...")

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for i, video_path in enumerate(video_files):
                    # We submit the task to the pool
                    future = executor.submit(
                        self._infer_video_worker, video_path, i, total_videos
                    )
                    futures.append(future)

                # Wait for all videos to finish and handle errors
                for future in as_completed(futures):
                    try:
                        worker_profiler_data = future.result()
                        if self.profiler and worker_profiler_data:
                            self.profiler.merge_data(worker_profiler_data)
                    except Exception as e:
                        print(f"Worker failed with error: {e}")

            self.after_infer_video_dir(video_dir)

    def _infer_video_worker(self, video_path: str, video_idx: int, total_videos: int):
        """
        Wrapper specifically for the worker process.
        Since 'self' is pickled and sent to the worker, self.model might be None
        initially in the new process.
        """
        # Ensure the model is loaded in THIS process
        self.prepare_model()

        # Call the original logic
        # ! remember to return profiler data if any
        return self.infer_video(video_path, video_idx, total_videos)

    def infer_video(self, video_path: str, video_idx: int, total_videos: int):
        """
        Processes each frame of a single video, performing inference and delegating
        the results to the registered handlers.
        """
        progress_str = (
            ""
            if (video_idx is None or total_videos is None)
            else f"[{video_idx + 1}/{total_videos}]"
        )
        pprint(f"{progress_str} Starting inference for: {video_path}")
        # ! Hook: Call the hook method before video inference
        self.before_infer_video(video_path)

        self.prepare_model()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error: Could not open video: {video_path}")

        # Get video properties to pass to handlers
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vfps = cap.get(cv2.CAP_PROP_FPS)
        frame_size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

        # Notify all handlers that a new video is starting
        for handler in self.result_handlers:
            handler.before_video(
                video_path,
                outdir=self.outdir,
                total_frames=total_frames,
                fps=vfps,
                frame_size=frame_size,
                skip_if_exists=self.cfg.inferCfg.skip_if_exists,
            )

        # find CsvRSHandler in self.result_handlers
        csv_handler = None
        for handler in self.result_handlers:
            if isinstance(handler, CsvRsProc):
                csv_handler = handler
                break
        assert csv_handler is not None, (
            "CsvRsProc not found in result_handlers, it is required."
        )
        SKIP_INFER = csv_handler.outfile_exists
        frame_idx = 0
        limit = self.cfg.inferCfg.limit if self.cfg.inferCfg.limit > 0 else total_frames
        if not SKIP_INFER:
            try:
                while cap.isOpened():
                    ret, frame_bgr = cap.read()
                    if not ret:
                        break  # End of video
                    frame_idx += 1
                    if limit > 0 and frame_idx > limit:
                        pprint(f"Frame limit reached: {limit}, stop")
                        break
                    self._log_progress(frame_idx, total_frames)

                    start_time = time.perf_counter()
                    infer_rs = self.infer_frame(frame_bgr, frame_idx)
                    if not all(key in infer_rs for key in BaseMethod.REQUIRED_INFER_RS):
                        raise ValueError(
                            f"Missing required inference results: {BaseMethod.REQUIRED_INFER_RS}"
                        )

                    elapsed_time = time.perf_counter() - start_time

                    # infer fps
                    fps = 1.0 / elapsed_time if elapsed_time > 0 else 0
                    fps = f"{fps:.2f}"

                    frame_rs_dict = {
                        "method": self.cfg.methodCfg.name,
                        "video": os.path.basename(video_path),
                        "num_frames": total_frames,
                        "frame_idx": frame_idx,
                        "elapsed_time": elapsed_time,
                        "infer_rs": infer_rs,
                        "vfps": vfps,
                        "frame_size": frame_size,
                        "fps": fps,
                    }
                    # --- Delegate the packet to all registered handlers ---
                    for handler in self.result_handlers:
                        handler.handle_frame_results(frame_bgr, frame_rs_dict)

            finally:
                cap.release()
                # Notify all handlers that the video processing is complete
        #! even if SKIP_INFER, we still need to call after_video to do some clean up
        for handler in self.result_handlers:
            handler.after_video(video_path=video_path)
        print(f"\nFinished inference for: {video_path}\n")

        # ! Hook: Call the hook method after video inference
        self.after_infer_video(video_path=video_path)

        print(f"\nFinished inference for: {video_path}\n")
        if self.num_infer_workers > 1:
            # Return profiler data for aggregation
            if self.profiler and self.profiler.enabled:
                return self.profiler.time_dict
        return {}
