from halib import *

import cv2
import time
import timm
import torch
from abc import ABC, abstractmethod
from halib.exp.perf.profiler import zProfiler

from src.results import *
from src.config import Config
from src.utils import get_cls
from src.metrics.base_metric_src import *
from src.results.base_rs_proc import BaseRSProc
from src.results.csv_proc import CsvRSProc
import sys

class MethodFactory:
    @staticmethod
    def create_method(config: Config, *args, **kwargs):
        def method_name_to_cls_name(name: str, suffix: str = "Method") -> str:
            """
            Convert snake_case string to PascalCase and append suffix.
            Example: "no_temp" -> "NoTempMethod"
            """
            parts = name.split("_")[:-1] # remove the 'mt' postfix
            # Capitalize the first letter fo each part, but keep the rest as is
            for i in range(len(parts)):
                word = parts[i]
                word = word[0].upper() + word[1:]
                parts[i] = word
            pascal = "".join(parts)
            return pascal + suffix

        pkg_name = "src.methods"
        # ! method_name == module_name
        method_postfix = "mt"
        module_name = f"{config.methodCfg.name}_{method_postfix}"
        cls_name = method_name_to_cls_name(module_name)
        pprint(f'Creating method class: {pkg_name}.{module_name}.{cls_name}')
        cls = get_cls(f"{pkg_name}.{module_name}.{cls_name}")
        assert cls is not None, f"Class '{cls_name}' not found in module '{pkg_name}'."

        rs_handler_list: list[BaseRSProc] = []
        if config.inferCfg.save_csv_results:
            rs_handler_list.append(CsvRSProc(config))
        if config.inferCfg.save_video_results:
            pkg_name = "src.results"
            chosen_video_handler = config.methodCfg.extra_cfgs.get(
                "video_rs_proc", "VideoRSProc"
            )
            rs_handler_list.append(
                get_cls(f"{pkg_name}.{chosen_video_handler}")(cfg=config)
            )

        kwargs = {"cfg": config, "rs_handlers": rs_handler_list}
        return cls(**kwargs)


class BaseMethod(ABC):
    """
    An abstract base class for video inference that decouples inference logic
    from output handling (e.g., saving CSVs or videos) via a handler system.
    """

    REQUIRED_INFER_RS = ["logits", "probs", "predLabelIdx", "predLabel"]

    def __init__(self, cfg: Config, rs_handlers: list[BaseRSProc] = None):
        """
        Initializes the detector.

        Args:
            cfg (Config): The configuration object.
            result_handlers (list[ResultHandlerBase], optional): A list of handlers to process the inference results. Defaults to None.
        """
        self.cfg = cfg
        self.model = None
        self.gpu_monitor = None
        self.outdir = os.path.abspath(cfg.get_outdir())
        os.makedirs(self.outdir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.profiler = zProfiler()

        # Store the list of handlers that will process the results
        self.result_handlers = rs_handlers if rs_handlers is not None else []

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
        pass

    def before_infer_video(self, video_path: str):
        """Hook method called before starting inference on a video."""
        pass

    def after_infer_video(self, video_path: str):
        """Hook method called after completing inference on a video."""
        pass

    # !------End HOOKS------------------------------------------------------

    def infer_video_dir(self, video_dir: str, recursive: bool = True):
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

    def infer_video(
        self, video_path: str, video_idx: int = None, total_videos: int = None
    ):
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
            if isinstance(handler, CsvRSProc):
                csv_handler = handler
                break
        assert csv_handler is not None, (
            "CsvRSProc not found in result_handlers, it is required."
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
