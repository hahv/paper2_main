import cv2
import time
import timm
import torch
from abc import ABC, abstractmethod

from halib import *

from temporal.config import Config
from temporal.rs_handler import *

from halib import *
from halib.common import seed_everything
from halib.research.base_exp import BaseExperiment
from halib.research.metrics import MetricsBackend

class EvalBase(ABC):
    """
    An abstract base class for video inference that decouples inference logic
    from output handling (e.g., saving CSVs or videos) via a handler system.
    """
    REQUIRED_INFER_RS = ["logits", "probs", "labelIdx", "predLabel"]

    def __init__(self, cfg: Config, rs_handlers: list[RSHandlerBase] = None):
        """
        Initializes the detector.

        Args:
            cfg (Config): The configuration object.
            result_handlers (list[ResultHandlerBase], optional): A list of handlers to process the inference results. Defaults to None.
        """
        self.cfg = cfg
        self.model = None
        self.gpu_monitor = None
        self.outdir = os.path.abspath(cfg.get_output_dir())
        os.makedirs(self.outdir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Store the list of handlers that will process the results
        self.result_handlers = rs_handlers if rs_handlers is not None else []

        # Save the configuration file to the output directory for reproducibility
        self.cfg_out_file = os.path.join(self.outdir, "__config.yaml")
        self.cfg.to_yaml_file(self.cfg_out_file)

    # --------------------------------------------------------------------------
    # Abstract Methods - To be implemented by subclasses
    # --------------------------------------------------------------------------

    # @abstractmethod
    # def infer_results_to_list(self, infer_results: dict) -> list:
    #     """Converts raw model output dictionary to a list for CSV logging."""
    #     pass

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

    @abstractmethod
    def annotate_frame(
        self,
        frame,
        frame_idx: int,
        total_frames: int,
        infer_results: dict = None,
        vis_data_results: dict = None,
        gpu_stats: dict = None,
    ) -> any:
        """Annotates the frame with detection results and returns the annotated frame."""
        pass

    # --------------------------------------------------------------------------
    # Core Methods
    # --------------------------------------------------------------------------

    def load_model(self):
        """Custom method to load the model, can be overridden if needed."""
        return timm.create_model(
            self.cfg.model_cfg.base_model,
            pretrained=False,
            num_classes=len(self.cfg.model_cfg.class_names),
            checkpoint_path=self.cfg.model_cfg.model_path,
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

    def infer_video_dir(self, video_dir: str, recursive: bool = True):
        """Processes all videos in a specified directory."""
        assert os.path.exists(video_dir), f"Video directory {video_dir} does not exist."
        video_files = fs.filter_files_by_extension(
            video_dir, [".mp4", ".avi", ".mov", ".mkv"], recursive=recursive
        )
        assert len(video_files) > 0, f"No video files found in {video_dir}."

        for i, video_path in enumerate(video_files):
            self.infer_video(video_path, video_idx=i, total_videos=len(video_files))

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
            )

        frame_idx = 0
        try:
            while cap.isOpened():
                ret, frame_bgr = cap.read()
                if not ret:
                    break  # End of video
                frame_idx += 1

                start_time = time.perf_counter()

                infer_rs = self.infer_frame(
                    frame_bgr, frame_idx
                )
                if not all(key in infer_rs for key in EvalBase.REQUIRED_INFER_RS):
                    raise ValueError(
                        f"Missing required inference results: {EvalBase.REQUIRED_INFER_RS}"
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
                    "fps": fps
                }
                # --- Delegate the packet to all registered handlers ---
                for handler in self.result_handlers:
                    handler.handle_frame_results(frame_bgr, frame_rs_dict)
                self._log_progress(frame_idx, total_frames)

        finally:
            cap.release()
            cv2.destroyAllWindows()
            # Notify all handlers that the video processing is complete
            for handler in self.result_handlers:
                handler.after_video()
            print(f"\nFinished inference for: {video_path}\n")
