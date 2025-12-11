import cv2
import time
import timm
import torch
from abc import ABC, abstractmethod

from halib import *

from temporal.archived.config_bk import Config
from halib.utils.gpu_mon import GPUMonitor


CSV_FIXED_COLUMNS = [
    "video",  # video name
    "num_frames",  # total number of frames in the video
    "frame_idx",  # frame index
    "do_infer",  # whether to perform inference on this frame
    "elapsed_time",  # infer elapsed time in seconds
]


# Abstract Base Class
class DetectorBase(ABC):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.csv_writer = None
        self.video_writer = None
        self.model = None
        self.gpu_monitor = None
        self.outdir = os.path.abspath(cfg.get_output_dir())
        # pprint(f"Output directory: {self.outdir}")
        # create output directory if it does not exist
        os.makedirs(self.outdir, exist_ok=True)
        self.cfg_out_file = os.path.join(self.outdir, "__config.yaml")
        self.cfg.to_yaml_file(
            self.cfg_out_file
        )  # save the config to the output directory
        if self.cfg.infer.save_results:
            self.dfmk = csvfile.DFCreator()
            self.table_name = None
            self.csv_rows = []
            self.out_csv_file = None
            if self.cfg.infer.gpu_monitor:
                # Initialize GPU monitor if configured
                self.gpu_monitor = GPUMonitor(
                    gpu_index=0,  # Default to the first GPU
                    interval=0.01,  # 10ms sampling
                )
            self.calc_perf_metrics = self.cfg.infer.calc_perf_metrics

    @abstractmethod
    def infer_results_to_list(self, infer_results):
        pass

    # ! should be overridden by the subclass if needed
    def custom_load_model(self):
        """Custom method to load the model if needed."""
        return timm.create_model(
            self.cfg.model.base_timm_model,
            pretrained=False,
            num_classes=len(self.cfg.model.class_names),
            checkpoint_path=self.cfg.model.model_path,
        )

    def load_model(self):
        if self.model is None:
            self.model = self.custom_load_model()
            self.model.eval()
            # convert to suitable device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(device)
        return self.model

    @abstractmethod
    def infer_frame(self, frame, frame_idx):
        # handle detection for a single frame
        # ! return shouldDoInfer: true, infer_result_dict: dict|None, vis_data_results: dict|None
        pass

    @abstractmethod
    def annotate_frame(
        self,
        frame,
        frame_idx,
        total_frames,
        infer_results=None,
        vis_data_results=None,
        gpu_stats=None,
    ):
        """Annotate the frame with detection results."""
        pass

    def after_detect_frame(
        self,
        frame,
        frame_idx,
        total_frames,
        infer_results,
        vis_data_results,
        gpu_stats=None,
    ):
        """Handle the results after detecting a frame."""
        pass

    # ! can be override by the subclass or just declear it in 'infer'of the config
    def csv_columns(
        self,
    ):  # appart from video and frame_idx, what columns to write to csv
        """Return the columns to be written to the CSV file."""
        fixed_column = (
            CSV_FIXED_COLUMNS + GPU_MONITOR_COLUMNS
            if self.gpu_monitor
            else CSV_FIXED_COLUMNS
        )
        return fixed_column + self.cfg.infer.csv_columns

    def _init_csv_writer(self, video_path):
        if self.cfg.infer.save_results:
            # console.rule()
            # pprint(video_path)
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            columns = self.csv_columns()
            self.dfmk.create_table(video_name, columns=columns)
            self.table_name = video_name
            self.out_csv_file = os.path.join(self.outdir, f"{video_name}_results.csv")
            # pprint(f"Output CSV file: {self.out_csv_file}")

    def _init_video_writer(self, video_path):
        fname = fs.get_file_name(video_path, split_file_ext=True)[0]
        # base info
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Error: Could not open video.")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        cap.release()
        if self.cfg.infer.save_out_video:
            out_video_path = os.path.join(self.outdir, f"{fname}_out.mp4")
            self.video_output_path = out_video_path
            self.video_writer = cv2.VideoWriter(out_video_path, fourcc, fps, frame_size)

    def handle_infer_results(
        self,
        video,
        frame_idx,
        total_frames,
        do_infer,
        infer_results,
        elapsed_time,
        gpu_stats=None,
    ):
        if self.cfg.infer.save_results:
            basic = [video, total_frames, frame_idx, do_infer, elapsed_time]
            if gpu_stats:
                gpu_avg_power = gpu_stats.get("gpu_avg_power", 0)
                gpu_avg_max_memory = gpu_stats.get("gpu_avg_max_memory", 0)
                basic += [gpu_avg_power, gpu_avg_max_memory]
            self.row = basic + self.infer_results_to_list(infer_results)
            self.csv_rows.append(self.row)
            # num_frame = self.cfg.infer.limit if self.cfg.infer.limit > 0 else total_frames

    def before_video_infer(self, video_path):
        """Perform any necessary setup before inference starts."""
        if hasattr(self, "csv_rows"):
            self.csv_rows.clear()  # clear previous rows if any

    def after_video_infer(self, video_path):
        """Perform any necessary cleanup after inference ends."""
        console.print(f"Finished infer: {video_path}")
        if (
            self.cfg.infer.save_results
            and hasattr(self, "dfmk")
            and self.dfmk is not None
        ):
            # write to csv file
            self.dfmk.insert_rows(self.table_name, self.csv_rows)
            self.dfmk.fill_table_from_row_pool(self.table_name)
            self.dfmk[self.table_name].to_csv(
                self.out_csv_file, index=False, sep=";", encoding="utf-8"
            )
            with ConsoleLog("Infer results"):
                pprint_local_path(self.out_csv_file)

        if hasattr(self, "video_writer") and self.video_writer is not None:
            self.video_writer.release()  # free the video writer resource
            with ConsoleLog("Video output"):
                pprint_local_path(self.video_output_path)

    def handle_vis_data_results(
        self,
        frame,
        frame_idx,
        total_frames,
        infer_results=None,
        vis_data_results=None,
        gpu_stats=None,
    ):
        frame_bgr = self.annotate_frame(
            frame=frame,
            frame_idx=frame_idx,
            total_frames=total_frames,
            infer_results=infer_results,
            vis_data_results=vis_data_results,
            gpu_stats=gpu_stats,
        )
        if hasattr(self, "video_writer") and self.video_writer is not None:
            # Write the annotated frame to the video output
            self.video_writer.write(frame_bgr)

    def _log_progress(self, frame_idx, total_frames, do_infer=True):
        """Log processing progress to console."""
        percentage = (frame_idx / total_frames) * 100
        prefix = "Infer" if do_infer else "[red]Skip[/red]"
        console.print(
            f"{prefix} frame {frame_idx}/{total_frames} ({percentage:.2f}%)...",
            end="\r",
            highlight=False,
        )

    def infer_video_dir(self, video_dir, recursive=True):
        """Process all videos in a directory."""
        assert os.path.exists(video_dir), f"Video directory {video_dir} does not exist."

        video_files = fs.filter_files_by_extension(
            video_dir, [".mp4", ".avi", ".mov"], recursive=recursive
        )
        assert len(video_files) > 0, f"No video files found in {video_dir}."
        num_videos = len(video_files)
        for video_idx, video_path in enumerate(video_files):
            # self._init_csv_writer(video_path)
            self.infer_video(video_path, video_idx=video_idx, total_videos=num_videos)

    def infer_video(self, video_path, video_idx=None, total_videos=None, verbose=True):
        """Process each frame of the video."""
        # create a txt file if it does not exist
        self._init_csv_writer(video_path)
        if self.cfg.infer.skip_if_exists:
            if os.path.exists(self.out_csv_file):
                console.print(
                    f"[yellow]<results existed> Skip infer: {video_path}[/yellow]"
                )
                return
        if verbose:
            progress_str = (
                ""
                if (video_idx is None or total_videos is None)
                else f" [{video_idx + 1}/{total_videos}]"
            )
            pprint(f"{progress_str} infer: {video_path}")

        self.load_model()
        # 2: read the video
        assert os.path.exists(video_path), f"Video file {video_path} does not exist."
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Error: Could not open video.")
        frame_idx = 0
        num_limit_frame = self.cfg.infer.limit
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if num_limit_frame > total_frames:
            num_limit_frame = total_frames
        self._init_video_writer(video_path)
        self.before_video_infer(video_path)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # End of video
                frame_idx += 1
                if num_limit_frame > 0 and frame_idx > num_limit_frame:
                    console.print(
                        f"Reached frame limit: {num_limit_frame}. Stopping inference."
                    )
                    time.sleep(2)  # pause a bit to let the user see the message
                    break
                start = time.perf_counter()
                if self.gpu_monitor:
                    self.gpu_monitor.start()
                do_infer, infer_results, vis_data_results = self.infer_frame(frame)
                elapsed_time = time.perf_counter() - start
                gpu_stats = None
                if self.gpu_monitor:
                    self.gpu_monitor.stop()
                    gpu_stats = self.gpu_monitor.get_stats()
                self._log_progress(frame_idx, total_frames, do_infer=do_infer)
                self.handle_infer_results(
                    video_path,
                    frame_idx,
                    total_frames,
                    do_infer,
                    infer_results,
                    elapsed_time,
                    gpu_stats,
                )
                self.handle_vis_data_results(
                    frame=frame,
                    frame_idx=frame_idx,
                    total_frames=total_frames,
                    infer_results=infer_results,
                    vis_data_results=vis_data_results,
                    gpu_stats=gpu_stats,
                )
                # ! do some extra task (like mask visualization) after detection
                self.after_detect_frame(
                    frame=frame,
                    frame_idx=frame_idx,
                    total_frames=total_frames,
                    infer_results=infer_results,
                    vis_data_results=vis_data_results,
                    gpu_stats=gpu_stats,
                )

        except Exception as e:
            console.print_exception()
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Inference completed.")
        self.after_video_infer(video_path)
