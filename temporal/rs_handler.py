from abc import ABC, abstractmethod
import os
import cv2
from temporal.config import *
from halib.system import filesys as fs
from halib.filetype import csvfile
import torch
import torch.nn.functional as F

class RSHandlerBase(ABC):
    """Abstract base class for handling inference results."""

    def before_video(
        self,
        video_path: str,
        **kwargs,
    ):
        """Called once before processing a video."""
        pass

    @abstractmethod
    def handle_frame_results(self, frame_bgr, frame_rs_dict: dict):
        """Called for each frame to handle its results."""
        pass

    def after_video(self):
        """Called once after processing a video is complete."""
        pass


class CsvRSHandler(RSHandlerBase):
    CSV_FIXED_COLUMNS = [
        "video",  # video name
        "num_frames",  # total number of frames in the video
        "frame_idx",  # frame index
        "do_infer",  # whether to perform inference on this frame
        "elapsed_time",  # infer elapsed time in seconds
    ]
    def infer_results_to_list(self, frame_rs_dict):
        # csv_infer_cols = [<class_names>, <logits>, <probs>, <pred_label_idx>, <pred_label>]
        infer_row_data = []
        className = self.cfg.model_cfg.class_names
        logits = frame_rs_dict["logits"] if frame_rs_dict else None
        probs = (
            torch.nn.functional.softmax(torch.tensor(logits), dim=0).numpy().tolist()
            if logits
            else None
        )
        pred_labelIdx = frame_rs_dict["pred_labelIdx"] if frame_rs_dict else None
        pred_label = (
            className[pred_labelIdx]
            if className and pred_labelIdx is not None
            else None
        )
        infer_row_data.append(className)
        infer_row_data.append(logits)
        infer_row_data.append(probs)
        infer_row_data.append(pred_labelIdx)
        infer_row_data.append(pred_label)
        return infer_row_data

    def __init__(self, cfg: Config):
        self.cfg = cfg
        assert self.cfg.infer_cfg.save_csv_results, "CSV saving is disabled in the config"
        self.dfmk = None
        self.table_name = None
        self.csv_rows = []
        self.out_csv_file = None
        self.outdir = os.path.abspath(cfg.get_outdir())

    def before_video(self, video_path: str, **kwargs):
        if not self.cfg.infer_cfg.save_csv_results:
            return

        extra_cols = kwargs.get("extra_csv_columns", [])
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.dfmk = csvfile.DFCreator()
        columns = CsvRSHandler.CSV_FIXED_COLUMNS + extra_cols
        self.dfmk.create_table(video_name, columns=columns)
        self.table_name = video_name
        self.out_csv_file = os.path.join(self.outdir, f"{video_name}_results.csv")
        self.csv_rows = []

    def handle_frame_results(self, frame_bgr, frame_rs_dict: dict):
        # Unpack data from the dictionary
        assert (
            "csv_row" in frame_rs_dict
        ), "Missing 'csv_row' in frame_data for CsvRSHandler"
        row_data = frame_rs_dict["csv_row"]
        self.csv_rows.append(row_data)

    def after_video(self):
        if not self.cfg.infer_cfg.save_csv_results or self.dfmk is None:
            return

        self.dfmk.insert_rows(self.table_name, self.csv_rows)
        self.dfmk.fill_table_from_row_pool(self.table_name)
        self.dfmk[self.table_name].to_csv(
            self.out_csv_file, index=False, sep=";", encoding="utf-8"
        )
        with ConsoleLog("Results saved to:"):
            pprint_local_path(self.out_csv_file)


class BaseVideoRSHandler(RSHandlerBase):
    @staticmethod
    def getColor(classIdx):
        # main palette
        palette = sns.color_palette(palette="gist_rainbow")
        color = palette[classIdx % len(palette)]
        # Convert to 255 scale
        r, g, b = color
        color_255 = (int(r * 255), int(g * 255), int(b * 255))
        return color_255

    @staticmethod
    def bgr_to_rgb(bgr):
        """Convert BGR color to RGB."""
        return (bgr[2], bgr[1], bgr[0])

    @staticmethod
    def annotate_frame(
        frame_bgr,  # ! make sure that this frame is in BGR format
        label_value_dict,  # {"label": value} pairs, e.g., {"FrameIdx": "1/100", etc.}
        vis_data_results=None,
    ):
        """Annotate frame with class names and probabilities; debug temporal stabilization info if available."""

        num_lines = len(label_value_dict)
        box_height = 30 + num_lines * 30
        overlay = frame_bgr.copy()

        # Draw semi-transparent black rectangle
        cv2.rectangle(overlay, (0, 0), (300, box_height), (0, 0, 0), thickness=-1)
        frame_bgr = cv2.addWeighted(overlay, 0.5, frame_bgr, 0.5, 0)
        # Add text annotations
        for i, (label, value) in enumerate(label_value_dict.items()):
            color = BaseVideoRSHandler.bgr_to_rgb(BaseVideoRSHandler.getColor(i))
            text = (
                f"{label}: {value:.2f}"
                if isinstance(value, float)
                else f"{label}: {value}"
            )
            cv2.putText(
                frame_bgr,
                text,
                (10, 30 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
        return frame_bgr

    def __init__(self, cfg: Config):
        self.cfg = cfg
        assert self.cfg.infer_cfg.save_video_results, "Video saving is disabled in the config"
        self.video_writer = None
        self.video_output_path = None
        self.outdir = os.path.abspath(cfg.get_output_dir())

    def before_video(self, video_path: str, **kwargs):
        fname = fs.get_file_name(video_path, split_file_ext=True)[0]
        self.video_output_path = os.path.join(self.outdir, f"{fname}_out.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        fps = kwargs['fps']
        frame_size = kwargs['frame_size']
        self.video_writer = cv2.VideoWriter(
            self.video_output_path, fourcc, fps, frame_size
        )
    def handle_frame_results(self, frame_bgr, frame_rs_dict: dict):
        lb_val_dict = {}
        frame_idx = frame_rs_dict['frame_idx']
        total_frames = frame_rs_dict['num_frames']
        lb_val_dict['fps'] = frame_rs_dict['fps']
        lb_val_dict["frameidx"] = f"{frame_idx + 1}/{total_frames}"
        infer_rs = frame_rs_dict['infer_rs']
        probs = infer_rs['probs']
        labelIdx = infer_rs['predLabelIdx']
        predLabel = infer_rs['predLabel']
        pred_str = f"{predLabel} ({probs[labelIdx]*100:.1f}%)"
        lb_val_dict["pred---"] = pred_str
        frame_vis = BaseVideoRSHandler.annotate_frame(frame_bgr, lb_val_dict)
        self.video_writer.write(frame_vis)

    def after_video(self):
        if self.video_writer is not None:
            self.video_writer.release()
            print(f"Annotated video saved to: {self.video_output_path}")
