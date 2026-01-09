from halib import *
import cv2
from src.config import *
from src.results.base_rs_proc import *


class VideoBaseRsProc(BaseRsProc):
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
        label_value_dict,
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
            color = VideoBaseRsProc.bgr_to_rgb(VideoBaseRsProc.getColor(i))
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
        assert self.cfg.inferCfg.save_video_results, (
            "Video saving is disabled in the config"
        )
        self.video_writer = None
        self.video_output_path = None
        self.outdir = os.path.abspath(cfg.get_outdir())
        self.outfile_exists = False

    def before_video(self, video_path: str, **kwargs):
        # if video_path is not a video (e.g., image folder), skip video writer creation
        if fs.is_dir(video_path):
            return
        if not self.cfg.inferCfg.save_csv_results:
            return
        fname = fs.get_file_name(video_path, split_file_ext=True)[0]
        self.video_output_path = os.path.join(self.outdir, f"{fname}_out.mp4")
        skip_if_exists = self.cfg.inferCfg.skip_if_exists
        if skip_if_exists and os.path.exists(self.video_output_path):
            self.outfile_exists = True
            print(f"Video file already exists, skipping: {self.video_output_path}")
            return  # skip creating dfmk and table
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # ty:ignore[unresolved-attribute]

        fps = kwargs["fps"]
        frame_size = kwargs["frame_size"]
        self.video_writer = cv2.VideoWriter(
            self.video_output_path, fourcc, fps, frame_size
        )

    def prepare_frame_vwriter(self, frame_bgr, frame_rs_dict: dict, extra_info=None):
        """Prepares the frame for writing to video. Override if needed."""
        lb_val_dict = {}
        frame_idx = frame_rs_dict["frame_idx"]
        total_frames = frame_rs_dict["num_frames"]
        lb_val_dict["fps"] = frame_rs_dict["fps"]
        lb_val_dict["frame_idx"] = f"{frame_idx}/{total_frames}"
        infer_rs = frame_rs_dict["infer_rs"]
        probs = infer_rs["probs"]
        labelIdx = infer_rs["predLabelIdx"]
        predLabel = infer_rs["predLabel"]
        pred_str = f"{predLabel} ({probs[labelIdx] * 100:.1f}%)"
        lb_val_dict["pred---"] = pred_str
        if extra_info is not None:
            lb_val_dict.update(extra_info)
        frame_vis = VideoBaseRsProc.annotate_frame(frame_bgr, lb_val_dict)
        return frame_vis, lb_val_dict

    def handle_frame_results(self, frame_bgr, frame_rs_dict: dict):
        frame_vis, lb_val_dict = self.prepare_frame_vwriter(frame_bgr, frame_rs_dict)
        self.video_writer.write(frame_vis)  # ty:ignore[possibly-missing-attribute]
        return lb_val_dict

    def after_video(self, video_path: str, **kwargs):
        if fs.is_dir(video_path):  # a image folder, skip
            return
        if self.outfile_exists:
            self.outfile_exists = False  # reset for next video
            return
        if self.video_writer is not None:
            self.video_writer.release()
            print(f"Annotated video saved to: {self.video_output_path}")
