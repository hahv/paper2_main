from abc import ABC, abstractmethod
import os
import cv2
from temporal.config import *
from halib.system import filesys as fs
from halib.filetype import csvfile
import torch
import torch.nn.functional as F
from collections import OrderedDict
from temporal.rs_hdl.rs_video_base import *

class FGMaskRSHandler(BaseVideoRSHandler):
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
        super().__init__(cfg)
        self.fg_mask_video_writer = None
        self.fg_mask_video_output_path = None

    def before_video(self, video_path: str, **kwargs):
        super().before_video(video_path, **kwargs)
        fname = fs.get_file_name(video_path, split_file_ext=True)[0]
        self.fg_mask_video_output_path = os.path.join(self.outdir, f"{fname}_fg_mask.mp4")
        self.fg_mask_video_writer = None
        self.fps = kwargs["fps"]

    def annotate_fg_mask(self, fg_mask_dict, vis_frame=None):
        """Annotate the foreground mask for visualization."""
        if vis_frame is None:
            vis_frame = fg_mask_dict["fg_mask"]
        assert vis_frame is not None, "Either fg_mask or vis_frame must be provided."
        block_size = fg_mask_dict["block_size"]
        active_blocks = fg_mask_dict.get("active_motion_blocks_info", [])
        firesmoke_info = fg_mask_dict.get("firesmoke_blocks_cls_info", {})
        roi_rect = fg_mask_dict.get("ROI_rect", None)

        # pprint(locals())

        H, W = vis_frame.shape[:2]
        if len(vis_frame.shape) == 2 or vis_frame.shape[2] == 1:
            # grayscale → convert to BGR
            vis = cv2.cvtColor(vis_frame, cv2.COLOR_GRAY2BGR)
        else:
            vis = vis_frame.copy()

        # --- 1. Draw grid lines ---
        for y in range(0, H, block_size):
            cv2.line(vis, (0, y), (W, y), (50, 50, 50), 1)
        for x in range(0, W, block_size):
            cv2.line(vis, (x, 0), (x, H), (50, 50, 50), 1)

        # --- 2. Mark active motion blocks ---
        for idx, percent in active_blocks:
            by = idx // (W // block_size)
            bx = idx % (W // block_size)
            x0, y0 = bx * block_size, by * block_size
            x1, y1 = x0 + block_size, y0 + block_size
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 2)  # yellow
            cv2.putText(
                vis,
                f"{percent:.1f}%",
                (x0 + 2, y0 + 16),  # shift a bit lower for larger text
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,  # bigger (default ~0.4)
                color=(0, 255, 255),
                thickness=2,  # bolder (default=1)
                lineType=cv2.LINE_AA,  # anti-aliased, smoother text
            )

        # --- 3. Overlay fire/smoke classification info ---
        all_active = firesmoke_info.get("all_active_blocks", [])
        firesmoke_active = set(firesmoke_info.get("firesmoke_active_blocks", []))

        for idx, prob in all_active:
            by = idx // (W // block_size)
            bx = idx % (W // block_size)
            x0, y0 = bx * block_size, by * block_size
            x1, y1 = x0 + block_size, y0 + block_size
            color = (
                (0, 0, 255) if idx in firesmoke_active else (0, 255, 0)
            )  # red for fire/smoke, green otherwise
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
            cv2.putText(
                vis,
                f"{prob:.2f}",
                (x0 + 2, y1 - 6),  # slight shift for bigger text
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,  # increase size
                color=color,
                thickness=2,  # make it bold
                lineType=cv2.LINE_AA,  # smooth edges
            )

        # --- 4. Draw final ROI if available ---
        if roi_rect is not None:
            x0, y0, x1, y1 = roi_rect
            cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 0, 0), 2)  # blue

        return vis

    def handle_frame_results(self, frame_bgr, frame_rs_dict: dict):
        # write to normal RGB video
        infer_rs = frame_rs_dict["infer_rs"]
        assert (
            "fg_mask_dict" in infer_rs
        ), "Foreground mask 'fg_mask_dict' not found in inference results"
        fg_mask_dict = infer_rs["fg_mask_dict"]
        # 1. write mask info into rgb video
        extra_info = {
            "active_block/frames": f"{fg_mask_dict.get('active_percent', 0.0)*100:.1f}%",
        }
        frame_vis, lb_val_dict = super().prepare_frame_vwriter(frame_bgr, frame_rs_dict, extra_info=extra_info)

        frame_vis = self.annotate_fg_mask(
            fg_mask_dict=fg_mask_dict, vis_frame=frame_vis
        )
        self.video_writer.write(frame_vis)
        # 2. do the anotation and write fg mask video
        fg_mask = fg_mask_dict["fg_mask"]
        if self.fg_mask_video_writer is None:
            # initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = self.fps
            frame_size = (fg_mask.shape[1], fg_mask.shape[0])
            self.fg_mask_video_writer = cv2.VideoWriter(
                    self.fg_mask_video_output_path, fourcc, self.fps, frame_size
                )
        # visualize fg mask (with block info: which block activated, which block classified as fire/smoke, ROI box, etc.)
        fg_mask_vis = self.annotate_fg_mask(fg_mask_dict=fg_mask_dict)
        # add annotations to fg mask video
        fg_mask_vis = BaseVideoRSHandler.annotate_frame(fg_mask_vis, lb_val_dict)
        self.fg_mask_video_writer.write(fg_mask_vis)

    def after_video(self):
        super().after_video()
        if self.fg_mask_video_writer is not None:
            self.fg_mask_video_writer.release()
            print(f"Foreground mask video saved to: {self.fg_mask_video_output_path}")
            self.fg_mask_video_writer = None
