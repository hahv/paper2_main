from halib import *
from typing import List
import os
from src.config import Config
from src.results.base_rs_proc import BaseRsProc

# Import your new classes
from src.results.viz.video_pipeline import VideoPipeline
from src.results.viz.infer_rs_renderer import InferRsRenderer


class VideoInferRsProc(BaseRsProc):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.outdir = os.path.abspath(cfg.get_outdir())
        self.pipelines: List[VideoPipeline] = []
        self.outfile_exists = False
        self.video_output_path: str = None
        self.out_video_postfix = "out"

    def prepare_pipelines_list(self, video_path: str, fps: float, frame_size: tuple):
        pipeline_ls = []
        infer_rs_pipe = VideoPipeline(self.video_output_path, fps, frame_size)
        infer_rs_pipe.add_renderer(InferRsRenderer())  # ty:ignore[invalid-argument-type]
        pipeline_ls.append(infer_rs_pipe)
        return pipeline_ls

    def update_video_output_path(self, video_path: str):
        fname = fs.get_file_name(video_path, split_file_ext=True)[0]
        self.video_output_path = os.path.join(
            self.outdir, f"{fname}_{self.out_video_postfix}.mp4"
        )

    # ! Normal cases: Frame size of out video = input frame size
    def calc_frame_size(self, video_path: str, **kwargs) -> tuple:
        return kwargs["frame_size"]

    def before_video(self, video_path: str, **kwargs):
        # if video_path is not a video (e.g., image folder), skip video writer creation
        if fs.is_dir(video_path):
            return
        if not self.cfg.inferCfg.save_csv_results:
            return
        self.update_video_output_path(video_path)
        skip_if_exists = self.cfg.inferCfg.skip_if_exists
        if skip_if_exists and os.path.exists(self.video_output_path):
            self.outfile_exists = True
            console.print(
                f"[red]Video file already exists, skipping: {self.video_output_path}[/red]"
            )
            return  # skip creating dfmk and table

        fps = kwargs["fps"]
        frame_size = self.calc_frame_size(video_path, **kwargs)
        pipeline_ls = self.prepare_pipelines_list(video_path, fps, frame_size)
        self.pipelines.extend(pipeline_ls)
        assert len(self.pipelines) > 0, "No video pipelines were created."

    def get_vis_frame(self, frame_bgr, frame_rs_dict: dict):
        return frame_bgr

    # ! abstract method implementation
    def handle_frame_results(self, frame_bgr, frame_rs_dict: dict):
        vis_frame = self.get_vis_frame(frame_bgr, frame_rs_dict)
        for pipe in self.pipelines:
            assert pipe.frame_size == (
                vis_frame.shape[1],
                vis_frame.shape[0],
            ), (
                f"Frame size mismatch: pipeline {pipe.frame_size} vs vis_frame {vis_frame.shape[1], vis_frame.shape[0]}"
            )
            pipe.process_and_write(vis_frame, frame_rs_dict)

    def after_video(self, video_path: str, **kwargs):
        if fs.is_dir(video_path):  # a image folder, skip
            return
        if self.outfile_exists:
            self.outfile_exists = False  # reset for next video
            return
        for pipe in self.pipelines:
            pipe.release()
        self.pipelines = []
