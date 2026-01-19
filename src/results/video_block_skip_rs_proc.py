from halib import *
from typing import List
import os
from src.config import Config
from src.results.base_rs_proc import BaseRsProc

# Import your new classes
from src.results.viz.video_pipeline import VideoPipeline
from src.results.viz.infer_rs_renderer import InferRsRenderer
from src.results.viz.grid_renderer import GridRenderer
from src.results.viz.block_rule_renderer import BlockRuleRenderer


class VideoBlockSkipRsProc(BaseRsProc):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.outdir = os.path.abspath(cfg.get_outdir())
        self.pipelines: List[VideoPipeline] = []
        self.outfile_exists = False
        self.video_output_path: str = None

    def prepare_pipelines_list(self, video_path: str, fps: float, frame_size: tuple):
        pipeline_ls = []
        infer_rs_pipe = VideoPipeline(self.video_output_path, fps, frame_size)
        infer_rs_pipe.add_renderer(InferRsRenderer())  # ty:ignore[invalid-argument-type]
        infer_rs_pipe.add_renderer(GridRenderer())  # ty:ignore[invalid-argument-type]
        infer_rs_pipe.add_renderer(BlockRuleRenderer())  # ty:ignore[invalid-argument-type]
        pipeline_ls.append(infer_rs_pipe)
        return pipeline_ls

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

        fps = kwargs["fps"]
        frame_size = kwargs["frame_size"]
        pipeline_ls = self.prepare_pipelines_list(video_path, fps, frame_size)
        self.pipelines.extend(pipeline_ls)
        assert len(self.pipelines) > 0, "No video pipelines were created."

    # ! abstract method implementation
    def handle_frame_results(self, frame_bgr, frame_rs_dict: dict):
        # pprint("Writing frame to video pipelines")
        for pipe in self.pipelines:
            pipe.process_and_write(frame_bgr, frame_rs_dict)

    def after_video(self, video_path: str, **kwargs):
        if fs.is_dir(video_path):  # a image folder, skip
            return
        if self.outfile_exists:
            self.outfile_exists = False  # reset for next video
            return
        for pipe in self.pipelines:
            pipe.release()
        self.pipelines = []
