from halib import *
from src.config import Config
from src.methods.skip.block_skip_proc import BlockSkipProc

# Import your new classes
from src.results.viz.video_pipeline import VideoPipeline
from src.results.viz.infer_rs_renderer import InferRsRenderer
from src.results.viz.grid_renderer import GridRenderer
from src.results.viz.block_rule_renderer import BlockRuleRenderer
from src.results.video_infer_rs_proc import VideoInferRsProc


class VideoBlockSkipRsProc(VideoInferRsProc):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.out_video_postfix = "out"

    # ! use Skip Proc: so frame size may differ (due to resize + pad in skip proc with respect to block size and scale_factor)
    def calc_frame_size(self, video_path: str, **kwargs) -> tuple:
        original_fsize = kwargs.get("frame_size")
        method_cfg = self.cfg.methodCfg.extra_cfgs.get("skip_proc").get("params")  # ty:ignore[possibly-missing-attribute]
        scale_factor = method_cfg.get("scale_factor")
        block_size = method_cfg.get("block_size")

        return BlockSkipProc.get_skip_proc_frame_size(
            frame_w_h=original_fsize,  # ty:ignore[invalid-argument-type]
            scale_factor=scale_factor,
            block_size=block_size,
        )
    # ! need to override to get correct frame to visualize
    def get_vis_frame(self, frame_bgr, frame_rs_dict: dict):
        mt_proc = frame_rs_dict["infer_rs"]["mt_proc"]
        vis_frame = mt_proc.get("vis_frame")
        return vis_frame

    def prepare_pipelines_list(self, video_path: str, fps: float, frame_size: tuple):
        pipeline_ls = []
        infer_rs_pipe = VideoPipeline(self.video_output_path, fps, frame_size)
        infer_rs_pipe.add_renderer(InferRsRenderer())  # ty:ignore[invalid-argument-type]
        infer_rs_pipe.add_renderer(GridRenderer())  # ty:ignore[invalid-argument-type]
        infer_rs_pipe.add_renderer(BlockRuleRenderer())  # ty:ignore[invalid-argument-type]
        pipeline_ls.append(infer_rs_pipe)
        return pipeline_ls
