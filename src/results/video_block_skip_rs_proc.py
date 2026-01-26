from halib import *
from src.config import Config
from src.methods.skip.base_block_skip_proc import BaseBlockSkipProc
from typing import List, Dict, Any

# Import your new classes
from src.results.viz.video_pipeline import VideoPipeline
from src.results.viz.base_renderer import BaseRenderer
from src.results.viz.infer_rs_renderer import InferRsRenderer
from src.results.viz.grid_renderer import GridRenderer
from src.results.video_infer_rs_proc import VideoInferRsProc
from src.results.viz.block_motion_only_renderer import BlockMontionOnlyRenderer
from src.results.viz.block_rule_based_renderer import BlockRuleBasedRenderer


class VideoBlockSkipRsProc(VideoInferRsProc):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.out_video_postfix = "out"
        self.renderer_context = "original_frame"

    # ! Normal cases: Frame size of out video = input frame size
    def calc_frame_size(self, video_path: str, **kwargs) -> tuple:
        if self.renderer_context == "original_frame":
            return kwargs["frame_size"]  # ty:ignore[invalid-return-type]
        else:
            original_fsize = kwargs.get("frame_size")
            method_cfg = self.cfg.methodCfg.extra_cfgs.get("skip_proc").get("params")  # ty:ignore[possibly-missing-attribute]
            scale_factor = method_cfg.get("scale_factor")
            block_size_orig = method_cfg.get("block_size_orig")
            return BaseBlockSkipProc.get_skip_proc_frame_size(
                frame_w_h=original_fsize,  # ty:ignore[invalid-argument-type]
                scale_factor=scale_factor,
                block_size_orig=block_size_orig,
            )

    # ! render in the orignal frame size
    def get_vis_frame(self, frame_bgr, frame_rs_dict: dict):
        return frame_bgr

    def get_block_extra_dict(self) -> Dict[str, Any]:
        return {"viz_in_fgmask": False}

    def get_custom_renderer(self) -> List[BaseRenderer]:
        method_cfg = self.cfg.methodCfg.extra_cfgs.get("skip_proc").get("params")  # ty:ignore[possibly-missing-attribute]
        # ! decide which renderer to use based on method config (which method is used)
        is_motion_only = "motion_only" in method_cfg["name"]
        # ! we alway render in original frame size
        if is_motion_only:
            return [BlockMontionOnlyRenderer(context=self.renderer_context)]
        else:
            return [BlockRuleBasedRenderer(context=self.renderer_context)]

    def prepare_pipelines_list(self, video_path: str, fps: float, frame_size: tuple):
        pipeline_ls = []
        infer_rs_pipe = VideoPipeline(self.video_output_path, fps, frame_size)
        infer_rs_pipe.add_renderer(InferRsRenderer())  # ty:ignore[invalid-argument-type]
        infer_rs_pipe.add_renderer(GridRenderer(context=self.renderer_context))  # ty:ignore[invalid-argument-type]
        custom_renderer_ls = self.get_custom_renderer()
        for renderer in custom_renderer_ls:
            infer_rs_pipe.add_renderer(renderer)  # ty:ignore[invalid-argument-type]
        pipeline_ls.append(infer_rs_pipe)
        return pipeline_ls
