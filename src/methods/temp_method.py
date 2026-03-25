from halib import *
from typing import Optional
import time
from src.config import Config
from src.results.base_rs_proc import BaseRsProc
from src.methods.no_temp_method import NoTempMethod
from src.methods.skip.base_skip_proc import BaseSkipProc, SkipProcFactory


class TempMethod(NoTempMethod):
    def __init__(self, cfg: Config, rs_handlers: Optional[list[BaseRsProc]] = None):
        super().__init__(cfg, rs_handlers)
        # Composition: We "have" a handler, we aren't "is" a handler
        assert "temp_method" in self.cfg.methodCfg.name, (  # ty:ignore[unsupported-operator]
            "only `temp_method` supported in yaml cfg `method_selector.selected_method`"
        )
        self.skip_proc: BaseSkipProc = SkipProcFactory.create_skip_proc(cfg)

    # ! This video done, reset the skip proc motion det if any
    def after_infer_video(self, video_path: str):
        if self.skip_proc.motion_det is not None:
            self.skip_proc.motion_det.reset()
        super().after_infer_video(video_path)

    def infer_frame(self, frame, frame_idx: int) -> dict:
        assert self.profiler is not None, "Profiler not initialized."
        assert self.profiler.enabled, "Profiler is not enabled."
        with self.profiler.measure("infer_wrapper") as ctx:
            mt_cfg_dict = {
                "mt_cfg": self.cfg.methodCfg.extra_cfgs.get("skip_proc", {}).copy()  # ty:ignore[unresolved-attribute]
            }
            meta_data = None
            infer_result = {}

            start_skip_check = time.perf_counter()

            # 1. Ask the handler: Should we skip?
            with ctx.step("skip_check"):
                should_skip, meta_data = self.skip_proc.should_skip(frame_idx, frame)
                meta_data.update(mt_cfg_dict)

            skip_check_time = time.perf_counter() - start_skip_check

            if should_skip:
                class_names = self.cfg.modelCfg.class_names
                infer_result = self.skip_proc.get_dummy_result(class_names)
            else:
                # 2. Ask the handler: Prepare the image (Crop/Resize/Etc)
                with ctx.step("prep_input"):
                    model_input = self.skip_proc.prepare_infer_input(frame, meta_data)

                # 3. Run the heavy model (Super class logic)
                # Note: We pass the *processed* input (e.g. the crop)
                with ctx.step("heavy_infer"):
                    infer_result = super().infer_frame(model_input, frame_idx)

                # The total time is the skip check + whatever the heavy infer took (either real CNN time or precomputed time)
                base_time = float(infer_result.get("elapsed_time", 0.0))
                if base_time == 0.0:
                    # Fallback if no precomputed time exists and we need to simulate actual CNN run
                    # Although BaseMethod usually handles real time, we need to inject it here if we want combined accuracy
                    pass # Handled gracefully by BaseMethod if we don't inject it when None

                if "elapsed_time" in infer_result:
                    infer_result["elapsed_time"] = skip_check_time + base_time

            # 4. Ask the handler: Fix the results (Coordinate mapping)
            # ! Merge meta data (of skip proc) into raw result
            # ! This is useful for logging or visualization later
            assert len(meta_data) > 0, "Meta data from skip proc is empty!"
            infer_result.update(meta_data)

            return infer_result
