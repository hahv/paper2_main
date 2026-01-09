from halib import *

from typing import Optional
from src.methods.no_temp_method import NoTempMethod
from src.methods.skip.base_skip_proc import BaseSkipProc

class TempMethod(NoTempMethod):
    def __init__(self, cfg, skip_proc: Optional[BaseSkipProc] = None):
        super().__init__(cfg)
        # Composition: We "have" a handler, we aren't "is" a handler
        self.skip_proc = skip_proc if skip_proc else BaseSkipProc()

    def infer_frame(self, frame, frame_idx: int) -> dict:
        with self.profiler.measure("infer_wrapper") as ctx:
            # 1. Ask the handler: Should we skip?
            with ctx.step("skip_check"):
                should_skip, meta_data = self.skip_proc.should_skip(frame_idx, frame)

            if should_skip:
                class_names = self.cfg.modelCfg.class_names
                return self.skip_proc.get_dummy_result(class_names)

            # 2. Ask the handler: Prepare the image (Crop/Resize/Etc)
            with ctx.step("prep_input"):
                model_input = self.skip_proc.prepare_input(frame, meta_data)

            # 3. Run the heavy model (Super class logic)
            # Note: We pass the *processed* input (e.g. the crop)
            with ctx.step("heavy_infer"):
                raw_result = super().infer_frame(model_input, frame_idx)

            # 4. Ask the handler: Fix the results (Coordinate mapping)
            final_result = self.skip_proc.post_process_result(raw_result, meta_data)

            return final_result