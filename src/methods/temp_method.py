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

    def _resolve_pre_calc_time(self, frame_idx: int) -> float | None:
        """Return precomputed elapsed_time for frame_idx, or None if unavailable."""
        if not (
            hasattr(self, "precomputed_rs_proc")
            and self.precomputed_rs_proc is not None
        ):
            return None
        pre_rs = self.precomputed_rs_proc.get_frame_data(frame_idx)
        if pre_rs is None:
            return None
        return pre_rs.get("elapsed_time", None)

    def infer_frame(self, frame, frame_idx: int) -> dict:
        assert self.profiler is not None, "Profiler not initialized."
        assert self.profiler.enabled, "Profiler is not enabled."

        with self.profiler.measure("infer_wrapper") as ctx:
            mt_cfg_dict = {
                "mt_cfg": self.cfg.methodCfg.extra_cfgs.get("skip_proc", {}).copy()  # ty:ignore[unresolved-attribute]
            }
            meta_data = None
            infer_result = {}

            # Tracks wall-clock cost of skip_check + prep_input combined
            overhead_start = time.perf_counter()

            # ── Step 1: Should we skip this frame? ────────────────────────────
            with ctx.step("skip_check"):
                should_skip, meta_data = self.skip_proc.should_skip(frame_idx, frame)
                meta_data.update(mt_cfg_dict)

            if should_skip:
                # Skip path: only skip_check is recorded — prep_input and
                # heavy_infer are intentionally absent (they did not run)
                infer_result = self.skip_proc.get_dummy_result(
                    self.cfg.modelCfg.class_names
                )

            else:
                # ── Step 2: Prepare model input ────────────────────────────────
                overhead_time = time.perf_counter() - overhead_start
                pre_calc_time = self._resolve_pre_calc_time(frame_idx)

                # ── Step 3: Heavy inference ────────────────────────────────────
                # pre_calc_time=None  → live wall-clock (normal path)
                # pre_calc_time=float → inject precomputed value; profiler
                #                       auto-accumulates into ctx duration ✅
                with ctx.step("heavy_infer", pre_calc_time=pre_calc_time):
                    infer_result = super().infer_frame(frame, frame_idx)

                # Propagate overhead into result so downstream consumers
                # (e.g. base_method.py) see the true combined elapsed cost
                if "elapsed_time" in infer_result:
                    infer_result["elapsed_time"] = overhead_time + float(
                        infer_result["elapsed_time"]
                    )

            # ── Step 4: Merge skip proc meta into result
            # ───────────────────────
            # ! Merge meta data (of skip proc) into raw result
            # ! This is useful for logging or visualization later
            assert meta_data and len(meta_data) > 0, (
                "Meta data from skip proc is empty!"
            )
            infer_result.update(meta_data)
            return infer_result