from halib.system.path import *


class GlobalConst:
    # csv column names
    COL_VIDEO = "video"
    COL_VIDEO_PATH = "video_path"
    COL_NUM_FRAMES = "num_frames"
    COL_FRAME_IDX = "frame_idx"
    COL_GT = "gt_label"
    COL_PRED = "pred_label"
    COL_ELAPSED_TIME = "elapsed_time"

    # label names
    FIRESMOKE_LABEL = "firesmoke"
    NONE_LABEL = "none"

    # file names / patterns
    GT_FILE_PATTERN = "__labels"
    INFER_FILE_PATTERN = "_results"

    # perf/analysis-related file
    PERF_FILE_PREFIX = "_"  # to make it appear at the top in file explorers

    # metrics
    METRIC_PER_FRAME = "per_frame"
    METRIC_PER_VIDEO = "per_video"

    # genaral
    # ! do not access directly, use proj_root() instead
    __PROJECT_ROOT = "/mnt/e/SyncData/paper2_main"

    @staticmethod
    def proj_root() -> str:
        return normalize_paths(GlobalConst.__PROJECT_ROOT)
