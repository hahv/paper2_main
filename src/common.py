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
    SKIP_LABEL = "skipped"

    # file names / patterns
    GT_FILE_PATTERN = "__labels"
    INFER_FILE_PATTERN = "_results"

    # perf/analysis-related file
    PERF_FILE_PREFIX = "_"  # to make it appear at the top in file explorers
    PERF_FILE_POSTFIX = "__perf"  # to make it easily identifiable as perf file; also used in report generation to find relevant perf files

    # metrics
    METRIC_PER_FRAME = "per_frame"
    METRIC_PER_VIDEO = "per_video"

    # general
    # ! do not access directly, use proj_root() instead
    __PROJECT_ROOT = "/mnt/e/SyncData/paper2_main"

    # method name pattern
    NOTEMP_MT_PATTERN = "no_temp_method"

    # timeline
    TL_TYPE_GT = "gt"
    TL_TYPE_NO_SKIP = "no_skip"
    TL_TYPE_SKIP = "skip"

    # timeline labels for ground truth
    TL_GT_FIRESMOKE = "FireSmoke"
    TL_GT_NONE = "None"

    # timeline labels for baseline methods (no-skip)
    TL_NOSKIP_CORRECT_POS = "Correct Pos."  # TP
    TL_NOSKIP_CORRECT_NEG = "Correct Neg."  # TN
    TL_NOSKIP_CORRECT = "Correct"  # TP+TN (combined for reporting)
    TL_NOSKIP_FALSE_ALARM_FP = "False Alarm"  # FP
    TL_NOSKIP_MISS_FN = "Miss"  # FN

    # timeline labels for skip/temporal method
    TL_SKIP_CORRECT_SKIP = "Correct Skip"  # no fire/smoke+correctly skipped
    TL_SKIP_CORRECT_INFER = "Correct Infer."  # fire/smoke+correctly processed
    TL_SKIP_FALSE_SKIP = "False Skip"  # fire/smoke but incorrectly skipped
    TL_SKIP_WASTED_INFER = "Wasted Infer."  # incorrectly proc (wasted resources)

    @staticmethod
    def proj_root() -> str:
        return normalize_paths(GlobalConst.__PROJECT_ROOT)
