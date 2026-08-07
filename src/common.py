from halib.system.path import *

# ! dynamically get <<PRJ_ROOT>> based on the current file path
PROJECT_DIR = Path(__file__).parent.parent

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
    NO_FRAMES_SKIP_IN_FPS_CALC = 3  # Number of initial frames to skip for FPS calculation, due to initialization overhead