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
    NO_FRAMES_SKIP_IN_FPS_CALC = 3  # Number of initial frames to skip for FPS calculation, due to initialization overhead

    # method name pattern
    NOTEMP_MT_PATTERN = "no_temp_method"

    # This is a template cfg file for external runs (fireNet, mobilenet, yolo)
    EXTERNAL_CFG = "config/zruns/_run_ext.yaml"

    # Config related
    OPTIM_OUTDIR = "zout/zoptim_val"

    # Optim
    METHOD_NAME = "method_name"
    COL_PARAM_SKIP_RATE = "skip_rate"
    COL_PARAM_RECALL = "metric_recall (tpr)"
    COL_PARAM_FAR = "metric_fpr (false alarm rate)"
    COL_PARAM_COMBINED_SCORE = "Combined_Score"
    COL_PARAM_RECALL_DROP = "Recall_Drop"
    COL_PARAM_W_S = "w_S"
    COL_PARAM_W_F = "w_F"
    COL_PARAM_W_R = "w_R"
    COL_PARAM_DELTA_R = "delta_R"
    COL_PARAM_RECALL_RET = "recall_retention"
    COL_PARAM_RECALL_RET_NORM = "recall_retention_norm"
    COL_PARAM_SKIP_RATE_NORM = "skip_rate_norm"
    COL_PARAM_FAR_REDUC_NORM = "far_reduction_norm"

    # Optim report file name
    OPT_ALL_RP_FILE_PREFIX = "_opt_all_rp__"
    OPT_SELECTED_RP_FILE_SUFFIX = "_opt_sel_rp__"
