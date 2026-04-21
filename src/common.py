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
    TL_NOSKIP_CORRECT_POS = "Recall"  # TP
    TL_NOSKIP_CORRECT_NEG = "True Negative"  # TN
    # TL_NOSKIP_CORRECT = "Correct"  # TP+TN (combined for reporting)
    TL_NOSKIP_FALSE_ALARM_FP = "False Alarm"  # FP
    TL_NOSKIP_MISS_FN = "Miss"  # FN

    # timeline labels for skip/temporal method
    TL_SKIP_CORRECT_SKIP = "Correct Skip"  # no fire/smoke+correctly skipped
    TL_SKIP_CORRECT_INFER = "Correct Infer."  # fire/smoke+correctly processed
    TL_SKIP_FALSE_SKIP = "False Skip"  # fire/smoke but incorrectly skipped
    TL_SKIP_WASTED_INFER = "Wasted Infer."  # incorrectly proc (wasted resources)

    TL_CSV_FILE_NAME = "_timeline_report.csv"
    # This is a template cfg file for external runs (fireNet, mobilenet, yolo)
    EXTERNAL_CFG = "config/zruns/_run_ext.yaml"

    # Config related
    OPTIM_OUTDIR = "zout/zoptim_sanity"

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

    @staticmethod
    def proj_root() -> str:
        return normalize_paths(GlobalConst.__PROJECT_ROOT)
