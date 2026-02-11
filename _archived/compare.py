# This script compares the performance of different video analysis methods

import os
import pandas as pd
from pprint import pprint
from halib import *
from halib import fs, csvfile
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
# from halib.filetype.videofile import VideoUtils
# VideoUtils.hstack()
# -----------------------------
# Configuration
# -----------------------------


POS = "O_Fire_smoke"
NEG = "X_None"


# -----------------------------
# FFmpeg Horizontal Stack
# -----------------------------


# -----------------------------
# Verify CSV and Video Mapping
# -----------------------------
def verify_csv_video(methods, indir, videodir):
    """Verify correspondence between GT, CSV, and video files."""
    rs_dict = {}
    for method in methods:
        method_dir = os.path.join(indir, method)
        pprint(f"Checking method directory: {method_dir}")

        video_files = fs.filter_files_by_extension(videodir, ".mp4", recursive=False)
        gt_files = fs.filter_files_by_extension(videodir, ".csv", recursive=False)
        csv_files = fs.filter_files_by_extension(method_dir, ".csv", recursive=False)
        # only '_results.csv' files
        csv_files = [
            f
            for f in csv_files
            if f.endswith("_results.csv")
            or f.endswith("_od.csv")
            or f.endswith("_pred.csv")
        ]
        vis_files = fs.filter_files_by_extension(method_dir, ".mp4", recursive=True)
        # remove all mask files
        vis_files = [
            f
            for f in vis_files
            if "fg_mask" not in fs.get_file_name(f, split_file_ext=True)[0]
        ]
        # ! if no vis files, use video files as placeholders
        if len(vis_files) == 0:
            vis_files = video_files.copy()
        def sort_key(x):
            return os.path.basename(x).split("_")[0]
        video_files, gt_files, csv_files, vis_files = map(
            lambda x: sorted(x, key=sort_key),
            [video_files, gt_files, csv_files, vis_files],
        )
        # pprint(f'[INFO] Method: {method}, CSV files: {len(csv_files)}, VIS files: {len(vis_files)}, VIDEO files: {len(video_files)}, GT files: {len(gt_files)}')

        assert len(csv_files) == len(video_files) == len(gt_files), (
            f"[ERROR] Mismatch in file counts for method {method}, "
            f"CSV: {len(csv_files)}, VIS: {len(vis_files)}, VIDEO: {len(video_files)}, GT: {len(gt_files)}"
        )

        rs_dict[method] = {
            "csv": csv_files,
            "vis": vis_files,
            "video": video_files,
            "gt": gt_files,
        }

    return rs_dict


# -----------------------------
# Convert Mapping to DataFrame
# -----------------------------
def rs_dict_to_dataframe(rs_dict):
    """Convert result dictionary to unified DataFrame."""
    data = {"video": []}
    for method in rs_dict.keys():
        for key in ["csv", "vis", "video", "gt"]:
            data[f"{method}_{key}"] = []

    # Reference videos (assumed identical across methods)
    base_method = list(rs_dict.keys())[0]
    video_files = rs_dict[base_method]["video"]
    video_names = [os.path.basename(v) for v in video_files]

    for i, video_name in enumerate(video_names):
        video_id = video_name.split(".")[0]
        data["video"].append(video_id)
        for method in rs_dict.keys():
            for key in ["csv", "vis", "video", "gt"]:
                data[f"{method}_{key}"].append(rs_dict[method][key][i])

    df = pd.DataFrame(data)

    # Keep single GT and video path columns
    df = df.rename(
        columns={f"{base_method}_gt": "gt", f"{base_method}_video": "video_path"}
    )
    for m in rs_dict.keys():
        for redundant in [f"{m}_gt", f"{m}_video"]:
            if redundant in df.columns and redundant not in ["gt", "video_path"]:
                del df[redundant]

    # Reorder columns for readability
    cols = ["video", "gt", "video_path"] + [
        c for c in df.columns if c not in ["video", "gt", "video_path"]
    ]
    df = df[cols]

    return df


# -----------------------------
# Performance Calculation
# -----------------------------
def calc_perf(video, gt_label, total_frames, mt_name, csv_path):
    """Calculate performance metrics per video per method."""
    is_YOLO_pred_csv = True if "yolo" in mt_name.lower() else False
    # pprint(f"[INFO] Calculating performance for video: {video}, method: {mt_name}")
    if is_YOLO_pred_csv:  # YOLO models: VP5_od.csv
        df_pred = pd.read_csv(csv_path, sep=";", encoding="utf-8")
    elif "firenet" in mt_name.lower() or "mobilenet" in mt_name.lower():
        df_pred = pd.read_csv(
            csv_path,
            sep=";",
            encoding="utf-8",
            dtype={"label": str, "elapsed_time": float},
            keep_default_na=False,
        )
        df_pred = df_pred.rename(columns={"label": "pred_label"})
        # convert 'none' to 'None'
        df_pred["pred_label"] = df_pred["pred_label"].apply(
            lambda x: "None" if x.lower() == "none" else x
        )
    else:
        df_pred = pd.read_csv(
            csv_path,
            sep=";",
            encoding="utf-8",
            dtype={"pred_label": str, "elapsed_time": float},
            keep_default_na=False,
        )
        # ensure if 'pred_label' (to_lower) == 'skipped', then set to 'None'
        df_pred["pred_label"] = df_pred["pred_label"].apply(
            lambda x: "None" if x.lower() == "skipped" else x
        )

    if len(df_pred) > total_frames:
        total_frames = len(df_pred)

    pred_lb, correct, num_wrong = None, False, 0

    if is_YOLO_pred_csv:
        df_pred.drop_duplicates(
            subset=["frame_id"], keep="first", inplace=True
        )  # keep first prediction of bboxes in a frame
        if gt_label == POS:
            correct = len(df_pred) > 0
            pred_lb = POS if correct else NEG
            num_wrong = total_frames - len(df_pred) if correct else total_frames
        elif gt_label == NEG:
            correct = len(df_pred) == 0
            pred_lb = NEG if correct else POS
            num_wrong = 0 if correct else len(df_pred)
        else:
            raise ValueError(f"Unknown GT label {gt_label} for {video}")

    else:  # YOLO models
        if gt_label == POS:
            pred_lb_pos = df_pred[df_pred["pred_label"] != "None"]
            if len(pred_lb_pos) > 0:
                pred_lb = POS
                correct = True
                num_wrong = len(df_pred[df_pred["pred_label"] == "None"])
            else:
                pred_lb = NEG
                correct = False
                num_wrong = len(df_pred)
        elif gt_label == NEG:
            pred_as_pos = df_pred[df_pred["pred_label"] != "None"]
            if len(pred_as_pos) > 0:
                pred_lb = POS
                correct = False
                num_wrong = len(pred_as_pos)
            else:
                pred_lb = NEG
                correct = True
                num_wrong = 0
        else:
            raise ValueError(f"Unknown gt label {gt_label} for video {video}")

    # pprint(
    #     f"[DEBUG] Video: {video}, GT: {gt_label}, Pred: {pred_lb}, Correct: {correct}, Num Wrong Frames: {num_wrong}"
    # )
    total_elapsed_time = (
        df_pred["elapsed_time"].sum() if "elapsed_time" in df_pred.columns else 0.0
    )
    total_elapsed_time = df_pred["elapsed_time"].sum()
    num_of_frame_with_time = len(df_pred)
    return pred_lb, correct, num_wrong, total_elapsed_time, num_of_frame_with_time


# -----------------------------
# Main Evaluation Process
# -----------------------------
# ! Input DataFrame format:
# video	        gt	            trial_01_csv
# VP11          VP11_gt.csv     trial_01/VP11_results.csv


def process_videos(df, methods):
    """Compute per-video performance across all methods."""
    dfmk = csvfile.DFCreator()
    dfmk.create_table(
        "perf",
        ["video", "gt", "total_frames"]
        + [
            f"{m}_{x}"
            for m in methods
            for x in [
                "pred",
                "correct",
                "num_wrong_frames",
                "total_elapsed_time",
                "num_of_frame_with_time",
            ]
        ],
    )
    rows = []

    for _, row in df.iterrows():
        video = row["video"]
        gt_csv = row["gt"]
        gt_df = pd.read_csv(gt_csv, sep=";", encoding="utf-8")
        total_frames = len(gt_df)
        gt_label = POS if "VP" in video else NEG

        row_data = [video, gt_label, total_frames]
        for mt_name in methods:
            pred_lb, correct, num_wrong, total_elapsed_time, num_of_frame_with_time = (
                calc_perf(video, gt_label, total_frames, mt_name, row[f"{mt_name}_csv"])
            )
            row_data += [
                pred_lb,
                correct,
                num_wrong,
                total_elapsed_time,
                num_of_frame_with_time,
            ]

        rows.append(row_data)

    dfmk.insert_rows("perf", rows)
    dfmk.fill_table_from_row_pool("perf")
    final_df = dfmk["perf"]

    # Remove redundant columns (pred labels)
    for m in methods:
        del final_df[f"{m}_pred"]

    return final_df


# -----------------------------
# Save Results
# -----------------------------
def save_raw_results(final_df, methods, outdir, filename="_cmp_raw_results.csv"):
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, filename)
    final_df.to_csv(outfile, index=False, sep=";", encoding="utf-8")

    correct_cols = [f"{m}_correct" for m in methods]
    all_correct_df = final_df[final_df[correct_cols].all(axis=1)]
    any_wrong_df = final_df[~final_df[correct_cols].all(axis=1)]

    all_correct_df.to_csv(
        f"{outdir}/all_correct.csv", index=False, sep=";", encoding="utf-8"
    )
    any_wrong_df.to_csv(
        f"{outdir}/method_wrong.csv", index=False, sep=";", encoding="utf-8"
    )

    print(f"[INFO] Results saved to {outdir}")
    return outfile


POS = "O_Fire_smoke"
NEG = "X_None"


def invert_label(label):
    if label == POS:
        return NEG
    elif label == NEG:
        return POS
    else:
        raise ValueError(f"Unknown label: {label}")


def get_gt_pred_and_time_info(
    df, method, mode="per_video"
):  # mode: per_frame or per_video
    assert mode in [
        "per_frame",
        "per_video",
    ], "mode should be 'per_frame' or 'per_video'"

    if mode == "per_video":
        y_true = df["gt"].tolist()
        y_correct = df[f"{method}_correct"].tolist()
        y_pred = []
        y_correct = df[f"{method}_correct"].tolist()
        y_true = df["gt"].tolist()
        y_pred = []
        for i in range(len(y_correct)):
            if y_correct[i] is True:  # predicted correctly
                y_pred.append(y_true[i])
            else:  # predicted incorrectly
                y_pred.append(invert_label(y_true[i]))
    elif mode == "per_frame":
        y_true_video = df["gt"].tolist()
        num_videos = len(y_true_video)
        frm_cnt_video = df["total_frames"].tolist()
        y_true = []
        num_wrongs_frames_video = df[f"{method}_num_wrong_frames"].tolist()
        y_pred = []
        for i in range(num_videos):
            num_total = frm_cnt_video[i]
            y_true.extend([y_true_video[i]] * num_total)
            num_wrong_frames = num_wrongs_frames_video[i]
            num_correct_frames = num_total - num_wrong_frames
            y_pred.extend([y_true_video[i]] * num_correct_frames)
            y_pred.extend([invert_label(y_true_video[i])] * num_wrong_frames)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    # num_frame_allvideos = df["total_frames"].sum()
    num_frame_with_time_allvideos = df[f"{method}_num_of_frame_with_time"].sum()
    elapsed_time_allvideos = df[f"{method}_total_elapsed_time"].sum()
    return y_true, y_pred, num_frame_with_time_allvideos, elapsed_time_allvideos


def cal_metric(df, methods, mode="per_video"):  # mode: per_frame or per_video
    assert mode in [
        "per_frame",
        "per_video",
    ], "mode should be 'per_frame' or 'per_video'"
    results = []
    for method in methods:
        method_rs_dict = {"method": method}
        y_true, y_pred, num_frame_with_time_allvideos, elapsed_time_allvideos = (
            get_gt_pred_and_time_info(df, method, mode=mode)
        )
        try:
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, pos_label=POS)
            recall = recall_score(y_true, y_pred, pos_label=POS)
            f1 = f1_score(y_true, y_pred, pos_label=POS)
            avg_proc_time = (
                elapsed_time_allvideos * 1.0 / num_frame_with_time_allvideos
                if num_frame_with_time_allvideos > 0
                else 0.0
            )
            fps = 1.0 / avg_proc_time if avg_proc_time > 0 else 0.0

            # Compute confusion matrix (labels ordered as [negative, positive])
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[NEG, POS]).ravel()
            false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
            # save results as a dict
            method_rs_dict.update(
                {
                    "accuracy": accuracy * 100.0,
                    "f1_score": f1 * 100.0,
                    "Recall": recall * 100.0,
                    "FAR": false_alarm_rate * 100.0,
                    "precision": precision * 100.0,
                    "fps": fps,
                }
            )
            results.append(method_rs_dict)

        except ValueError as e:
            print(
                f"Error: {e}. Please ensure labels are valid strings and lists have the same length."
            )
    # Convert results to DataFrame for better visualization
    # pprint(results)
    results_df = pd.DataFrame(results)
    return results_df


def calc_metrics_and_report(
    raw_cmp_csv, outdir, modes=["per_video", "per_frame"], log=True
):
    df = pd.read_csv(raw_cmp_csv, sep=";", encoding="utf-8")
    methods_used = []
    methods_used = [
        col.replace("_correct", "") for col in df.columns if "_correct" in col
    ]
    for mode in modes:
        per_mode_df = cal_metric(df, methods_used, mode=mode)
        out_mode_csv = os.path.join(outdir, f"__perf_{mode}_results.csv")
        per_mode_df.to_csv(out_mode_csv, index=False, sep=";", encoding="utf-8")
        print(f"[INFO] <<{mode}>> metrics saved to {out_mode_csv}")
        if log:
            console.rule(f"Mode={mode}")
            csvfile.fn_display_df(per_mode_df, max_col_width=60)


# -----------------------------
# Main Entry
# -----------------------------
def main():
    # METHODS = [
    #     "prof_hgnetv2b5_2classes_notemp",
    #     # "prof_hgnetv2b5_3classes_notemp",
    #     "yolov5s_notemp",
    #     "yolov5l_notemp",
    # ]

    # METHODS = [
    #     "prof_no_temp",
    #     "prof_temp_stabilize",
    # ]
    # INDIR = "/mnt/e/SyncData/paper2_main/zcompare/znew_model/__all_cases"
    INDIR = "/mnt/e/SyncData/paper2_main/zcompare/all_new_no_temp/__archived"
    METHODS = fs.list_dirs(INDIR)
    IGNORE = ["__archived", "_archived_static", "__error"]
    METHODS = [m for m in METHODS if m not in IGNORE]
    # pprint(METHODS)
    # INDIR = "/mnt/e/SyncData/paper2_baseline/zout/DFire_test"

    # VIDEODIR = "/mnt/e/SyncData/paper2_main/datasets/DFireVal/valid"
    # VIDEODIR = "/mnt/e/SyncData/paper2_main/datasets/DFireNonStatic/test"
    VIDEODIR = "/mnt/e/SyncData/paper2_main/datasets/DFire/test"
    # VIDEODIR = "/mnt/e/SyncData/paper2_main/datasets/DFireStatic/test"

    OUTDIR = INDIR
    os.makedirs(OUTDIR, exist_ok=True)
    rs_dict = verify_csv_video(methods=METHODS, indir=INDIR, videodir=VIDEODIR)
    df = rs_dict_to_dataframe(rs_dict)
    df.to_csv(
        os.path.join(OUTDIR, "cmp_raw_data_source.csv"),
        index=False,
        sep=";",
        encoding="utf-8",
    )

    final_df = process_videos(df=df, methods=METHODS)
    raw_cmp_csv = save_raw_results(
        final_df=final_df,
        methods=METHODS,
        outdir=OUTDIR,
        filename="_cmp_raw_results.csv",
    )
    calc_metrics_and_report(raw_cmp_csv, OUTDIR)


# -----------------------------
if __name__ == "__main__":
    main()
