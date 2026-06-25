from alembic.util import err
from bottle import ext
from torch.utils.tensorboard.summary import video
from requests.packages import target
from zz_df_unify import get_df
from halib import *
from halib.filetype import csvfile
from tap import *
from halib.filetype.videofile import VideoUtils

class CustomArgs(Tap):
    indir: str = "./zout/zruns/AIPro_task/old_vs_new"

def main():
    args = CustomArgs().parse_args()
    BASE_DIR = args.indir
    outfile_name = "hgnetv5_hgnetv6_preds.csv"
    outfile = os.path.join(BASE_DIR, outfile_name)
    df = get_df(
        args.indir, outdir=BASE_DIR, outfile_name=outfile_name, copy_raw_data=True
    )
    # csvfile.fn_display_df(df.head(10))
    columns = df.columns.tolist()
    columns = [c for c in columns if "_dir" not in c]
    df = df[columns]
    df.to_csv(outfile, index=False, sep=";", encoding="utf-8")
    
    TARGET_MODEL = "hgnetv2_b6"
    target_col = None
    for c in columns:
        if TARGET_MODEL in c:
            target_col = c
            break
    assert target_col is not None, f"Could not find column for target model '{TARGET_MODEL}'"
    columns_need = ["video", "frame_idx", "gt_label", "video_path", target_col]
    # GT_LABELS = ["fire_smoke", "none"]
    target_col_normalized = f"{target_col}_converted" 
    df[target_col_normalized] = df[target_col].map(
        lambda x: "fire_smoke" if str(x).lower() in ["fire", "smokeonly"] else "none"
    )
    ERROR_TYPE_A = "Missed"
    ERROR_TYPE_B = "False Alarm"
    
    error_type_col = f"error_type"
    df[error_type_col] = df.apply(
        lambda row: ERROR_TYPE_A if row[target_col_normalized] == "none" and row["gt_label"] == "fire_smoke"
        else ERROR_TYPE_B if row[target_col_normalized] == "fire_smoke" and row["gt_label"] == "none"
        else None,
        axis=1
    )

    wrong_df = df[df[error_type_col].notnull()][columns_need + [target_col_normalized, error_type_col]]
    wrong_outfile = os.path.join(BASE_DIR, f"{TARGET_MODEL}_wrong_preds.csv")
    wrong_df.to_csv(wrong_outfile, index=False, sep=";", encoding="utf-8")
    print(f"Saved wrong predictions to: {wrong_outfile}")

    # Group by video and aggregate

    extract_df = (
        df.groupby(["video", "error_type"])
        .agg(
            video_path=(
                "video_path",
                "first",
            ),  # Grab the common path for this video
            frame_list=(
                "frame_idx",
                list,
            ),  # Aggregate frame indices into a clean Python list
        )
        .reset_index()
    )

    # Reorder columns to match your exact request format
    extract_df = extract_df[["video", "video_path", "error_type", "frame_list"]]
    extract_outfile = os.path.join(BASE_DIR, f"{TARGET_MODEL}_wrong_frames_summary.csv")
    extract_df.to_csv(extract_outfile, index=False, sep=";", encoding="utf-8")
    print(f"Saved wrong frames summary to: {extract_outfile}")

    # csvfile.fn_display_df(extract_df.head(10))

    # # Optional: if you want to look at a specific video's results
    # csvfile.fn_display_df(extract_df.head(10))
    
    def extract_frame_each_video(row):
        # video_name = row["video"]
        video_path = row["video_path"]
        error_type = row["error_type"]
        frame_indices = row["frame_list"]
        print(f"Video: {row['video']}, Path: {video_path}, Frames with wrong preds: {frame_indices}")
        OUTDIR_BASE = os.path.join(BASE_DIR, f"wrong_frames")
        os.makedirs(OUTDIR_BASE, exist_ok=True)
        outdir = os.path.join(OUTDIR_BASE, error_type)
        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
        VideoUtils.extract_frame(video_path, frame_indices, outdir=outdir)
    # 2. Apply it to each row
    extract_df.apply(extract_frame_each_video, axis=1)
    
if __name__ == "__main__":
    main()