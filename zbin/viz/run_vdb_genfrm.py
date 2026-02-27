import cv2
from halib import *
from halib.common.common import seed_everything
from halib.filetype import csvfile
from tap import Tap

"""
Builds a visual inspection folder from a categorized video directory.

Pipeline:
  1. Scan `video_dir` and generate a metadata CSV via `get_video_dir_meta_csv`
     (video filenames must follow the pattern: `{data_source}__{category}__...`).
  2. Print a summary table of video counts by data source and category.
  3. For each video, uniformly sample `num_frames_per_video` frames within the
     specified portion of the video (e.g., `part="0.2:0.8"` → middle 60%).
  4. Save each frame as a JPEG under:
     `{out_dir}/{category}/{category}__{data_source}__{video_name}__frm{idx}.jpg`
"""


class PlotVideoArgs(Tap):
    video_dir: str = "./datasets/UFireIndoorFull/val"
    out_dir: str = "./zout/reports/viz/vdb_screenshots/raw"  # Output directory
    seed: int = 1  # Random seed for reproducibility
    part = "0.2:0.8"  # part to visualize, e.g., "0.2:0.8" means visualize the middle 60% of the videos
    num_frames_per_video = (
        1  # Number of frames to sample from each video for visualization
    )


META_DATA_CSV = "video_metadata"
FILE_NAME_SEP = "__"
TABLE_NAME = "video_meta"


def get_meta_csv_file_path(video_dir, out_dir):
    video_dir_name = fs.get_dir_name(video_dir)
    return os.path.join(out_dir, f"{video_dir_name}_{META_DATA_CSV}.csv")


def get_video_dir_meta_csv(video_dir, out_dir):
    """
    Main logic to extract features, cluster, and split videos.
    """
    # 1. Gather all video paths
    if not os.path.exists(video_dir):
        pprint(f"Directory not found: {video_dir}")
        return [], [], []

    video_paths = fs.filter_files_by_extension(
        directory=video_dir, ext=[".mp4", ".avi", ".mov", ".mkv"], recursive=True
    )
    assert len(video_paths) > 0, f"No video files found in {video_dir}"
    pprint(
        f"Found {len(video_paths)} videos. Extracting frames and computing embeddings..."
    )
    # the new csv will have columns: video_path, data_source, category
    dfmk = csvfile.DFCreator()

    dfmk.create_table(
        table_name=TABLE_NAME, columns=["video_path", "data_source", "category"]
    )
    rows = []
    for vp in tqdm(video_paths, desc="Proc video ..."):
        vpath = os.path.abspath(vp)
        vfile_name = fs.get_file_name(vpath, split_file_ext=True)[0]
        vfile_name_parts = vfile_name.split(FILE_NAME_SEP)
        data_source = vfile_name_parts[0]
        category = vfile_name_parts[1]
        rows.append([vpath, data_source, category])
    dfmk.insert_rows(TABLE_NAME, rows)
    dfmk.fill_table_from_row_pool(TABLE_NAME)
    outfile = get_meta_csv_file_path(video_dir, out_dir)
    df = dfmk[TABLE_NAME].copy()
    df = df[["data_source", "category", "video_path"]]
    df.sort_values(by=["data_source", "category"], inplace=True)
    df.to_csv(
        outfile,
        index=False,
        sep=";",
        encoding="utf-8",
    )
    pprint(f"Saved video metadata CSV to: ⏬")
    pprint_local_path(outfile, get_wins_path=True)
    return df


def make_video_screenshots(df, out_dir, part, num_frames_per_video):
    """
    Sample frames from each video in `df` and save them as JPEGs.
    """
    part_start, part_end = (float(x) for x in part.split(":"))

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Sampling frames"):
        video_path = row["video_path"]
        data_source = row["data_source"]
        category = row["category"]
        video_name = fs.get_file_name(video_path, split_file_ext=True)[0]

        out_subdir = os.path.join(out_dir, category)
        os.makedirs(out_subdir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            continue

        start_frame = int(total_frames * part_start)
        end_frame = int(total_frames * part_end)
        frame_range = end_frame - start_frame
        if frame_range <= 0:
            cap.release()
            continue

        n = num_frames_per_video
        indices = [
            start_frame + int(i * frame_range / (n - 1))
            if n > 1
            else start_frame + frame_range // 2
            for i in range(n)
        ]

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            fname = f"{data_source}{FILE_NAME_SEP}{category}{FILE_NAME_SEP}{video_name}__frm{idx}.jpg"
            cv2.imwrite(os.path.join(out_subdir, fname), frame)

        cap.release()

    pprint(f"Done. Frames saved to: {out_dir}")


def main():
    args = PlotVideoArgs().parse_args()
    seed_everything(args.seed)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)
    console.rule("[bold red] Generate images for video dataset visualization")

    # Step 1: Generate metadata CSV
    df = get_video_dir_meta_csv(args.video_dir, args.out_dir)
    if df is None or len(df) == 0:
        pprint("No videos found. Exiting.")
        return

    # Step 2: Print summary table
    summary = df.groupby(["data_source", "category"]).size().reset_index(name="count")
    csvfile.fn_display_df(summary)

    # Step 3 & 4: Sample frames and save as JPEG
    make_video_screenshots(df, args.out_dir, args.part, args.num_frames_per_video)


if __name__ == "__main__":
    main()
