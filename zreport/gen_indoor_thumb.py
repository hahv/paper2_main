import os
from halib import *
from halib.common import seed_everything
from halib.utils.video import VideoUtils

from argparse import ArgumentParser
import random
import cv2

def parse_args():
    parser = ArgumentParser(description="desc text")
    parser.add_argument(
        "-db",
        "--db",
        type=str,
        help="dataset path",
        default="/mnt/d/zdataset_paper2/build_video_dataset/my_firesmoke_indoor",
    )
    parser.add_argument(
        "-outdir",
        "--outdir",
        type=str,
        help="output directory",
        default="./zreport/indoor_thumbs",
    )
    return parser.parse_args()


def main():
    seed_everything(42)
    args = parse_args()
    db = args.db

    labels = os.listdir(db)
    outdir = args.outdir
    for label in labels:
        label_dir = os.path.join(outdir, label)
        os.makedirs(label_dir, exist_ok=True)

    df = VideoUtils.get_video_dir_meta_df(db, search_recursive=True)
    # csvfile.fn_display_df(df)
    for idx, row in tqdm(df.iterrows()):
        video_path = row["video_path"]
        # get parent dir name as label
        label = os.path.basename(os.path.dirname(video_path))
        frame_cnt = row["frame_count"]

        # thumbnail at middle range (30% ~ 70%) but random
        t_start = int(frame_cnt * 0.3)
        t_end = int(frame_cnt * 0.7)
        thumb_idx = random.randint(t_start, t_end)
        fname = fs.get_file_name(video_path,split_file_ext=True)[0]

        # Open the video
        cap = cv2.VideoCapture(video_path)

        # Check if video opened successfully
        if not cap.isOpened():
            raise IOError("Cannot open video")

        # Go directly to frame 199 (0-based indexing!)
        cap.set(cv2.CAP_PROP_POS_FRAMES, thumb_idx)

        ret, frame = cap.read()
        if ret:
            output_frame_path = os.path.join(outdir, label, f"{fname}_thumb_frmidx_{thumb_idx}.jpg")
            cv2.imwrite(output_frame_path, frame)
            print(f"Frame {thumb_idx} saved as", output_frame_path)
        else:
            print(f"Failed to read frame {thumb_idx}")

        cap.release()


if __name__ == "__main__":
    main()
