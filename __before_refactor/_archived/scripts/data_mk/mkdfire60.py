from halib import *
from argparse import ArgumentParser
import cv2
import os


def parse_args():
    parser = ArgumentParser(description="desc text")
    parser.add_argument(
        "-indir",
        "--indir",
        type=str,
        help="arg1 description",
        default="datasets/DFire60/test",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    indir = args.indir

    video_files = fs.filter_files_by_extension(indir, ext=".mp4", recursive=False)
    pprint(video_files)

    # if num_frame of video > 60, then make a copy with only first 60 frames
    # del the src video after making the copy
    for video_file in video_files:
        pprint(f"Processing {video_file}...")
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            continue
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if num_frames <= 60:
            cap.release()
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        cap.release()

        temp_file = video_file + ".tmp60.mp4"
        out = cv2.VideoWriter(temp_file, fourcc, fps, (width, height))

        cap = cv2.VideoCapture(video_file)
        frame_count = 0
        while frame_count < 60:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frame_count += 1
        cap.release()
        out.release()

        # os.remove(video_file)
        # os.rename(temp_file, video_file)


if __name__ == "__main__":
    main()
