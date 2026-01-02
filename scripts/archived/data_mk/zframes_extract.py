from halib import *
from argparse import ArgumentParser
import cv2


def parse_args():
    parser = ArgumentParser(description="desc text")
    parser.add_argument(
        "-indir",
        "--indir",
        type=str,
        help="test video dir",
        default="./datasets/DFire/test",
    )
    parser.add_argument(
        "-outdir", "--outdir", type=str, help="output dir", default="./zout/dfire_vimgs"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    indir = args.indir
    outdir = args.outdir
    assert os.path.exists(indir), f"{indir} not exists"
    os.makedirs(outdir, exist_ok=True)
    video_list = fs.filter_files_by_extension(indir, [".mp4", ".avi"], recursive=False)

    def video_name_to_cls(video_name):
        if "FP" in video_name:
            return "none"
        else:
            return "fire_smoke"

    for video_path in tqdm(video_list):
        video_name = fs.get_file_name(video_path, split_file_ext=True)[0]
        # video_outdir = os.path.join(outdir, fname)
        # os.makedirs(video_outdir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # divide num_frames into 10 parts of equal length (if possible)
        frame_indices_parts = np.array_split(np.arange(num_frames), 10)
        # get random index from each part
        selected_frame_indices = [
            np.random.choice(part) for part in frame_indices_parts if len(part) > 0
        ]
        cls_name = video_name_to_cls(video_name)
        video_outdir = os.path.join(outdir, cls_name)
        os.makedirs(video_outdir, exist_ok=True)
        for frame_idx in selected_frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame {frame_idx} from {video_path}")
                continue
            frame_fname = f"{video_name}_frame{frame_idx:04d}.png"
            frame_path = os.path.join(video_outdir, frame_fname)
            cv2.imwrite(frame_path, frame)
        cap.release()
        print(
            f"Extracted {len(selected_frame_indices)} frames from {video_path} to {video_outdir}"
        )


if __name__ == "__main__":
    main()
