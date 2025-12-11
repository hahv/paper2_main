from halib import *
from argparse import ArgumentParser
import cv2
import os


def parse_args():
    parser = ArgumentParser(
        description="desc text")
    parser.add_argument('-indir', '--indir', type=str,
                        help='test video dir', default='./datasets/DFire/test')
    return parser.parse_args()


def main():
    args = parse_args()
    indir = args.indir
    video_list = fs.filter_files_by_extension(indir, ['.mp4', '.avi'], recursive=False)
    # ignore files with _crop in name
    video_list = [v for v in video_list if '_crop' not in fs.get_file_name(v)]
    crop_df = pd.read_csv(os.path.join(indir, "crop.csv"), sep=";")
    for idx,  video in enumerate(tqdm(video_list)):
        video_name = fs.get_file_name(video, split_file_ext=True)[0]
        df_row = crop_df[crop_df['video'] == video_name] # single row
        if df_row.empty:
            print(f"No crop info for {video_name}, skipping.")
            continue
        x1, y1, x2, y2 = df_row.iloc[0][['x1', 'y1', 'x2', 'y2']]
        # use ffmpeg to crop video
        out_video = os.path.join(indir, f"{video_name}_crop.mp4")
        if os.path.exists(out_video):
            print(f"{out_video} exists, skipping.")
            continue
        cmd = f"ffmpeg -i {video} -filter:v \"crop={x2 - x1}:{y2 - y1}:{x1}:{y1}\" -c:a copy {out_video} -y"
        os.system(cmd)
        print(f"Cropped video saved to {out_video}")


if __name__ == "__main__":
    main()
