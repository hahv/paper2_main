from halib import *
from argparse import ArgumentParser
from halib import *
from halib.utils.video import VideoUtils
import subprocess


import os
import random
from moviepy.editor import VideoFileClip
from loguru import logger


def split_non_overlap_clips(
    video_path, target_dir, num_clips, min_duration, max_duration
):
    """
    Split video into sequential clips, divide into groups = num_clips,
    then pick one random clip per group (if available).

    Args:
        video_path (str): Path to input video.
        target_dir (str): Directory where selected clips will be saved.
        num_clips (int): Desired number of clips to output.
        min_duration (float): Minimum clip duration (sec).
        max_duration (float): Maximum clip duration (sec).

    Returns:
        list[str]: Paths to saved clips.
    """
    os.makedirs(target_dir, exist_ok=True)

    video = VideoFileClip(video_path)
    total_duration = video.duration
    pprint(f"Video duration: {total_duration} seconds")

    # Step 1: Split into sequential clips
    split_clips = []
    start = 0.0
    while start < total_duration:
        duration = random.uniform(min_duration, max_duration)
        end = min(start + duration, total_duration)

        # Merge last small fragment
        if end - start < min_duration and split_clips:
            split_clips[-1] = (split_clips[-1][0], end)
            break

        split_clips.append((start, end))
        start = end
    with ConsoleLog(f"{fs.get_file_name(video_path, split_file_ext=True)[0]}:"):
        print(f"Total {len(split_clips)} split clips created.")

        # Step 2: Divide into groups
        groups = [[] for _ in range(num_clips)]
        for i, clip in enumerate(split_clips):
            group_idx = min(i * num_clips // len(split_clips), num_clips - 1)
            groups[group_idx].append(clip)

        # Step 3: Pick one random clip per group (skip empty groups)
        chosen = []
        for g in groups:
            if g:  # only if group has clips
                chosen.append(random.choice(g))

        # Handle case when we can't get enough
        if len(chosen) < num_clips:
            print(
                f"Warning: only {len(chosen)} clips could be chosen (requested {num_clips})."
            )

        # Step 4: Save chosen clips
        output_paths = []
        video_name = fs.get_file_name(video_path, split_file_ext=True)[0]
        for i, (s, e) in tqdm(enumerate(chosen, 1), "Saving clips"):
            pprint(f"Saving clip {i} start={s}, end={e}")
            clip = video.subclip(s, e)
            out_path = os.path.join(target_dir, f"{video_name}_clip_{i}.mp4")
            if os.path.exists(out_path):
                pprint(f"{out_path} already existed. Skip")
            else:
                # clip.write_videofile(
                #     out_path, codec="libx264", audio_codec="aac", fps=video.fps
                # )
                try:
                    cmd = [
                        "ffmpeg",
                        "-y",
                        "-i",
                        video_path,
                        "-ss",
                        str(s),
                        "-to",
                        str(e),
                        "-c",
                        "copy",
                        out_path,
                    ]
                    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                except:
                    pprint(f"Error when saving clip {i}")
            output_paths.append(out_path)

    return output_paths


def get_video_dict(indir):
    sub_dirs = fs.list_dirs(indir)
    video_subdir_dict = {}
    num_videos = 0
    for sd in sub_dirs:
        video_in_subdir = fs.filter_files_by_extension(
            os.path.join(indir, sd),
            ext=[".mp4", ".avi", ".mov", ".mkv", ".mpg"],
            recursive=False,
        )
        seg_videos = []
        for v in video_in_subdir:
            if "-seg" in v:
                seg_videos.append(v)
        num_videos += len(seg_videos)
        video_subdir_dict[sd] = seg_videos

    pprint(f"Total {num_videos} videos in {len(sub_dirs)} subdirs.")
    return num_videos, video_subdir_dict


def main():
    # INDIR = r"/mnt/d/ZDev/zpaper_db_prof_Park_selected"
    INDIR = r"D:\zDatasets\zpaper_db_prof_Park_selected"
    total_video, video_subdir_dict = get_video_dict(INDIR)
    # OUTDIR = r"/mnt/e/zDatasets/zzNoneFireProf"
    OUTDIR = r"D:\shared_folder\zzNoneFireProf"
    os.makedirs(OUTDIR, exist_ok=True)
    proc_video = 0
    for sub_dir in video_subdir_dict:
        video_ls = video_subdir_dict[sub_dir]
        for test_video in video_ls:
            proc_video += 1
            # test_video = list(video_subdir_dict.values())[0][0]
            console.rule(f"Proc {proc_video}/{total_video}")
            min_duration_secs = 180
            max_duration_secs = 360
            try:
                split_non_overlap_clips(
                    test_video,
                    OUTDIR,
                    num_clips=20,
                    min_duration=min_duration_secs,
                    max_duration=max_duration_secs,
                )
            except Exception as e:
                logger.error(f"{test_video} --- error = {e}")


if __name__ == "__main__":
    main()
