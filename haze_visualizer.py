#!/usr/bin/env python3
"""
Haze Detection Visualizer
For each video in in_dir, produces a side-by-side output video:
  Left  : original RGB frame
  Right : FrameDiff foreground mask
  Overlay: haze score, haze threshold, avg haze time, and HAZE DETECTED (red, center) if triggered

Usage:
  python haze_visualizer.py <in_dir> [--diff_thresh 15] [--haze_thresh 60]
                            [--haze_resize 80] [--haze_patch 5]
"""

from halib.filetype.csvfile import DFCreator

import argparse
import os
import time
import glob
import cv2
import numpy as np
from halib import *


# --------------------------------------------------------------------------- #
# Dark Channel Haze Score
# --------------------------------------------------------------------------- #
# def dark_channel_score(
#     frame_bgr: np.ndarray, resize_w: int = 80, patch_size: int = 5
# ) -> float:
#     """
#     Compute the mean dark channel of a BGR frame.
#     Higher score = more haze / whitish content.

#     Args:
#         frame_bgr  : input BGR frame (any resolution)
#         resize_w   : width to downsample to before computing (smaller = faster)
#                      Recommended: 80 (~0.3-0.5ms), 160 (~1ms), 320 (~4ms)
#         patch_size : erosion kernel size for neighbourhood minimum
#                      Recommended: 5 for resize_w=80, 15 for resize_w=320
#     """
#     h, w = frame_bgr.shape[:2]
#     scale = min(1.0, resize_w / w)
#     if scale < 1.0:
#         new_w = int(w * scale)
#         new_h = int(h * scale)
#         small = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
#     else:
#         small = frame_bgr

#     # Per-pixel minimum across B, G, R channels
#     min_channel = np.min(small, axis=2).astype(np.uint8)

#     # Minimum filter over patch_size x patch_size neighbourhood (erosion)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
#     dark = cv2.erode(min_channel, kernel)

#     return float(np.mean(dark))


def dark_channel_score(
    frame_bgr: np.ndarray, resize_w: int = 80, patch_size: int = 5
) -> float:
    """
    Compute the mean dark channel of a BGR frame.
    Higher score = more haze / whitish content.

    Args:
        frame_bgr  : input BGR frame (any resolution)
        resize_w   : width to downsample to before computing (smaller = faster)
                     Recommended: 80 (~0.3-0.5ms), 160 (~1ms), 320 (~4ms)
        patch_size : erosion kernel size for neighbourhood minimum
                     Recommended: 5 for resize_w=80, 15 for resize_w=320
    """
    small_frame = cv2.resize(frame_bgr, (64, 36), interpolation=cv2.INTER_NEAREST)

    # 2. Get the minimum value across the color channels (axis=2)
    # This creates our approximated "Dark Channel"
    dark_channel = np.min(small_frame, axis=2)

    # 3. The single score is the average of this dark channel.
    # Range is 0.0 to 255.0
    haze_score = np.mean(dark_channel)

    return haze_score


# --------------------------------------------------------------------------- #
# FrameDiff foreground mask
# --------------------------------------------------------------------------- #
def framediff_mask(
    gray_curr: np.ndarray, gray_prev: np.ndarray, diff_thresh: int
) -> np.ndarray:
    """Return a binary uint8 mask (0 or 255) of pixels that changed."""
    diff = cv2.absdiff(gray_curr, gray_prev)
    _, mask = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)
    return mask


# --------------------------------------------------------------------------- #
# Draw helpers
# --------------------------------------------------------------------------- #
FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE = 0.65
THICKNESS = 2
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (0, 0, 255)  # BGR


def put_text_shadow(img, text, org, color=WHITE, scale=FONT_SCALE, thickness=THICKNESS):
    """Draw text with a thin black shadow for readability on any background."""
    cv2.putText(
        img,
        text,
        (org[0] + 1, org[1] + 1),
        FONT,
        scale,
        BLACK,
        thickness + 1,
        cv2.LINE_AA,
    )
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_overlays(
    composite: np.ndarray,
    haze_score: float,
    haze_thresh: float,
    avg_haze_ms: float,
    resize_w: int,
    patch_size: int,
    is_hazy: bool,
    frame_w: int,
    frame_h: int,
):
    """Overlay all text onto the composite (in-place)."""
    # -- top-left info block --------------------------------------------------
    lines = [
        f"Haze Score  : {haze_score:.1f}",
        f"Haze Thresh : {haze_thresh:.1f}",
        f"Avg Haze T  : {avg_haze_ms:.3f} ms/frame",
        f"Haze Resize : {resize_w}px  Patch: {patch_size}px",
    ]
    y = 28
    for line in lines:
        put_text_shadow(composite, line, (10, y))
        y += 26

    # -- center HAZE DETECTED -------------------------------------------------
    if is_hazy:
        label = "HAZE DETECTED"
        big_scale = 1.8
        big_thick = 3
        (tw, th), _ = cv2.getTextSize(label, FONT, big_scale, big_thick)
        cx = (composite.shape[1] // 2) - tw // 2
        cy = (frame_h // 2) + th // 2
        # shadow
        cv2.putText(
            composite,
            label,
            (cx + 2, cy + 2),
            FONT,
            big_scale,
            BLACK,
            big_thick + 2,
            cv2.LINE_AA,
        )
        # red text
        cv2.putText(
            composite, label, (cx, cy), FONT, big_scale, RED, big_thick, cv2.LINE_AA
        )


# --------------------------------------------------------------------------- #
# Process one video
# --------------------------------------------------------------------------- #
def process_video(
    video_path: str,
    out_dir: str,
    diff_thresh: int,
    haze_thresh: float,
    haze_resize: int,
    haze_patch: int,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [WARN] Cannot open {video_path}, skipping.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    base = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(out_dir, f"{base}_out.mp4")

    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # ty:ignore[unresolved-attribute]
    # writer = cv2.VideoWriter(out_path, fourcc, fps, (w * 2, h))

    prev_gray = None
    haze_times = []
    frame_idx = 0
    # running_avg_ms = 0.0

    print(f"  Processing : {os.path.basename(video_path)}")
    print(f"  Resolution : {w}x{h} @ {fps:.1f} fps  (~{total} frames)")
    print(
        f"  Haze params: resize_w={haze_resize}px  patch={haze_patch}px  thresh={haze_thresh}"
    )
    haze_frame_count = 0
    haze_scores = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # -- Haze score -------------------------------------------------------
        t0 = time.perf_counter()
        haze_score = dark_channel_score(
            frame, resize_w=haze_resize, patch_size=haze_patch
        )
        haze_times.append((time.perf_counter() - t0) * 1000.0)
        # running_avg_ms = float(np.mean(haze_times))

        is_hazy = haze_score > haze_thresh
        if is_hazy:
            haze_frame_count += 1
        haze_scores.append(haze_score)

        # # -- FrameDiff mask ---------------------------------------------------
        # if prev_gray is not None:
        #     mask = framediff_mask(gray, prev_gray, diff_thresh)
        # else:
        #     mask = np.zeros((h, w), dtype=np.uint8)

        # prev_gray = gray

        # # -- Build composite --------------------------------------------------
        # mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # composite = np.concatenate([frame, mask_bgr], axis=1)

        # # -- Separator line ---------------------------------------------------
        # cv2.line(composite, (w, 0), (w, h), (180, 180, 180), 1)

        # # -- Overlays ---------------------------------------------------------
        # draw_overlays(
        #     composite,
        #     haze_score,
        #     haze_thresh,
        #     running_avg_ms,
        #     haze_resize,
        #     haze_patch,
        #     is_hazy,
        #     w,
        #     h,
        # )

        # writer.write(composite)
    video_name = os.path.basename(video_path)
    total_frame = frame_idx
    avg_haze_time_ms = float(np.mean(haze_times)) if haze_times else 0
    avg_haze_score = float(np.mean(haze_scores)) if haze_scores else 0
    total_hazy_frames = haze_frame_count
    # ["video_name", "avg_haze_time_ms", "avg_haze_score", "total_frames", "total_hazy_frames"])

    cap.release()
    # writer.release()

    # avg_ms = float(np.mean(haze_times)) if haze_times else 0.0
    # print(
    #     f"  Result     : {frame_idx} frames | "
    #     f"Avg haze time = {avg_ms:.3f} ms/frame | "
    #     f"Saved -> {out_path}"
    # )
    video_type = "safe_video" if "__lb_none__" in video_name else "firesmoke_video"
    row = [
        video_name,
        video_type,
        avg_haze_time_ms,
        avg_haze_score,
        total_frame,
        total_hazy_frames,
    ]
    dict_data = {
        "video_name": video_name,
        "video_type": video_type,
        "avg_haze_time_ms": avg_haze_time_ms,
        "avg_haze_score": avg_haze_score,
        "total_frames": total_frame,
        "total_hazy_frames": total_hazy_frames,
    }
    pprint_box(dict_data, title="Video Summary")
    return row


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Haze Detection Visualizer — side-by-side RGB + FrameDiff mask",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("in_dir", help="Input directory containing video files")
    parser.add_argument(
        "--diff_thresh",
        type=int,
        default=15,
        help="FrameDiff pixel-difference threshold",
    )
    parser.add_argument(
        "--haze_thresh",
        type=float,
        default=60.0,
        help="Dark-channel haze score threshold for HAZE DETECTED",
    )
    parser.add_argument(
        "--haze_resize",
        type=int,
        default=80,
        help="Width (px) to downsample frame before dark channel "
        "[80=fastest ~0.3ms | 160=~1ms | 320=~4ms]",
    )
    parser.add_argument(
        "--haze_patch",
        type=int,
        default=5,
        help="Erosion patch size for dark channel "
        "[5 for resize=80 | 15 for resize=320]",
    )

    args = parser.parse_args()

    in_dir = os.path.abspath(args.in_dir)
    if not os.path.isdir(in_dir):
        print(f"[ERROR] '{in_dir}' is not a valid directory.")
        return

    out_dir = os.path.join(in_dir, "out")
    os.makedirs(out_dir, exist_ok=True)

    # Collect all video files
    exts = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.wmv", "*.flv", "*.m4v")
    videos = []
    for ext in exts:
        videos += glob.glob(os.path.join(in_dir, ext))
        videos += glob.glob(os.path.join(in_dir, ext.upper()))
    videos = sorted(set(videos))

    if not videos:
        print(f"[INFO] No video files found in '{in_dir}'.")
        return

    print("=" * 60)
    print(f"  Haze Detection Visualizer")
    print(f"  Found    : {len(videos)} video(s) in '{in_dir}'")
    print(
        f"  Settings : diff_thresh={args.diff_thresh}  "
        f"haze_thresh={args.haze_thresh}  "
        f"haze_resize={args.haze_resize}px  "
        f"haze_patch={args.haze_patch}px"
    )
    print(f"  Output   : {out_dir}")
    print("=" * 60)
    dfmk = DFCreator()
    dfmk.create_table(
        "haze_videos",
        [
            "video_name",
            "video_type",
            "avg_haze_time_ms",
            "avg_haze_score",
            "total_frames",
            "total_hazy_frames",
        ],
    )
    rows = []
    for vp in tqdm(videos):
        print(f"\nProcessing video: {os.path.basename(vp)}")
        row = process_video(
            vp,
            out_dir,
            args.diff_thresh,
            args.haze_thresh,
            args.haze_resize,
            args.haze_patch,
        )
        rows.append(row)
    dfmk.insert_rows("haze_videos", rows)
    dfmk.fill_table_from_row_pool("haze_videos")
    dfmk["haze_videos"].to_csv(
        os.path.join(in_dir, "haze_video_summary.csv"),
        sep=";",
        encoding="utf-8",
        index=False,
    )

    print("All done.")


if __name__ == "__main__":
    main()
