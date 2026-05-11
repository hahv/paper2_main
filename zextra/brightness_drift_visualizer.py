#!/usr/bin/env python3
"""
Brightness Drift Visualizer (Method 2 — Temporal Brightness Drift)
Based on haze_visualizer.py

For each video in in_dir, runs the Temporal Brightness Drift detector:
  - Maintains a rolling buffer of K frames
  - On every no-motion frame, computes:
      delta_brightness = |mean_V(current) - mean_V(rolling_buffer)|
      delta_color      = |mean_H(current) - mean_H(rolling_buffer)|
  - Triggers FORCE PROBE if either delta exceeds its threshold

Output per video: CSV summary + optional side-by-side visualisation video

Usage:
    python brightness_drift_visualizer.py <in_dir>
        [--buf_len 15]
        [--th_brightness 0.05]
        [--th_color 0.04]
        [--diff_thresh 5]
        [--small_width 120]
        [--vis]
"""

import argparse
import glob
import os
import time
from collections import deque

import cv2
import numpy as np

try:
    from halib.filetype.csvfile import DFCreator
    from halib import pprint_box, tqdm

    HAS_HALIB = True
except ImportError:
    HAS_HALIB = False
    from tqdm import tqdm as _tqdm

    def tqdm(x):
        return _tqdm(x)

    def pprint_box(d, title=""):
        print(f"\n{'=' * 40}\n{title}\n{d}\n{'=' * 40}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FONT = cv2.FONT_HERSHEY_DUPLEX
FSCALE = 0.60
THICKNESS = 2
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (0, 0, 255)
GREEN = (0, 200, 0)
YELLOW = (0, 200, 200)


def _text(img, text, org, color=RED, scale=FSCALE, thick=THICKNESS):
    """Shadow text for readability."""
    cv2.putText(
        img, text, (org[0] + 1, org[1] + 1), FONT, scale, BLACK, thick + 1, cv2.LINE_AA
    )
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Temporal Brightness Drift Score
# ---------------------------------------------------------------------------


class TemporalBrightnessDrift:
    """
    Rolling-buffer temporal brightness + hue drift detector.

    Every frame call update(frame_bgr, has_motion).
    Returns (mean_v, mean_h, delta_v, delta_h, force_probe).

    Logic:
      - Buffer stores (mean_V, mean_H) of last `buf_len` frames.
      - On a no-motion frame:
            delta_V = |V_current - mean(V_buffer)|
            delta_H = |H_current - mean(H_buffer)|   (circular, 0-1 range)
            force_probe = (delta_V > th_b) OR (delta_H > th_c)
      - Buffer is updated every frame (motion or not) to stay current.
    """

    def __init__(
        self,
        buf_len: int = 15,
        th_brightness: float = 0.05,
        th_color: float = 0.04,
        small_width: int = 640,
    ):
        self.buf_len = buf_len
        self.th_b = th_brightness
        self.th_c = th_color
        self.small_width = small_width
        self._buf_v: deque = deque(maxlen=buf_len)
        self._buf_h: deque = deque(maxlen=buf_len)

    # def _hsv_stats(self, frame_bgr: np.ndarray):
    #     """Return (mean_V, mean_H) in [0,1] from a downsampled BGR frame."""
    #     h, w = frame_bgr.shape[:2]
    #     new_w = min(self.small_width, w)
    #     new_h = max(1, int(h * new_w / w))
    #     small = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    #     hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.float32)
    #     mean_v = float(np.mean(hsv[:, :, 2])) / 255.0
    #     mean_h = float(np.mean(hsv[:, :, 0])) / 180.0
    #     return mean_v, mean_h

    def _hsv_stats(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        new_w = min(self.small_width, w)
        new_h = max(1, int(h * new_w / w))
        small  = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        # Use grayscale mean instead of HSV — no cvtColor needed
        gray   = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        mean_v = float(np.mean(gray)) / 255.0
        return mean_v, 0.0   # hue always 0 — disable delta_H

    @staticmethod
    def _circular_dist(a: float, b: float) -> float:
        """Circular distance for hue in [0,1] space (wraps at 1)."""
        d = abs(a - b)
        return min(d, 1.0 - d)

    def update(self, frame_bgr: np.ndarray, has_motion: bool):
        """
        Update buffer and compute drift.

        Returns:
            mean_v      : current frame mean brightness (0-1)
            mean_h      : current frame mean hue (0-1)
            delta_v     : brightness drift from buffer mean (0-1)
            delta_h     : hue drift from buffer mean (0-1, circular)
            force_probe : True if no-motion AND drift exceeds thresholds
        """
        mean_v, mean_h = self._hsv_stats(frame_bgr)

        force_probe = False
        delta_v = 0.0
        delta_h = 0.0

        if len(self._buf_v) > 0:
            buf_mean_v = float(np.mean(self._buf_v))
            buf_mean_h = float(np.mean(self._buf_h))
            delta_v = abs(mean_v - buf_mean_v)
            delta_h = self._circular_dist(mean_h, buf_mean_h)

            if not has_motion:
                force_probe = (delta_v > self.th_b) or (delta_h > self.th_c)

        self._buf_v.append(mean_v)
        self._buf_h.append(mean_h)

        return mean_v, mean_h, delta_v, delta_h, force_probe

    def reset(self):
        self._buf_v.clear()
        self._buf_h.clear()


# ---------------------------------------------------------------------------
# Motion gate
# ---------------------------------------------------------------------------


def has_motion_simple(
    gray_curr: np.ndarray, gray_prev: np.ndarray, diff_thresh: int = 5
) -> bool:
    """True if mean absolute frame-diff > diff_thresh."""
    if gray_prev is None:
        return False
    diff = cv2.absdiff(gray_curr, gray_prev)
    return float(np.mean(diff)) > diff_thresh


# ---------------------------------------------------------------------------
# Visualisation overlay
# ---------------------------------------------------------------------------


def draw_overlays(composite, info: dict, frame_w: int, frame_h: int):
    """Draw HUD on the composite frame (in-place)."""
    lines = [
        f"Motion       : {'YES' if info['motion'] else 'NO'}",
        f"mean_V       : {info['mean_v']:.3f}",
        f"mean_H       : {info['mean_h']:.3f}",
        f"delta_V      : {info['delta_v']:.4f}  (th={info['th_b']:.3f})",
        f"delta_H      : {info['delta_h']:.4f}  (th={info['th_c']:.3f})",
        f"Buf len      : {info['buf_len']}",
        f"Avg drift t  : {info['avg_ms']:.3f} ms/frame",
    ]
    y = 28
    for line in lines:
        _text(composite, line, (10, y))
        y += 26

    if info["force_probe"]:
        label = "FORCE PROBE (DRIFT)"
        bscale = 1.5
        bthick = 3
        (tw, th), _ = cv2.getTextSize(label, FONT, bscale, bthick)
        cx = (composite.shape[1] // 2) - tw // 2
        cy = (frame_h // 2) + th // 2
        cv2.putText(
            composite,
            label,
            (cx + 2, cy + 2),
            FONT,
            bscale,
            BLACK,
            bthick + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            composite, label, (cx, cy), FONT, bscale, YELLOW, bthick, cv2.LINE_AA
        )

    bar_h = 8
    bar_y = frame_h - bar_h - 4
    bar_w_v = int(min(1.0, info["delta_v"] / max(info["th_b"], 1e-6)) * frame_w)
    bar_w_h = int(min(1.0, info["delta_h"] / max(info["th_c"], 1e-6)) * frame_w)
    cv2.rectangle(composite, (0, bar_y), (bar_w_v, bar_y + bar_h // 2), GREEN, -1)
    cv2.rectangle(
        composite, (0, bar_y + bar_h // 2), (bar_w_h, bar_y + bar_h), YELLOW, -1
    )


# ---------------------------------------------------------------------------
# Process one video
# ---------------------------------------------------------------------------


def process_video(video_path: str, out_dir: str, args) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [WARN] Cannot open {video_path}, skipping.")
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    base = os.path.splitext(os.path.basename(video_path))[0]

    print(f"  Processing : {os.path.basename(video_path)}")
    print(f"  Resolution : {w}x{h} @ {fps:.1f} fps (~{total} frames)")

    writer = None
    if args.vis:
        out_path = os.path.join(out_dir, f"{base}_drift.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w * 2, h))

    detector = TemporalBrightnessDrift(
        buf_len=args.buf_len,
        th_brightness=args.th_brightness,
        th_color=args.th_color,
        small_width=args.small_width,
    )

    prev_gray = None
    drift_times = []
    frame_idx = 0
    force_probe_cnt = 0
    no_motion_cnt = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        motion = has_motion_simple(gray, prev_gray, args.diff_thresh)
        if not motion:
            no_motion_cnt += 1
        prev_gray = gray

        t0 = time.perf_counter()
        mean_v, mean_h, delta_v, delta_h, force_probe = detector.update(frame, motion)
        drift_times.append((time.perf_counter() - t0) * 1000.0)

        if force_probe:
            force_probe_cnt += 1

        if writer is not None:
            right = frame.copy()
            info = dict(
                motion=motion,
                mean_v=mean_v,
                mean_h=mean_h,
                delta_v=delta_v,
                delta_h=delta_h,
                th_b=args.th_brightness,
                th_c=args.th_color,
                buf_len=args.buf_len,
                avg_ms=float(np.mean(drift_times)),
                force_probe=force_probe,
            )
            draw_overlays(right, info, w, h)
            composite = np.concatenate([frame, right], axis=1)
            cv2.line(composite, (w, 0), (w, h), (180, 180, 180), 1)
            writer.write(composite)

    cap.release()
    if writer:
        writer.release()

    avg_drift_ms = float(np.mean(drift_times)) if drift_times else 0.0
    video_name = os.path.basename(video_path)
    video_type = "safe_video" if "__lb_none__" in video_name else "firesmoke_video"
    probe_rate_pct = (force_probe_cnt / max(no_motion_cnt, 1)) * 100.0

    summary = {
        "video_name": video_name,
        "video_type": video_type,
        "total_frames": frame_idx,
        "no_motion_frames": no_motion_cnt,
        "force_probe_frames": force_probe_cnt,
        "probe_rate_pct": round(probe_rate_pct, 2),
        "avg_drift_ms": round(avg_drift_ms, 4),
    }
    pprint_box(summary, title="Video Summary")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Temporal Brightness Drift Detector — Method 2 smoke gate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("in_dir", help="Input directory containing video files")
    parser.add_argument(
        "--buf_len", type=int, default=15, help="Rolling buffer length K"
    )
    parser.add_argument(
        "--th_brightness", type=float, default=0.05, help="Delta-V threshold (0-1)"
    )
    parser.add_argument(
        "--th_color", type=float, default=0.04, help="Delta-H threshold (0-1, circular)"
    )
    parser.add_argument(
        "--diff_thresh",
        type=int,
        default=5,
        help="Motion gate mean-diff threshold (uint8 pixels)",
    )
    parser.add_argument(
        "--small_width",
        type=int,
        default=120,
        help="Resize width (px) before HSV analysis",
    )
    parser.add_argument(
        "--vis", action="store_true", help="Write side-by-side debug video"
    )
    args = parser.parse_args()

    in_dir = os.path.abspath(args.in_dir)
    if not os.path.isdir(in_dir):
        print(f"[ERROR] '{in_dir}' is not a valid directory.")
        return

    out_dir = os.path.join(in_dir, "out")
    os.makedirs(out_dir, exist_ok=True)

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
    print("  Temporal Brightness Drift Visualizer")
    print(f"  Found     : {len(videos)} video(s) in '{in_dir}'")
    print(
        f"  Settings  : buf_len={args.buf_len}  th_V={args.th_brightness}"
        f"  th_H={args.th_color}  diff_thresh={args.diff_thresh}"
        f"  small_width={args.small_width}px  vis={args.vis}"
    )
    print(f"  Output    : {out_dir}")
    print("=" * 60)

    rows = []
    for vp in tqdm(videos):
        print(f"\nProcessing video: {os.path.basename(vp)}")
        row = process_video(vp, out_dir, args)
        if row:
            rows.append(row)

    if rows:
        import csv

        csv_path = os.path.join(in_dir, "drift_summary.csv")
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[INFO] Summary saved -> {csv_path}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
