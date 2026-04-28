#!/usr/bin/env python3
"""
Smoke/White-ish Frame Detector via Saturation Drift
-----------------------------------------------------
Processes a video file and flags frames that should NOT be skipped
because they contain white-ish / smoke-settled content.

Usage:
    python smoke_frame_detector.py --input video.mp4 --output result.mp4

Arguments:
    --input        : Path to input video file
    --output       : Path to output annotated video (default: output_annotated.mp4)
    --scale        : Downscale factor for fast computation (default: 0.25)
    --abs_thresh   : Absolute whitish ratio threshold (default: 0.25)
    --sat_thresh   : Saturation level below which a pixel is "white-ish" (default: 30)
    --val_thresh   : Brightness level above which a pixel is "bright" (default: 180)
    --history_len  : Number of frames in rolling history for drift (default: 30)
    --drift_thresh : Saturation drift from baseline to flag smoke (default: 15.0)
    --min_history  : Minimum frames before drift check activates (default: 10)
"""

import cv2
import numpy as np
import argparse
import time
from collections import deque


# ─────────────────────────────────────────────
# Core Detection Functions
# ─────────────────────────────────────────────

def whitish_score(small_hsv: np.ndarray, sat_thresh: int = 30, val_thresh: int = 180) -> float:
    """
    Fast per-frame whitish ratio.
    Returns fraction of pixels that are bright AND low-saturation.
    Smoke/haze desaturates the scene while keeping it bright.
    Camera blur does NOT significantly desaturate.
    """
    S = small_hsv[:, :, 1]
    V = small_hsv[:, :, 2]
    mask = (S < sat_thresh) & (V > val_thresh)
    return float(mask.mean())


def compute_sat_stats(small_hsv: np.ndarray):
    """Returns (mean, std) of saturation channel."""
    S = small_hsv[:, :, 1].astype(np.float32)
    return float(S.mean()), float(S.std())

from halib import *

class SaturationDriftDetector:
    """
    Tracks rolling saturation history and detects gradual drops
    (smoke settling, which has no motion but desaturates the scene).
    This is immune to cameras that are always blurry — because the
    baseline itself already includes the blur constant offset.
    """
    DRIFT_LIST = []
    def __init__(self, history_len: int = 5, drift_thresh: float = 15.0, min_history: int = 2):
        self.history = deque(maxlen=history_len)
        self.drift_thresh = drift_thresh
        self.min_history = min_history

    def update(self, sat_mean: float) -> tuple:
        """
        Push new saturation mean. Returns (is_smoke_drift, drift_value).
        drift > drift_thresh => saturation dropped from baseline => smoke settling.
        """
        self.history.append(sat_mean)
        if len(self.history) < self.min_history:
            return False, 0.0

        # Use the max saturation seen in the window as the "clean baseline"
        baseline = float(max(self.history))
        drift = baseline - sat_mean
        self.DRIFT_LIST.append(abs(drift))
        print(f"{drift=:.1f}", end="\r")
        # print('baseline=%.1f  current=%.1f  drift=%.1f' % (baseline, sat_mean, drift))
        # pprint('running drift history (abs): ' + str(np.mean(self.DRIFT_LIST)))
        return drift > self.drift_thresh, drift


# ─────────────────────────────────────────────
# Frame Classification
# ─────────────────────────────────────────────

def classify_frame(
    frame: np.ndarray,
    drift_detector: SaturationDriftDetector,
    scale: float = 0.25,
    abs_thresh: float = 0.25,
    sat_thresh: int = 30,
    val_thresh: int = 180,
) -> dict:
    """
    Runs all checks on a single frame. Returns a result dict:
      - whitish_ratio  : float  (per-frame bright+desaturated pixel ratio)
      - sat_mean       : float
      - sat_std        : float
      - drift          : float  (sat drop from baseline)
      - is_whitish_abs : bool   (absolute threshold triggered)
      - is_drift       : bool   (drift threshold triggered)
      - do_not_skip    : bool   (True => run inference, smoke/whitish detected)
      - label          : str    (human-readable classification)
      - elapsed_ms     : float  (compute time in ms)
    """
    t0 = time.perf_counter()

    small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    ratio = whitish_score(hsv, sat_thresh=sat_thresh, val_thresh=val_thresh)
    sat_mean, sat_std = compute_sat_stats(hsv)
    is_drift, drift = drift_detector.update(sat_mean)

    is_whitish_abs = ratio > abs_thresh
    do_not_skip = is_whitish_abs or is_drift

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Human-readable classification
    if is_whitish_abs and is_drift:
        label = "SMOKE (abs+drift)"
    elif is_whitish_abs:
        label = "SMOKE (abs)"
    elif is_drift:
        label = "SMOKE (drift)"
    elif sat_mean > 40 and sat_std < 25:
        label = "BLUR (clean)"
    else:
        label = "CLEAN"

    return {
        "whitish_ratio": ratio,
        "sat_mean": sat_mean,
        "sat_std": sat_std,
        "drift": drift,
        "is_whitish_abs": is_whitish_abs,
        "is_drift": is_drift,
        "do_not_skip": do_not_skip,
        "label": label,
        "elapsed_ms": elapsed_ms,
    }


# ─────────────────────────────────────────────
# Overlay Drawing
# ─────────────────────────────────────────────

COLORS = {
    "CLEAN":              (50, 200, 50),    # green
    "BLUR (clean)":       (200, 160, 50),   # amber
    "SMOKE (abs)":        (0, 60, 220),     # red
    "SMOKE (drift)":      (0, 100, 255),    # orange-red
    "SMOKE (abs+drift)":  (0, 0, 255),      # bright red
}


def draw_overlay(frame: np.ndarray, result: dict, frame_idx: int) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    label = result["label"]
    color = COLORS.get(label, (200, 200, 200))
    do_not_skip = result["do_not_skip"]

    # Top status banner
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, 200), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)

    # Label text
    cv2.putText(out, f"Status: {label}", (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

    # Big metrics text at top-left
    metrics_lines = [
        f"Frame Index      : {frame_idx:05d}",
        f"Whitish Ratio    : {result['whitish_ratio']:.3f}",
        f"Saturation Mean  : {result['sat_mean']:.1f}",
        f"Saturation Std   : {result['sat_std']:.1f}",
        f"Saturation Drift : {result['drift']:.1f}",
        f"Compute Time     : {result['elapsed_ms']:.2f} ms"
    ]
    
    y_offset = 70
    for line in metrics_lines:
        cv2.putText(out, line, (12, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (210, 210, 210), 2, cv2.LINE_AA)
        y_offset += 25

    # Colored border when smoke detected
    if do_not_skip:
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), color, 6)

    return out


# ─────────────────────────────────────────────
# Main Video Processor
# ─────────────────────────────────────────────

def process_video(args, video_path: str, outdir: str):
    import os
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    base_name = os.path.basename(video_path)
    output_path = os.path.join(outdir, base_name)

    print(f"[INFO] Input : {video_path}")
    print(f"[INFO] Size  : {orig_w}x{orig_h}  FPS: {fps:.1f}  Frames: {total_frames}")
    print(f"[INFO] Output: {output_path}")
    print(f"[INFO] Params: scale={args.scale}, abs_thresh={args.abs_thresh}, "
          f"drift_thresh={args.drift_thresh}, history={args.history_len}")
    print()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # ty:ignore[unresolved-attribute]
    writer = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))

    drift_detector = SaturationDriftDetector(
        history_len=args.history_len,
        drift_thresh=args.drift_thresh,
        min_history=args.min_history,
    )

    frame_idx = 0
    smoke_count = 0
    clean_count = 0
    total_ms = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = classify_frame(
            frame,
            drift_detector,
            scale=args.scale,
            abs_thresh=args.abs_thresh,
            sat_thresh=args.sat_thresh,
            val_thresh=args.val_thresh,
        )

        total_ms += result["elapsed_ms"]
        if result["do_not_skip"]:
            smoke_count += 1
        else:
            clean_count += 1

        annotated = draw_overlay(frame, result, frame_idx)
        writer.write(annotated)

        if frame_idx % 100 == 0:
            avg_ms = total_ms / (frame_idx + 1)
            pct = (frame_idx / max(total_frames, 1)) * 100
            print(f"  [{pct:5.1f}%] Frame {frame_idx:05d} | "
                  f"{result['label']:<22} | "
                  f"ratio={result['whitish_ratio']:.3f}  "
                  f"drift={result['drift']:.1f}  "
                  f"avg={avg_ms:.2f}ms/frame")

        frame_idx += 1

    cap.release()
    writer.release()

    avg_ms = total_ms / max(frame_idx, 1)
    print()
    print("=" * 60)
    print(f"  Total frames   : {frame_idx}")
    print(f"  Smoke/whitish  : {smoke_count}  ({100*smoke_count/max(frame_idx,1):.1f}%)")
    print(f"  Clean/blur     : {clean_count}  ({100*clean_count/max(frame_idx,1):.1f}%)")
    print(f"  Avg compute    : {avg_ms:.3f} ms/frame  (at scale={args.scale})")
    print(f"  Output saved   : {args.output}")
    print("=" * 60)


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Smoke/White-ish Frame Detector — Saturation Drift Method"
    )
    p.add_argument("--input",        type=str,   required=True,  help="Input video path")
    p.add_argument("--output",       type=str,   default="output_annotated.mp4", help="Output video path")
    p.add_argument("--scale",        type=float, default=0.25,   help="Frame downscale factor (default: 0.25)")
    p.add_argument("--abs_thresh",   type=float, default=0.25,   help="Abs whitish pixel ratio to flag smoke (default: 0.25)")
    p.add_argument("--sat_thresh",   type=int,   default=30,     help="Saturation < this = white-ish pixel (default: 30)")
    p.add_argument("--val_thresh",   type=int,   default=180,    help="Brightness > this = bright pixel (default: 180)")
    p.add_argument("--history_len",  type=int,   default=30,     help="Rolling history window in frames (default: 30)")
    p.add_argument("--drift_thresh", type=float, default=15.0,   help="Sat drift from baseline to flag smoke (default: 15.0)")
    p.add_argument("--min_history",  type=int,   default=10,     help="Min frames before drift activates (default: 10)")
    return p.parse_args()


if __name__ == "__main__":
    import os
    from rich.console import Console
    console = Console()
    args = parse_args()
    indir = args.input
    outdir = os.path.join(indir, "out")
    os.makedirs(outdir, exist_ok=True)

    from halib.system import filesys as fs
    video_files = fs.filter_files_by_extension(indir, [".mp4", ".avi", ".mov", ".mkv"])
    for video_path in video_files:
        console.rule(f"\nProcessing video: {video_path}")
        process_video(args, video_path, outdir)
