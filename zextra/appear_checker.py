import cv2
import numpy as np
import sys


class AppearanceAnomalyChecker:
    def __init__(self, bg_alpha=0.01, brightness_lift_th=8.0):
        self.bg_alpha = bg_alpha
        self.brightness_lift_th = brightness_lift_th
        self._bg_mean = None

    def update(self, gray: np.ndarray):
        cur_mean = float(np.mean(gray))
        if self._bg_mean is None:
            self._bg_mean = cur_mean
        else:
            self._bg_mean += self.bg_alpha * (cur_mean - self._bg_mean)

    def is_anomalous(self, gray: np.ndarray):
        if self._bg_mean is None:
            return False, 0.0, 0.0
        cur_mean = float(np.mean(gray))
        diff = cur_mean - self._bg_mean
        return diff > self.brightness_lift_th, cur_mean, diff

    def reset(self):
        self._bg_mean = None


def draw_overlay(frame, cur_mean, bg_mean, diff, is_anomalous, frame_idx, th):
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # Semi-transparent dark box background for text
    box_h, box_w = 150, 320
    cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    color_ok      = (100, 220, 100)
    color_warn    = (0, 80, 255)
    color_text    = (220, 220, 220)
    color_label   = (160, 160, 160)

    status_color  = color_warn if is_anomalous else color_ok
    status_text   = "ANOMALY DETECTED" if is_anomalous else "NORMAL"

    lines = [
        (f"Frame:      {frame_idx}",              color_text),
        (f"Cur Mean:   {cur_mean:.2f}",            color_text),
        (f"BG Mean:    {bg_mean:.2f}",             color_label),
        (f"Diff:       {diff:+.2f}  (th={th:.1f})", color_text),
        (f"Status:     {status_text}",              status_color),
    ]

    y = 32
    for text, color in lines:
        cv2.putText(frame, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, color, 1, cv2.LINE_AA)
        y += 26

    # Border flash on anomaly
    if is_anomalous:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color_warn, 3)


def process_video(input_path: str, outdir: str,
                  bg_alpha: float = 0.01, brightness_lift_th: float = 8.0):
    import os
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {input_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    base_name = os.path.basename(input_path)
    output_path = os.path.join(outdir, base_name)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # ty:ignore[unresolved-attribute]
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    checker    = AppearanceAnomalyChecker(bg_alpha, brightness_lift_th)
    frame_idx  = 0
    anomaly_count = 0

    print(f"[INFO] Input : {input_path}")
    print(f"[INFO] Output: {output_path}")
    print(f"[INFO] FPS={fps:.1f}  Size={width}x{height}  Frames={total}")
    print(f"[INFO] bg_alpha={bg_alpha}  brightness_lift_th={brightness_lift_th}")
    print("-" * 60)
    print(f"{'Frame':>6} {'CurMean':>9} {'BGMean':>9} {'Diff':>8} {'Status'}")
    print("-" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scaled_gray = cv2.resize(gray, (width // 4, height // 4), interpolation=cv2.INTER_AREA)

        # Update background BEFORE checking (don't absorb anomaly into bg too
        # fast)
        start_t = cv2.getTickCount()
        checker.update(scaled_gray)
        anomalous, cur_mean, diff = checker.is_anomalous(scaled_gray)
        bg_mean = checker._bg_mean if checker._bg_mean is not None else 0.0
        end_t = cv2.getTickCount()
        elapsed_ms = (end_t - start_t) / cv2.getTickFrequency() * 1000
        print(f"Time for frame {frame_idx}: {elapsed_ms:.2f} ms", end="\r")

        if anomalous:
            anomaly_count += 1

        draw_overlay(frame, cur_mean, bg_mean, diff, anomalous,
                     frame_idx, brightness_lift_th)
        writer.write(frame)

        # Print every 30 frames (or on anomaly)
        if frame_idx % 30 == 0 or anomalous:
            status = "!! ANOMALY" if anomalous else "   normal"
            print(f"{frame_idx:>6} {cur_mean:>9.2f} {bg_mean:>9.2f} "
                  f"{diff:>+8.2f}  {status}")

        frame_idx += 1

    cap.release()
    writer.release()

    print("-" * 60)
    print(f"[DONE] Total frames: {frame_idx}  |  Anomaly frames: {anomaly_count}"
          f"  ({100*anomaly_count/max(frame_idx,1):.1f}%)")
    print(f"[DONE] Saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    import os
    from rich.console import Console

    p = argparse.ArgumentParser(description="Appearance Anomaly Checker")
    p.add_argument("--input", type=str, required=True, help="Input video directory")
    p.add_argument("--bg_alpha", type=float, default=0.01, help="EMA speed (0.001-0.05)")
    p.add_argument("--brightness_lift_th", type=float, default=8.0, help="diff threshold to flag anomaly")
    args = p.parse_args()

    indir = args.input
    outdir = os.path.join(indir, "out")
    os.makedirs(outdir, exist_ok=True)

    console = Console()
    from halib.system import filesys as fs
    video_files = fs.filter_files_by_extension(indir, [".mp4", ".avi", ".mov", ".mkv"])

    for video_path in video_files:
        console.rule(f"\nProcessing video: {video_path}")
        process_video(video_path, outdir, args.bg_alpha, args.brightness_lift_th)
