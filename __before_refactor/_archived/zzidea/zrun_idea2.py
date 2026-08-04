import cv2
import numpy as np
import pywt
from collections import deque
from halib import *  # noqa: F403
from tap import *
from halib.exp.perf.profiler import zProfiler


class RunIdeaArgs(Tap):
    video: str = r"./datasets/UFireIndoor/firesmoke/aihub__lb_fire_0178.mp4"
    block_size: int = 32
    scale_factor: float = 1.0  # New argument for scaling


class SimpleFrameDifference:
    """
    A simple Frame Difference implementation using OpenCV to replace MOG2/pybgs.
    Logic: |Current_Frame - Previous_Frame| > Threshold
    """

    def __init__(self, diff_threshold=5):
        self.diff_threshold = diff_threshold
        self.prev_frame = None

    def apply(self, frame_bgr):
        # Convert to grayscale for simple differencing
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self.prev_frame is None:
            self.prev_frame = gray
            return np.zeros_like(gray)

        # Compute absolute difference
        diff = cv2.absdiff(self.prev_frame, gray)

        # Threshold to create binary mask
        _, fgmask = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)

        # Update background
        self.prev_frame = gray

        return fgmask


class BlockAnalyzer:
    def __init__(self, history_len=30):
        self.spatial_energy_history = {}
        self.background_energy_history = {}
        self.history_len = history_len

    def get_spatial_wavelet_energy(self, roi, channel="r"):
        try:
            if channel == "r":
                c_idx = 2
            elif channel == "g":
                c_idx = 1

            gray = roi[:, :, c_idx]
            coeffs = pywt.dwt2(gray, "haar")
            cA, (cH, cV, cD) = coeffs
            energy = np.mean(np.square(cH) + np.square(cV) + np.square(cD))
            return energy
        except Exception:
            return 0.0

    def check_fire_candidate(self, roi, block_id):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        fire_pixels = cv2.countNonZero(mask1 + mask2)
        pixel_count = roi.shape[0] * roi.shape[1]
        color_prob = fire_pixels / pixel_count

        if color_prob < 0.2:
            return False

        energy = self.get_spatial_wavelet_energy(roi, channel="r")
        if energy > 50.0:
            return True
        return False

    def check_smoke_candidate(self, roi, block_id):
        b, g, r = cv2.split(roi)
        chrominance = np.abs(r.astype(float) - g.astype(float)) + np.abs(
            g.astype(float) - b.astype(float)
        )
        avg_chroma = np.mean(chrominance)
        avg_intensity = np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))

        if avg_chroma > 40 or avg_intensity < 30:
            return False

        current_energy = self.get_spatial_wavelet_energy(roi, channel="g")
        if block_id not in self.background_energy_history:
            self.background_energy_history[block_id] = current_energy
            return False

        bg_energy = self.background_energy_history[block_id]
        self.background_energy_history[block_id] = (
            0.95 * bg_energy + 0.05 * current_energy
        )

        if current_energy < (bg_energy * 0.6) and bg_energy > 10.0:
            return True
        return False


def _resize_and_pad(frame, scale_factor, blk_size):
    """
    Resizes frame and adds padding to ensure dimensions are multiples of blk_size.
    Ported from tempStabilize_mt.py
    """
    if scale_factor != 1.0:
        scaled_frame = cv2.resize(
            frame,
            None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_AREA,
        )
    else:
        scaled_frame = frame

    H, W = scaled_frame.shape[:2]
    pad_h = (blk_size - (H % blk_size)) % blk_size
    pad_w = (blk_size - (W % blk_size)) % blk_size

    if pad_h > 0 or pad_w > 0:
        scaled_frame = cv2.copyMakeBorder(
            scaled_frame,
            0,
            pad_h,
            0,
            pad_w,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
    return scaled_frame


def _get_active_blocks(fg_mask, blk_size, base_threshold=0.1):
    """
    Vectorized calculation of active blocks.
    Ported from _active_motion_blocks in tempStabilize_mt.py
    """
    H, W = fg_mask.shape
    blk_h, blk_w = H // blk_size, W // blk_size

    # View as blocks (N_blocks_y, N_blocks_x, blk_size, blk_size)
    # Note: Requires dimensions to be exact multiples (handled by _resize_and_pad)
    blocks = fg_mask.reshape(blk_h, blk_size, blk_w, blk_size).swapaxes(1, 2)

    # Count active pixels (255s) per block
    # blocks > 0 creates a boolean mask, sum over last 2 dims (pixels in block)
    counts = (blocks > 0).sum(axis=(2, 3))
    total_pixels = blk_size * blk_size
    percentages = counts / total_pixels

    # Adaptive Thresholding Logic
    avg_percentage = np.mean(percentages) if percentages.size > 0 else 0
    # Increase threshold if scene is chaotic (wind/pan)
    adapted_threshold = max(base_threshold, base_threshold + (avg_percentage * 0.1))

    # Boolean mask of active blocks
    active_mask = percentages > adapted_threshold

    # Return indices of active blocks
    # np.argwhere returns [[row, col], [row, col]...]
    active_indices_2d = np.argwhere(active_mask)

    return active_indices_2d, adapted_threshold


def process_video(input_path, output_path, block_size=32, scale_factor=1.0):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # ty:ignore[unresolved-attribute]
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    # 2. Motion Mask Output
    mask_output_path = output_path.replace(".mp4", "_mask.mp4")
    out_mask = cv2.VideoWriter(mask_output_path, fourcc, fps, (width, height))

    # ! Changed: Using simple Frame Difference instead of MOG2
    # Logic taken from tempStabilize (bgs.FrameDifference equivalent)
    bg_subtractor = SimpleFrameDifference(diff_threshold=25)

    analyzer = BlockAnalyzer()
    frame_idx = 0
    profiler = zProfiler()

    print(f"Start processing: {input_path} | Scale: {scale_factor}")

    with profiler.measure("video_proc") as ctx:
        while True:
            with ctx.step("frame_read"):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                print(f"Processing frame {frame_idx}", end="\r")

            with ctx.step("motion_proc"):
                # 1. Scaling & Padding
                scaled_frame = _resize_and_pad(frame, scale_factor, block_size)

                # 2. Frame Difference
                fgmask = bg_subtractor.apply(scaled_frame)

                # --- Write Mask Video ---
                # Resize mask back to original resolution for visualization
                # Use INTER_NEAREST to keep binary look
                vis_mask = cv2.resize(
                    fgmask, (width, height), interpolation=cv2.INTER_NEAREST
                )
                vis_mask_bgr = cv2.cvtColor(vis_mask, cv2.COLOR_GRAY2BGR)
                out_mask.write(vis_mask_bgr)

                # Clean up mask (Optional, keep lightweight)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

                # 3. Get Active Blocks (Vectorized & Adaptive)
                active_blocks_indices, adapt_thres = _get_active_blocks(
                    fgmask, block_size, base_threshold=0.05
                )

            with ctx.step("block_analysis"):
                vis_frame = frame.copy()

                # Only iterate through ACTIVE blocks
                for r, c in active_blocks_indices:
                    # Calculate scaled coordinates
                    y1 = r * block_size
                    y2 = y1 + block_size
                    x1 = c * block_size
                    x2 = x1 + block_size

                    # Extract ROI from Scaled Frame for analysis
                    block_roi = scaled_frame[y1:y2, x1:x2]

                    # Map coordinates back to Original Frame for visualization
                    vis_x1 = int(x1 / scale_factor)
                    vis_y1 = int(y1 / scale_factor)
                    vis_x2 = int(x2 / scale_factor)
                    vis_y2 = int(y2 / scale_factor)

                    # Ensure within bounds
                    vis_x2 = min(vis_x2, width)
                    vis_y2 = min(vis_y2, height)

                    # Draw Motion Box (Yellow)
                    cv2.rectangle(
                        vis_frame, (vis_x1, vis_y1), (vis_x2, vis_y2), (0, 255, 255), 1
                    )

                    block_id = (r, c)
                    is_fire = False
                    is_smoke = False

                    # Analysis
                    if analyzer.check_fire_candidate(block_roi, block_id):
                        is_fire = True
                    if analyzer.check_smoke_candidate(block_roi, block_id):
                        is_smoke = True

                    # Labels
                    label = ""
                    color = (0, 255, 255)
                    if is_fire and is_smoke:
                        label = "F+S"
                        color = (0, 0, 255)
                    elif is_fire:
                        label = "FIRE"
                        color = (0, 0, 255)
                    elif is_smoke:
                        label = "SMOKE"
                        color = (200, 200, 200)

                    if label:
                        cv2.rectangle(
                            vis_frame, (vis_x1, vis_y1), (vis_x2, vis_y2), color, 2
                        )
                        cv2.putText(
                            vis_frame,
                            label,
                            (vis_x1, vis_y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            1,
                        )

            with ctx.step("write_frame"):
                out.write(vis_frame)

        cap.release()
        out.release()
        out_mask.release()
        print(f"\nProcessing complete. Saved to {output_path}")
        profiler.report_and_plot(".", tag="z_runIdea2")


def main():
    args = RunIdeaArgs().parse_args()
    process_video(
        args.video,
        "./z_run2_output.mp4",
        block_size=args.block_size,
        scale_factor=args.scale_factor,
    )


if __name__ == "__main__":
    main()
