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
    scale_factor: float = 1.0


class TemporalMotionDetector:
    """
    Port of C++ FireDetector::temporalStabilization logic.
    Implements Accumulation + Decay for robust motion detection.
    """

    def __init__(self, shape):
        self.prev_frame = None
        # deltaMasks[vchID] -> persistent accumulation buffer
        self.delta_mask = np.zeros(shape, dtype=np.uint8)

        # Constants from C++ Code
        self.DIFF_FRAME_TH = 1  # Sensitivity to pixel change
        self.IMPACK_PLUS_ONE = 5  # Weight added per change
        self.MASK_TH = 10  # Threshold to activate motion
        self.MAX_VAL = 25  # Cap for accumulation
        self.DECAY = 1  # Decay per frame

    def apply(self, frame_bgr):
        # 1. Convert to Gray
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self.prev_frame is None:
            self.prev_frame = gray.copy()
            return np.zeros_like(gray)

        # 2. Frame Difference (absdiff)
        delta = cv2.absdiff(self.prev_frame, gray)

        # 3. Threshold Difference
        # C++: threshold(delta, delta, DIFF_FRAME_TH, IMPACK_PLUS_ONE, THRESH_BINARY);
        # Result: Pixels > 1 become 5, others 0
        _, binary_delta = cv2.threshold(
            delta, self.DIFF_FRAME_TH, self.IMPACK_PLUS_ONE, cv2.THRESH_BINARY
        )

        # 4. Accumulation (deltaMask = deltaMask + delta)
        # We use cv2.add to ensure saturation logic (though max is small here)
        self.delta_mask = cv2.add(self.delta_mask, binary_delta)

        # 5. Cap values (cv::min(deltaMask, 25, deltaMask))
        # cv2.threshold with THRESH_TRUNC caps values at MAX_VAL
        _, self.delta_mask = cv2.threshold(
            self.delta_mask, self.MAX_VAL, self.MAX_VAL, cv2.THRESH_TRUNC
        )

        # 6. Decay (subtract(deltaMask, 1, deltaMask))
        # Using cv2.subtract handles underflow (0 - 1 = 0) automatically for uint8
        self.delta_mask = cv2.subtract(self.delta_mask, self.DECAY)  # ty:ignore[no-matching-overload]

        # 7. Generate Current Mask (compare(deltaMask, MASK_TH, curMask, cv::CMP_GE))
        _, cur_mask = cv2.threshold(
            self.delta_mask, self.MASK_TH, 255, cv2.THRESH_BINARY
        )

        # Update previous frame
        self.prev_frame = gray.copy()

        return cur_mask


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
            return np.mean(np.square(cH) + np.square(cV) + np.square(cD))
        except Exception:
            return 0.0

    def check_fire_candidate(self, roi, block_id):
        # 1. Color (Red/Yellow)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(
            hsv, lower_red2, upper_red2
        )
        color_prob = cv2.countNonZero(mask) / (roi.shape[0] * roi.shape[1])

        if color_prob < 0.2:
            return False

        # 2. Texture (Wavelet)
        if self.get_spatial_wavelet_energy(roi, channel="r") > 50.0:
            return True
        return False

    def check_smoke_candidate(self, roi, block_id):
        # 1. Color (Gray)
        b, g, r = cv2.split(roi)
        chroma = np.mean(np.abs(r.astype(float) - g) + np.abs(g - b))
        intensity = np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))

        if chroma > 40 or intensity < 30:
            return False

        # 2. Texture (Blurring / Energy Loss)
        curr_e = self.get_spatial_wavelet_energy(roi, channel="g")
        if block_id not in self.background_energy_history:
            self.background_energy_history[block_id] = curr_e
            return False

        bg_e = self.background_energy_history[block_id]
        self.background_energy_history[block_id] = 0.95 * bg_e + 0.05 * curr_e

        if curr_e < (bg_e * 0.6) and bg_e > 10.0:
            return True
        return False


def _resize_and_pad(frame, scale_factor, blk_size):
    if scale_factor != 1.0:
        scaled_frame = cv2.resize(
            frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA
        )
    else:
        scaled_frame = frame

    H, W = scaled_frame.shape[:2]
    pad_h = (blk_size - (H % blk_size)) % blk_size
    pad_w = (blk_size - (W % blk_size)) % blk_size

    if pad_h > 0 or pad_w > 0:
        scaled_frame = cv2.copyMakeBorder(
            scaled_frame, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
    return scaled_frame


def _get_active_blocks(fg_mask, blk_size, roi_percent_thresh=0.2):
    """
    Identifies blocks with sufficient motion using C++ logic (ROI_TH).
    """
    H, W = fg_mask.shape
    blk_h, blk_w = H // blk_size, W // blk_size

    # Reshape to (N_blocks_y, N_blocks_x, blk_size, blk_size)
    blocks = fg_mask.reshape(blk_h, blk_size, blk_w, blk_size).swapaxes(1, 2)

    # Count non-zero pixels per block
    counts = (
        cv2.countNonZero(blocks.reshape(-1, blk_size * blk_size).T).T
        if 0
        else (blocks > 0).sum(axis=(2, 3))
    )
    total_pixels_per_block = blk_size * blk_size

    # Determine active blocks
    active_mask = counts >= (roi_percent_thresh / 100.0) * total_pixels_per_block

    # Return indices
    return np.argwhere(active_mask)


def process_video(input_path, output_path, block_size=32, scale_factor=1.0):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # ty:ignore[unresolved-attribute]
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize Temporal Detector with scaled size
    # We need to know scaled size first
    ret, frame = cap.read()
    if not ret:
        return
    scaled_frame = _resize_and_pad(frame, scale_factor, block_size)

    motion_detector = TemporalMotionDetector(scaled_frame.shape[:2])

    # Reset video pointer
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    analyzer = BlockAnalyzer()
    frame_idx = 0
    profiler = zProfiler()

    print(f"Processing: {input_path} | Scale: {scale_factor}")

    with profiler.measure("video_proc") as ctx:
        while True:
            with ctx.step("read"):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                print(f"Frame {frame_idx}", end="\r")

            with ctx.step("motion"):
                scaled_frame = _resize_and_pad(frame, scale_factor, block_size)

                # Apply C++ Temporal Stabilization Logic
                motion_mask = motion_detector.apply(scaled_frame)

                # Get Active Blocks using ROI_TH
                active_indices = _get_active_blocks(
                    motion_mask, block_size, roi_percent_thresh=50
                )
                # Note: Adjusted threshold to 50 for 32x32 blocks to be safer (C++ used 200 likely for larger blocks)

            with ctx.step("analysis"):
                vis_frame = frame.copy()

                for r, c in active_indices:
                    # Coords
                    y1, y2 = r * block_size, (r + 1) * block_size
                    x1, x2 = c * block_size, (c + 1) * block_size

                    block_roi = scaled_frame[y1:y2, x1:x2]

                    # Map back to original for Vis
                    vx1, vy1 = int(x1 / scale_factor), int(y1 / scale_factor)
                    vx2, vy2 = int(x2 / scale_factor), int(y2 / scale_factor)

                    # Draw Motion Block (Yellow)
                    cv2.rectangle(vis_frame, (vx1, vy1), (vx2, vy2), (0, 255, 255), 1)

                    block_id = (r, c)
                    is_fire = analyzer.check_fire_candidate(block_roi, block_id)
                    is_smoke = analyzer.check_smoke_candidate(block_roi, block_id)

                    label = ""
                    color = (0, 255, 255)
                    if is_fire and is_smoke:
                        label, color = "F+S", (0, 0, 255)
                    elif is_fire:
                        label, color = "FIRE", (0, 0, 255)
                    elif is_smoke:
                        label, color = "SMOKE", (200, 200, 200)

                    if label:
                        cv2.rectangle(vis_frame, (vx1, vy1), (vx2, vy2), color, 2)
                        cv2.putText(
                            vis_frame,
                            label,
                            (vx1, vy1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            1,
                        )

            out.write(vis_frame)

    cap.release()
    out.release()
    print(f"\nSaved to {output_path}")
    profiler.report_and_plot(".", tag="z_runIdea3")


def main():
    args = RunIdeaArgs().parse_args()
    process_video(
        args.video,
        "./z_runIdea3_output.mp4",
        block_size=args.block_size,
        scale_factor=args.scale_factor,
    )


if __name__ == "__main__":
    main()
