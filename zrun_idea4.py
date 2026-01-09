import cv2
import numpy as np
import pywt
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
        _, binary_delta = cv2.threshold(
            delta, self.DIFF_FRAME_TH, self.IMPACK_PLUS_ONE, cv2.THRESH_BINARY
        )

        # 4. Accumulation (deltaMask = deltaMask + delta)
        self.delta_mask = cv2.add(self.delta_mask, binary_delta)

        # 5. Cap values
        _, self.delta_mask = cv2.threshold(
            self.delta_mask, self.MAX_VAL, self.MAX_VAL, cv2.THRESH_TRUNC
        )

        # 6. Decay
        self.delta_mask = cv2.subtract(self.delta_mask, self.DECAY)  # ty:ignore[no-matching-overload]

        # 7. Generate Current Mask
        _, cur_mask = cv2.threshold(
            self.delta_mask, self.MASK_TH, 255, cv2.THRESH_BINARY
        )

        # Update previous frame
        self.prev_frame = gray.copy()

        return cur_mask


class FastIntegralAnalyzer:
    """
    Optimized Analyzer using Integral Images (Summed Area Tables).
    Calculates features for the whole frame once, allowing O(1) lookups per block.
    """

    def __init__(self, rows, cols):
        # State for smoke background energy (stored as 2D float array for speed)
        # We initialize with -1.0 to indicate "no history yet"
        self.bg_energy_history = np.full((rows, cols), -1.0, dtype=np.float32)

        # Buffers for integral images
        self.int_fire_mask = None
        self.int_energy_r = None
        self.int_energy_g = None
        self.int_chroma = None
        self.int_intensity = None

    def _compute_integral(self, map_data):
        """Helper to compute integral image. Output is (H+1, W+1)."""
        return cv2.integral(map_data)

    def _get_block_sum(self, integral_img, x, y, w, h):
        """
        O(1) calculation of sum over a rectangular area.
        Formula: Sum = D - B - C + A
        """
        # Integral image has 1px padding at top/left
        y1, x1 = y, x
        y2, x2 = y + h, x + w

        # Safety clipping
        h_int, w_int = integral_img.shape
        y2 = min(y2, h_int - 1)
        x2 = min(x2, w_int - 1)

        A = integral_img[y1, x1]
        B = integral_img[y1, x2]
        C = integral_img[y2, x1]
        D = integral_img[y2, x2]

        return D - B - C + A

    def precompute_frame_features(self, frame_bgr):
        """
        Run expensive operations ONCE for the entire frame.
        """
        # --- 1. FIRE COLOR INTEGRAL ---
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        # Combine masks
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(
            hsv, lower_red2, upper_red2
        )  # ty:ignore[unsupported-operator]
        # Convert to 0/1 float
        mask_bin = (mask > 0).astype(np.float32)
        self.int_fire_mask = self._compute_integral(mask_bin)

        # --- 2. SMOKE FEATURES INTEGRAL (Chroma + Intensity) ---
        b, g, r = cv2.split(frame_bgr)
        # Chroma: |R-G| + |G-B|
        chroma = cv2.absdiff(r, g) + cv2.absdiff(g, b)
        # Intensity: Gray
        intensity = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        self.int_chroma = self._compute_integral(chroma.astype(np.float32))
        self.int_intensity = self._compute_integral(intensity.astype(np.float32))

        # --- 3. WAVELET ENERGY INTEGRAL (Texture) ---
        # We process Red (Fire) and Green (Smoke) channels.
        # DWT2 reduces size by half (w/2, h/2).

        # Red Channel Energy
        coeffs_r = pywt.dwt2(r, "haar")
        _, (cH, cV, cD) = coeffs_r
        energy_map_r = np.square(cH) + np.square(cV) + np.square(cD)
        self.int_energy_r = self._compute_integral(energy_map_r)

        # Green Channel Energy
        coeffs_g = pywt.dwt2(g, "haar")
        _, (cH, cV, cD) = coeffs_g
        energy_map_g = np.square(cH) + np.square(cV) + np.square(cD)
        self.int_energy_g = self._compute_integral(energy_map_g)

    def analyze_block(self, r, c, block_size):
        """
        Analyze a specific block using O(1) lookups.
        """
        # Coordinates in full resolution
        x, y = c * block_size, r * block_size
        w, h = block_size, block_size
        area = w * h

        # --- CHECK FIRE ---
        is_fire = False
        # 1. Color Probability
        fire_pixels = self._get_block_sum(self.int_fire_mask, x, y, w, h)
        if (fire_pixels / area) >= 0.2:
            # 2. Texture Energy (Red)
            # Wavelet map is HALF size. Divide coords by 2.
            w_w, w_h = w // 2, h // 2
            w_x, w_y = x // 2, y // 2
            w_area = max(1, w_w * w_h)

            total_energy = self._get_block_sum(self.int_energy_r, w_x, w_y, w_w, w_h)
            avg_energy = total_energy / w_area

            if avg_energy > 50.0:
                is_fire = True

        # --- CHECK SMOKE ---
        is_smoke = False
        # 1. Color Properties
        total_chroma = self._get_block_sum(self.int_chroma, x, y, w, h)
        total_intensity = self._get_block_sum(self.int_intensity, x, y, w, h)

        avg_chroma = total_chroma / area
        avg_intensity = total_intensity / area

        if avg_chroma <= 40 and avg_intensity >= 30:
            # 2. Texture Energy (Green) & Blurring
            w_w, w_h = w // 2, h // 2
            w_x, w_y = x // 2, y // 2
            w_area = max(1, w_w * w_h)

            curr_e_sum = self._get_block_sum(self.int_energy_g, w_x, w_y, w_w, w_h)
            curr_e = curr_e_sum / w_area

            # History Logic (Stateful via 2D array)
            bg_e = self.bg_energy_history[r, c]

            if bg_e == -1.0:
                # Initialize
                self.bg_energy_history[r, c] = curr_e
            else:
                # Update
                self.bg_energy_history[r, c] = 0.95 * bg_e + 0.05 * curr_e
                bg_e = self.bg_energy_history[r, c]  # Get updated value

                # Check Energy Drop (Blurring)
                if curr_e < (bg_e * 0.6) and bg_e > 10.0:
                    is_smoke = True

        return is_fire, is_smoke


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
    counts = (blocks > 0).sum(axis=(2, 3))
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

    # Initialize Initialization Phase
    ret, frame = cap.read()
    if not ret:
        return
    scaled_frame = _resize_and_pad(frame, scale_factor, block_size)

    # 1. Init Motion Detector
    motion_detector = TemporalMotionDetector(scaled_frame.shape[:2])

    # 2. Init Integral Analyzer
    rows = scaled_frame.shape[0] // block_size
    cols = scaled_frame.shape[1] // block_size
    analyzer = FastIntegralAnalyzer(rows, cols)

    # Reset video pointer
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

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

                # Get Active Blocks using ROI_TH (50 pixels approx ~5% of 32x32 block)
                active_indices = _get_active_blocks(
                    motion_mask, block_size, roi_percent_thresh=50
                )

            with ctx.step("analysis_precompute"):
                # !! OPTIMIZATION: Compute all features once per frame !!
                analyzer.precompute_frame_features(scaled_frame)

            with ctx.step("analysis_loop"):
                vis_frame = frame.copy()

                for r, c in active_indices:
                    # !! OPTIMIZATION: O(1) Lookup instead of heavy calculation
                    is_fire, is_smoke = analyzer.analyze_block(r, c, block_size)

                    # Visualization Logic
                    label = ""
                    color = (0, 255, 255)  # Default Motion Yellow

                    if is_fire and is_smoke:
                        label, color = "F+S", (0, 0, 255)
                    elif is_fire:
                        label, color = "FIRE", (0, 0, 255)
                    elif is_smoke:
                        label, color = "SMOKE", (200, 200, 200)

                    # Coords for drawing
                    y1, y2 = r * block_size, (r + 1) * block_size
                    x1, x2 = c * block_size, (c + 1) * block_size
                    vx1, vy1 = int(x1 / scale_factor), int(y1 / scale_factor)
                    vx2, vy2 = int(x2 / scale_factor), int(y2 / scale_factor)

                    # Draw box
                    cv2.rectangle(vis_frame, (vx1, vy1), (vx2, vy2), color, 1)

                    if label:
                        # Thicker box for detection
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
    profiler.report_and_plot(".", tag="z_runIdea4")


def main():
    args = RunIdeaArgs().parse_args()
    process_video(
        args.video,
        "./z_runIdea4_output.mp4",
        block_size=args.block_size,
        scale_factor=args.scale_factor,
    )


if __name__ == "__main__":
    main()
