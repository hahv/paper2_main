from torch.distributed.elastic.metrics.api import prof
import cv2
import numpy as np
import pywt
from halib import *  # noqa: F403
from tap import *
from halib.exp.perf.profiler import zProfiler

class RunIdeaArgs(Tap):
    video: str = r"./datasets/UFireIndoor/firesmoke/aihub__lb_fire_0178.mp4"  # Path to input video file
    block_size: int = 32  # Size of blocks to analyze

# !Paper: Flame Detection for Video-based Early Fire Warning Systems and 3D Visualization of Fire Propagation

class BlockAnalyzer:
    def __init__(self, history_len=30):
        # Stores history for temporal features (Variance/Flicker)
        # Key: (row, col), Value: specific history buffer
        self.spatial_energy_history = {}
        self.background_energy_history = {}  # For smoke blurring detection
        self.history_len = history_len

    def get_spatial_wavelet_energy(self, roi, channel="r"):
        """
        Calculates texture 'roughness' using Wavelets[cite: 376].
        Fire = High Energy (Rough). Smoke = Low Energy (Blur).
        """
        try:
            if channel == "r":
                c_idx = 2  # Red in BGR
            elif channel == "g":
                c_idx = 1  # Green (good for general detail)

            gray = roi[:, :, c_idx]

            # Haar wavelet is fast and effective for edge detection
            coeffs = pywt.dwt2(gray, "haar")
            cA, (cH, cV, cD) = coeffs

            # Energy = sum of squared detail coefficients [cite: 477]
            energy = np.mean(np.square(cH) + np.square(cV) + np.square(cD))
            return energy
        except Exception:
            return 0.0

    def check_fire_candidate(self, roi, block_id):
        """
        Checks for Fire:
        1. Color: High Saturation Red/Yellow [cite: 66, 437]
        2. Texture: High Spatial Energy (Roughness) [cite: 470]
        """
        # --- 1. Color Analysis ---
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Define Fire Color Range (Red/Yellow/Orange)
        # Note: In OpenCV HSV, Red wraps around 0/180
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        fire_pixels = cv2.countNonZero(mask1 + mask2)
        pixel_count = roi.shape[0] * roi.shape[1]

        color_prob = fire_pixels / pixel_count

        if color_prob < 0.2:  # If less than 20% is fire-colored, skip
            return False

        # --- 2. Spatial Wavelet Analysis (Roughness) ---
        energy = self.get_spatial_wavelet_energy(roi, channel="r")

        # Fire creates HIGH energy (messy texture)
        # Threshold depends on video res, 50.0 is a conservative starting point
        if energy > 50.0:
            return True

        return False

    def check_smoke_candidate(self, roi, block_id):
        """
        Checks for Smoke:
        1. Color: Low Saturation (Gray)
        2. Texture: Background Energy Loss (Blurring)
        """
        # --- 1. Color Analysis (Grayness) ---
        b, g, r = cv2.split(roi)
        # Chrominance: How colorful is it? Smoke is low chrominance.
        chrominance = np.abs(r.astype(float) - g.astype(float)) + np.abs(
            g.astype(float) - b.astype(float)
        )
        avg_chroma = np.mean(chrominance)
        avg_intensity = np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))

        # Rule: Low Color (Gray) AND Moderate Brightness (Not Shadow)
        if avg_chroma > 40 or avg_intensity < 30:
            return False

        # --- 2. Background Blurring (Energy Loss) ---
        current_energy = self.get_spatial_wavelet_energy(roi, channel="g")

        # Initialize background history if new
        if block_id not in self.background_energy_history:
            self.background_energy_history[block_id] = current_energy
            return False  # Need history to compare

        bg_energy = self.background_energy_history[block_id]

        # Update background model (slowly learn new textures)
        self.background_energy_history[block_id] = (
            0.95 * bg_energy + 0.05 * current_energy
        )

        # Smoke Check: Did energy DROP significantly? (Blurring effect)
        # If current energy is < 60% of usual background energy -> Smoke
        if current_energy < (bg_energy * 0.6) and bg_energy > 10.0:
            return True

        return False


def process_video(input_path, output_path, block_size=32):
    cap = cv2.VideoCapture(input_path)

    # Get video info
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define Output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # ty:ignore[unresolved-attribute]
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Background Subtractor (Adaptive - similar to paper suggestion)
    # DetectShadows=False is faster
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=50, detectShadows=False
    )

    analyzer = BlockAnalyzer()
    frame_idx = 0
    profiler = zProfiler()
    with profiler.measure("video_proc") as ctx:
        while True:
            with ctx.step("fram_motion_proc"):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                print(f"Processing frame {frame_idx}", end="\r")

                # 1. Background Subtraction (Motion Detection) [cite: 60, 382]
                fgmask = fgbg.apply(frame)

                # Clean up mask (remove noise)
                # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

            with ctx.step("block_analysis"):
                # Visualization overlay
                vis_frame = frame.copy()

                # Loop through blocks
                rows = height // block_size
                cols = width // block_size

                for r in range(rows):
                    for c in range(cols):
                        # Define Block Coordinates
                        y1 = r * block_size
                        y2 = y1 + block_size
                        x1 = c * block_size
                        x2 = x1 + block_size

                        # Extract Block Data
                        block_mask = fgmask[y1:y2, x1:x2]
                        block_roi = frame[y1:y2, x1:x2]

                        # --- Step A: Motion Check ---
                        # Count moving pixels in this block
                        motion_pixels = cv2.countNonZero(block_mask)
                        total_pixels = block_size * block_size
                        motion_ratio = motion_pixels / total_pixels

                        # If > 20% of block is moving, process it
                        if motion_ratio > 0.2:
                            # Draw Motion Box (Yellow)
                            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

                            block_id = (r, c)
                            is_fire = False
                            is_smoke = False

                            # --- Step B: Fire Check ---
                            if analyzer.check_fire_candidate(block_roi, block_id):
                                is_fire = True

                            # --- Step C: Smoke Check ---
                            if analyzer.check_smoke_candidate(block_roi, block_id):
                                is_smoke = True

                            # --- Visualization ---
                            label = ""
                            color = (0, 255, 255)  # Default Yellow

                            if is_fire and is_smoke:
                                label = "F+S"
                                color = (0, 0, 255)  # Red
                            elif is_fire:
                                label = "FIRE"
                                color = (0, 0, 255)  # Red
                            elif is_smoke:
                                label = "SMOKE"
                                color = (200, 200, 200)  # Gray

                            if label:
                                # Draw thicker box and text for candidates
                                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                                cv2.putText(
                                    vis_frame,
                                    label,
                                    (x1, y1 + 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.4,
                                    color,
                                    1,
                                )
            with ctx.step("write_frame"):
                # Write frame
                out.write(vis_frame)

            # Optional: Show real-time (press 'q' to quit)
            # cv2.imshow("Analysis", vis_frame)
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break

        cap.release()
        out.release()
        # cv2.destroyAllWindows()
        print(f"Processing complete. Saved to {output_path}")
        pprint_local_path(output_path, get_wins_path=True)
        profiler.report_and_plot(".", tag='z_runIdea1')


def main():
    # Parse arguments
    args = RunIdeaArgs().parse_args()
    video_path = args.video
    block_size = args.block_size
    output_path = "./z_run1_output.mp4"
    process_video(video_path, output_path, block_size=block_size)

if __name__ == "__main__":
    main()