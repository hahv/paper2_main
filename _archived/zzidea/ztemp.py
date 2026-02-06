import cv2
import numpy as np
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
    """

    def __init__(self, shape):
        self.prev_frame = None
        self.delta_mask = np.zeros(shape, dtype=np.uint8)
        self.DIFF_FRAME_TH = 1
        self.IMPACK_PLUS_ONE = 5
        self.MASK_TH = 10
        self.MAX_VAL = 25
        self.DECAY = 1

    def apply(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.prev_frame is None:
            self.prev_frame = gray.copy()
            return np.zeros_like(gray)

        delta = cv2.absdiff(self.prev_frame, gray)
        _, binary_delta = cv2.threshold(
            delta, self.DIFF_FRAME_TH, self.IMPACK_PLUS_ONE, cv2.THRESH_BINARY
        )
        self.delta_mask = cv2.add(self.delta_mask, binary_delta)
        _, self.delta_mask = cv2.threshold(
            self.delta_mask, self.MAX_VAL, self.MAX_VAL, cv2.THRESH_TRUNC
        )
        self.delta_mask = cv2.subtract(self.delta_mask, self.DECAY)
        _, cur_mask = cv2.threshold(
            self.delta_mask, self.MASK_TH, 255, cv2.THRESH_BINARY
        )
        self.prev_frame = gray.copy()
        return cur_mask


class VectorizedAnalyzer:
    """
    Ultra-Fast Analyzer using 'Resize-as-Average'.
    Uses Laplacian instead of Wavelets for speed.
    """

    def __init__(self, rows, cols):
        # Grid History for Smoke Energy (rows, cols)
        self.bg_energy_history = np.full((rows, cols), -1.0, dtype=np.float32)
        self.rows = rows
        self.cols = cols

        # Constants
        # Texture Threshold: Laplacian implies edges.
        # > 5.0 usually means significant texture (edges/noise) in the block
        self.TEXTURE_THRESH = 5.0

    def analyze_frame(self, frame_bgr, motion_mask):
        """
        Returns two boolean grids: is_fire_grid, is_smoke_grid
        """
        # H, W of the scaled processing frame
        h, w = frame_bgr.shape[:2]

        # Target grid size (tiny image)
        grid_shape = (self.cols, self.rows)  # (width, height) for resize

        # --- PREPARATION (Vectorized) ---
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Texture Map (The Bottleneck Fix)
        # Use Laplacian (optimized C++) instead of Wavelets
        # Result is a map where bright pixels = edges/roughness
        laplacian = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
        texture_energy_map = cv2.convertScaleAbs(laplacian).astype(np.float32)

        # 2. Downsample Texture to Grid
        # INTER_AREA calculates the average of the pixels in the block
        grid_texture = cv2.resize(
            texture_energy_map, grid_shape, interpolation=cv2.INTER_AREA
        )

        # 3. Downsample Motion to Grid
        # Check which blocks have motion > 20%
        # Divide by 255 because mask is 0-255
        motion_float = motion_mask.astype(np.float32) / 255.0
        grid_motion_avg = cv2.resize(
            motion_float, grid_shape, interpolation=cv2.INTER_AREA
        )
        grid_active_motion = grid_motion_avg > 0.2  # ROI_TH equivalent

        # --- FIRE LOGIC ---
        # 1. Color Mask
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        # Binary mask (0 or 1.0)
        fire_mask = (
            cv2.inRange(hsv, lower_red1, upper_red1)
            | cv2.inRange(hsv, lower_red2, upper_red2)
        ).astype(np.float32) / 255.0

        # Downsample to get % of red per block
        grid_fire_color = cv2.resize(
            fire_mask, grid_shape, interpolation=cv2.INTER_AREA
        )

        # Fire Decision: Active Motion + High Red % + High Texture
        grid_is_fire = (
            grid_active_motion
            & (grid_fire_color > 0.2)
            & (grid_texture > self.TEXTURE_THRESH)  # Rough texture
        )

        # --- SMOKE LOGIC ---
        # 1. Color Mask (Gray)
        b, g, r = cv2.split(frame_bgr)
        chroma = cv2.absdiff(r, g) + cv2.absdiff(g, b)

        # Low chroma (<40) and moderate intensity (>30)
        # Note: We do this check on the grid level for speed approximation
        grid_chroma = cv2.resize(
            chroma.astype(np.float32), grid_shape, interpolation=cv2.INTER_AREA
        )
        grid_intensity = cv2.resize(
            gray.astype(np.float32), grid_shape, interpolation=cv2.INTER_AREA
        )

        grid_is_gray = (grid_chroma < 40) & (grid_intensity > 30)

        # 2. History Update (Blurring Check)
        # Update history where it's uninitialized (-1)
        init_mask = self.bg_energy_history == -1.0
        self.bg_energy_history[init_mask] = grid_texture[init_mask]

        # Update Running Average
        # history = 0.95*history + 0.05*current
        self.bg_energy_history = 0.95 * self.bg_energy_history + 0.05 * grid_texture

        # Blurring Decision:
        # Active Motion + Gray Color + Energy Drop (Current < 60% of History)
        grid_energy_drop = (grid_texture < (self.bg_energy_history * 0.6)) & (
            self.bg_energy_history > 5.0
        )

        grid_is_smoke = grid_active_motion & grid_is_gray & grid_energy_drop

        return grid_is_fire, grid_is_smoke


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


def process_video(input_path, output_path, block_size=32, scale_factor=1.0):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # ty:ignore[unresolved-attribute]
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize
    ret, frame = cap.read()
    if not ret:
        return
    scaled_frame = _resize_and_pad(frame, scale_factor, block_size)

    # Init Motion Detector
    motion_detector = TemporalMotionDetector(scaled_frame.shape[:2])

    # Init Vectorized Analyzer
    rows = scaled_frame.shape[0] // block_size
    cols = scaled_frame.shape[1] // block_size
    analyzer = VectorizedAnalyzer(rows, cols)

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

            with ctx.step("motion_and_analysis"):
                scaled_frame = _resize_and_pad(frame, scale_factor, block_size)

                # 1. Motion
                motion_mask = motion_detector.apply(scaled_frame)

                # 2. Vectorized Analysis (Returns boolean grids)
                # This does EVERYTHING: Motion Check, Color Check, Texture Check
                grid_is_fire, grid_is_smoke = analyzer.analyze_frame(
                    scaled_frame, motion_mask
                )

                # Combine to find any active blocks (Fire OR Smoke) for visualization
                # np.argwhere returns indices of True values
                fire_indices = np.argwhere(grid_is_fire)
                smoke_indices = np.argwhere(grid_is_smoke)

            with ctx.step("visualization"):
                vis_frame = frame.copy()

                # Helper to draw boxes
                def draw_boxes(indices, label, color):
                    for r, c in indices:
                        y1, y2 = r * block_size, (r + 1) * block_size
                        x1, x2 = c * block_size, (c + 1) * block_size
                        vx1, vy1 = int(x1 / scale_factor), int(y1 / scale_factor)
                        vx2, vy2 = int(x2 / scale_factor), int(y2 / scale_factor)

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

                draw_boxes(smoke_indices, "SMOKE", (200, 200, 200))
                draw_boxes(fire_indices, "FIRE", (0, 0, 255))

            out.write(vis_frame)

    cap.release()
    out.release()
    print(f"\nSaved to {output_path}")
    profiler.report_and_plot(".", tag="z_runIdea_Vectorized")


def main():
    args = RunIdeaArgs().parse_args()
    process_video(
        args.video,
        "./z_runIdea_Vectorized.mp4",
        block_size=args.block_size,
        scale_factor=args.scale_factor,
    )


if __name__ == "__main__":
    main()
