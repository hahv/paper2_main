import cv2
import numpy as np
import time
from halib import *

SMALL_SIZE = (320, 180)  # Standard small size for < 1ms processing

def detect_haze_simplified_dcp(img, small_size=SMALL_SIZE):
    """
    Method A: Simplified Dark Channel Prior (Pixel-wise only).
    Extremely fast, good for general brightness/haze detection.
    """
    # Downsample to 320x180 (Standard for < 1ms processing)
    small = cv2.resize(img, small_size, interpolation=cv2.INTER_NEAREST)

    # Pixel-wise dark channel
    dark_channel = np.min(small, axis=2)

    return np.mean(dark_channel) / 255.0


def detect_haze_traditional_dcp(img, small_size=SMALL_SIZE):
    """
    Method B: Traditional Dark Channel Prior (Pixel-wise + Patch-wise).
    More robust to noise, follows original research papers.
    """
    small = cv2.resize(img, small_size, interpolation=cv2.INTER_NEAREST)

    # 1. Pixel-wise min
    pixel_min = np.min(small, axis=2)

    # 2. Patch-wise min (Erosion) - 5x5 kernel
    kernel = np.ones((5, 5), np.uint8)
    dark_channel = cv2.erode(pixel_min, kernel)

    return np.mean(dark_channel) / 255.0


def detect_haze_v_s_ratio(img, small_size=SMALL_SIZE):
    """
    Method C: Value-Saturation Difference.
    Great for ignoring bright objects that aren't fog (like white cars).
    """
    small = cv2.resize(img, small_size, interpolation=cv2.INTER_NEAREST)

    # Convert to HSV
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # Haze = High Brightness (V) - Low Saturation (S)
    score = (np.mean(v).astype(float) - np.mean(s).astype(float)) / 255.0
    return max(0.0, min(1.0, score))


def benchmark_methods(image_hd, small_size=SMALL_SIZE):
    print(f"--- Benchmarking on {image_hd.shape[1]}x{image_hd.shape[0]} Image ---")

    methods = [
        ("Simplified DCP", detect_haze_simplified_dcp),
        ("Traditional DCP", detect_haze_traditional_dcp),
        ("V-S Ratio     ", detect_haze_v_s_ratio),
    ]

    for name, func in methods:
        # Warm up
        func(image_hd, small_size=small_size)

        # Timing
        start = time.perf_counter()
        iterations = 500
        for _ in range(iterations):
            score = func(image_hd, small_size=small_size)
        end = time.perf_counter()

        avg_time_ms = ((end - start) / iterations) * 1000
        print(f"{name} | Score: {score:.4f} | Avg Time: {avg_time_ms:.4f} ms")


if __name__ == "__main__":
    scale_factor_list = [0.5, 0.25, 0.125, 0.0625]  # For testing different sizes
    for scale in scale_factor_list:
        small_size = (int(1920 * scale), int(1080 * scale))
        console.rule(f"\nTesting on {small_size[0]}x{small_size[1]} image:")
        # Create a synthetic image of the specified size
        # 1. Create a synthetic 'Clear' HD image (1080p)
        # A simple gradient to simulate a landscape
        clear_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        clear_img[:, :, 0] = np.linspace(200, 50, 1080)[:, None]  # Blue sky
        clear_img[:, :, 1] = np.linspace(150, 100, 1080)[:, None]  # Greenish
        clear_img[:, :, 2] = 50  # Red

        # 2. Create a 'Hazy' version of the same image
        # Haze adds 'Airlight' (white/gray) and reduces original signal
        haze_intensity = 0.6
        airlight = 220
        hazy_img = cv2.addWeighted(
            clear_img,
            1 - haze_intensity,
            np.full_like(clear_img, airlight),
            haze_intensity,
            0,
        )

        print("TESTING CLEAR IMAGE:")
        benchmark_methods(clear_img, small_size=small_size)

        print("\nTESTING HAZY IMAGE:")
        benchmark_methods(hazy_img, small_size=small_size)

        print("\nNote: Times include the overhead of cv2.resize.")
