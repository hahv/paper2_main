import cv2
import numpy as np


def get_mask_simple(frame1, frame2, sensitivity=30):
    """
    Simple, noise-resistant motion mask.
    Automatically converts color frames to grayscale.
    One tunable parameter: sensitivity (higher → stricter detection).
    """
    # --- Ensure grayscale ---
    if len(frame1.shape) == 3:
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    if len(frame2.shape) == 3:
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # --- Frame difference ---
    diff = cv2.absdiff(frame2, frame1)

    # --- Noise reduction ---
    diff = cv2.GaussianBlur(diff, (5, 5), 0)

    # --- Threshold based on sensitivity ---
    thresh_val = max(5, min(255, sensitivity))
    _, mask = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)

    # --- Morphological cleaning ---
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    return mask


def play_motion_mask(video_path, sensitivity=30):
    """Play original and motion mask side-by-side."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open video:", video_path)
        return

    ret, prev_frame = cap.read()
    if not ret:
        print("❌ Cannot read first frame.")
        return

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        # Compute mask
        mask = get_mask_simple(prev_frame, curr_frame, sensitivity=sensitivity)

        # Display
        cv2.imshow("Original Video", curr_frame)
        cv2.imshow("Motion Mask", mask)

        # Quit with 'q'
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

        prev_frame = curr_frame

    cap.release()
    cv2.destroyAllWindows()

from halib import *
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(
        description="desc text")
    parser.add_argument('-v', '--video', type=str,
                        help='Video file path')
    return parser.parse_args()


def main():
    args = parse_args()
    video = args.video
    assert os.path.isfile(video), f"❌ Video file not found: {video}"
    play_motion_mask(video, sensitivity=30)


if __name__ == "__main__":
    main()