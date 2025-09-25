import cv2
import numpy as np

# Video source (replace with webcam 0 or video path)
# video_source = r"E:\NextCloud\paper3\datasets\Custom_Test\test\smallfire.mp4"
# video_source = r"C:\Users\ha\Desktop\bgslibrary2_qtgui_opencv320_x64\test\d0 - Trim.mp4"
video_source = r"C:\Users\ha\Desktop\3.mp4"

# Thresholds and params
flicker_thresh = 0.5
upward_thresh = 1.0
min_area = 100  # smaller area for grid cells
grid_rows, grid_cols = 20, 20  # grid size

# LK Optical Flow params
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)
feature_params = dict(maxCorners=10, qualityLevel=0.3, minDistance=7, blockSize=7)

# Background subtractor (MOG2)
fgbg = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=16, detectShadows=False
)

# Open video
cap = cv2.VideoCapture(video_source)
ret, prev_frame = cap.read()
if not ret:
    print("Error: Cannot read video")
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
height, width = prev_gray.shape
cell_h = height // grid_rows
cell_w = width // grid_cols

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fg_mask = fgbg.apply(frame)

    # Use a color copy to draw bounding boxes and labels
    vis_frame = frame.copy()

    for r in range(grid_rows):
        for c in range(grid_cols):
            x = c * cell_w
            y = r * cell_h
            roi_fg = fg_mask[y : y + cell_h, x : x + cell_w]

            # Check motion presence in grid cell
            motion_area = cv2.countNonZero(roi_fg)
            if motion_area < min_area:
                continue

            roi_prev = prev_gray[y : y + cell_h, x : x + cell_w]
            roi_curr = gray[y : y + cell_h, x : x + cell_w]

            # Get good features to track in this cell
            p0 = cv2.goodFeaturesToTrack(roi_prev, mask=None, **feature_params)
            if p0 is None:
                continue

            p1, st, _ = cv2.calcOpticalFlowPyrLK(
                roi_prev, roi_curr, p0, None, **lk_params
            )
            if p1 is None:
                continue

            good_new = p1[st == 1]
            good_old = p0[st == 1]

            if len(good_new) < 2:
                continue

            dx = good_new[:, 0] - good_old[:, 0]
            dy = good_new[:, 1] - good_old[:, 1]
            angles = np.arctan2(dy, dx)

            flicker_score = np.var(angles)
            upward_score = -np.mean(dy)  # Negative dy = upward motion

            if flicker_score > flicker_thresh and upward_score > upward_thresh:
                color = (0, 0, 255)
                label = "FIRE/SMOKE"
            else:
                color = (0, 255, 0)
                label = "Other Motion"

            cv2.rectangle(vis_frame, (x, y), (x + cell_w, y + cell_h), color, 2)
            cv2.putText(
                vis_frame,
                label,
                (x + 2, y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

    cv2.imshow("Grid-based Fire/Smoke ROI Detection", vis_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    prev_gray = gray.copy()

cap.release()
cv2.destroyAllWindows()
