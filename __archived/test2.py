import cv2
import numpy as np
import time
import pybgs as bgs

# Video source
video_source = r"/mnt/e/NextCloud/paper3/datasets/fire video - VDS3-20250805T111138Z-1-001/fire video - VDS3/video16.avi"

# Thresholds
flicker_thresh = 0.5
upward_thresh = 1.0
min_area = 400
max_rois = 5  # Max ROIs per frame to process

# LK Optical Flow params
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Background subtractor
fgbg = bgs.ViBe()

# Open video
cap = cv2.VideoCapture(video_source)
ret, prev_frame = cap.read()
if not ret:
    print("Error: Cannot read video")
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 1: Background subtraction
    fg_mask = fgbg.apply(frame)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter & sort contours by area
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
    contours.sort(key=cv2.contourArea, reverse=True)
    contours = contours[:max_rois]  # Only process top N ROIs

    vis_frame = frame.copy()

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        roi_prev = prev_gray[y : y + h, x : x + w]
        roi_curr = gray[y : y + h, x : x + w]

        # Skip if very little change
        if np.mean(cv2.absdiff(roi_prev, roi_curr)) < 2:
            continue

        # Step 2: Sparse optical flow
        p0 = cv2.goodFeaturesToTrack(roi_prev, mask=None, **feature_params)
        if p0 is None:
            continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(roi_prev, roi_curr, p0, None, **lk_params)
        if p1 is None:
            continue

        good_new = p1[st == 1]
        good_old = p0[st == 1]
        if len(good_new) < 2:
            continue

        dx = good_new[:, 0] - good_old[:, 0]
        dy = good_new[:, 1] - good_old[:, 1]
        angles = np.arctan2(dy, dx)

        # Step 3: Analyze motion pattern
        flicker_score = np.var(angles)
        upward_score = -np.mean(dy)

        if flicker_score > flicker_thresh and upward_score > upward_thresh:
            label_color = (0, 0, 255)
            label = "FIRE/SMOKE"
        else:
            label_color = (0, 255, 0)
            label = "Other Motion"

        # Step 4: Visualization
        cv2.rectangle(vis_frame, (x, y), (x + w, y + h), label_color, 2)
        cv2.putText(
            vis_frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2
        )

    # Display FPS
    fps = 1.0 / (time.time() - start_time + 1e-8)
    cv2.putText(
        vis_frame,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    print(f"FPS: {fps:.2f}", end="\r")

    cv2.imshow("Fire/Smoke ROI Detection", vis_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    prev_gray = gray.copy()

cap.release()
cv2.destroyAllWindows()
