import cv2
import numpy as np

# Use webcam (0) or replace with video path: 'video.mp4'
video_source = 0

# Thresholds
flicker_thresh = 0.5
upward_thresh = 1.0
min_area = 400

# LK Optical Flow params
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Background subtractor (MOG2 as ViBe substitute)
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

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 1: Background subtraction (ViBe substitute)
    fg_mask = fgbg.apply(frame)

    # Convert to color for visualization (so we can draw colored boxes)
    fg_vis = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < min_area:
            continue

        roi_prev = prev_gray[y : y + h, x : x + w]
        roi_curr = gray[y : y + h, x : x + w]

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
        upward_score = -np.mean(dy)  # Negative dy = upward motion

        if flicker_score > flicker_thresh and upward_score > upward_thresh:
            color = (0, 0, 255)
            label = "FIRE/SMOKE"
        else:
            color = (0, 255, 0)
            label = "Other Motion"

        # Step 4: Draw on foreground mask
        cv2.rectangle(fg_vis, (x, y), (x + w, y + h), color, 2)
        cv2.putText(fg_vis, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show foreground mask with annotations
    cv2.imshow("Fire/Smoke Detection on Foreground Mask", fg_vis)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    prev_gray = gray.copy()

cap.release()
cv2.destroyAllWindows()
