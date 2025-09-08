import cv2
import numpy as np

# --- Parameters ---
VIDEO_PATH = r"E:\NextCloud\paper3\datasets\fire video - VDS3-20250805T111138Z-1-001\fire video - VDS3\video8.avi"  # or 0 for webcam
MOTION_THRESHOLD = 1.0  # tune: avg flow magnitude below this → skip

# --- Setup video ---
cap = cv2.VideoCapture(VIDEO_PATH)

# Shi-Tomasi corner detection params (to pick sparse keypoints)
feature_params = dict(maxCorners=200, qualityLevel=0.3, minDistance=7, blockSize=7)

# Lucas-Kanade optical flow params
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

# Colors for visualization
color = (0, 255, 0)

# Read first frame
ret, old_frame = cap.read()
if not ret:
    print("Error: Cannot read video")
    cap.release()
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow (track sparse points)
    if p0 is not None:
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )

        if p1 is not None:
            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # Compute average flow magnitude
            flow = good_new - good_old
            magnitudes = np.linalg.norm(flow, axis=1)
            avg_magnitude = np.mean(magnitudes) if len(magnitudes) > 0 else 0

            # --- Pre-check decision ---
            if avg_magnitude < MOTION_THRESHOLD:
                decision = "SKIP (no significant motion)"
            else:
                decision = "PROCESS (possible fire/smoke)"

            # --- Visualization ---
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                a, b, c, d = int(a), int(b), int(c), int(d)  # <-- convert to int
                frame = cv2.line(frame, (a, b), (c, d), color, 2)
                frame = cv2.circle(frame, (a, b), 5, color, -1)

            cv2.putText(
                frame,
                f"Avg Flow: {avg_magnitude:.2f} | {decision}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

            # Update old points
            p0 = good_new.reshape(-1, 1, 2)

        else:
            # No points tracked → reinitialize
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)

    else:
        # No keypoints → reinitialize
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)

    # Update reference frame
    old_gray = frame_gray.copy()

    cv2.imshow("Optical Flow Pre-check", frame)
    if cv2.waitKey(30) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
