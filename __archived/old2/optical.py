import cv2
import numpy as np

# Video source (replace with webcam 0 or video path)
# video_source = r"E:\NextCloud\paper3\datasets\Custom_Test\test\smallfire.mp4"
# video_source = r"C:\Users\ha\Desktop\bgslibrary2_qtgui_opencv320_x64\test\d0 - Trim.mp4"
video_source = r"C:\Users\ha\Desktop\1.mp4"
video_source = r"C:\Users\ha\Desktop\bgslibrary2_qtgui_opencv320_x64\test\d0.mp4"

# Parameters
THRESHOLD_MOTION_MAGNITUDE = 0.5  # Threshold for motion magnitude
MIN_MOTION_POINTS_RATIO = 0.1  # Minimum ratio of feature points with significant motion
MAX_CORNERS = 100  # Maximum number of feature points to track
MIN_CORNERS = 10  # Minimum number of corners to proceed with flow calculation

# Lucas-Kanade optical flow parameters
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

# Shi-Tomasi corner detection parameters
feature_params = dict(
    maxCorners=MAX_CORNERS, qualityLevel=0.3, minDistance=7, blockSize=7
)


def check_skip_using_optical_flow(video_source):
    # Initialize video capture
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video stream")
        return

    # Initialize variables
    previous_frame = None
    previous_points = None
    skip_detection = True

    while cap.isOpened():
        # Read current frame
        ret, current_frame = cap.read()
        if not ret:
            print("End of video stream")
            break

        # Convert current frame to grayscale
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Skip first frame (no previous frame/points to compare)
        if previous_frame is None:
            previous_frame = current_gray
            # Detect initial feature points
            previous_points = cv2.goodFeaturesToTrack(
                current_gray, mask=None, **feature_params
            )
            skip_detection = False
        else:
            # Detect feature points in current frame if previous points are insufficient
            if previous_points is None or len(previous_points) < MIN_CORNERS:
                previous_points = cv2.goodFeaturesToTrack(
                    previous_frame, mask=None, **feature_params
                )

            # Skip if too few feature points are detected
            if previous_points is None or len(previous_points) < MIN_CORNERS:
                print("Skipping: Too few feature points")
                skip_detection = True
            else:
                # Compute sparse optical flow using Lucas-Kanade
                current_points, status, err = cv2.calcOpticalFlowPyrLK(
                    previous_frame, current_gray, previous_points, None, **lk_params
                )

                # Select points with valid tracking (status == 1)
                good_points_prev = previous_points[status == 1]
                good_points_curr = current_points[status == 1]

                # Calculate motion magnitudes
                if len(good_points_prev) > 0:
                    motion_vectors = good_points_curr - good_points_prev
                    magnitudes = np.sqrt(np.sum(motion_vectors**2, axis=1))
                    significant_motion_points = np.sum(
                        magnitudes > THRESHOLD_MOTION_MAGNITUDE
                    )
                    motion_ratio = significant_motion_points / len(good_points_prev)

                    # Decide if detection can be skipped
                    skip_detection = motion_ratio < MIN_MOTION_POINTS_RATIO
                else:
                    # No valid points tracked, assume static scene
                    skip_detection = True

            # Update feature points for next iteration
            previous_points = cv2.goodFeaturesToTrack(
                current_gray, mask=None, **feature_params
            )

        # Output decision
        if skip_detection:
            print("Skipping fire/smoke detection: No significant motion")
        else:
            print("Proceeding with fire/smoke detection")
            # Placeholder for fire/smoke detection
            # perform_fire_smoke_detection(current_frame)

        # Update previous frame
        previous_frame = current_gray

        # Optional: Display frame with tracked points for debugging
        debug_frame = current_frame.copy()
        if previous_points is not None:
            for pt in previous_points:
                x, y = pt.ravel().astype(int)
                cv2.circle(debug_frame, (x, y), 3, (0, 255, 0), -1)
        cv2.imshow("Frame", debug_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


def perform_fire_smoke_detection(frame):
    # Placeholder for fire/smoke detection logic
    # Replace with actual detection algorithm
    pass


# Example usage
if __name__ == "__main__":
    # Replace 'video.mp4' with your video file or camera index (e.g., 0 for webcam)
    check_skip_using_optical_flow(video_source)
