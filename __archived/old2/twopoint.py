import cv2
import numpy as np

# Initialize video capture (replace with your video path or camera)
video_file = r"E:/NextCloud/paper3/datasets/fire video - VDS3-20250805T111138Z-1-001/fire video - VDS3/video1.avi"

cap = cv2.VideoCapture(video_file)

# Two-point model placeholders
min_model = None
max_model = None
first_frame = True

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot read the frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if first_frame:
        # Initialize min and max with the first frame
        min_model = gray.copy()
        max_model = gray.copy()
        first_frame = False

    # Foreground mask: 255 = foreground, 0 = background
    fg_mask = np.where((gray < min_model) | (gray > max_model), 255, 0).astype(np.uint8)

    # Update min/max for background pixels
    min_model = np.where(fg_mask == 0, np.minimum(min_model, gray), min_model)
    max_model = np.where(fg_mask == 0, np.maximum(max_model, gray), max_model)

    cv2.imshow("Foreground Mask", fg_mask)

    if cv2.waitKey(30) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
