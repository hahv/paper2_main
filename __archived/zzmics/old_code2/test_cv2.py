import cv2
import time

# Open video (0 for webcam, or replace with "video.mp4")
cap = cv2.VideoCapture(r"E:\NextCloud\paper3\datasets\Custom_Test\test\f6.mp4")

if not cap.isOpened():
    print("Error: Cannot open video source")
    exit()

# Variables for FPS calculation
prev_time = 0
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get current time
    curr_time = time.time()
    # Calculate FPS
    if prev_time != 0:
        fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time

    # Put FPS text on frame
    cv2.putText(
        frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )

    # Show video
    cv2.imshow("Video with FPS", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
