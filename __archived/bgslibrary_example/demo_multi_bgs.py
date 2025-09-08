"""
Advanced Background Subtraction Demo using pybgs Library

This advanced demo script showcases the use of multiple background subtraction algorithms
from the pybgs library to process video files and image sequences for moving object detection.
It leverages the utility functions defined in `utils.py` for a clean and efficient demonstration,
automatically adapting to the installed OpenCV version to utilize the appropriate algorithms.

Features:
- Processes both video files and image sequences with ease.
- Dynamically selects background subtraction algorithms based on the OpenCV version.
- Demonstrates a wide range of algorithms for comprehensive analysis and comparison.

Usage:
- For video processing: python demo2.py
- For image sequence processing: python demo2.py image

Preparation:
- Ensure the paths to the video file and image directory are correctly set in the script.
- Ensure the `helper/utils.py` module is correctly located and accessible for importing utility functions.
"""

import sys
import cv2
import glob

from helper.utils import *

# Display the installed OpenCV version
print("OpenCV Version: {}".format(cv2.__version__))

# Initialize available algorithms based on the OpenCV version
algorithms = initialize_algorithms()

# Notify the user about the number of algorithms initialized
print("Number of available algorithms: ", len(algorithms))

# Detect operational mode (video or image sequence) based on command-line arguments
image_mode = len(sys.argv) == 2 and sys.argv[1] == "image"
img_folder = "dataset/frames" if image_mode else ""
# video_file = "dataset/real_cam1.mp4"
video_file = "dataset/wrong_fd_cam_stream.mp4"
# video_file = r"/mnt/e/NextCloud/paper3/datasets/TestVideos/d0-Trim.mp4"
# video_file = r"/mnt/e/NextCloud/paper3/datasets/TestVideos/smallfire.mp4"
# video_file = r"/mnt/e/NextCloud/paper3/datasets/fire video - VDS3-20250805T111138Z-1-001/fire video - VDS3/video16.avi"
# video_file = r"/mnt/e/NextCloud/paper3/datasets/DFire_Static/test/VP1.mp4"
# video_file = r"/mnt/e/NextCloud/paper3/datasets/TestVideos/video5.mp4"
# video_file = r"/mnt/e/NextCloud/paper3/datasets/TestVideos/d0-Trim.mp4"
# video_file = r"/mnt/e/NextCloud/paper3/datasets/DFire_Static/test/VP13.mp4"
# video_file = r"/mnt/e/NextCloud/paper3/datasets/TestVideos/f6.mp4"

# video_file = (
# r"/mnt/e/NextCloud/paper3/datasets/TestVideos/5816532-hd_1920_1080_25fps.mp4"
# )
img_array = sorted(glob.iglob(img_folder + "/*.png")) if image_mode else []

# Process each algorithm on the chosen media type
for algorithm in algorithms:
    print("Processing with: ", type(algorithm).__name__)
    if image_mode:
        process_images(img_array, algorithm)
    else:
        process_video(video_file, algorithm)

print("Demo Completed Successfully")
cv2.destroyAllWindows()
