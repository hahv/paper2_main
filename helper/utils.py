"""
Utility Functions for Background Subtraction with pybgs Library

This script provides utility functions supporting the main demonstration of background subtraction
using various algorithms from the pybgs library. It includes OpenCV version checks, initialization
of a comprehensive list of background subtraction algorithms, and functions to process both video
files and image sequences using these algorithms.

Key Functions:
- is_cv2, is_cv3, is_cv4, is_lower_or_equals_cv347: Check the OpenCV version installed.
- check_opencv_version: Helper function to check for specific OpenCV major version.
- initialize_algorithms: Initializes a list of background subtraction algorithms available in the pybgs library, adjusted for the installed version of OpenCV.
- process_images: Processes a sequence of images using a specified background subtraction algorithm, displaying the original image, foreground mask, and background model.
- process_video: Processes video files frame by frame using a specified background subtraction algorithm, displaying the original frame, foreground mask, and background model.

Usage:
These utility functions are designed to be imported and used in a main script that demonstrates
the application of background subtraction techniques on video or image data. They handle the heavy
lifting of algorithm initialization and frame processing, simplifying the main script's logic.
"""

import cv2
import pybgs as bgs
import time
from halib import *
from halib.filetype import csvfile


# Functions to check OpenCV version
def is_cv2():
    return check_opencv_version("2.")


def is_cv3():
    return check_opencv_version("3.")


def is_lower_or_equals_cv347():
    [major, minor, revision] = cv2.__version__.split(".")
    return int(major) == 3 and int(minor) <= 4 and int(revision) <= 7


def is_cv4():
    return check_opencv_version("4.")


def check_opencv_version(major):
    return cv2.__version__.startswith(major)


def initialize_algorithms():
    """
    Initialize and return a list of background subtraction algorithms based on the installed OpenCV version.
    """
    global CUSTOM_ALGORITHM_LIST
    if CUSTOM_ALGORITHM_LIST:
        # Custom list of algorithms to test
        return [bgs.FrameDifference()]

    ######################Default list of algorithms to test
    algos = [
        bgs.FrameDifference(),
        bgs.StaticFrameDifference(),
        bgs.WeightedMovingMean(),
        bgs.WeightedMovingVariance(),
        bgs.AdaptiveBackgroundLearning(),
        bgs.AdaptiveSelectiveBackgroundLearning(),
        bgs.MixtureOfGaussianV2(),
        bgs.PixelBasedAdaptiveSegmenter(),
        bgs.SigmaDelta(),
        bgs.SuBSENSE(),
        bgs.LOBSTER(),
        bgs.PAWCS(),
        bgs.TwoPoints(),
        bgs.ViBe(),
        bgs.CodeBook(),
        bgs.FuzzySugenoIntegral(),
        bgs.FuzzyChoquetIntegral(),
        bgs.LBSimpleGaussian(),
        bgs.LBFuzzyGaussian(),
        bgs.LBMixtureOfGaussians(),
        bgs.LBAdaptiveSOM(),
        bgs.LBFuzzyAdaptiveSOM(),
        bgs.VuMeter(),
        bgs.KDE(),
        bgs.IndependentMultimodal(),
    ]

    if is_cv2():
        # pass
        algos.extend([bgs.MixtureOfGaussianV1(), bgs.GMG()])  # OpenCV 2.x specific

    if not is_cv2():
        # pass
        algos.append(bgs.KNN())  # OpenCV > 2.x specific

    if is_cv2() or is_cv3():
        algos.extend(
            [
                bgs.DPAdaptiveMedian(),
                bgs.DPGrimsonGMM(),
                bgs.DPZivkovicAGMM(),
                bgs.DPMean(),
                bgs.DPWrenGA(),
                bgs.DPPratiMediod(),
                bgs.DPEigenbackground(),
                bgs.DPTexture(),
                bgs.T2FGMM_UM(),
                bgs.T2FGMM_UV(),
                bgs.T2FMRF_UM(),
                bgs.T2FMRF_UV(),
                bgs.MultiCue(),
            ]
        )

    if is_cv2() or is_lower_or_equals_cv347():
        algos.extend([bgs.LBP_MRF(), bgs.MultiLayer()])

    # algos = [bgs.ViBe()] # force to test

    return algos


def process_images(img_array, algorithm):
    """
    Process each image in img_array with the specified algorithm.
    Displays original image, foreground mask, and background model.
    """
    for img_path in img_array:
        img = cv2.imread(img_path)
        img_output = algorithm.apply(img)
        img_bgmodel = algorithm.getBackgroundModel()

        cv2.imshow("Original Image", img)
        cv2.imshow("Foreground Mask", img_output)
        cv2.imshow("Background Model", img_bgmodel)

        if cv2.waitKey(10) & 0xFF == 27:  # Exit if ESC is pressed
            break

        print("Frames left: " + str(len(img_array) - img_array.index(img_path)))


outfile = "./speed_exlude_cap_read.csv"
dfmk = None

SAVE_CSV = False  # Set to True to save results to CSV file
FRAME_BY_FRAME = False  # Set to True to process video frame by frame
SHOW_FOREGROUND_MASK = True
SCALE_VIDEO_OUTPUT = 0.25

CUSTOM_ALGORITHM_LIST = True
GRID_SIZE = 0

def process_video(video_file, algorithm):
    """
    Process each frame of the specified video file with the given algorithm.
    Displays original video frame, foreground mask, and background model.
    """
    global dfmk, outfile, SAVE_CSV, FRAME_BY_FRAME, SHOW_FOREGROUND_MASK, GRID_SIZE, SCALE_VIDEO_OUTPUT
    frame_res = None
    dfmk = csvfile.DFCreator()
    dfmk.create_table("speed", ["Algorithm", "FPS", "Video File"])
    capture = cv2.VideoCapture(video_file)
    while not capture.isOpened():
        capture = cv2.VideoCapture(video_file)
        cv2.waitKey(1000)
        print("Waiting for the video to be loaded...")
    print(f"Proc video: {video_file}, Algorithm: {type(algorithm).__name__}")
    fps_ls = []
    frame_cnt = 0
    SLOW_CNT_CHECK = 10  # number of frames to check for slow processing
    while True:
        flag, frame = capture.read()
        start_time = time.time()  # time before processing frame
        if frame_res is None:
            frame_res = frame.shape[:2][::-1]
        if not flag:
            pprint(">>No more frames to read or error in reading the frame.")
            break

        fg_mask = algorithm.apply(frame)
        end_time = time.time()
        fps = 1.0 / (end_time - start_time)
        fps_ls.append(fps)
        # img_bgmodel = algorithm.getBackgroundModel()
        # print(f"FPS: {fps:.2f}")
        # Overlay FPS on foreground mask
        alg_name = str(type(algorithm).__name__)
        cv2.putText(
            frame,
            f"{alg_name} - FPS: {fps:.2f} - res: {frame_res}",
            org=(20, 100),  # position: top-left corner
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=(0, 255, 255),  # yellow in BGR
            thickness=2,
        )
        oframe = frame
        if SCALE_VIDEO_OUTPUT < 1.0:
            oframe = cv2.resize(
                frame,
                (0, 0),  # 0,0 means the size is computed from fx, fy
                fx=SCALE_VIDEO_OUTPUT,
                fy=SCALE_VIDEO_OUTPUT,
                interpolation=cv2.INTER_LINEAR,
            )
        cv2.imshow("Original Video", oframe)

        if SHOW_FOREGROUND_MASK:
            if GRID_SIZE > 0:
                # Draw grid on the foreground mask
                block_size = GRID_SIZE
                # Convert to BGR if it's a single channel
                if len(fg_mask.shape) == 2:
                    img_grid = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
                else:
                    img_grid = fg_mask.copy()

                height, width = img_grid.shape[:2]

                # Draw vertical lines
                for x in range(0, width, block_size):
                    cv2.line(img_grid, (x, 0), (x, height), color=(0, 255, 0), thickness=1)

                # Draw horizontal lines
                for y in range(0, height, block_size):
                    cv2.line(img_grid, (0, y), (width, y), color=(0, 255, 0), thickness=1)

                fg_mask = img_grid
            if SCALE_VIDEO_OUTPUT < 1.0:
                    # Resize the foreground mask if SCALE_VIDEO_OUTPUT is less than 1.0
                fg_mask = cv2.resize(
                    fg_mask,
                    (0, 0),  # 0,0 means the size is computed from fx, fy
                    fx=SCALE_VIDEO_OUTPUT,
                    fy=SCALE_VIDEO_OUTPUT,
                    interpolation=cv2.INTER_LINEAR,
                )

            cv2.imshow("Foreground Mask", fg_mask)

        frame_cnt += 1

        if frame_cnt >= SLOW_CNT_CHECK:
            current_avg_fps = sum(fps_ls)/ len(fps_ls)
            if current_avg_fps < 5.0:
                print(f"FPS too low ({current_avg_fps:.2f}), stopping processing.")
                break
        if FRAME_BY_FRAME:
            # Wait for a key press to proceed to the next frame
            key = cv2.waitKey(0)
            if key & 0xFF == 27:
                # ESC to exit
                break
        else:
            if cv2.waitKey(10) & 0xFF == 27:  # Exit if ESC is pressed
                break
    # end of while loop
    fps_avg = sum(fps_ls) / len(fps_ls)
    print(f"Average FPS: {fps_avg:.2f}")
    if SAVE_CSV and (dfmk is not None):
        print("Saving results to CSV file...")
        dfmk.insert_rows("speed", [alg_name, fps_avg, video_file])
        dfmk.fill_table_from_row_pool("speed")
        df = dfmk["speed"]
        if not os.path.exists(outfile):
            df.to_csv(outfile, index=False, sep=";")
        else:
            df.to_csv(outfile, mode="a", header=False, index=False, sep=";")
    dfmk = None
