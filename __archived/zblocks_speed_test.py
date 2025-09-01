from halib import *

import cv2
import timm
import time
import torch
import line_profiler
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from timebudget import timebudget
import pybgs as bgs
from pyinstrument import Profiler

# import torchvision.transforms.v2 as transforms
from halib.research.profiler import zProfiler


BLOCK_SIZE = 32  # size of each block to extract
NUM_FRAME_TRIAL = 100  # number of frames to process for profiling
SCALE = 0.5
SKIP_ENABLED = True  # skip module enabled
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
video_source = r"dataset/smallfire.mp4"

def frame_to_batch_blocks(frame, block_size=32):
    # Step 2: Convert BGR to RGB and scale to [0, 1]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Shape: (H, W, 3)
    frame_rgb = frame_rgb.astype(np.float32) / 255.0  # Scale to [0, 1]

    # Step 3: Normalize frame (vectorized)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    frame_normalized = (frame_rgb - mean[None, None, :]) / std[
        None, None, :
    ]  # Broadcasting

    # Step 4: Transpose to channel-first (C, H, W) for PyTorch
    frame_normalized = frame_normalized.transpose(2, 0, 1)  # Shape: (3, H, W)

    # Step 5: Extract 32x32 blocks
    C, H, W = frame_normalized.shape
    # Ensure H and W are divisible by block_size
    H_new = H - (H % block_size)
    W_new = W - (W % block_size)
    frame_normalized = frame_normalized[:, :H_new, :W_new]  # Crop to divisible size

    # Reshape to (C, H//block_size, block_size, W//block_size, block_size)
    blocks = frame_normalized.reshape(
        C, H_new // block_size, block_size, W_new // block_size, block_size
    )
    # Transpose to (H//block_size, W//block_size, C, block_size, block_size)
    blocks = blocks.transpose(1, 3, 0, 2, 4)
    # Reshape to (num_blocks, C, block_size, block_size)
    num_blocks = (H_new // block_size) * (W_new // block_size)
    blocks = blocks.reshape(num_blocks, C, block_size, block_size)

    # Step 6: Convert to PyTorch tensor (CPU)
    batch_blocks = torch.from_numpy(blocks)  # Shape: (num_blocks, 3, 32, 32)

    return batch_blocks

zprofile = zProfiler()

def proc_data_prepare(frame, rgb_frame, proc_func):
    global zprofile
    zprofile.ctx_start("proc_data_prepare")
    zprofile.step_start("proc_data_prepare", "frame_to_batch_blocks")
    proc_func(frame=frame, block_size=BLOCK_SIZE)
    zprofile.step_end("proc_data_prepare", "frame_to_batch_blocks")
    zprofile.ctx_end("frame_to_batch_blocks")

    # get the name of the function
    func_name = proc_func.__name__
    return func_name

# @line_profiler.profile
def main():
    global video_source, NUM_FRAME_TRIAL, SCALE, SKIP_ENABLED, BLOCK_SIZE
    global zprofile

    cap = cv2.VideoCapture(video_source)
    frame_idx = 0
    time_ls = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    avg_fps = -1

    while True:
        ret, frame = cap.read()
        time_start = time.time()
        if not ret:
            break
        frame_idx += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        func_name = proc_data_prepare(frame, rgb_frame, frame_to_batch_blocks)
        elapsed_time = time.time() - time_start
        fps = 1.0 / elapsed_time
        time_ls.append(elapsed_time)
        avg_fps = 1.0 / (sum(time_ls) / len(time_ls)) if time_ls else 0
        cv2.putText(
            frame,
            f"{frame_idx}/{total_frames} - FPS: {fps:.2f} (Avg: {avg_fps:.2f})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        frame = cv2.resize(
            frame, (0, 0), fx=SCALE, fy=SCALE, interpolation=cv2.INTER_LINEAR
        )
        cv2.imshow("Prediction", frame)
        # stop after NUM_FRAME_TRIAL or ESC
        if frame_idx >= NUM_FRAME_TRIAL:
            break
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    # pprint(zprofile.get_report_dict(with_detail=True))
    zprofile.report_and_plot(outdir="./docs/profiler", tag=func_name)


if __name__ == "__main__":
    pprint("Starting infer...")
    main()
