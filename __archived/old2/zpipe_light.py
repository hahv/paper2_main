import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import time
from timebudget import timebudget
from halib import *
import torchvision.models as models
import line_profiler
import cv2

class FireSmokeCNN(nn.Module):
    def __init__(self):
        super(FireSmokeCNN, self).__init__()
        # First convolutional layer: 3 input channels (for RGB) to 8 output channels
        # Input size: (batch_size, 3, 16, 16)
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        # First pooling layer
        # Output after pooling: (batch_size, 8, 8, 8)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional layer: 8 input channels to 16 output channels
        # Input size: (batch_size, 8, 8, 8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        # Second pooling layer
        # Output after pooling: (batch_size, 16, 4, 4)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flattened layer size: 16 channels * 4 * 4 spatial dimensions
        self.fc1 = nn.Linear(16 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 2)  # 2 classes: Fire/Smoke and No Fire/Smoke
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Apply first conv and pool with ReLU activation
        x = self.pool1(F.relu(self.conv1(x)))
        # Apply second conv and pool with ReLU activation
        x = self.pool2(F.relu(self.conv2(x)))

        x = x.reshape(x.size(0), -1)  # Ensure compatibility with batch size

        # Fully connected layers with ReLU and Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# Function to divide image into 16x16 grid and select blocks

# @line_profiler.profile
# --- Block extraction ---
def extract_and_stack_blocks(normed_input_tensor, num_blocks=100, block_size=16):
    """
    Extract random normalized blocks from an input tensor.
    normed_input_tensor: torch.Tensor with shape (3, H, W), already normalized.
    Returns: torch.Tensor with shape (num_blocks, 3, block_size, block_size)
    """
    # Convert to numpy for easier block splitting
    img = normed_input_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
    height, width = img.shape[:2]

    # Calculate grid dimensions
    grid_height = height // block_size
    grid_width = width // block_size

    # Ensure divisibility
    img = img[: grid_height * block_size, : grid_width * block_size, :]
    height, width = img.shape[:2]

    # Reshape into blocks
    blocks = img.reshape(grid_height, block_size, grid_width, block_size, 3)
    blocks = blocks.transpose(0, 2, 1, 3, 4)  # (grid_h, grid_w, b, b, 3)
    blocks = blocks.reshape(-1, block_size, block_size, 3)  # (total_blocks, b, b, 3)

    # Randomly select
    total_blocks = grid_height * grid_width
    num_blocks = min(num_blocks, total_blocks)
    selected_idx = np.random.choice(total_blocks, size=num_blocks, replace=False)
    selected_blocks = blocks[selected_idx]  # (num_blocks, b, b, 3)

    # Back to torch, normalized, shape: (num_blocks, 3, b, b)
    selected_blocks = torch.from_numpy(selected_blocks).permute(0, 3, 1, 2).float()
    return selected_blocks


# Function for batch inference
# @line_profiler.profile
def infer_batch(model, blocks, device):
    blocks = blocks.to(device)
    with torch.no_grad():
        outputs = model(blocks)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
    return preds


def get_video_info(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()
    return width, height, fps, frames

@line_profiler.profile
def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = FireSmokeCNN().to(device)

    video_source = r"E:\NextCloud\paper3\datasets\TestVideos\smallfire.mp4"
    width, height, fps, frames = get_video_info(video_source)
    maximum_block_count = (height // 16) * (width // 16)

    # Generate synthetic block images from input image
    NUM_FRAME_TRIAL = 20
    frame_idx = 0
    time_ls = []

    # Open video source (0 = webcam, or replace with filename)
    cap = cv2.VideoCapture(video_source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        start_time = time.time()
        frame_idx += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        # Normalize the full frame
        preprocess = transforms.Compose([
            transforms.ToTensor(),  # (C,H,W) in [0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
        # num_block = np.random.randint(1, maximum_block_count + 1)  # Random number of blocks
        num_block = maximum_block_count
        normed_tensor = preprocess(pil_img)  # (3,H,W)
        with timebudget(f"--data prepare: {num_block} blocks"):
            data_start_time = time.time()
            # Extract and stack blocks from the input image
            synthetic_block_img = extract_and_stack_blocks(normed_tensor, num_block)
            data_time = time.time() - data_start_time
        with timebudget(f">> Infer - block {num_block}"):
            start_infer_time = time.time()
            # Perform inference on the synthetic blocks
            infer_batch(
                model=model, blocks=synthetic_block_img, device=device
            )
            infer_time = time.time() - start_infer_time

            total_time = time.time() - start_time
            time_ls.append((frame_idx, num_block, data_time, infer_time, total_time))
        if frame_idx >= NUM_FRAME_TRIAL:
            print("Processed enough frames, exiting...")
            break
    # time_ls = time_ls[:-2]  # last frame very slow do not know why, so remove it
    time_ls = time_ls[1:]  # remove the first frame as it has no previous points
    header = ("frame_idx", "num_block", "data_time", "infer_time", "total_time")
    pprint([header] + time_ls)
    avg_data_time = np.mean([x[2] for x in time_ls])
    avg_infer_time = np.mean([x[3] for x in time_ls])
    avg_total_time = np.mean([x[4] for x in time_ls])
    print(f"Average Data Preparation Time: {avg_data_time:.4f} seconds")
    print(f"Average Inference Time: {avg_infer_time:.4f} seconds")
    print(f"Average Total Time: {avg_total_time:.4f} seconds")

# Main execution
if __name__ == "__main__":
    main()
