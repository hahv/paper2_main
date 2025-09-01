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
from numpy.lib.stride_tricks import as_strided


base_timm_model = "hgnetv2_b5.ssld_stage2_ft_in1k"
model_path = "models/prof_hgnetv2_b5.ssld_stage2_ft_in1k-20250620-50-98.78.pth"
class_names = [
    "Fire",
    "Normal",
    "SmokeOnly",
]
algorithm = bgs.FrameDifference() #background subtraction algorithm

extract_block_func_name = ""

BLOCK_SIZE = 32  # size of each block to extract
NUM_FRAME_TRIAL = 100  # number of frames to process for profiling
SCALE = 0.5
SKIP_ENABLED = True  # skip module enabled
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
video_source = r"dataset/smallfire.mp4"
# video_source = r"./dataset/real_cam.mp4"
# video_source = r"dataset/wrong_fd_cam_stream.mp4"
# video_source = r"dataset/f6.mp4"
zprofiler = zProfiler()
big_model_transform = transforms.Compose(
    [
        transforms.Resize((360, 640)),  # keep ratio fixed for inference
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
small_model_transform = transforms.Compose(
    [
        transforms.ToTensor(),  # (C,H,W) in [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

def load_model(base_timm_model, class_names, model_path):
    model = timm.create_model(
        base_timm_model,
        pretrained=False,
        num_classes=len(class_names),
        checkpoint_path=model_path,
    )
    model.eval()  # evaluation mode
    global device
    model = model.to(device=device)
    return model


class FireSmokeCNN16(nn.Module):
    def __init__(self):
        super(FireSmokeCNN16, self).__init__()
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


class FireSmokeCNN32(nn.Module):
    def __init__(self):
        super(FireSmokeCNN32, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 8 * 8, 64)  # updated flattened size
        self.fc2 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.reshape(x.size(0), -1)  # Ensure compatibility with batch size
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TinyFireSmokeCNN(nn.Module):
    def __init__(self):
        super(TinyFireSmokeCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)  # fewer channels
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)  # fewer channels
        self.pool2 = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(8 * 8 * 8, 16)  # smaller fc layer
        self.fc2 = nn.Linear(16, 2)  # 2 classes
        self.dropout = nn.Dropout(0.2)  # lighter dropout

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        # x = x.view(x.size(0), -1)
        x = x.reshape(x.size(0), -1)  # Ensure compatibility with batch size
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def extract_all_blocks_cpu_transform(pil_img, block_size=16):
    """
    Input: pil_img (PIL Image)
    Output: blocks (num_blocks, C, block_size, block_size) float32 PyTorch tensor
    """
    # 1. Apply transform to get normalized tensor (C,H,W)
    zprofiler.step_start("extract_all_blocks_cpu_transform", "transform")
    normed_tensor = small_model_transform(pil_img)  # (C,H,W) float32, normalized
    zprofiler.step_end("extract_all_blocks_cpu_transform", "transform")

    # 2. Convert to NumPy for as_strided
    zprofiler.step_start("extract_all_blocks_cpu_transform", "to_numpy")
    tensor = normed_tensor.numpy()  # (C,H,W) float32
    zprofiler.step_end("extract_all_blocks_cpu_transform", "to_numpy")

    # 3. Extract blocks using as_strided
    zprofiler.step_start("extract_all_blocks_cpu_transform", "unfold")
    C, H, W = tensor.shape
    stride = tensor.strides
    blocks = as_strided(
        tensor,
        shape=(H // block_size, W // block_size, C, block_size, block_size),
        strides=(
            stride[1] * block_size,
            stride[2] * block_size,
            stride[0],
            stride[1],
            stride[2],
        ),
    )
    blocks = blocks.reshape(-1, C, block_size, block_size)
    zprofiler.step_end("extract_all_blocks_cpu_transform", "unfold")

    # 4. Convert back to PyTorch tensor and return
    zprofiler.step_start("extract_all_blocks_cpu_transform", "to_torch")
    result = torch.from_numpy(blocks).float()
    zprofiler.step_end("extract_all_blocks_cpu_transform", "to_torch")
    return result


def extract_all_blocks_cpu(frame_bgr, block_size=16):
    """
    Input: frame_bgr (H,W,3) uint8
    Output: blocks (num_blocks, C, block_size, block_size) float32 normalized
    """

    # 1. Convert to float32 once
    frame = torch.from_numpy(frame_bgr).float() / 255.0  # (H,W,3)

    # 2. BGR -> RGB
    frame = frame[..., [2, 1, 0]]  # (H,W,3)

    # 3. Normalize (vectorized)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3)
    frame = (frame - mean) / std  # (H,W,3)

    # 4. HWC -> CHW
    tensor = frame.permute(2, 0, 1).contiguous()  # (C,H,W)

    # 5. Use unfold (optimized C++ kernel) to split blocks
    C, H, W = tensor.shape
    blocks = tensor.unfold(1, block_size, block_size).unfold(2, block_size, block_size)
    # Shape: (C, H//B, W//B, B, B)

    # 6. Rearrange into (num_blocks, C, B, B) without extra copies
    blocks = blocks.permute(1, 2, 0, 3, 4).reshape(-1, C, block_size, block_size)

    return blocks


def extract_all_blocks_cpu_cv2(frame_bgr, block_size=16):
    global zprofiler
    zprofiler.ctx_start("extract_all_blocks_cpu_cv2")
    # 1. BGR → RGB
    zprofiler.step_start("extract_all_blocks_cpu_cv2", "bgr_to_rgb")
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    zprofiler.step_end("extract_all_blocks_cpu_cv2", "bgr_to_rgb")

    # 2. Convert to float and normalize
    zprofiler.step_start("extract_all_blocks_cpu_cv2", "normalize")
    frame = frame.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    frame = (frame - mean) / std  # (H,W,3)
    zprofiler.step_end("extract_all_blocks_cpu_cv2", "normalize")

    zprofiler.step_start("extract_all_blocks_cpu_cv2", "transpose")
    # 3. HWC → CHW
    tensor = np.transpose(frame, (2, 0, 1))  # (C,H,W)
    zprofiler.step_end("extract_all_blocks_cpu_cv2", "transpose")

    # 4. Block extraction using as_strided (no copies)
    zprofiler.step_start("extract_all_blocks_cpu_cv2", "reshape_blocks")
    C, H, W = tensor.shape
    new_shape = (C, H // block_size, W // block_size, block_size, block_size)
    new_strides = (
        tensor.strides[0],
        block_size * tensor.strides[1],
        block_size * tensor.strides[2],
        tensor.strides[1],
        tensor.strides[2],
    )
    blocks = as_strided(tensor, shape=new_shape, strides=new_strides)
    blocks = blocks.transpose(1, 2, 0, 3, 4).reshape(-1, C, block_size, block_size)
    zprofiler.step_end("extract_all_blocks_cpu_cv2", "reshape_blocks")
    zprofiler.ctx_end("extract_all_blocks_cpu_cv2")

    return torch.from_numpy(blocks)


def extract_all_blocks_cpu_as_strided(frame_bgr, block_size=16):
    """
    Input: frame_bgr (H,W,3) uint8
    Output: blocks (num_blocks, C, block_size, block_size) float32 PyTorch tensor
    """
    zprofiler.step_start("extract_all_blocks_cpu_cv2", "bgr2rgb")

    # 1. Convert to float32 and normalize to [0,1]
    frame = frame_bgr.astype(np.float32) / 255.0  # (H,W,3)

    # 2. BGR -> RGB
    frame = frame[..., [2, 1, 0]]  # (H,W,3)

    # 3. Normalize with mean and std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    frame = (frame - mean) / std  # (H,W,3)

    # 4. HWC -> CHW
    tensor = np.transpose(frame, (2, 0, 1))  # (C,H,W)

    # 5. Use as_strided to extract blocks
    C, H, W = tensor.shape
    stride = tensor.strides
    blocks = as_strided(
        tensor,
        shape=(H // block_size, W // block_size, C, block_size, block_size),
        strides=(
            stride[1] * block_size,
            stride[2] * block_size,
            stride[0],
            stride[1],
            stride[2],
        ),
    )

    # 6. Reshape to (num_blocks, C, block_size, block_size)
    blocks = blocks.reshape(-1, C, block_size, block_size)

    # 7. Convert to PyTorch tensor (float32) and return
    return torch.from_numpy(blocks).float()


def extract_all_blocks_cpu_as_strided_no_norm(frame_bgr, block_size=16):
    """
    Input: frame_bgr (H,W,3) uint8
    Output: blocks (num_blocks, C, block_size, block_size) float32 PyTorch tensor
    """
    zprofiler.step_start("extract_all_blocks_cpu_cv2", "bgr2rgb")

    # 1. Convert to float32 and normalize to [0,1]
    frame = frame_bgr.astype(np.float32) / 255.0  # (H,W,3)

    # 2. BGR -> RGB
    frame = frame[..., [2, 1, 0]]  # (H,W,3)

    # 4. HWC -> CHW
    tensor = np.transpose(frame, (2, 0, 1))  # (C,H,W)

    # 5. Use as_strided to extract blocks
    C, H, W = tensor.shape
    stride = tensor.strides
    blocks = as_strided(
        tensor,
        shape=(H // block_size, W // block_size, C, block_size, block_size),
        strides=(
            stride[1] * block_size,
            stride[2] * block_size,
            stride[0],
            stride[1],
            stride[2],
        ),
    )

    # 6. Reshape to (num_blocks, C, block_size, block_size)
    blocks = blocks.reshape(-1, C, block_size, block_size)

    # 7. Convert to PyTorch tensor (float32) and return
    return torch.from_numpy(blocks).float()


# @line_profiler.profile
def skip_module(tiny_model, cv2_bgr_frame, pil_img_frame, extract_func_name):
    # pprint('Running skip module...')
    should_skip = False
    roi_rect = None
    global small_model_transform
    global zprofiler
    zprofiler.ctx_start("skip_module")
    zprofiler.step_start("skip_module", "bg_subtract")
    scale_factor = 0.75  # scale factor for resizing
    cv2_bgr_frame = cv2.resize(
        cv2_bgr_frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA
    )
    fg_mask = algorithm.apply(cv2_bgr_frame)
    zprofiler.step_end("skip_module", "bg_subtract")
    zprofiler.step_start("skip_module", "block_data_prepare")
    H, W = cv2_bgr_frame.shape[:2]

    # Maximum block count
    max_blocks = (H // BLOCK_SIZE) * (W // BLOCK_SIZE)
    num_blocks = max_blocks  # for benchmarking
    # blocks_cpu = extract_all_blocks_cpu_cv2(frame, block_size=BLOCK_SIZE)
    extract_func = globals()[extract_func_name]
    blocks_cpu = extract_func(cv2_bgr_frame, block_size=BLOCK_SIZE)
    global device

    # Move to GPU just before inference
    blocks = blocks_cpu.to(device)

    zprofiler.step_end("skip_module", "block_data_prepare")
    zprofiler.step_start("skip_module", "infer_tiny_model")
    preds = do_infer(tiny_model, blocks)
    zprofiler.step_end("skip_module", "infer_tiny_model")
    zprofiler.ctx_end("skip_module")
    return should_skip, roi_rect, fg_mask


def do_infer(model, input_tensor):
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
    return preds


# @line_profiler.profile
def main():
    global base_timm_model, class_names, model_path, video_source, NUM_FRAME_TRIAL, SCALE, SKIP_ENABLED, big_model_transform
    global extract_block_func_name

    global zProfiler

    # Load model + transforms
    model = load_model(base_timm_model, class_names, model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if BLOCK_SIZE == 16:
        tiny_model = FireSmokeCNN16().to(device)
    else:
        assert BLOCK_SIZE == 32, "BLOCK_SIZE must be either 16 or 32"
        # tiny_model = FireSmokeCNN32().to(device)
        tiny_model = TinyFireSmokeCNN().to(device)
    tiny_model.eval()  # Set to evaluation mode

    cap = cv2.VideoCapture(video_source)
    frame_idx = 0
    time_ls = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    avg_fps = -1

    while True:
        zprofiler.ctx_start("pipeline")
        zprofiler.step_start("pipeline", "read_frame")
        ret, cv2_bgr_frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        start_time = time.time()

        # Preprocess frame
        img_rgb = cv2.cvtColor(cv2_bgr_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        zprofiler.step_end("pipeline", "read_frame")
        should_skip = False
        fg_mask = None
        zprofiler.step_start("pipeline", "skip_module")
        if SKIP_ENABLED:
            should_skip, _, fg_mask = skip_module(tiny_model, cv2_bgr_frame, pil_img)
        zprofiler.step_end("pipeline", "skip_module")

        should_skip = False  # ! always run big model for profiling
        zprofiler.step_start("pipeline", "big_model_infer")
        if not should_skip:
            zprofiler.ctx_start("big_model_infer")
            zprofiler.step_start("big_model_infer", "preprocess")
            img_tensor = big_model_transform(pil_img).unsqueeze(0).to(device)
            zprofiler.step_end("big_model_infer", "preprocess")
            zprofiler.step_start("big_model_infer", "infer")
            do_infer(model, img_tensor)
            zprofiler.step_end("big_model_infer", "infer")
        zprofiler.step_end("pipeline", "big_model_infer")

        zprofiler.ctx_end("pipeline")

        # !Measure elapsed time
        elapsed = time.time() - start_time
        time_ls.append(elapsed)

        # Overlay prediction on frame
        fps = 1.0 / elapsed if elapsed > 0 else 0
        avg_fps = 1.0 / np.mean(time_ls) if len(time_ls) > 0 else 0
        avg_fps_str = f" (avg {avg_fps:.2f}) " if avg_fps > 0 else ""
        cv2.putText(
            cv2_bgr_frame,
            f"{frame_idx}/{total_frames}-FPS: {fps:.2f}{avg_fps_str}- SKIP_MODULE: {'ON' if SKIP_ENABLED else 'OFF'}- BLOCK_SIZE: {BLOCK_SIZE}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv2_bgr_frame = cv2.resize(
            cv2_bgr_frame, (0, 0), fx=SCALE, fy=SCALE, interpolation=cv2.INTER_LINEAR
        )
        cv2.imshow("Prediction", cv2_bgr_frame)

        if fg_mask is not None:
            # draw grid on the foreground mask
            # convert fg_mask to BGR for display
            fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
            H, W = fg_mask.shape[:2]
            grid_h = H // BLOCK_SIZE
            grid_w = W // BLOCK_SIZE
            for i in range(grid_h):
                cv2.line(fg_mask, (0, i * BLOCK_SIZE), (W, i * BLOCK_SIZE), (255, 0, 0), 1)
            for j in range(grid_w):
                cv2.line(fg_mask, (j * BLOCK_SIZE, 0), (j * BLOCK_SIZE, H), (255, 0, 0), 1)
            fg_mask = cv2.resize(fg_mask, (0, 0), fx=SCALE, fy=SCALE, interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Foreground Mask", fg_mask)

        # stop after NUM_FRAME_TRIAL or ESC
        if frame_idx >= NUM_FRAME_TRIAL:
            break
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

    # drop first frame (cold start) timing
    if len(time_ls) > 1:
        time_ls = time_ls[1:]

    avg_fps = 1.0 / np.mean(time_ls) if len(time_ls) > 0 else 0
    print(f"Processed {frame_idx} frames. Avg FPS: {avg_fps:.2f}")
    zprofiler.ctx_end("big_model_infer")
    zprofiler.meta_info()
    zprofiler.report_and_plot(outdir="docs/profiler", tag=extract_block_func_name)


if __name__ == "__main__":
    pprint("Starting inference profiling...")
    main()
    # with Profiler() as p:
    #     main()
    # with open("profile.html", "w") as f:
    #     f.write(p.output_html())
