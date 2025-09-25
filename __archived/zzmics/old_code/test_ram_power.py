import torch
import torchvision.models as models
import time
import random
import numpy as np
import os

# ---- CONFIG ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True).eval().to(device)


# Simulate 224x224 RGB input
def generate_random_frame():
    return torch.rand(1, 3, 224, 224)


# Motion detection stub (random)
def motion_detected():
    return random.random() < 0.5  # 50% chance


# GPU mem and power logging
def log_gpu_usage(label):
    mem = torch.cuda.memory_allocated(device) / (1024 * 1024)
    print(f"{label} | GPU Memory: {mem:.2f} MiB")


# Optional: NVIDIA-SMI power reading
def get_power_draw():
    try:
        out = os.popen(
            "nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits"
        ).read()
        return float(out.strip())
    except:
        return None


# ---- MAIN LOOP ----
torch.cuda.empty_cache()
print("Starting test loop...\n")

for i in range(2000):  # simulate 20 frames
    frame = generate_random_frame()
    is_motion = motion_detected()

    if is_motion:
        with torch.no_grad():
            input_tensor = frame.to(device)
            _ = model(input_tensor)
        log_gpu_usage(f"[{i:02d}] Inference")
    else:
        time.sleep(0.05)  # simulate idle delay
        log_gpu_usage(f"[{i:02d}] Skipped")
        torch.cuda.empty_cache()  # Uncomment to force memory release

    # Optional: log power
    power = get_power_draw()
    if power:
        print(f"       Power Draw: {power:.2f} W\n")
