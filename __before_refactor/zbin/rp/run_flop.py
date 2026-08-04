from halib import *
import torch
import timm
from fvcore.nn import FlopCountAnalysis

# ── Config ────────────────────────────────────────────────────────────
MODEL_NAME = "hgnetv2_b5.ssld_stage2_ft_in1k"
NUM_CLASSES = 3
INPUT_SIZE = (3, 360, 640)  # (C, H, W)

# ── Build model — no pretrained weights ───────────────────────────────
model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
model.eval()

# ── Custom dummy input ────────────────────────────────────────────────
dummy_input = torch.randn(1, *INPUT_SIZE)

# ── FLOPs calculation ─────────────────────────────────────────────────
flops = FlopCountAnalysis(model, dummy_input)
gflops = flops.total() / 1e9
mflops = flops.total() / 1e6
with ConsoleLog("FLOPs Calculation"):
    pprint(f"Model      : {MODEL_NAME}")
    pprint(f"Classes    : {NUM_CLASSES}")
    pprint(f"Input size : {INPUT_SIZE}")
    pprint(f"FlOPs = Floating-Point Operations")
    pprint(f"MFLOPs = 10^6 FLOPs, GFLOPs = 10^9 FLOPs")
    pprint(f"Total FLOPs: {gflops:.3f} GFLOPs = {mflops:.3f} MFLOPs")
