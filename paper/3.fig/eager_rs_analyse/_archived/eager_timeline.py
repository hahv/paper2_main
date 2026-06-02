import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import ConnectionPatch
import numpy as np
import os
import cv2

matplotlib.use("Agg")

# 1. Load the dataset
df = pd.read_csv("paper/3.fig/eager_analyse/eager_timeline.csv", sep=";")

# Increased size to accommodate multiple insets alongside the legends
fig, axes = plt.subplots(
    3, 1, figsize=(16, 8.5), sharex=True, gridspec_kw={"height_ratios": [3.2, 1, 1]}
)
fig.subplots_adjust(hspace=0.1)

frames = df["frame"].values

# ── Row 1: Classifier prediction ──────────────────────────
ax1 = axes[0]
for _, row in df.iterrows():
    f, val, prob = row["frame"], row["classifier"], row["prob"]
    color = "#d62728" if val == 1 else "#aec7e8"
    ax1.bar(f, prob, color=color, width=0.8, align="center", alpha=0.85)

    # Rotated probability text inside bar
    ax1.text(
        f,
        prob / 2,
        f"{prob:.2f}",
        ha="center",
        va="center",
        fontsize=6.5,
        fontweight="bold",
        color="white" if val == 1 else "#333333",
        rotation=-90,
    )

ax1.set_ylabel("Classifier $\\mathcal{M}$\n(confidence)", fontsize=9, labelpad=8)
ax1.set_ylim(0, 1.9)  # Higher limits to clear space for the multi-insets
ax1.set_yticks([0, 0.5, 1.0])
ax1.tick_params(axis="y", labelsize=8)
ax1.spines[["top", "right"]].set_visible(False)
ax1.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
ax1.text(35.6, 0.5, "threshold", fontsize=7, color="gray", va="center")

# ── Row 2 & 3: Decisions & Eager Mode ──────────────────────
for i, col, label, colors in zip(
    [1, 2],
    ["skipped", "eager_mode"],
    ["Skip\ndecision $s_t$", "Eager\nmode"],
    [("#98df8a", "#ff9896"), ("#ff7f0e", "#c7c7c7")],
):
    ax = axes[i]
    for _, row in df.iterrows():
        f, val = row["frame"], row[col]
        color = colors[0] if val == 1 else colors[1]
        ax.bar(f, 1, color=color, width=0.8, align="center", alpha=0.85)
        if col == "skipped":
            ax.text(
                f,
                0.5,
                "S" if val == 1 else "R",
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
            )
    ax.set_yticks([])
    ax.set_ylabel(label, fontsize=9, labelpad=8)
    ax.set_ylim(0, 1.4)
    ax.spines[["top", "right", "left"]].set_visible(False)

# ── Eager mode boundary lines ──────────────────────────────
eager_start = df[df["eager_mode"] == 1]["frame"].min()
eager_end = df[df["eager_mode"] == 1]["frame"].max()

for ax in axes:
    ax.axvline(
        eager_start - 0.5, color="darkorange", linestyle="--", linewidth=1.3, alpha=0.9
    )
    ax.axvline(
        eager_end + 0.5, color="steelblue", linestyle="--", linewidth=1.3, alpha=0.9
    )

axes[0].annotate(
    "Eager triggered",
    xy=(eager_start, 1.05),
    fontsize=8,
    color="darkorange",
    ha="center",
    fontweight="bold",
)
axes[0].annotate(
    "Eager cleared",
    xy=(eager_end, 1.05),
    fontsize=8,
    color="steelblue",
    ha="center",
    fontweight="bold",
)

# ── 🔍 Multiple Inset Video Frames (No Overlap) ──────────
target_frames = [14, 15]  # Array of frames to visualize

# Spaced out towards the center/right area to prevent overlap with the top-left legend
inset_x_positions = np.linspace(0.40, 0.75, len(target_frames))

for idx, t_frame in enumerate(target_frames):
    # --- Frame Extraction (Mock or Real) ---
    img = None
    if "video" in df.columns:
        video_path = df.loc[df["frame"] == t_frame, "video"].values[0]
        if os.path.exists(str(video_path)):
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, t_frame - 1)
            ret, frame = cap.read()
            if ret:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cap.release()

    if img is None:  # Fallback image layout
        img = np.ones((50, 70, 3), dtype=np.uint8) * 240
        img[15:35, 25:45, 0 if idx == 0 else 2] = 200

    # --- Create Inset ---
    ax_ins = ax1.inset_axes([inset_x_positions[idx], 0.65, 0.14, 0.3])
    ax_ins.imshow(img)
    ax_ins.set_xticks([])
    ax_ins.set_yticks([])
    ax_ins.set_title(f"Frame {t_frame}", fontsize=8, fontweight="bold")
    for spine in ax_ins.spines.values():
        spine.set_linewidth(0.8)

    # --- Connection Lines ---
    prob_val = df[df["frame"] == t_frame]["prob"].values[0]
    for corner_x in [0, 1]:  # Left and right corners of the inset image box
        con = ConnectionPatch(
            xyA=(corner_x, 0),
            xyB=(t_frame + (0.4 if corner_x else -0.4), prob_val),
            coordsA="axes fraction",
            coordsB="data",
            axesA=ax_ins,
            axesB=ax1,
            color="#555555",
            linestyle="--",
            linewidth=0.8,
        )
        ax1.add_artist(con)

# ── Legends (Restored) ─────────────────────────────────────
leg1 = [
    mpatches.Patch(color="#d62728", alpha=0.85, label="Positive (fire/smoke)"),
    mpatches.Patch(color="#aec7e8", alpha=0.85, label="Negative"),
]
leg2 = [
    mpatches.Patch(color="#98df8a", alpha=0.85, label="Skipped (S)"),
    mpatches.Patch(color="#ff9896", alpha=0.85, label="Inference run (R)"),
]
leg3 = [
    mpatches.Patch(color="#ff7f0e", alpha=0.85, label="Eager ON"),
    mpatches.Patch(color="#c7c7c7", alpha=0.85, label="Eager OFF"),
]

axes[0].legend(handles=leg1, fontsize=7.5, loc="upper left", framealpha=0.8)
axes[1].legend(handles=leg2, fontsize=7.5, loc="upper left", framealpha=0.8)
axes[2].legend(handles=leg3, fontsize=7.5, loc="upper left", framealpha=0.8)

plt.suptitle(
    "Eager Mode Timeline: False-Alarm Scenario", fontsize=11, fontweight="bold", y=1.01
)
axes[2].set_xticks(frames)
axes[2].set_xticklabels(frames, fontsize=7)
axes[2].set_xlabel("Frame index", fontsize=10)

plt.tight_layout()
plt.savefig("paper/3.fig/eager_analyse/eager_timeline.png", dpi=180, bbox_inches="tight")
print("Saved.")