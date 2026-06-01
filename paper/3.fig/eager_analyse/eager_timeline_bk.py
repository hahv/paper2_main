import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Load the dataset
df = pd.read_csv("paper/3.fig/eager_analyse/eager_timeline.csv", sep=";")

fig, axes = plt.subplots(
    3, 1, figsize=(14, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]}
)
fig.subplots_adjust(hspace=0.08)

frames = df["frame"].values

# ── Row 1: Classifier prediction + prob inside bar ────────
ax1 = axes[0]
for _, row in df.iterrows():
    f, val, prob = row["frame"], row["classifier"], row["prob"]
    color = "#d62728" if val == 1 else "#aec7e8"
    bar_h = prob  # bar height = probability
    ax1.bar(f, bar_h, color=color, width=0.8, align="center", alpha=0.85)

    # Show prob value inside bar (rotated 90 degrees to write from top to bottom)
    txt_y = bar_h / 2
    txt_color = "white" if val == 1 else "#333333"
    ax1.text(
        f,
        txt_y,
        f"{prob:.2f}",
        ha="center",
        va="center",
        fontsize=6.5,
        fontweight="bold",
        color=txt_color,
        rotation=-90,  # Rotates the text to write from top to bottom
    )

ax1.set_ylabel("Classifier $\\mathcal{M}$\n(confidence)", fontsize=9, labelpad=8)
ax1.set_ylim(0, 1.1)
ax1.set_yticks([0, 0.5, 1.0])
ax1.tick_params(axis="y", labelsize=8)
ax1.spines[["top", "right"]].set_visible(False)
ax1.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
ax1.text(35.6, 0.5, "threshold", fontsize=7, color="gray", va="center")

# ── Row 2: Skip decision ───────────────────────────────────
ax2 = axes[1]
for _, row in df.iterrows():
    f, val = row["frame"], row["skipped"]
    color = "#98df8a" if val == 1 else "#ff9896"
    ax2.bar(f, 1, color=color, width=0.8, align="center", alpha=0.85)
    label = "S" if val == 1 else "R"
    ax2.text(
        f,
        0.5,
        label,
        ha="center",
        va="center",
        fontsize=7,
        fontweight="bold",
        color="#333333",
    )
ax2.set_yticks([])
ax2.set_ylabel("Skip\ndecision $s_t$", fontsize=9, labelpad=8)
ax2.set_ylim(0, 1.4)
ax2.spines[["top", "right", "left"]].set_visible(False)

# ── Row 3: Eager mode state ────────────────────────────────
ax3 = axes[2]
for _, row in df.iterrows():
    f, val = row["frame"], row["eager_mode"]
    color = "#ff7f0e" if val == 1 else "#c7c7c7"
    ax3.bar(f, 1, color=color, width=0.8, align="center", alpha=0.85)
ax3.set_yticks([])
ax3.set_ylabel("Eager\nmode", fontsize=9, labelpad=8)
ax3.set_ylim(0, 1.4)
ax3.spines[["top", "right", "left"]].set_visible(False)
ax3.set_xlabel("Frame index", fontsize=10)

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

# ── Legends ────────────────────────────────────────────────
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
ax3.set_xticks(frames)
ax3.set_xticklabels(frames, fontsize=7)
plt.tight_layout()
plt.savefig(
    "paper/3.fig/eager_analyse/eager_timeline.png", dpi=180, bbox_inches="tight"
)
print("Saved.")
