"""
eager_timeline_caseB.py  -  CASE B: Eager Mode Rescues Missed Smoke Detections

Video   : aihub__lb_smoke__0208  (slow-developing smoke, ground truth = SmokeOnly)
Frames  : 55-115
Story   : Slow-developing smoke produces minimal inter-frame motion, so the skip
          module keeps blocking frames. Skip-only would miss most detections.
          Around frame 71 the skip module passes a frame; DL correctly detects
          SmokeOnly (Wfire=1) -> Eager activates -> all subsequent frames reach
          the classifier -> smoke is continuously and correctly detected.

Run:
  python eager_timeline_success.py
Output:
  eager_timeline_success.png
"""
from halib import *
import os

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.use("Agg")
BASE_CSV = "paper/3.fig/eager_rs_analyse/raw_csv/"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(BASE_CSV + "_eager_timeline_success.csv", sep=";")
frames = df["frame"].values

fig, axes = plt.subplots(
    3, 1, figsize=(18, 8), sharex=True, gridspec_kw={"height_ratios": [3.2, 1, 1]}
)
fig.subplots_adjust(hspace=0.06)
ax1, ax2, ax3 = axes

# -- Row 1: Classifier confidence --------------------------------------------
for _, row in df.iterrows():
    f, val, prob, sk = (
        int(row["frame"]),
        int(row["classifier"]),
        float(row["prob"]),
        int(row["skipped"]),
    )
    color = (
        "#d62728" if (val == 1 and sk == 0) else ("#c7c7c7" if sk == 1 else "#aec7e8")
    )
    ax1.bar(f, max(prob, 0.04), color=color, width=0.8, align="center", alpha=0.85)
    if f % 5 == 0 and prob > 0.1:
        ax1.text(
            f,
            max(prob, 0.04) / 2,
            f"{prob:.2f}",
            ha="center",
            va="center",
            fontsize=6,
            fontweight="bold",
            color="white" if val == 1 else "#333333",
            rotation=-90,
        )

ax1.set_ylabel("Classifier $\\mathcal{M}$\n(SmokeOnly conf.)", fontsize=10, labelpad=8)
ax1.set_ylim(0, 1.65)
ax1.set_yticks([0, 0.5, 1.0])
ax1.tick_params(axis="y", labelsize=9)
ax1.spines[["top", "right"]].set_visible(False)
ax1.axhline(0.5, color="gray", linestyle=":", linewidth=0.9, alpha=0.6)
ax1.text(
    frames[-1] + 0.8, 0.51, "0.5 threshold", fontsize=7.5, color="gray", va="bottom"
)
ax1.text(
    0.01,
    0.97,
    "Ground truth: SmokeOnly - grey bars would be MISSED without Eager mode (skip-only failure)",
    transform=ax1.transAxes,
    fontsize=8.5,
    color="#005500",
    fontstyle="italic",
    va="top",
    bbox=dict(boxstyle="round,pad=0.3", fc="#f0fff0", ec="#2ca02c", lw=0.8),
)

# -- Row 2: Skip decision ----------------------------------------------------
for _, row in df.iterrows():
    f, val = int(row["frame"]), int(row["skipped"])
    color = "#98df8a" if val == 1 else "#ff9896"
    ax2.bar(f, 1, color=color, width=0.8, align="center", alpha=0.85)
    if f % 5 == 0:
        ax2.text(
            f,
            0.5,
            "S" if val == 1 else "R",
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
        )
ax2.set_yticks([])
ax2.set_ylabel("Skip\ndecision $s_t$", fontsize=10, labelpad=8)
ax2.set_ylim(0, 1.4)
ax2.spines[["top", "right", "left"]].set_visible(False)

# -- Row 3: Eager mode -------------------------------------------------------
for _, row in df.iterrows():
    f, val = int(row["frame"]), int(row["eager_mode"])
    color = "#ff7f0e" if val == 1 else "#c7c7c7"
    ax3.bar(f, 1, color=color, width=0.8, align="center", alpha=0.85)
ax3.set_yticks([])
ax3.set_ylabel("Eager\nmode", fontsize=10, labelpad=8)
ax3.set_ylim(0, 1.4)
ax3.spines[["top", "right", "left"]].set_visible(False)

# -- Pre-eager shading (skip-only miss zone) ---------------------------------
pre = df[df["eager_mode"] == 0]
if len(pre):
    x0, x1 = int(pre["frame"].min()) - 0.5, int(pre["frame"].max()) + 0.5
    for ax in axes:
        ax.axvspan(x0, x1, color="#ffcccc", alpha=0.18, zorder=0)
    axes[0].text(
        (x0 + x1) / 2,
        1.45,
        "Skip-only zone: all frames skipped\n-> smoke undetected",
        ha="center",
        fontsize=8,
        color="#cc0000",
        fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff5f5", ec="#ffaaaa", lw=0.8),
    )

# -- Eager activation line ---------------------------------------------------
eager_on = df[df["eager_mode"] == 1]
if len(eager_on):
    eager_start = int(eager_on["frame"].min())
    for ax in axes:
        ax.axvline(
            eager_start - 0.5,
            color="darkorange",
            linestyle="--",
            linewidth=1.6,
            alpha=0.9,
            zorder=5,
        )
    prob_here = (
        float(df[df["frame"] == eager_start]["prob"].values[0])
        if len(df[df["frame"] == eager_start])
        else 0.7
    )
    axes[0].annotate(
        f"Eager mode ON (frame {eager_start})\nSmoke saved!",
        xy=(eager_start - 0.5, prob_here),
        xytext=(eager_start + 6, 1.38),
        fontsize=8.5,
        color="darkorange",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="darkorange", lw=0.8),
    )

# -- Legends -----------------------------------------------------------------
ax1.legend(
    handles=[
        mpatches.Patch(
            color="#d62728",
            alpha=0.85,
            label="Positive - TP (correctly detected by Eager)",
        ),
        mpatches.Patch(
            color="#c7c7c7", alpha=0.85, label="Skipped - would be MISSED by skip-only"
        ),
        mpatches.Patch(color="#aec7e8", alpha=0.85, label="Run - Negative"),
    ],
    fontsize=8,
    loc="lower right",
    framealpha=0.9,
)
ax2.legend(
    handles=[
        mpatches.Patch(color="#98df8a", alpha=0.85, label="Skipped by skip module (S)"),
        mpatches.Patch(color="#ff9896", alpha=0.85, label="Inference run (R)"),
    ],
    fontsize=8,
    loc="lower right",
    framealpha=0.9,
)
ax3.legend(
    handles=[
        mpatches.Patch(color="#ff7f0e", alpha=0.85, label="Eager mode ON"),
        mpatches.Patch(color="#c7c7c7", alpha=0.85, label="Eager mode OFF"),
    ],
    fontsize=8,
    loc="lower right",
    framealpha=0.9,
)

# -- X-axis & title ----------------------------------------------------------
ax3.set_xticks(frames[::5])
ax3.set_xticklabels(frames[::5], fontsize=8)
ax3.set_xlabel("Frame index", fontsize=11)
plt.suptitle(
    "Case B - Eager Mode Success: Rescuing Missed Smoke Detections (Slow-Developing Smoke)\n"
    "Video: aihub__lb_smoke__0208   |   Ground truth: SmokeOnly   |   Frames 55-115",
    fontsize=11,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()

outfile = BASE_DIR + "/eager_timeline_success.png"
plt.savefig(outfile, dpi=180, bbox_inches="tight")
pprint_local_path(outfile, get_wins_path=True, using_box=True)
