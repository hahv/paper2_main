"""
eager_timeline_failure.py  -  CASE A: False-Alarm / Eager Stuck in Static Scene

Video   : aihub__lb_none__0175  (background-only, ground truth = None)
Frames  : 1-50
Story   : Skip module correctly skips static background frames. The periodic
          Nchk forced-inference check runs DL on frame 34; the classifier wrongly
          predicts SmokeOnly (Wfire=1) triggering Eager mode immediately. Eager
          stays locked ON for the rest of the video -- a false-alarm failure mode.

Layout (2 rows):
  Row 1 (tall) : Classifier M confidence bars
  Row 2 (short): Skip decision per frame
  Eager ON/OFF encoded as background tint on both rows (no 3rd subplot needed).

Run:
  python eager_timeline_failure.py
Output:
  eager_timeline_failure.png
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

df = pd.read_csv(BASE_CSV + "_eager_timeline_failure.csv", sep=";")
frames = df["frame"].values

fig, axes = plt.subplots(
    2,
    1,
    figsize=(18, 6),
    sharex=True,
    gridspec_kw={"height_ratios": [3.5, 1]},
)
fig.subplots_adjust(hspace=0.06)
ax1, ax2 = axes

# ── Background tint: normal mode (pre-eager, both rows) ──────────────────────
pre = df[df["eager_mode"] == 0]
if len(pre):
    x0, x1 = int(pre["frame"].min()) - 0.5, int(pre["frame"].max()) + 0.5
    for ax in axes:
        ax.axvspan(x0, x1, color="#e8f5e9", alpha=0.45, zorder=0)

# ── Background tint: Eager ON region — orange wash shows the problem ─────────
eager_on = df[df["eager_mode"] == 1]
if len(eager_on):
    xe0 = int(eager_on["frame"].min()) - 0.5
    xe1 = int(eager_on["frame"].max()) + 0.5
    for ax in axes:
        ax.axvspan(xe0, xe1, color="#fff3e0", alpha=0.55, zorder=0)

# ── Row 1: Classifier confidence ─────────────────────────────────────────────
for _, row in df.iterrows():
    f = int(row["frame"])
    val = int(row["classifier"])
    prob = float(row["prob"])

    color = "#d62728" if val == 1 else "#aec7e8"
    ax1.bar(f, max(prob, 0.04), color=color, width=0.8, align="center", alpha=0.88)

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
ax1.set_ylim(0, 1.62)
ax1.set_yticks([0, 0.5, 1.0])
ax1.tick_params(axis="y", labelsize=9)
ax1.spines[["top", "right"]].set_visible(False)
ax1.axhline(0.5, color="gray", linestyle=":", linewidth=0.9, alpha=0.6)
ax1.text(
    frames[-1] + 0.8, 0.51, "0.5 threshold", fontsize=7.5, color="gray", va="bottom"
)

# Ground truth note
ax1.text(
    0.01,
    0.97,
    "Ground truth: background only (None)  |  ALL positive predictions are FALSE ALARMS",
    transform=ax1.transAxes,
    fontsize=8.5,
    color="#7f0000",
    fontstyle="italic",
    va="top",
    bbox=dict(boxstyle="round,pad=0.3", fc="#fff0f0", ec="#d62728", lw=0.8),
)

# ── Row 2: Skip decision ──────────────────────────────────────────────────────
for _, row in df.iterrows():
    f, val = int(row["frame"]), int(row["skipped"])
    color = "#98df8a" if val == 1 else "#ff9896"
    ax2.bar(f, 1, color=color, width=0.8, align="center", alpha=0.88)
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

# ── Zone labels ───────────────────────────────────────────────────────────────
if len(pre):
    axes[0].text(
        (x0 + x1) / 2,
        1.47,
        "Normal mode\n(skip active)",
        ha="center",
        fontsize=8,
        color="#1b5e20",
        fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.25", fc="#f1f8f1", ec="#81c784", lw=0.8),
    )

if len(eager_on):
    axes[0].text(
        (xe0 + xe1) / 2,
        1.47,
        "Eager mode ON — locked\n(FAR sustained)",
        ha="center",
        fontsize=8,
        color="#b85c00",
        fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.25", fc="#fff8f0", ec="darkorange", lw=0.8),
    )

# ── Eager activation line ─────────────────────────────────────────────────────
eager_start = int(eager_on["frame"].min())
for ax in axes:
    ax.axvline(
        eager_start - 0.5,
        color="darkorange",
        linestyle="--",
        linewidth=1.8,
        alpha=0.95,
        zorder=5,
    )

prob_at_start = float(df[df["frame"] == eager_start]["prob"].values[0])
axes[0].annotate(
    f"Eager ON\n(frame {eager_start})\nstuck, never clears",
    xy=(eager_start - 0.5, prob_at_start),
    xytext=(eager_start + 5, 1.32),
    fontsize=8.5,
    color="darkorange",
    fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.3),
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="darkorange", lw=0.9),
)

# ── Nchk forced-check false alarm marker ─────────────────────────────────────
fp_df = df[(df["skipped"] == 0) & (df["classifier"] == 1)]
if len(fp_df):
    first_fp = int(fp_df["frame"].min())
    for ax in axes:
        ax.axvline(
            first_fp, color="#aa0000", linestyle=":", linewidth=1.3, alpha=0.9, zorder=5
        )
    prob_fp = float(df[df["frame"] == first_fp]["prob"].values[0])
    axes[0].annotate(
        f"$N_{{chk}}$ forced check\n→ false alarm\n(frame {first_fp})",
        xy=(first_fp, prob_fp),
        xytext=(first_fp - 11, 1.28),
        fontsize=8,
        color="#aa0000",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#aa0000", lw=1.1),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#aa0000", lw=0.8),
    )

# ── Legends ───────────────────────────────────────────────────────────────────
ax1.legend(
    handles=[
        mpatches.Patch(
            color="#d62728", alpha=0.88, label="Positive prediction (FALSE ALARM — FP)"
        ),
        mpatches.Patch(
            color="#aec7e8", alpha=0.88, label="Negative prediction (correct)"
        ),
        mpatches.Patch(
            color="#fff3e0",
            alpha=0.8,
            ec="darkorange",
            label="Eager mode ON — locked (orange tint)",
        ),
        mpatches.Patch(
            color="#e8f5e9",
            alpha=0.8,
            ec="#81c784",
            label="Normal mode — skip active (green tint)",
        ),
    ],
    fontsize=8,
    loc="upper left",
    framealpha=0.92,
    bbox_to_anchor=(0.01, 0.93),
)
ax2.legend(
    handles=[
        mpatches.Patch(color="#98df8a", alpha=0.88, label="Skipped (S)"),
        mpatches.Patch(color="#ff9896", alpha=0.88, label="Inference run (R)"),
    ],
    fontsize=8,
    loc="lower right",
    framealpha=0.9,
)

# ── X-axis & title ────────────────────────────────────────────────────────────
ax2.set_xticks(frames[::5])
ax2.set_xticklabels(frames[::5], fontsize=8)
ax2.set_xlabel("Frame index", fontsize=11)

plt.suptitle(
    "Case A — Eager Mode Failure: False-Alarm Locks Eager ON in Static Background Scene\n"
    "Video: aihub__lb_none__0175   |   Ground truth: background only (None)   |   Frames 1–50",
    fontsize=11,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()

outfile = BASE_DIR + "/eager_timeline_failure.png"
plt.savefig(outfile, dpi=180, bbox_inches="tight")
pprint_local_path(outfile, get_wins_path=True, using_box=True)
