"""
eager_timeline_success.py  -  CASE B: Eager Mode Rescues Missed Smoke Detections

Video   : aihub__lb_smoke__0208  (slow-developing smoke, ground truth = SmokeOnly)
Frames  : 55-115
Story   : Slow-developing smoke produces minimal inter-frame motion, so the skip
          module keeps blocking frames. Skip-only would miss most detections.
          Around frame 71 the skip module passes a frame; DL correctly detects
          SmokeOnly (Wfire=1) -> Eager activates -> all subsequent frames reach
          the classifier -> smoke is continuously and correctly detected.

Layout (2 rows):
  Row 1 (tall) : Classifier M confidence bars  [Pipeline with Eager]
  Row 2 (short): Skip decision per frame       [Skip-only counterfactual]
  Eager ON/OFF encoded as background tint on both rows (no 3rd subplot needed).

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
    2,
    1,
    figsize=(18, 6),
    sharex=True,
    gridspec_kw={"height_ratios": [3.5, 1]},
)
fig.subplots_adjust(hspace=0.06)
ax1, ax2 = axes

# ── Background tint: Eager ON region (both rows) ─────────────────────────────
eager_on = df[df["eager_mode"] == 1]
if len(eager_on):
    xe0 = int(eager_on["frame"].min()) - 0.5
    xe1 = int(eager_on["frame"].max()) + 0.5
    for ax in axes:
        ax.axvspan(xe0, xe1, color="#fff3e0", alpha=0.55, zorder=0)

# ── Background tint: Skip-only miss zone (pre-eager, both rows) ──────────────
pre = df[df["eager_mode"] == 0]
if len(pre):
    x0, x1 = int(pre["frame"].min()) - 0.5, int(pre["frame"].max()) + 0.5
    for ax in axes:
        ax.axvspan(x0, x1, color="#ffcccc", alpha=0.22, zorder=0)

# ── Row 1: Classifier confidence ─────────────────────────────────────────────
for _, row in df.iterrows():
    f = int(row["frame"])
    val = int(row["classifier"])
    prob = float(row["prob"])
    sk = int(row["skipped"])

    if sk == 1:
        color = "#c7c7c7"  # skipped — counterfactual miss
    elif val == 1:
        color = "#d62728"  # TP detected by Eager
    else:
        color = "#aec7e8"  # run but negative

    ax1.bar(f, max(prob, 0.04), color=color, width=0.8, align="center", alpha=0.88)

    if f % 5 == 0 and prob > 0.12 and sk == 0:
        ax1.text(
            f,
            max(prob, 0.04) / 2,
            f"{prob:.2f}",
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="white" if val == 1 else "#333333",
            rotation=-90,
        )

ax1.set_ylabel("Classifier $\\mathcal{M}$\n(SmokeOnly conf.)", fontsize=13, labelpad=8)
ax1.set_ylim(0, 1.62)
ax1.set_yticks([0, 0.5, 1.0])
ax1.tick_params(axis="y", labelsize=9)
ax1.spines[["top", "right"]].set_visible(False)
ax1.axhline(0.5, color="gray", linestyle=":", linewidth=0.9, alpha=0.6)
ax1.text(
    frames[-1] + 0.8, 0.51, "0.5 threshold", fontsize=13.5, color="gray", va="bottom"
)

# Ground truth note
ax1.text(
    0.01,
    0.97,
    "Ground truth: SmokeOnly  |  Grey bars = frames skipped by skip module"
    " — would be MISSED without Eager mode",
    transform=ax1.transAxes,
    fontsize=13.5,
    color="#005500",
    fontstyle="italic",
    va="top",
    bbox=dict(boxstyle="round,pad=0.3", fc="#f0fff0", ec="#2ca02c", lw=0.8),
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
            fontsize=13,
            fontweight="bold",
        )

ax2.set_yticks([])
ax2.set_ylabel("Skip\ndecision $s_t$", fontsize=13, labelpad=8)
ax2.set_ylim(0, 1.4)
ax2.spines[["top", "right", "left"]].set_visible(False)

# ── Zone labels ───────────────────────────────────────────────────────────────
if len(pre):
    axes[0].text(
        (x0 + x1) / 2,
        1.47,
        "Skip-only zone\n(smoke undetected)",
        ha="center",
        fontsize=13,
        color="#cc0000",
        fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.25", fc="#fff5f5", ec="#ffaaaa", lw=0.8),
    )

if len(eager_on):
    mid_eager = (xe0 + xe1) / 2
    axes[0].text(
        mid_eager,
        1.47,
        "Eager mode ON\n(smoke recovered)",
        ha="center",
        fontsize=13,
        color="#b85c00",
        fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.25", fc="#fff8f0", ec="darkorange", lw=0.8),
    )

# ── Eager activation line ─────────────────────────────────────────────────────
if len(eager_on):
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
    prob_here = (
        float(df[df["frame"] == eager_start]["prob"].values[0])
        if len(df[df["frame"] == eager_start])
        else 0.7
    )
    axes[0].annotate(
        f"Eager ON\n(frame {eager_start})",
        xy=(eager_start - 0.5, prob_here),
        xytext=(eager_start + 5, 1.32),
        fontsize=13.5,
        color="darkorange",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.3),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="darkorange", lw=0.9),
    )

# ── Legends ───────────────────────────────────────────────────────────────────
ax1.legend(
    handles=[
        mpatches.Patch(
            color="#d62728", alpha=0.88, label="SmokeOnly — TP (detected by Eager)"
        ),
        mpatches.Patch(
            color="#c7c7c7", alpha=0.88, label="Skipped — would be MISSED by skip-only"
        ),
        mpatches.Patch(color="#aec7e8", alpha=0.88, label="Run — Negative"),
        mpatches.Patch(
            color="#fff3e0",
            alpha=0.8,
            ec="darkorange",
            label="Eager mode ON (orange tint)",
        ),
        mpatches.Patch(
            color="#ffcccc",
            alpha=0.6,
            ec="#ffaaaa",
            label="Skip-only miss zone (red tint)",
        ),
    ],
    fontsize=13,
    loc="upper left",
    framealpha=0.92,
    bbox_to_anchor=(0.01, 0.93),
)
ax2.legend(
    handles=[
        mpatches.Patch(color="#98df8a", alpha=0.88, label="Skipped (S)"),
        mpatches.Patch(color="#ff9896", alpha=0.88, label="Inference run (R)"),
    ],
    fontsize=13,
    loc="lower right",
    framealpha=0.9,
)

# ── X-axis & title ────────────────────────────────────────────────────────────
ax2.set_xticks(frames[::5])
ax2.set_xticklabels(frames[::5], fontsize=13)
ax2.set_xlabel("Frame index", fontsize=13)

plt.suptitle(
    "Eager Mode Success: Rescuing Missed Smoke Detections (slow-developing or nearly stationary smoke event)\n"
    "Video: aihub__lb_smoke__0208   |   Ground truth: FireSmoke",
    fontsize=15,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()

outfile = BASE_DIR + "/eager_timeline_success.png"
plt.savefig(outfile, dpi=180, bbox_inches="tight")
pprint_local_path(outfile, get_wins_path=True, using_box=True)
