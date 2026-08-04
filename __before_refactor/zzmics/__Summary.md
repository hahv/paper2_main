<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Paper 2 — Complete Summary for Newcomers

## 🎯 The Big Picture

You are building a **real-time fire/smoke detection system**. The core detector is a deep learning (DL) model — it is accurate but **slow** because it runs on every single frame.

**The problem:** In most video frames, nothing is happening. Running DL inference on a static, empty scene wastes computation.

**The solution:** Build a **skip module** — a lightweight pre-filter that looks at each frame first and says *"skip this frame, nothing is moving"* before the DL model ever sees it.

***

## 🏗️ System Architecture

```
Video Frame
    │
    ▼
┌─────────────────┐
│   Skip Module   │  ← Your contribution
│  (Motion Det.)  │
└────────┬────────┘
         │
    Motion?
    /       \
  YES        NO
   │          │
   ▼          ▼
 Run DL     SKIP
 Model    (save compute)
   │
   ▼
Fire/Smoke Decision
```

The skip module works at the **block level** — it divides the frame into blocks and checks what fraction of blocks contain motion. If the fraction is below a threshold (`block_ratio_th`), the frame is skipped.

***

## 📐 Two Motion Detectors You Built

### Detector 1 — FrameDiffDet (Simple Baseline)

**Idea:** Compare two consecutive frames pixel by pixel. If the difference is big enough, call it motion.

**One parameter:** `τ_d` — how sensitive it is to change.

**Problem:** It has **no memory**. Every frame is judged independently, so:

- A single noisy frame → false motion → frame not skipped (false skip risk)
- A very subtle fire → small diff → missed motion → wrongly skipped (recall risk)
- The "safe zone" (high recall + meaningful skip) is **very narrow** in practice

***

### Detector 2 — AccMotionDet (Your Main Method)

**Idea:** Instead of deciding on a single frame, **accumulate evidence over time**. A pixel must move consistently across multiple frames before it is declared as motion.

**How it works — step by step:**

```
Frame t arrives
       │
       ▼
Compute |F_t - F_{t-1}|   ← absolute difference
       │
       ▼
If diff > τ_d → add α to that pixel's heat   ← instantaneous delta Δ_t
       │
       ▼
Heat map M_t = clip(M_{t-1} + Δ_t, M_max)   ← accumulate + cap
       │
       ▼
M_t = max(M_t - δ, 0)                        ← decay every frame
       │
       ▼
If M_t(pixel) ≥ τ_m → motion pixel           ← threshold to binary mask C_t
```

**Five parameters:**


| Symbol | Name | Role |
| :-- | :-- | :-- |
| $\tau_d$ | Diff sensitivity | Ignores tiny/noisy pixel changes |
| $\alpha$ | Motion increment | How much heat one detected frame adds |
| $\tau_m$ | Activation threshold | Minimum heat to call a pixel "motion" |
| $M_{\max}$ | Accumulation cap | Prevents stale heat from building forever |
| $\delta$ | Decay rate | How fast heat fades when motion stops |

**Why it is better than FrameDiffDet:**

- Single noisy frames never accumulate enough heat to cross `τ_m` → noise suppressed
- Consistent motion (real fire/smoke) quickly builds heat → reliably detected
- After motion stops, heat decays over ~15 frames → smooth, stable output

***

## 📊 Experimental Results

### What you measured

| Metric | Meaning |
| :-- | :-- |
| Recall | Did we catch all real fire/smoke frames? (must stay high) |
| FPR | Did we falsely alarm on safe frames? (lower is better) |
| Correct Skip | Frames we safely skipped (higher = more efficient) |
| False Skip | Fire frames we wrongly skipped (must stay near zero) |
| FPS | Processing speed (higher = better real-time performance) |

### FrameDiffDet sweep result

The grid search showed the trade-off is very harsh: to keep recall safe, the skip rate collapses to nearly nothing (~1–2%). Not useful.

### AccMotionDet result (your best config)

|  | Baseline (No Skip) | AccMotionDet Skip |
| :-- | :-- | :-- |
| Recall | 95.62% | 94.85% |
| FPR | 0.251% | 0.160% |
| FPS | 24.97 | 28.51 |
| Speedup | — | **+14.2%** |

**The story:** You sacrifice only 0.77pp recall in exchange for 14.2% faster processing and 36% fewer false alarms. The FPR actually *improves* because skipped static frames can never produce false positives.

***

## 🔧 Parameter Search Strategy

You did a **grid search** over the key operating-point parameters:

- `scale_factor`, `block_size_orig`, `block_ratio_th` (block-level skip control)
- `diff_frame_th` / `τ_d` (motion sensitivity)

You **fixed** `M_max=25` and `decay=1` because:

- They are coupled — together they control memory tail length (~15 frames after saturation)
- The reference C++ implementation used these values
- Results were already strong without searching them

***

## 📝 What Is Written So Far

- ✅ AccMotionDet algorithm in LaTeX (clean notation)
- ✅ FrameDiffDet algorithm in LaTeX
- ✅ Notation paragraph explaining all symbols
- ✅ Baseline vs AccMotionDet comparison numbers
- ✅ Parameter justification arguments


## 📝 What Still Needs Writing

- ⬜ Motivation paragraph (why skip matters for real-time fire detection)
- ⬜ FrameDiff limitation paragraph (prose)
- ⬜ AccMotionDet method paragraph (prose)
- ⬜ Results table (Baseline vs FrameDiff vs AccMotionDet)
- ⬜ Analysis/discussion paragraph
- ⬜ Parameter table with final values

