# Baseline Refs for fire/smoke detection (for dataset, model choices, etc.)
https://arxiv.org/abs/2011.04863
https://github.com/ChangyWen/STCNet-for-Smoke-Detection

# To-do workflow

```text
+----------------------------------------------------------------------------------+
|                    🚀 STORY-DRIVEN EXPERIMENT WORKFLOW / TIMELINE                |
+----------------------------------------------------------------------------------+
Note:
0️⃣ Our (no-skip) = A heavy classifier trained with million images ("heavy but safe")
1️⃣ M1 = A lightweight image classifier ("fast but unsafe")
2️⃣ M2 = A lightweight object detector;  ("fast but unsafe")
3️⃣ M3 = A temporal voting method (video-level baseline) ("accurate but delay to detection")

The emojis make it instantly scannable: 🚦 for decisions, ✅/❌ for pass/fail, 🎯 for recall/safety, ⚡ for speed, etc.—while preserving the exact sequential dependencies from the table and paper outline

🔍 [^0] DATASET CHECK ✅
    |
    |-- Verify split and balance:
    |      Train/Val = 90 videos 📊
    |      Test      = 60 videos 📊
    |      Classes   = FireSmoke / Safe
    |
    v
⚡ [^1] TEST-SET BASELINES FIRST 🏁
    |
    |-- Run BIG MODEL (no skip) on TEST
    |      -> get reference accuracy / recall / latency ⏱️
    |
    |-- Run M1 on TEST
    |      -> get lightweight speed / recall tradeoff ⚖️
    |
    |-- Run M2 on TEST
    |      -> get lightweight speed / recall tradeoff ⚖️
    |
    +--> 🚦 Decision A:
         "Do we clearly see:
          BIG MODEL = accurate but slow 🐌,
          M1/M2 = faster but miss anomalies ❌?"
               |
               +-- ❌ NO --> 🔧 re-check data / labels / eval pipeline
               |
               +-- ✅ YES --> continue
                              |
                              v
📊 [^2] DEFINE METRICS + SAFETY RULE 🛡️
    |
    |-- Recall(Anomaly) 🎯
    |-- Filter Rate on Safe frames 🚫
    |-- System FAR
    |-- Safety constraint: Recall >= 99%
    |
    v
🔧 [^3] VALIDATION-SET TUNING 🎛️
    |
    |-- Approach 1: naive motion 🏃
    |-- Approach 2: motion + color + texture 🎨🔥💨
    |-- Grid search on VAL only
    |
    +--> 🚦 Decision B:
         "Is there a config satisfying Recall >= 99%? 🛡️"
               |
               +-- ❌ NO --> 🔧 revise thresholds / rules
               |
               +-- ✅ YES --> choose best config by SkipScore ⭐
                              |
                              v
✅ [^4] COMPONENT TEST ON TEST SET 🧪
    |
    |-- Run final Approach 1 alone
    |-- Run final Approach 2 alone
    |
    |-- Measure:
    |      Recall(Anomaly) 🎯
    |      Filter Rate (Safe) 🚫
    |      Overhead (ms) ⏱️
    |
    +--> 🚦 Decision C:
         "Does Approach 2 behave like a safe gatekeeper? 🛡️"
               |
               +-- ❌ NO --> go back to [^3] 🔄
               |
               +-- ✅ YES --> continue
                              |
                              v
🚀 [^5] FULL PIPELINE TEST ON TEST SET 💥
    |
    |-- Approach 1 + BIG MODEL
    |-- Approach 2 + BIG MODEL
    |-- Compare against:
    |      BIG MODEL only
    |      M1
    |      M2
    |
    |-- Measure:
    |      system recall / F1 🎯
    |      latency ⏱️
    |      FPS ⚡
    |      compute reduction 📉
    |
    +--> 🎯 Core Claim Check:
         "Does Approach 2 + BIG keep BIG MODEL accuracy
          while reducing compute and latency? ⚖️"
               |
               +-- ❌ NO --> go back to [^3] or [^4] 🔄
               |
               +-- ✅ YES --> continue
                              |
                              v
⏱️ [^6] TEMPORAL BASELINE COMPARISON 🥇
    |
    |-- Compare against M3
    |-- Measure:
    |      time-to-first-alarm 🏃‍♂️
    |      video-level recall
    |      false alarm rate
    |
    v
🔬 [^7] ABLATION + QUALITATIVE EVIDENCE 📸
    |
    |-- motion only
    |-- + fire rules 🔥
    |-- + smoke rules 💨
    |-- full rules
    |
    |-- collect edge cases:
    |      smoke missed by Appr.1 ❌
    |      safe motion falsely triggered by Appr.1 🚫
    |
    v
⚠️ [^8] LIMITATION CHECK 🧪
    |
    |-- low light / IR 🌙
    |-- fog / weather 🌫️
    |-- unusual lighting
    |
    v
🏆 +----------------------------------------------------------------------------------+
| FINAL STORY: redundancy is the bottleneck; a safe skip module lets BIG MODEL    |
| run much faster without losing safety-critical recall. 🎯⚡                       |
+----------------------------------------------------------------------------------+
```


## Compact paper version (with emojis)

Perfect for pasting into manuscripts or presentations:

```text
+----------------------------------------------------------------------------------+
|                    🚀 PAPER2 TARGET TABLES WORKFLOW (All-in-One)                 |
+----------------------------------------------------------------------------------+

🔍 [Baselines + Component Test] → Table 1 ✅
    |
    +-- Table 1: Standalone Recall and Filtering Performance
    |     | Method                        | Overhead | Recall | Filter Rate |
    |     |-------------------------------|----------|--------|-------------|
    |     | M1 (MobileNetV2 Classifier)   | 15.0ms   | 85.3%  | N/A         |
    |     | M2 (YOLOv8-Nano Detector)     | 22.0ms   | 88.2%  | N/A         |
    |     | Appr.1 (Naive Motion)         | 1.2ms    | 97.2%  | 65.0%       |
    |     | Appr.2 (Rule-Based)           | 2.5ms    | 99.1%  | 72.1%       |
    |
    v
🎛️ [VAL Grid Search] → Tables 5&6 ⭐
    |
    +-- Table 5: Hyperparam Search Space (Target: 16x16, 1.5%, smoke rules)
    |     | Param                       | Appr.1 | Appr.2  |
    |     |-----------------------------|--------|---------|
    |     | Grid Size (NxN)             | 16x16  | 16x16   |
    |     | Motion Threshold            | 2%     | 1.5%    |
    |     | Grayness Threshold (Smoke)  | N/A    | [0.7,0.95]|
    |
    +-- Table 6: Top 3 Configs (Val Set)
    |     | Rank | Config                     | Recall | Filter | SkipScore |
    |     |------|----------------------------|--------|--------|-----------|
    |     | 1    | 16x16,1.5%,Smoke Rules     | 99.1%  | 72.1%  | 0.714     | ✅ Appr.2
    |     | 2    | 16x16,2.0%,Smoke Rules     | 99.0%  | 68.3%  | 0.676     | ❌
    |     | 3    | 32x32,1.0%,Smoke Rules     | 99.2%  | 65.7%  | 0.652     | ❌
    |
    v
💥 [Full Pipeline Test] → Table 2 🚀
    |
    +-- Table 2: End-to-End System Performance
    |     | Config                      | Latency | FPS  | Recall | Cost Red. |
    |     |-----------------------------|---------|------|--------|-----------|
    |     | BIG MODEL only              | 50.0ms  | 20   | 98.5%  | 0%        |
    |     | M1                          | 15.0ms  | 66   | 76.0%  | 70%       |
    |     | M2                          | 22.0ms  | 45   | 81.5%  | 56%       |
    |     | Ours (Appr.1 + BIG)         | 18.3ms  | 54   | 98.3%  | 63%       |
    |     | Ours (Appr.2 + BIG)         | 16.5ms  | 60   | 98.5%  | 67%       | ✅
    |
    v
🥇 [vs M3 Temporal] → Tables 3&4 ⏱️
    |
    +-- Table 3: Time-to-Alarm Analysis (*M3 using BIG Model*)
    |     | Method             | Scope      | Delay   | Min Latency | First Alarm |
    |     |--------------------|------------|---------|-------------|-------------|
    |     | M3 Temporal Voting | Video(30f) | 25ms/f  | 30 Frames   | >750ms      |
    |     | Ours (Appr.2+BIG)  | Frame-level| 16.5ms/f| 1 Frame     | ~52.5ms     | ✅
    |
    +-- Table 4: Video-Level Metrics
    |     | Method             | Video Recall| False Alarm | Compute/Sec |
    |     |--------------------|-------------|-------------|-------------|
    |     | M3 Temporal Voting | 97.5%       | 1.2%        | 750ms       |
    |     | Ours (Appr.2+BIG)  | 98.4%       | 1.5%        | 495ms       | ✅
    |
    v
📸 [Ablation + Limits] → Qualitative Figure 4 🔬⚠️

🏆 STORY: BIG MODEL accuracy ⚡ 3x faster with safe skip module!
+----------------------------------------------------------------------------------+

```

# The main document describing the project (goals, progress, etc.)

## Task
Design a lightweight skip module for real-time fire and smoke detection in static camera surveillance systems.

### Problem Statement
Current deep learning (DL) pipelines are computationally inefficient because they process every video frame through heavy models (~50ms per frame), even for static scenes with no fire or smoke activity. The goal is to implement a fast pre-check mechanism that filters out irrelevant frames before running expensive inference, ensuring that indicators for *both* fire and smoke are considered.

**System Context:**
- **Current Workflow:** Read frame → Preprocess → DL inference (detect fire/smoke) → Alarm if detected
- **Proposed Workflow:** Read frame → Fast pre-check → Skip if negative *or* DL inference if positive → Alarm if detected

### Data Specifications
- **Input:** Video stream from static cameras (no camera movement or additional sensors).
- **Expected Output:** The skip module must correctly bypass frames without fire/smoke and trigger DL inference for frames with potential indicators.

### Requirements & Constraints
- Methods must be computationally lightweight for real-time processing.
- Crucially, the system must minimize false negatives (missing actual events) while maximizing skipped frames.
- Must identify indicators for both fire and smoke.

### Current Proposed Approaches

**Approach 1: Naive Block-Based Motion Analysis**
1. **Grid Formulation:** Divide the motion mask into an *N x N* grid and calculate the percentage of motion pixels per block.
2. **Active Block Detection:** Mark a block as "active" if motion exceeds a defined threshold.
3. **Skip Decision:** Skip the frame if there are no active blocks; otherwise, run DL inference.
4. **Output:** "Skip" or "DL Inference Output."

**Approach 2: Rule-Based Block-Based Motion Analysis**
1. **Grid Formulation:** Same as Approach 1.
2. **Active Block Detection:** Same as Approach 1.
3. **Rule-Based Skip Decision:**
    - Skip if no active blocks are found.
    - If an active block is detected, apply heuristic rules to check for fire/smoke indicators. Examples include:
        - **Color Rules:** Check for high red channel intensity (fire) or grayish colors (smoke).
        - **Temporal Consistency:** Analyze texture and color changes over time (e.g., flickering for fire, growing/dissipating for smoke).
        - **Additional Rules:** Other domain-specific heuristics.
    - Run DL inference if rules suggest potential fire/smoke; otherwise, skip the frame.
4. **Output:** "Skip" or "DL Inference Output."

## Current Progress

### Progress Summary
- **Dataset:** Created a high-quality custom dataset of 150 HD videos (balanced 50/50/50 split) from static cameras.
- **Implementation:** Developed two lightweight skip modules:
  - *Approach 1:* Naive Block-Based Motion Analysis.
  - *Approach 2:* Rule-Based Motion Analysis (color/texture heuristics).
- **The "Villain":** The "BIG MODEL"—a heavy, highly accurate, but slow (50ms/frame) deep learning model trained on millions of images.
- **The Goal:** Prove that your skip module makes the *system* faster without compromising the BIG MODEL's accuracy.

***

### Strategy: Choosing Baselines & Defining the Story
You are worried about baselines because you are comparing a "pre-filter" against "models." This is a category error. **You are not competing with M1, M2, or M3 on accuracy.** You are competing on **Efficiency vs. Accuracy Trade-off.**

**The Narrative Arc:**
1. **The Status Quo (Baseline 0):** Running "BIG MODEL" on *every single frame*. This is accurate but computationally wasteful (100% load).
2. **The Weak Competitor (M1/M2/M3):** Using a lightweight model *instead* of the BIG MODEL. These are fast but have poor accuracy (high False Negatives) because they were trained on small datasets (2k-5k images vs millions).
3. **The Solution (Your Method):** Running "Skip Module + BIG MODEL." This retains the high accuracy of the BIG MODEL but approaches the speed of the lightweight models.

**How to handle M3 (Temporal/Voting):**
Since M3 uses temporal voting to reduce false alarms *after* inference, it is actually complementary, not a direct competitor to a *pre-inference* skip module. However, you can frame M3 as a "Post-Processing Optimization" and your method as "Pre-Processing Optimization." *Recommendation: Keep M3 as a discussion point or secondary comparison, but focus on M1/M2 as the primary "Lightweight Alternatives."*
