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
