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
🎛️ [VAL Grid Search] → Tables 5A, 5B, 6A, 6B ⭐
    |
    +-- Table 5A: Approach 1 Search Space and Best Validation Setting
    |     | Hyperparameter                | Candidate Values        | Selected | Note                    |
    |     |-------------------------------|-------------------------|----------|-------------------------|
    |     | Grid Size (N×N)               | {8×8,16×16,32×32}      | 16×16    | Good detail/cost tradeoff |
    |     | Motion Threshold              | {1.0%,2.0%,5.0%}       | 2.0%     | Simpler baseline setting |
    |     | Fire-specific Color Rule      | Not used                | N/A      | Pure motion baseline     |
    |     | Smoke Grayness Rule           | Not used                | N/A      | Cannot explicitly model smoke |
    |     | Temporal Consistency Window   | Minimal / simple        | N/A      | Keep baseline lightweight |
    |
    +-- Table 5B: Approach 2 Search Space and Best Validation Setting
    |     | Hyperparameter                | Candidate Values        | Selected     | Note                         |
    |     |-------------------------------|-------------------------|--------------|------------------------------|
    |     | Grid Size (N×N)               | {8×8,16×16,32×32}      | 16×16        | Good detail/cost tradeoff    |
    |     | Motion Threshold              | {1.0%,1.5%,2.0%,5.0%}  | 1.5%         | Better for subtle smoke      |
    |     | Fire-specific Color Rule      | Tuned range             | Selected     | Helps detect flame regions   |
    |     | Smoke Grayness Rule           | Candidate intervals     | [0.7,0.95]  | Captures semi-transparent smoke |
    |     | Temporal Consistency Window   | {3,5}                   | 5            | Improves robustness          |
    |
    +-- Table 6A: Approach 1 Validation Ranking
    |     | Rank | Configuration              | Recall | Filter | Safety ≥99% | Note                    |
    |     |------|----------------------------|--------|--------|-------------|-------------------------|
    |     | 1    | 16×16, 2.0% motion         | 97.2%  | 65.0%  | No          | Best Appr.1, but unsafe |
    |     | 2    | 16×16, 1.0% motion         | TBD    | TBD    | TBD         | More sensitive/noisier  |
    |     | 3    | 32×32, 1.0% motion         | TBD    | TBD    | TBD         | Coarser blocks          |
    |
    +-- Table 6B: Approach 2 Validation Ranking
    |     | Rank | Configuration                    | Recall | Filter | SkipScore | Selected | Note                  |
    |     |------|----------------------------------|--------|--------|-----------|----------|-----------------------|
    |     | 1    | 16×16,1.5%,Smoke Rules Enabled   | 99.1%  | 72.1%  | 0.714     | ✅ Yes   | Best safe config      |
    |     | 2    | 16×16,2.0%,Smoke Rules Enabled   | 99.0%  | 68.3%  | 0.676     | ❌ No    | Safe but less efficient |
    |     | 3    | 32×32,1.0%,Smoke Rules Enabled   | 99.2%  | 65.7%  | 0.652     | ❌ No    | Higher recall, lower efficiency |
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
    |     | Method             | Scope       | Delay    | Min Latency | First Alarm |
    |     |--------------------|-------------|----------|-------------|-------------|
    |     | M3 Temporal Voting | Video(30f)  | 25ms/f   | 30 Frames   | >750ms      |
    |     | Ours (Appr.2+BIG)  | Frame-level | 16.5ms/f | 1 Frame     | ~52.5ms     | ✅
    |
    +-- Table 4: Video-Level Metrics
    |     | Method             | Video Recall | False Alarm | Compute/Sec |
    |     |--------------------|--------------|-------------|-------------|
    |     | M3 Temporal Voting | 97.5%        | 1.2%        | 750ms       |
    |     | Ours (Appr.2+BIG)  | 98.4%        | 1.5%        | 495ms       | ✅
    |
    v
📸 [Ablation + Limits] → Qualitative Figure 4 🔬⚠️

🏆 STORY: BIG MODEL accuracy ⚡ 3x faster with safe skip module!
+----------------------------------------------------------------------------------+
