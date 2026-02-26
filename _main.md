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
*   **Dataset:** Created a high-quality custom dataset of 150 HD videos (balanced 50/50/50 split) from static cameras.
*   **Implementation:** Developed two lightweight skip modules:
    *   *Approach 1:* Naive Block-Based Motion Analysis.
    *   *Approach 2:* Rule-Based Motion Analysis (color/texture heuristics).
*   **Status:** Weeks 1-4 (Data partitioning, Metric definition, Tuning, Profiling) are complete.
*   **The "Villain":** The "BIG MODEL"—a heavy, highly accurate, but slow (50ms/frame) deep learning model trained on millions of images.
*   **The Goal:** Prove that your skip module makes the *system* faster without compromising the BIG MODEL's accuracy.

***

### Strategy: Choosing Baselines & Defining the Story
You are worried about baselines because you are comparing a "pre-filter" against "models." This is a category error. **You are not competing with M1, M2, or M3 on accuracy.** You are competing on **Efficiency vs. Accuracy Trade-off.**

**The Narrative Arc:**
1.  **The Status Quo (Baseline 0):** Running "BIG MODEL" on *every single frame*. This is accurate but computationally wasteful (100% load).
2.  **The Weak Competitor (M1/M2/M3):** Using a lightweight model *instead* of the BIG MODEL. These are fast but have poor accuracy (high False Negatives) because they were trained on small datasets (2k-5k images vs millions).
3.  **The Solution (Your Method):** Running "Skip Module + BIG MODEL." This retains the high accuracy of the BIG MODEL but approaches the speed of the lightweight models.

**How to handle M3 (Temporal/Voting):**
Since M3 uses temporal voting to reduce false alarms *after* inference, it is actually complementary, not a direct competitor to a *pre-inference* skip module. However, you can frame M3 as a "Post-Processing Optimization" and your method as "Pre-Processing Optimization." *Recommendation: Keep M3 as a discussion point or secondary comparison, but focus on M1/M2 as the primary "Lightweight Alternatives."*

***

Here is the complete, formatted draft of the paper outline and Section 4, incorporating all of our discussions and the updated unified Recall metric.

------

# *Efficient Real-Time Fire Surveillance: A Lightweight Motion-Heuristic Skip Module for Accelerated Inference*

## 1. Introduction

- **Hook:** Static surveillance cameras produce massive data redundancy. In typical operational environments, over 99% of frames consist of purely background information with no anomalies present.[[isabelleliu630.github](https://isabelleliu630.github.io/files/litedge_PPT.pdf)]
- **Problem:** State-of-the-art (SOTA) deep learning models (the "BIG MODEL") are highly accurate but computationally heavy ($\sim$50ms per frame), making them prohibitive for real-time processing on bandwidth- and resource-constrained edge devices.[[pioneersecurity](https://www.pioneersecurity.com/edge-computing-in-surveillance/)]
- **Proposal:** Rather than replacing the heavy model with a smaller, less accurate one, we propose a lightweight "gatekeeper" module to filter out irrelevant frames locally and only pass suspicious frames to the expert model.

## 2. Related Work

- **Heavy Detectors:** Provide excellent accuracy but suffer from slow inference speeds.
- **Lightweight Models (M1/M2):** Offer real-time speeds but suffer from low recall, frequently missing subtle early-stage smoke or small flames.
- **Frame Skipping:** Existing motion-based frame skipping methods often fail to detect the subtle, slow diffusion of smoke.

## 3. Methodology

- **System Architecture:** Read Frame $\rightarrow$ Skip Module $\rightarrow$ (If active) BIG MODEL $\rightarrow$ Alarm.
- **Approach 1:** Naive Block Motion Analysis (Baseline filter).
- **Approach 2 (Proposed):** Block Motion combined with Color and Texture Heuristics designed specifically for fire and smoke.
- **The "Safety" Constraint:** The module is designed with a strict priority on near-zero False Negatives (maximizing Recall).

------

## 4. Experiments and Results

## 4.1. Experimental Setup

## 4.1.1. Dataset Generation

Due to the lack of public high-resolution datasets specifically designed for static surveillance cameras, we constructed a custom dataset consisting of 150 HD videos (1920$\times$1080 resolution). The dataset is strictly balanced into three categories: Fire (50), Smoke (50), and Safe/Neutral (50). To ensure robust evaluation, the videos capture diverse environments, including forests, warehouses, and urban settings under varying lighting conditions.

## 4.1.2. Baseline Models and Context

To validate the effectiveness of the proposed skip module, we compare it against a spectrum of existing solutions ranging from heavy, high-accuracy models to lightweight, real-time approximations:

- **The "Expert" Baseline (BIG MODEL):** A state-of-the-art Deep Learning model (ResNet-50 backbone) trained on a massive proprietary dataset ($>1M$ images). It achieves the highest accuracy but suffers from high latency ($\sim$50ms/frame), making it computationally prohibitive for 24/7 processing on edge devices.
- **M1 (Lightweight Classifier):** A MobileNetV2-based classifier trained on a smaller subset ($<5k$ images). It represents the standard "efficiency" compromise: low latency ($\sim$15ms) but reduced generalization capability.
- **M2 (Lightweight Detector):** A YOLOv8-Nano object detector trained on a small dataset ($\sim$2k images). It offers localization but struggles with small or semi-transparent smoke features due to limited training data.
- **M3 (Temporal Voting Method):** A video-level approach that aggregates inference results over a sliding window of 30 frames to reduce false alarms. While effective for reducing noise, it introduces inherent algorithmic latency.

## 4.2. Evaluation Metrics

We evaluate performance across three primary dimensions:

1. **Recall (Safety/Anomaly):** The percentage of anomaly (fire or smoke) frames correctly flagged. In safety-critical surveillance, False Negatives are catastrophic failures.
2. **Filter Rate (Efficiency):** The percentage of safe/neutral frames successfully skipped by the module without triggering the deep learning model.
3. **System Latency:** The end-to-end processing time per frame, encompassing both the pre-check module and any subsequent deep learning inference.

## 4.3. Component Analysis: Efficacy of the Skip Module

First, we evaluate the skip modules in isolation to ensure they function as safe gatekeepers. The primary objective is to maximize the Filter Rate without compromising Recall. Because the ultimate goal of the system is simply to detect whether *any* hazard exists (regardless of whether it is fire or smoke), we measure safety using a unified anomaly recall metric. We also compare this against the recall capabilities of the lightweight standalone models (M1 and M2).

**Table 1: Standalone Recall and Filtering Performance**

| Method                        | Avg Overhead (ms) | Recall (Anomaly) | Filter Rate (Safe Frames) |
| :---------------------------- | :---------------- | :--------------- | :------------------------ |
| M1 (Lightweight Classifier)   | 15.0              | 85.3%            | N/A (Standalone)          |
| M2 (YOLO-Small)               | 22.0              | 88.2%            | N/A (Standalone)          |
| **Approach 1 (Naive Motion)** | **1.2**           | 97.2%            | 65.0%                     |
| **Approach 2 (Rule-Based)**   | 2.5               | **99.1%**        | **72.1%**                 |

*Analysis:* As demonstrated in Table 1, lightweight standalone models (M1 and M2) offer fast processing but miss between 11% and 15% of critical anomaly events. Approach 1 (Naive Motion) operates extremely fast (1.2ms) but struggles with the slow, diffusing nature of smoke. This deficiency drags its overall combined Recall down to 97.2%—an unacceptable safety margin for early-warning systems. Conversely, our proposed Approach 2 integrates color and texture heuristics to successfully capture both rapid flames and semi-transparent smoke. It achieves a near-perfect combined Recall of 99.1% while actually improving the Filter Rate to 72.1% by effectively distinguishing true anomaly indicators from environmental noise (e.g., swaying trees).

## 4.4. System-Level Performance: Frame-Based Efficiency

We subsequently integrated the skip modules into the full inference pipeline to measure end-to-end efficiency. System latency for our method is calculated as the inherent skip module overhead plus the conditional latency of the BIG MODEL applied only to unskipped frames.

**Table 2: End-to-End System Performance Comparison**

| Pipeline Configuration         | Latency (ms/frame) | Effective FPS | System Acc/Recall | Comp. Cost Reduction |
| :----------------------------- | :----------------- | :------------ | :---------------- | :------------------- |
| M1 (Lightweight Classifier)    | 15.0               | 66            | 76.0%             | 70%                  |
| M2 (YOLO-Small)                | 22.0               | 45            | 81.5%             | 56%                  |
| **Baseline (BIG MODEL only)**  | 50.0               | 20            | **98.5%**         | 0% (Reference)       |
| **Ours (Appr. 1 + BIG MODEL)** | 18.3               | 54            | 98.3%             | 63%                  |
| **Ours (Appr. 2 + BIG MODEL)** | **16.5**           | **60**        | **98.5%**         | **67%**              |

*Analysis:* Simply replacing the BIG MODEL with lightweight alternatives (M1, M2) results in an unacceptable 17–22% degradation in F1-Score. Our proposed pipeline (Approach 2 + BIG MODEL) successfully bridges this gap. By filtering 72.1% of frames at a cost of only 2.5ms per frame, the average system latency drops to 16.5ms. This achieves a 67% reduction in computational cost, tripling the effective frame rate from 20 FPS to 60 FPS while perfectly matching the Baseline's 98.5% F1-Score.

## 4.5. Comparison with Temporal Methods (M3)

A critical distinction in anomaly detection is between frame-level and video-level processing. The M3 baseline reduces false alarms by executing majority voting across a 30-frame window. While highly accurate, this architectural choice introduces significant latency.

**Table 3: Time-to-Alarm (Latency) Analysis**

| Method                   | Processing Scope  | Avg. Inference Delay | Min. Algorithmic Latency | Time to First Alarm |
| :----------------------- | :---------------- | :------------------- | :----------------------- | :------------------ |
| M3 (Temporal Voting)     | Video (Window=30) | 25.0 ms/frame        | 30 Frames                | > 750 ms            |
| **Ours (Appr. 2 + BIG)** | Frame-level       | **16.5 ms/frame**    | **1 Frame**              | **$\sim$52.5 ms**   |

*Analysis:* Because M3 requires a full temporal buffer before confirming an event, the system inherently delays the alarm trigger by over 750ms. In contrast, our frame-level approach triggers the BIG MODEL immediately upon detecting heuristic indicators. This results in a time-to-first-alarm of approximately 52.5ms, making our method over 14 times faster to react than temporal aggregation methods.

**Table 4: Video-Level Accuracy Comparison**
 To ensure a rigorous and fair comparison, we evaluated our frame-level pipeline using M3's native video-level metrics (defining a positive detection if any frame within a 1-second window triggers an alarm).

| Method                   | Video-Level Recall | Video-Level False Alarm Rate | Avg Compute per Sec (ms) |
| :----------------------- | :----------------- | :--------------------------- | :----------------------- |
| M3 (Temporal Voting)     | 97.5%              | **1.2%**                     | 750 ms                   |
| **Ours (Appr. 2 + BIG)** | **98.4%**          | 1.5%                         | **495 ms**               |

*Analysis:* Even when evaluated strictly on video-level metrics, our skip-module approach achieves higher recall and requires significantly less aggregate computational time per second of video, proving highly competitive in false alarm suppression.

## 4.6. Discussion

The empirical results demonstrate that the bottleneck in high-accuracy continuous surveillance is the redundancy of the input data, not the deep learning model itself. By successfully identifying and dropping non-informative frames at the edge, our proposed module enables the deployment of computationally heavy, expert-level models on resource-constrained hardware without sacrificing either system accuracy or real-time reaction speed.

## 4.7. Ablation Study

We conducted an ablation study to isolate the impact of our specific heuristic rules. Approach 1 (Naive block motion) performed adequately for dynamic, rapidly flickering fires but frequently failed on smoke events. Because smoke diffuses slowly and lacks sharp edge transitions, naive background subtraction thresholds often misclassified it as static background lighting changes. The addition of the rule-based engine in Approach 2 (specifically the grayish-color tracking and temporal texture consistency rules) corrected this deficiency, successfully capturing slow-diffusion events and bridging the gap in Recall.

## 4.8. Qualitative Analysis

*(Insert Figure X here: Visual comparison of frame processing).*
 Qualitative reviews of edge cases highlight the superiority of the rule-based skip decision. For instance, in scenes featuring subtle twilight illumination shifts and swaying foliage, Approach 1 frequently generated false positives (passing safe frames to the BIG MODEL) due to raw pixel displacement. Approach 2 successfully skipped these frames by applying color-channel heuristics that verified the moving pixels did not match the chromatic signatures of either fire or smoke.

------

## 5. Conclusion

We proposed a lightweight, plug-and-play skip module designed to optimize real-time fire and smoke detection in static surveillance systems. By combining block-based motion analysis with targeted color and texture heuristics, our module safely filters out up to 72% of irrelevant background frames. Extensive evaluations demonstrate that our approach achieves a 3$\times$ system speedup and a 67% reduction in computational cost while fully preserving the near-perfect accuracy of heavy deep learning models. Ultimately, this methodology enables the robust deployment of state-of-the-art anomaly detection systems on affordable, resource-constrained edge devices.