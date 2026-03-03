Below is the complete, formatted draft of the paper outline and Section 4, incorporating all of our discussions and the updated unified Recall metric.

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

**Image placeholders for diagrams**:
```
textSystem Architecture:
[ASCII Workflow Diagram #1]

Approach 1/2:
[ASCII Block Diagrams #2]
[ASCII Motion Grid #4]
```

------

## 4. Experiments and Results

## 4.1. Experimental Setup

## 4.1.1. Dataset Generation and Partitioning

Due to the lack of public high-resolution datasets specifically designed for static surveillance cameras, we constructed a custom dataset consisting of 150 HD videos (1920$\times$1080 resolution). The dataset is strictly balanced into three categories: Fire (50), Smoke (50), and Safe/Neutral (50). To ensure robust evaluation, the videos capture diverse environments, including forests, warehouses, and urban settings under varying lighting conditions.

The 150 videos were randomly split into Training/Validation (60%, n=90: 30 Fire, 30 Smoke, 30 Safe) for hyperparameter tuning and hyperparameter selection, and Test (40%, n=60: 20 per class) for unbiased final performance evaluation. Stratified sampling ensured balance across classes and environments.

**Video Dataset Image Samples:**

```
text...under varying lighting conditions.

[ASCII Dataset Montage #3]

Figure 2: Representative frames from our custom HD dataset.
```

## 4.1.2. Baseline Models and Context

To validate the effectiveness of the proposed skip module, we compare it against a spectrum of existing solutions ranging from heavy, high-accuracy models to lightweight, real-time approximations:

- **The "Expert" Baseline (BIG MODEL):** A state-of-the-art Deep Learning model (ResNet-50 backbone) trained on a massive proprietary dataset ($>1M$ images). It achieves the highest accuracy but suffers from high latency ($\sim$50ms/frame), making it computationally prohibitive for 24/7 processing on edge devices.
- **M1 (Lightweight Classifier):** A MobileNetV2-based classifier trained on a smaller subset ($<5k$ images). It represents the standard "efficiency" compromise: low latency ($\sim$15ms) but reduced generalization capability.
- **M2 (Lightweight Detector):** A YOLOv8-Nano object detector trained on a small dataset ($\sim$2k images). It offers localization but struggles with small or semi-transparent smoke features due to limited training data.
- **M3 (Temporal Voting Method):** A video-level approach that aggregates inference results over a sliding window of 30 frames to reduce false alarms. While effective for reducing noise, it introduces inherent algorithmic latency.

All inference latencies were measured on an NVIDIA Jetson Nano / RTX 3060 to simulate edge deployment

**TODO**: add hardware and software context here

## 4.1.3. Hyperparameter Selection Strategy

Both proposed skip modules contain tunable hyperparameters (e.g., grid size $N$, motion intensity threshold, color channel limits). To ensure rigorous, automated, and reproducible optimization, we employed **Grid Search** on the Validation set (60% of total dataset, ~90 videos).

The primary **safety constraint** was **Recall(Anomaly) ≥ 99%**. To enable objective selection among feasible configurations, we introduce **SkipScore**, a unified composite metric analogous to mAP in object detection:

Let, for a given hyperparameter configuration \(h\):

- \(R_{\text{sys}}(h)\): system-level anomaly recall (frame-level)
  \[
  R_{\text{sys}}(h) = \frac{\#\{\text{anomaly frames that produce at least one alarm}\}}{\#\{\text{anomaly frames}\}}
  \]
- \(\text{FR}(h)\): filter rate on safe frames (efficiency of gate)
  \[
  \text{FR}(h) = \frac{\#\{\text{safe frames skipped by gate}\}}{\#\{\text{safe frames}\}}
  \]
- \(\text{FAR}_{\text{sys}}(h)\): system false alarm rate on safe frames
  \[
  \text{FAR}_{\text{sys}}(h) = \frac{\#\{\text{safe frames that produce at least one alarm in full pipeline}\}}{\#\{\text{safe frames}\}}
  \]

With a target recall \(R_{\text{target}}\) (e.g. \(0.99\)), your selection rule is:

1. **Safety constraint**
   \[
   R_{\text{sys}}(h) \ge R_{\text{target}}
   \]

2. **Objective (SkipScore\(_B\)) among feasible configs**
   \[
   \text{SkipScore}_B(h) = \text{FR}(h)\cdot \left(1 - \text{FAR}_{\text{sys}}(h)\right)
   \]

3. **Final choice**
   \[
   h^\* = \arg\max_{h \in G,\; R_{\text{sys}}(h) \ge R_{\text{target}}} \text{SkipScore}_B(h)
   \]

Here, all three quantities are frame-based, satisfying the “same unit” caution, and recall is enforced as a hard constraint rather than traded off in the product. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/12933512/4c0b5553-c23e-43c4-8fdf-970746bb8760/paper_outline.md)


**SkipScore** ∈; higher values indicate superior safety-efficiency trade-offs. The optimal configuration maximizes SkipScore among those satisfying the Recall constraint.[[sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0020025525006139)]

The complete Grid Search algorithm is shown in Algorithm 1. The Test set (40%, ~60 videos) was reserved exclusively for final performance evaluation.

**Algorithm 1: Grid Search for Hyperparameter Optimization**

```
textInput: Validation Dataset (D_val), Hyperparameter Grid G, Recall_Threshold = 0.99
Output: Best hyperparameters h_best

1. Initialize: best_score = 0, h_best = None
2. For each hyperparameter combination h ∈ G:
   a. Apply skip module with h to D_val
   b. Compute metrics: Recall(R), Filter_Rate(FR), FPR = 1 - FR
   c. If R >= Recall_Threshold:
      i.   SkipScore = R * FR * (1 - FPR)  // Note: simplifies to R * FR^2
      ii.  If SkipScore > best_score:
           best_score = SkipScore
           h_best = h
3. Return h_best
```

**Table 5: Hyperparameter Search Space and Final Selected Values**

| Hyperparameter                    | Possible Values | Approach 1 Final | Approach 2 Final | Rationale                                             |
| :-------------------------------- | :-------------- | :--------------- | :--------------- | :---------------------------------------------------- |
| Grid Size ($N \times N$)          | {8, 16, 32}     | **16×16**        | **16×16**        | Balances spatial granularity with computational cost  |
| Motion Threshold (% pixels/block) | {1%, 2%, 5%}    | **2%**           | **1.5%**         | Lowered for Approach 2 to capture subtle smoke motion |
| Red Channel Threshold (Fire)      | N/A             | N/A              | ****             | Targets high-intensity flame regions                  |
| Grayness Threshold (Smoke)        | N/A             | N/A              | **[0.7, 0.95]**  | Identifies semi-transparent smoke (low saturation)    |
| Temporal Window (frames)          | N/A             | N/A              | **{3, 5} → 5**   | For consistency checking (flicker vs. noise)          |

*Grid Search Scale:* 27 combinations (3×3×3 core params). 12 satisfied Recall ≥ 99%.

**Table 6: Grid Search Results Summary (Top 3 Configurations on Validation Set)**

| Rank | Configuration                         | Recall (Val) | Filter Rate (Val) | SkipScore (Val)        | Selected       |
| :--- | :------------------------------------ | :----------- | :---------------- | :--------------------- | :------------- |
| 1    | 16×16, 1.5%, Smoke Rules Enabled      | **99.1%**    | **72.1%**         | **0.714**              | ✅ (Approach 2) |
| 2    | 16×16, 2.0%, Smoke Rules Enabled      | 99.0%        | 68.3%             | 0.676                  | ❌              |
| 3    | 32×32, 1.0%, Smoke Rules Enabled      | 99.2%        | 65.7%             | 0.652                  | ❌              |
| --   | 16×16, 2.0%, No Smoke Rules (Appr. 1) | 97.2%        | 65.0%             | **N/A** (Recall < 99%) | ❌              |


*Note:* SkipScore simplifies to `Recall × Filter_Rate²` due to FPR = 1 - Filter_Rate. Approach 1's best config failed the safety constraint (97.2% Recall), so we relaxed it slightly for that ablation but prioritized Approach 2.

**Heat map visualization**
```
text...after Table 6.

[ASCII Heatmap #6]

Figure 3: SkipScore heatmap from Grid Search (darker = higher).
```


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
| **Baseline (BIG MODEL only)**  | 50.0               | 20            | **98.5%**         | 0% (Reference)       |
| M1 (Lightweight Classifier)    | 15.0               | 66            | 76.0%             | 70%                  |
| M2 (YOLO-Small)                | 22.0               | 45            | 81.5%             | 56%                  |
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

## 4.7. Ablation and Qualitative Analysis

To isolate the impact of our specific heuristic rules, we conducted an ablation study comparing the naive motion baseline (Approach 1) against the full rule-based engine (Approach 2). While Approach 1 performed adequately for dynamic, rapidly flickering fires, its recall dropped significantly on smoke events. Because smoke diffuses slowly and lacks sharp edge transitions, naive background subtraction thresholds frequently misclassified it as static background lighting changes.

The integration of the rule-based engine in Approach 2—specifically the grayish-color tracking and temporal texture consistency rules—corrected this deficiency, bridging the gap in Recall. This quantitative improvement is supported by qualitative reviews of edge cases (see Figure 4).

*(Insert Figure 4 here: 2x2 grid showing a frame with smoke missed by Appr 1 but caught by Appr 2, and a safe frame with swaying trees flagged by Appr 1 but skipped by Appr 2).*

As illustrated in Figure 4, Approach 2 successfully captures slow-diffusion smoke events that lack the raw pixel displacement required to trigger Approach 1. Furthermore, in scenes featuring subtle twilight illumination shifts and swaying foliage, Approach 1 frequently generated false positives (unnecessarily passing safe frames to the BIG MODEL). Approach 2 successfully filtered these frames by applying color-channel heuristics, verifying that the moving pixels did not match the chromatic signatures of either fire or smoke.



## 5. Discussion

*(Focus on the "Why" and the "Limits")*
 The empirical results demonstrate that the bottleneck in high-accuracy continuous surveillance is the redundancy of the input data, not the deep learning model itself. By successfully identifying and dropping non-informative frames at the edge, our proposed module enables the deployment of computationally heavy, expert-level models on resource-constrained hardware.

**Limitations and Generalization:** While Approach 2 proved highly robust, the reliance on color heuristics means the system is currently constrained to daytime or well-lit surveillance. In zero-light environments utilizing IR cameras, the red-channel heuristics for fire detection would fail, requiring a fallback to pure motion or thermal thresholds. Furthermore, extreme weather conditions such as heavy, moving fog can mimic the grayish diffusion of smoke, occasionally leading to false positives that reduce the overall Filter Rate.

## 6. Conclusion

*(Focus on the "What" and "Next")*
 We proposed a lightweight, plug-and-play skip module designed to accelerate real-time fire and smoke detection in static surveillance systems. By combining block-based motion analysis with targeted color and texture heuristics, our module safely filters out up to 72% of irrelevant background frames without compromising safety. Extensive evaluations demonstrate that our approach achieves a 3$\times$ system speedup and a 67% reduction in computational cost, while fully preserving the near-perfect accuracy (98.5% F1-Score) of heavy deep learning models. Future work will focus on integrating adaptive, unsupervised thresholding to dynamically adjust heuristic rules based on real-time environmental lighting and weather shifts.