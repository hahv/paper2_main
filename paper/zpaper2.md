---
# ---------------- START ABSTRACT SYNC BLOCK ----------------#
# <!-- !START_SYNC_BLOCK -->
<!-- TARGET_PROJECT: G:\My Drive\1_PhD\Obsidian\Home\3. Writing\paperfire2 -->
<!-- SYNC_TARGET_FILE: 00_Meta_Abstract.md-->
<!-- BLOCK_ID: abstract -->
date: 2026.04.29
title: "Efficient Real-Time Fire Surveillance: A Lightweight Motion-Heuristic
Skip Module for Accelerated Inference in Indoor Environments"
abstract: "Conventional deep learning (DL)-based fire and smoke detection systems process
every frame of a video stream through a computationally intensive model to classify the
presence of fire or smoke. However, in surveillance scenarios — particularly indoor
environments — video streams frequently contain static, background-only frames devoid of
motion or relevant events, rendering per-frame inference redundant and wasteful of
computational resources. To address this limitation, this study proposes a novel
skip-module mechanism that leverages motion detection to selectively bypass DL inference
on non-informative frames in fire and smoke classification pipelines. Evaluation on a
large-scale dataset comprising 150 indoor videos demonstrates that the proposed method
substantially improves processing throughput while preserving detection performance.
Specifically, the skip module enable the system a nearly 30% increase in frames per second (FPS)
while incurring a marginal relative recall reduction of less than 1.5% relative to the baseline (recall ≥ 94%),
demonstrating the feasibility of plug-and-play inference acceleration for real-time fire
and smoke surveillance systems."

# <!-- !END_SYNC_BLOCK -->

author: Hoang Van-Ha, Jong Weon Lee, Park Chun-Su
documentclass: article
fontsize: 10pt
geometry:
  - a4paper
  - margin=1cm
link-citations: true
secPrefix:
  - Section
  - Sections
figPrefix:
  - Fig.
  - Figs.
tblPrefix:
  - Table
  - Tables
eqnPrefix:
  - Eq.
  - Eqs.

---

# Introduction {#sec:introduction}

<!-- !START_SYNC_BLOCK -->
<!-- TARGET_PROJECT: G:\My Drive\1_PhD\Obsidian\Home\3. Writing\paperfire2 -->
<!-- SYNC_TARGET_FILE: 01_Introduction.md-->
<!-- BLOCK_ID: intro -->

<!-- !Why matter -->

Fires and wildfires, if not detected and controlled in their early stages, can
quickly escalate, especially under dry and windy conditions, resulting in
catastrophic consequences, including loss of life, destruction of property, and
damage to natural ecosystems. For instance, in 2017, urban fires and wildfires
in California, USA, caused an estimated economic loss of 10 billion USD
[@californiafire:online]. More recently, in 2025, a forest fire in Gyeongsang
Province, South Korea, burned approximately 90,000 acres, resulting in at least
27 fatalities and the evacuation of nearly 40,000 residents
[@southkoreafire:online]. Automated early-stage fire and smoke detection systems
are therefore critical for enabling timely intervention and minimizing damage.

<!-- !Existing DL approaches -->

Numerous fire and smoke detection methods leveraging deep learning (DL) have
been proposed in recent years [@cheng2024visual; @gragnaniello2024fire]. In
practical deployments, these systems are commonly applied to CCTV video streams,
where DL-based classifiers or object detectors perform inference on each frame
individually. Although this frame-wise paradigm is straightforward to implement,
it exhibits two key limitations. First, it relies exclusively on spatial
information within individual frames, neglecting temporal cues, such as
motion, inherent in video data. Second, in surveillance scenarios,
particularly in **indoor** environments, consecutive frames often exhibit minimal
variation due to static scene content. Applying a DL model indiscriminately to
every frame under such conditions is computationally redundant, increasing
processing cost and latency without contributing to detection performance. This
inefficiency is further compounded by the well-known accuracy-efficiency
trade-off in DL: more accurate models are generally more computationally
intensive, and redundant frame processing amplifies these demands, rendering
real-time performance difficult to achieve.

<!-- !What we propose -->

A common strategy to reduce inference cost is to substitute a heavy,
high-accuracy DL model with a lighter alternative; however, this typically
degrades detection reliability, an unacceptable compromise in safety-critical
applications such as fire and smoke detection. This study takes a complementary
approach: rather than replacing the classifier, we introduce a lightweight
skip-module that acts as a computational gate, selectively forwarding only
frames with significant scene activity to the high-capacity classifier while
bypassing static, non-informative frames. As illustrated in Fig.
\ref{fig:pipeline}, the conventional pipeline processes every frame through the
classifier, whereas the proposed pipeline inserts the skip-module upstream to
conditionally suppress redundant inference calls.

<!-- !Main Contributions -->

In particular, our main contributions are summarized as follows:

- **Skip-Module Mechanism:** We propose a lightweight skip-module that exploits
  motion estimation via frame differencing to identify static scenes and bypass
  DL inference on static frames, reducing computational cost without
  modifying the underlying classifier. The module is designed as a plug-and-play
  component compatible with existing fire and smoke detection pipelines. To
  identify the optimal operating configuration, we further develop a systematic
  hyperparameter optimization procedure that maximizes throughput while
  preserving detection performance.

- **Indoor Fire and Smoke Video Dataset:** We construct an annotated dataset
  comprising 150 indoor fire and smoke videos (including fire, smoke, and
  background-only classes) captured by static surveillance cameras at
  resolutions ranging from $814 \times 720$ to $3840 \times 2160$. The dataset
  covers diverse environments such as warehouses, parking areas, and offices
  under varying lighting conditions, providing a realistic benchmark for
  evaluating detection systems in static surveillance scenarios.

- **Comprehensive System Evaluation:** We integrate the skip-module with a
  high-capacity DL classifier (BIG model) and evaluate the combined system on
  our video dataset against the baseline (BIG model without skipping) and
  existing methods. Results demonstrate a 30% improvement in FPS with a recall
  reduction of less than 1%, alongside an ablation study that provides insights
  into system performance and limitations.

<!-- !Paper Organization -->

The rest of this paper is organized as follows: [@sec:relatedWork] reviews
existing fire and smoke detection methods for video and relevant motion
detection techniques, contextualizing the need for efficient processing in
static surveillance scenarios. [@sec:method] describes the proposed system
architecture, the design of the skip-module, its integration with the BIG model,
and the hyperparameter optimization strategy. [@sec:results] presents the
experimental setup, evaluation metrics, and performance results on our
large-scale indoor video dataset. Finally, [@sec:conclusion] summarizes the key
findings and discusses limitations and future research directions.

<!-- !END_SYNC_BLOCK -->

# Related Work {#sec:relatedWork}

<!-- !START_SYNC_BLOCK -->
<!-- TARGET_PROJECT: G:\My Drive\1_PhD\Obsidian\Home\3. Writing\paperfire2 -->
<!-- SYNC_TARGET_FILE: 02_aRelated_work.md-->
<!-- BLOCK_ID: related -->

<!-- !Sample writing -->

**Fire and Smoke Detection in Images and Videos**: Automated fire and smoke
detection has attracted considerable research interest, with deep learning
emerging as the dominant paradigm. The work [@cheng2024visual] provides a
comprehensive survey of visual fire detection methods, highlighting the rapid
adoption of DL-based approaches in this domain. The majority of these methods
formulate the problem as image-level binary classification (fire/smoke vs. none)
or object detection, in which a model receives a single RGB image and outputs
either a class label with confidence score or localized bounding boxes
[@geng2024yolofm; @khan2025optimized]. Research efforts have primarily focused
on improving model accuracy and inference speed through architectural
modifications and advanced training strategies. In video-based deployments,
these image classifiers are applied directly to each frame, enabling integration
with existing pipelines without architectural changes.

Relying exclusively on individual frames, however, discards temporal information
inherent in video data that is potentially informative for detection.
[@khan2025beyond] survey video-based fire and smoke detection approaches and
highlight the benefit of modeling temporal dynamics. Representative methods
include the work of [@cao2019attention], who combine a CNN for spatial feature
extraction with a bidirectional LSTM for temporal modeling, and
[@ali2025toward], who employ 3D CNNs with attention mechanisms to jointly
capture spatio-temporal patterns. While these approaches demonstrate improved
detection performance over purely image-based methods, they incur greater
computational cost and require more laborious dataset construction involving
temporal annotation. Hybrid methods address this partially by combining
image-based classifiers with lightweight temporal post-processing --- such as
temporal voting [@de2023hybrid] or motion-based false positive suppression
[@gragnaniello2025flame] --- yet still apply DL inference unconditionally to
every frame, regardless of scene content.

Across both paradigms, the dominant deployment pattern applies DL inference to
each frame or short clip, without considering whether inference is warranted
given the frame content. In surveillance scenarios, particularly **indoors**,
video streams frequently consist of purely background frames with no motion, in
which fire or smoke is a priori unlikely. Motivated by this observation, we
propose a skip-module that filters such low-activity frames prior to DL
inference, reducing computational cost while with minimal degradation in
detection reliability. The module prioritizes retaining positive frames
(fire/smoke present) while skipping negative frames (background only),
functioning as a complementary accelerator for any existing frame-wise detection
pipeline.

**Motion Detection and the Proposed Skip Module**: The skip decision in our
module relies on motion estimation, which connects to the well-studied problem
of background subtraction. [@sobral2014comprehensive] and
[@garcia2020background] provide comprehensive reviews of background subtraction
methods, ranging from simple statistical models to deep learning-based
approaches. More robust methods such as MOG2 and KNN [@zivkovic2006efficient]
model the scene background statistically and are resilient to gradual
illumination changes; however, they introduce non-trivial memory overhead and
initialization latency. As the skip-module is designed primarily as a fast,
lightweight gate, we instead adopt motion estimation based on frame
differencing, which offers minimal computational overhead and requires no
background model initialization.

Existing efficient video inference methods, such as FrameExit
[@ghodrati2021frameexit], employ learned gating mechanisms to selectively
process frames based on prediction confidence, thereby reducing redundant
inference; however, they require joint training of the gating mechanism and the
underlying classifier, limiting plug-and-play applicability to pre-trained
models. In contrast, this study adopts a training-free approach by integrating lightweight frame-differencing-based motion estimation as a plug-and-play gate, requiring no retraining of the underlying classifier, as described in Section \ref{sec:method}.

<!-- !END_SYNC_BLOCK -->

# The proposed method {#sec:method}

<!-- !START_SYNC_BLOCK -->
<!-- TARGET_PROJECT: G:\My Drive\1_PhD\Obsidian\Home\3. Writing\paperfire2 -->
<!-- SYNC_TARGET_FILE: 02_Method.md-->
<!-- BLOCK_ID: method -->

## System Overview

```{=latex}
\input{./3.fig/fig_pipeline.tex}
```
The proposed system is a fire and smoke detection pipeline that processes a
continuous video stream and classifies each frame as either containing
fire/smoke or not. The core idea is to augment the conventional frame-wise
inference pipeline with a lightweight skip module $\mathcal{S}$, which serves as
a gating mechanism that selectively suppresses DL inference on
frames with low likelihood of containing fire or smoke — most commonly, static
background frames — thereby reducing redundant computation and improving overall
throughput. Formally, let $f_t$ denote the video frame at time step $t$,
$\mathcal{M}$ the DL classifier, and $\hat{y}_t \in {0, 1}$ the predicted label
for $f_t$.

**Baseline system.** Each frame is directly passed to the classifier:

$$\hat{y}_t = \mathcal{M}(f_t)$$

**Proposed system.** The skip module $\mathcal{S}$ first evaluates $f_t$ using
inter-frame motion cues and outputs a binary gate decision $s_t$. While this work instantiates $\mathcal{S}$ using motion-based cues exclusively, the formulation is general and the skip module can incorporate other types of cues such as color, texture, or learned features. The full system output becomes:
$$
\hat{y}_t = \begin{cases} \mathcal{M}(f_t) & \text{if } s_t = 1 \quad
\text{(run inference)} \\ 0 & \text{if } s_t = 0 \quad \text{(skip, label as
negative)} \end{cases}
$$

In this work, skip module $\mathcal{S}$ derives $s_t$ from the motion estimated
between $f_t$ and the previous frame $f_{t-1}$. Frames with little or no motion
are unlikely to contain fire or smoke, and are therefore skipped without
invoking the classifier $\mathcal{M}$. The model $\mathcal{M}$ is typically a
high-capacity, accurate classifier but is computationally expensive. The skip
module $\mathcal{S}$ is deliberately designed to be computationally lightweight,
adding negligible overhead. The goal of the skip module $\mathcal{S}$ is to skip
as many background/safe frames as possible while minimizing the risk of skipping
fire/smoke frames, thus improving overall system throughput (FPS) while
maintaining high detection accuracy.

## Skip Module Design

```{=latex}
\input{./3.fig/fig_skipmodule.tex}
```

```{=latex}
\input{6.algo/skip_module.tex}
```

As illustrated in Figure \ref{fig:skipmodule} and formalized in Algorithm \ref{alg:skipmodule}, the skip module $\mathcal{S}$ can
leverage any available information — such as motion, color, or texture — to
compute the skip decision $s_t$. In this work, we focus on motion information
derived from consecutive frames as the primary cue. Specifically, we design and
evaluate two motion-based approaches of $\mathcal{S}$: (1) FrameDiffDet — a
frame differencing method, and (2) AccMotionDet — a motion accumulation method.
Both approaches are selected for their simplicity and low computational cost,
making them well-suited for real-time deployment. The details of each approach
are described in the following subsections.

### FrameDiffDet — Naive Motion Detection

```{=latex}
\input{6.algo/frame_diff_det.tex}
```

### AccMotionDet — Motion Detection with Accumulation

```{=latex}
\input{6.algo/acc_motion_det.tex}
```

## Skip Module Hyperparameter Optimization {#sec:hyperparam}

```{=latex}
\input{./6.algo/hyperparam_algo.tex}
```

<!-- !END_SYNC_BLOCK -->

# Experiments and Results {#sec:results label="results"}

<!-- !START_SYNC_BLOCK -->
<!-- TARGET_PROJECT: G:\My Drive\1_PhD\Obsidian\Home\3. Writing\paperfire2 -->
<!-- SYNC_TARGET_FILE: 03_Results.md-->
<!-- BLOCK_ID: results -->

## Fire and Smoke Indoor Video Dataset

```{=latex}
\input{./3.fig/fig_videodb.tex}
```

```{=latex}
 \input{./4.table/tb_ufireindoor.tex}
```

Several existing benchmarks address this detection task, including FireNet
[@jadon2019firenet], Firesense [@Firesens4:online], FiSmo [@cazzolato2017fismo],
and FURG [@steffens2015unconstrained]. However, these were recorded with moving
cameras and are therefore unsuitable for static surveillance scenarios.
Static-camera datasets such as DFire [@dfiredataset], VisiFire Bilkent
[@VisiFireBilkent:online], KMU Fire and Smoke [@KMUFireSmokeDataset], Mivia Fire
and Mivia Smoke datasets [@foggia2015real], and USTC Smoke [@lin2017smoke] do
exist; however, they collectively suffer from several limitations: a
predominance of outdoor scenes, a small number of video samples, low spatial
resolution, and insufficient diversity in fire/smoke appearances and
environmental conditions.

To address these deficiencies, we constructed a dedicated static indoor video
dataset by aggregating clips from multiple heterogeneous sources. Fire and smoke
samples were sourced from the Korea AI Fire Dataset [@AIHub87:online], the USTC
Smoke Dataset [@lin2017smoke], and the VDS3 Dataset [@huang2022fire],
supplemented by videos collected from open online platforms (Pexels, Pixabay,
and YouTube). Non-fire/smoke (negative) samples were compiled from self-recorded
footage captured by real CCTV cameras in various indoor environments (e.g.,
parking areas), along with videos from the Safe & Unsafe Behavior in Workplaces
dataset [@onal2024video], the Indoor Action dataset [@deniz2024optimized], the
MPII Cooking 2 Dataset [@rohrbach2016recognizing], and the WiseNet dataset
[@marroquin2019wisenet]. Table \ref{tb:ufireindoor} summarizes the properties of the
self-collected video dataset alongside a comparison with existing datasets, and
Figure \ref{fig:videodb} presents representative sample frames from our dataset.

For the hyperparameter optimization described in Section \ref{sec:hyperparam},
the video dataset was further partitioned into a validation set,
$D_{\text{val}}$ (46 videos), used to select the optimal skip-module
hyperparameters, and a test set, $D_{\text{test}}$ (104 videos), used for the
final evaluation of the proposed system against the baseline and existing
methods. The partition adhered to a 30:70 ratio (validation:test), following the
protocol of [@de2023hybrid], and was performed using stratified sampling to
preserve balanced class distributions in both subsets.

## Evaluation Metrics {#sec:metrics label="metrics"}

Performance is evaluated under **frame-level** evaluation, following the
standard protocol for fire and smoke detection in videos [@steffens2016non].
Frame-level evaluation measures detection performance for each individual frame,
providing a strict assessment of classification accuracy. The following label
definitions apply to this evaluation protocol:

- True Positive ($TP$): Correct detection of fire or smoke.
- True Negative ($TN$): Correct identification of the absence of fire or smoke.
- False Positive ($FP$): Incorrect detection of fire or smoke when none is present.
- False Negative ($FN$): Failure to detect fire or smoke when present.

Quantitative results are reported using standard classification metrics — accuracy, recall, false alarm rate ($\mathrm{FAR}$), precision, F1-score, and frames per second (FPS) — as defined below.

```{=latex}
\input{./5.eq/eq_metrics.tex}
```

## Models and Implementation Details {#sec:baselines label="baselines"}

**Baseline Models**: To demonstrate both the superiority of the high-capacity classifier over lightweight alternatives and the effectiveness of the proposed skip module in accelerating its inference, we evaluate the following baseline models alongside our BIG model:

- **MobileNet** [@mukhopadhyay2019fpga]: A modified MobileNet architecture
  fine-tuned for fire and smoke detection.

- **FireNet** [@jadon2019firenet]: A lightweight CNN-based image classifier comprising 14 layers, with a model size of 7.5 MB and approximately 650k trainable parameters.

- **YOLOv5s** [@de2023hybrid]: A lightweight object detector based on YOLOv5
  [@JocherYOLOv5byUltralytics2020], trained with hyperparameters selected via
  grid search on the D-Fire dataset [@dfiredataset].

- **YOLOv5l** [@de2023hybrid]: A heavier variant of the above, based on the larger YOLOv5l architecture, offering higher capacity at increased computational cost.

**Proposed Classifier (BIG Model $\mathcal{M}$ — HGNetV2)**: The BIG model
$\mathcal{M}$ is obtained by fine-tuning a pretrained High-Performance GPU
Network v2 (HGNetV2), specifically the
\texttt{hgnetv2\_b5.ssld\_stage2\_ft\_in1k} checkpoint [@hgnetv2timm:online]
from the Timm library [@rw2019timm], which was originally trained using SSLD
knowledge distillation [@cui2021beyond]. HGNetV2 [@hgnetv2PaddleCl7:online] is a
high-capacity CNN architecture designed to achieve substantially higher accuracy
than models of comparable inference speed on NVIDIA GPUs, making it well-suited
for deployment in our target environment. For fine-tuning, we compiled a dataset
of over 1 million fire and smoke images collected from internet sources,
partitioned into training (80%) and test (20%) subsets. The model was trained
using the Adam optimizer with an initial learning rate of $1 \times 10^{-4}$, a
weight decay of $1 \times 10^{-5}$, a batch size of 32, and 100 epochs. On the
held-out test set, the final model achieved an accuracy of 98.66%. The model
accepts inputs of size $3 \times 360 \times 640$.

<!-- !! MUST update with Prof. Park -->
 <!-- \textcolor{red}{For fine-tuning, we compiled a dataset of
1,000,000 fire and smoke images collected from internet sources, partitioned
into training (80\%) and test (20\%) subsets. The model was optimized using
Adam with an initial learning rate of $1 \times 10^{-4}$ and weight decay of
$1 \times 10^{-5}$ for 100 epochs with a batch size of 32. On the held-out
test set, the final model achieved a recall of xx.xx\% and a false alarm rate
of xx.xx\%.}. The model accepts inputs of size $3 \times 360 \times 640$. -->

**Implementation Details**: All experiments were conducted on a workstation
equipped with an Intel Core i9-10900K CPU, 64 GB DDR4 system memory, and an
NVIDIA GeForce RTX 3090 GPU (24 GB VRAM), running Windows 10 Pro (22H2, build
19045). Deep learning inference was performed using PyTorch 2.7.1 under CUDA
12.9, and video processing and motion estimation were carried out using OpenCV
4.11 (CPU-only).

## Results

### Hyperparameter Optimization Results
In this work, the recall tolerance is set to $\delta_R = 0.015$, permitting a
maximum absolute recall drop of $1.5\%$ relative to the baseline. At the
baseline recall of approximately $95\%$, this ensures the worst-case deployed
system retains a recall of at least $93.5\%$, while affording the hyperparameter
search sufficient flexibility to identify configurations with meaningful
efficiency gains. The optimization weights are set to $w_S = 0.7$ and $w_R =
0.3$, prioritizing skip rate (the primary determinant of throughput gain) while
using recall retention as a secondary criterion to favor configurations closer
to baseline performance among feasible candidates.


<!-- **Frame Diff Parameter Grid Search:** -->

#### Hyperparameter Optimization Results for the FrameDiffDet-based Skip Module
<!-- To identify an optimal configuration for the `motion_only_block_skip_proc`
module using `FrameDiffDet` as its motion estimator, we conducted a systematic
grid search over four parameters: `scale_factor`, `block_size_orig`,
`block_ratio_th`, and `diff_thresh`. The search space was defined as follows:

```{=latex}
\input{./4.table/tb_gridsearch_framediff.tex}
``` -->


To identify the optimal configuration for the FrameDiffDet-based skip module,
we performed a systematic grid search over four hyperparameters: scale factor
$\alpha$, block size $B$, block active threshold $\tau$, and difference
sensitivity threshold $\tau_d$, yielding a total of
$2 \times 2 \times 3 \times 4 = 48$ candidate configurations. The search
space is as follows:

+ **Scale factor** $\alpha \in \{0.5, 1.0\}$: controls the downsampling ratio
  applied to input frames prior to motion computation. A smaller $\alpha$
  reduces computational cost but may lose subtle motion detail. We evaluate
  full resolution ($\alpha = 1.0$) and half resolution ($\alpha = 0.5$).

+ **Block size** $B \in \{16, 32\}$: determines the spatial granularity of
  motion detection, expressed in pixels of the original frame (the effective
  block size on the downsampled frame is $B_s = B \cdot \alpha$). Smaller
  blocks ($B = 16$) capture localized motion from small fire or smoke regions,
  while larger blocks ($B = 32$) aggregate motion over a broader spatial
  context, providing greater robustness against isolated pixel-level noise.

+ **Block active threshold** $\tau \in \{0.05, 0.10, 0.15\}$: defines the
  minimum foreground pixel ratio within a block for it to be declared active;
  inference is triggered if at least one active block is detected. A low value
  ($\tau = 0.05$) is sensitive to sparse motion within a block, minimizing
  missed detections, while a higher value ($\tau = 0.15$) demands denser local
  motion, suppressing responses to minor pixel-level disturbances. The three
  values provide a systematic sweep from fine to coarse sensitivity.

- **Difference sensitivity threshold** $\tau_d \in \{3, 5, 7, 10\}$: defines
  the minimum absolute inter-frame intensity change required for a pixel to be
  declared active. This range is chosen to cover diverse sensitivity levels,
  from near-noise-level detection ($\tau_d = 3$, responding to subtle
  illumination changes) to robust large-motion detection ($\tau_d = 10$,
  requiring strong, unambiguous motion).


```{=latex}
\input{./4.table/tb_val_results_frameDiff.tex}
```
Table \ref{tb:val_results_frameDiff} presents representative configurations
illustrating the recall--efficiency trade-off; the full search space of 48
configurations is omitted for brevity. The results reveal a fundamental
limitation of \textsc{FrameDiffDet}: it fails to deliver meaningful throughput
improvement under the safety constraint $\delta_R = 0.015$. 

Among all 48 evaluated configurations, only three satisfy the recall constraint,
all requiring the lowest sensitivity setting $\tau_d = 3$. The best-ranked
feasible configuration ($\alpha{=}0.5,\ B{=}16,\ \tau{=}0.05,\ \tau_d{=}3$)
achieves a skip rate of merely $0.94\%$, bypassing fewer than 1 in 100
background frames. Consequently, the FPS of all feasible configurations
($22.0$--$23.3$) falls below the no-skip baseline ($24.7$), indicating that
the skip module overhead outweighs the benefit of skipped inference calls.
Conversely, configurations that do achieve substantial skip rates (reaching
$47$--$60\%$ at higher $\tau_d$ values of 7 and 10) suffer severe recall
degradation of $5$--$10\%$, far exceeding the safety tolerance.

These results demonstrate that \textsc{FrameDiffDet}, without noise suppression
or robust sustained-motion confirmation, is unsuitable as a skip gate: it
either skips too few frames to yield any efficiency gain, or skips too
aggressively and suffer unacceptable recall loss.



#### Hyperparameter Optimization Results for the AccMotionDet-based Skip Module

```{=latex}
\input{./4.table/tb_val_results_accMotionDet.tex}
```

The hyperparameter search for AccMotionDet is designed to mirror the FrameDiffDet study while accounting for the temporal accumulation of motion scores over consecutive frames. Details of the search space (of 96 configurations) are as follows:

+ **Scale factor ($\alpha \in \{0.5, 1.0\}$)** and **Block size ($B \in \{16, 32\}$)**: Retained from FrameDiffDet to maintain consistency in the spatial resolution and grid density trade-offs.

+ **Block active threshold ($\tau \in \{0.05, 0.10\}$)**: Restricted to a narrower range than in FrameDiffDet, as temporal accumulation naturally suppresses false activations, rendering more aggressive thresholds unnecessary.

+ **Diff threshold ($\tau_d \in \{3, 5\}$)**: Limited to the lower end of the FrameDiffDet search space, as higher values were found to cause excessive frame skipping and unacceptable recall degradation.

+ **Accumulation step ($\omega=5$)** and **Decay rate ($\delta=1$)**: We fix these
parameters to streamline the hyperparameter search, allowing us to focus on the
primary sensitivity controls: the activation threshold ($\tau_m$) and
accumulation cap ($K_{\max}$). Since the optimal tuning of $\tau_m$ and
$K_{\max}$ depends directly on by $\omega$, keeping $\omega$ constant provides a
stable baseline for optimization. Furthermore, setting $\delta=1$ ensures
consistent, linear decay behavior while reducing the dimensionality of the
search space.

+ **Activation threshold ($\tau_m \in \{5, 10\}$)**: This threshold serves as the decision gate for the motion mask, labeling pixels as active only when their accumulated motion score exceeds $\tau_m$. Given the fixed accumulation parameters ($\omega=5, \delta=1$), $\tau_m=5$ and $\tau_m=10$ are approximately equivalent to requiring motion signals over 2 and 3 consecutive frames, respectively.

+ **Accumulation cap ($K_{\max} \in \{15, 25, 35\}$)**: This parameter limits the maximum accumulated motion score to prevent runaway accumulation during prolonged activity. Without this cap, the system could remain triggered indefinitely even after motion ceases. By selecting a range of values, we diversify the system's saturation behavior, allowing us to tune how quickly the module resets following high-activity periods.

<!-- !! TO REWRITE -->
Table \ref{tb:val_results_accMotionDet} presents representative configurations
illustrating the recall--efficiency trade-off for \textsc{AccMotionDet}; the
full search space is omitted for brevity. These results demonstrate a
substantial improvement over the naive  over the naive \textsc{FrameDiffDet}
skip module: by leveraging temporal accumulation, \textsc{AccMotionDet}
maintains high recall while consistently achieving skip rates between $39\%$ and
$56\%$. The best-ranked configuration ($\alpha{=}0.5,\ B{=}32,\ \tau{=}0.05,\
\tau_d{=}5,\ \tau_m{=}5,\ K_{\max}{=}15$) achieves a skip rate of $44.39\%$ with
only a $0.45\%$ drop in recall, yielding a throughput of $36.3$ FPS—a
significant improvement over the $24.7$ FPS baseline. In contrast to earlier
configurations that failed to bridge the throughput gap, the
\textsc{AccMotionDet} configurations successfully operate within the safety
constraint $\delta_R = 0.015$, confirming that temporal accumulation is
requisite for robust motion-based gating.

<!-- ! Analysis the hyperparameters search results of AccMotionDet -->

<!-- ## Hyperparameter Optimization Results: AccMotionDet

Table \ref{tb:val*results_accMotionDet} shows the top-10 ranked configurations
for the AccMotionDet skip module on the validation set $\mathcal{D}*\text{val}$,
ordered by combined score $\Phi$. The baseline system (no skip module) achieves
a recall of 94.35\% at 24.1~FPS.

### Selected Configuration

The optimal configuration (Exp.~1) uses a half-resolution scale ($\alpha =
0.5$), coarse block size ($B = 32$), conservative block-active threshold ($\tau
= 0.05$), moderate pixel sensitivity ($\tau_d = 5$), low activation threshold
($\tau_m = 5$), and a small accumulation cap ($K_{\max} = 15$). This
configuration achieves a skip rate of **44.39\%** and a recall of **93.90\%**
($\Delta = -0.45\%$), well within the tolerance $\delta_R = 0.015$. The
resulting FPS improves from 24.1 to **36.6**, a **52\% throughput gain**, with
no change in FPR (0.74\% throughout).

### Hyperparameter Sensitivity Analysis

**Scale factor $\alpha$.** All top-10 configurations consistently use $\alpha =
0.5$ (half resolution). Full-resolution processing ($\alpha = 1.0$) does not
appear in any feasible high-scoring configuration, confirming that downscaling
reduces sensitivity to high-frequency pixel noise while lowering skip-module
latency --- both effects are beneficial.

**Block size $B$.** $B = 32$ dominates the top-10 rankings, with only one entry
using $B = 16$ (Exp.~10, score 0.4906). Coarser blocks aggregate motion evidence
spatially, providing greater robustness to isolated pixel-level disturbances
that could otherwise trigger unnecessary inference calls. The result confirms
the prior expectation in Section~\ref{secmethod} that temporal accumulation
already suppresses transient noise, making fine-grained $B = 16$ blocks less
necessary for AccMotionDet.

**Block active threshold $\tau$.** Configurations with $\tau = 0.05$ yield the
highest skip rates (42--44\%), while $\tau = 0.10$ achieves slightly lower skip
rates (39--40\%) but marginally better recall retention. The top-ranked
configuration (score 0.5203) uses $\tau = 0.05$, confirming that a conservative
trigger policy --- requiring only sparse motion evidence before invoking
inference --- is preferred for maximizing the combined score under the recall
constraint.

**Activation threshold $\tau_m$.** Configurations with $\tau_m = 5$ (Exp.~1--6)
consistently outscore those with $\tau_m = 10$ (Exp.~7--8). A higher activation
threshold $\tau_m = 10$ requires stronger sustained motion before triggering
inference, which slightly improves the skip rate but incurs a larger recall drop
(up to $-0.53\%$), reducing the recall retention term in $\Phi$.

**Accumulation cap $K_{\max}$.** Across all three values evaluated ($K_{\max}
\in \{15, 25, 35\}$), the skip rate decreases monotonically with increasing
$K_{\max}$ (e.g., 44.39\% $\to$ 42.89\% $\to$ 42.38\% for Exp.~1--3). A lower
cap allows the accumulator to reset more readily after high-motion periods,
enabling faster recognition of subsequent static frames and thus higher skip
rates. The difference in combined score across the three values is small ($\leq
0.007$), indicating low sensitivity to this parameter within the evaluated
range. -->

<!-- ### Comparison with FrameDiffDet

A fundamental contrast emerges when comparing AccMotionDet and FrameDiffDet on
the validation set (Table @tbl:frameDiff-val). All feasible FrameDiffDet
configurations --- those satisfying $\Delta R \leq \delta_R$ --- achieve skip
rates below **1.4\%**, yielding no meaningful FPS improvement over the baseline
(22.4--23.5~FPS vs.\ 24.1~FPS baseline). This reveals a structural limitation of
naive frame differencing: in the static indoor surveillance setting, the
per-pixel sensitivity required to safely detect slow-onset smoke events forces
the threshold $\tau_d$ to remain low (i.e., $\tau_d = 3$), which in turn flags
even subtle illumination changes as motion, suppressing skip decisions on nearly
all frames. By contrast, the temporal accumulation in AccMotionDet absorbs
transient pixel fluctuations over multiple frames, enabling confident skip
decisions on genuinely static frames while preserving sensitivity to sustained
motion signatures characteristic of fire and smoke. This results in a
**30$\times$ higher skip rate** (44.39\% vs.\ 0.72\%) and a **63\% higher FPS**
(36.6 vs.\ 22.4) for the respective best configurations, at a comparable recall
cost ($-0.45\%$ vs.\ $-0.17\%$). FrameDiffDet achieves high skip rates only at
the cost of severe recall degradation ($\geq 5.5\%$ for skip rates $\geq 47\%$),
confirming that it lacks the temporal smoothing necessary for safe operation in
this domain. AccMotionDet is therefore selected as the skip module for all
subsequent system-level evaluations. -->

### Frame-Level Accuracy and Throughput: Baselines and Skip Module {#sec:e2e-perf}

```{=latex}
\input{./4.table/tb_perf_per_frame.tex}
```
Table \ref{tb:perf_per_frame} benchmarks the proposed pipeline against
lightweight fire-and-smoke detectors, described in Section \ref{sec:baselines}.
The baseline BIG Model (without skip module) achieves the highest recall
(95.62%), accuracy (98.77%), and F1-score (0.97), substantially outperforming
all lightweight alternatives. This gap reflects the consequences of neural
scaling laws [@hestness2017deep; @alabdulmohsin2022revisiting;
@bahri2024explaining]: the BIG model benefits from both a more expressive
architecture and a significantly larger training dataset. Notably, MobileNet
and FireNet exhibit near-degenerate recall of 5.84% and 34.1%, respectively,
rendering them unsuitable for safety-critical fire and smoke detection despite
their speed and compactness. YOLOv5s achieves the highest throughput (109.94
FPS) but still suffers a recall of 68.58%, roughly 27 percentage points below
the BIG model, confirming that no lightweight alternative achieves an acceptable
accuracy--efficiency balance on this task.

**Effect of the Skip Module:** Integrating the skip module yields a consistent
set of improvements across nearly all metrics. Throughput increases from 24.97
to 32.32 FPS, a gain of approximately 29%, while the F1-score is fully preserved
at 0.97. Recall drops by only 1.24 percentage points (95.62% $\to$ 94.38%),
remaining within the safety constraint $\delta_R = 0.015$. A less obvious but
notable benefit is the reduction in FAR from 0.25% to 0.14%, accompanied by a
precision increase from 99.17% to 99.54%: by suppressing inference on
near-static background frames, the skip module also eliminates a class of
spurious detections that the BIG model would otherwise generate.

### Comparison with Temporal Post-Processing {#sec:cmp-temporal}

Temporal Persistence Thresholding (TPT) [@de2023hybrid] augments the standard
per-frame inference pipeline with a lightweight post-processing stage designed
to suppress isolated false alarms. Unlike the proposed skip module, TPT does
not reduce the number of frames forwarded to the classifier --- every frame is
still processed by the full BIG model. Instead, a circular boolean buffer of
length $W$ records the raw per-frame predictions over a rolling window. A
detection is confirmed only if the fraction of positive predictions within the
buffer exceeds a persistence threshold $\tau_\text{persist}$; otherwise, the
prediction is suppressed and the frame is re-labelled as negative.

```{=latex}
\input{./4.table/tb_val_results_Tpt.tex}
```
```{=latex}
\input{./4.table/tb_cmp_other_temp.tex}
```

We follow the hyperparameter selection protocol of [@de2023hybrid], evaluating
TPT on the validation set $\mathcal{D}_\text{val}$. Two representative
configurations are reported: a strict variant
($\text{TPT}_\text{strict}$: $W = 5,\ \tau_\text{persist} = 0.2$) and a
balanced variant ($\text{TPT}_\text{balanced}$: $W = 10,\
\tau_\text{persist} = 0.5$), with the full grid search results shown in
Table \ref{tb:gridsearch_tpt}. Both configurations are then compared against
the proposed pipeline in Table \ref{tb:cmp_other_temp}.

Table \ref{tb:cmp_other_temp} highlights a clear distinction between temporal
post-processing and the proposed skip module in terms of the
efficiency--accuracy trade-off. TPT processes every frame through the full Big
Model and therefore does not improve throughput, with FPS remaining essentially
unchanged (24.97 to 24.96). In contrast, the AccMotionDet-based skip module
skips 34.93% of negative frames and increases FPS from 24.97 to 32.32, corresponding to
an approximately 29% throughput gain. It also achieves the largest reduction in
FAR, decreasing from 0.251% to 0.136% (about 46% relative reduction), whereas
the TPT variants provide only negligible FAR improvement. Among the two TPT
settings, $\text{TPT}_\text{balanced}$ attains slightly lower recall than
$\text{TPT}_\text{strict}$ (95.123% vs. 95.519%) while producing nearly the
same FAR (0.247% vs. 0.250%), indicating limited sensitivity to the persistence
threshold. Finally, the recall reduction of AccMotionDet is modest
(95.62% $\to$ 94.38%) and remains within the safety constraint
$\delta_R = 0.015$.

### Quantitative and Qualitative Analysis

#### Efficiency of Skip Module

```{=latex}
\input{./3.fig/fig_fps_increase.tex}
```
To evaluate the efficiency of the skip module, the BIG model was run on the test
set $\mathcal{D}_\text{test}$ with and without the skip module. The average
processing time of the skip module, the BIG model inference time, and the
overall system FPS were recorded in both cases. The results are visualized in
Figure \ref{fig:fps_increase}.

As shown in Figure \ref{fig:fps_increase}, the skip module requires an average processing time of only 1.79 ms per frame, which is approximately ×22 lower than the BIG model inference time of 39.88 ms. In the best case, the skip module correctly identifies and bypasses negative frames, contributing only 1.79 ms of overhead. In the worst case, where the skip module fails to bypass a frame and the BIG model inference is subsequently invoked, the additional latency introduced by the skip module is 1.79 ms — a mere 4.5% increase relative to the BIG model inference time. Overall, integrating the skip module improves system throughput by approximately 30% on $\mathcal{D}_\text{test}$, demonstrating its effectiveness for real-time fire and smoke detection in indoor surveillance scenarios.


<!--! GUIDE: #### Qualitative Analysis:
  + Success case analysis: Side-by-side frames: correctly skipped static
  background vs. correctly passed fire/smoke frame, one for each class -- Gives
  reviewers visual intuition of what the gating looks like in practice
  + Failure case analysis: Pick 2-3 failure videos from test set: (1) slow smoke
  onset over-skipped, (2) persistent background motion (zero skip), show
  representative frames with caption -- Scientific honesty; safety-critical
  papers are expected to show where the system fails -->

#### Qualitative Analysis

Figure~\ref{fig:qualitative} illustrates representative success and failure
cases of the AccMotionDet skip module on the test set $\mathcal{D}_\text{test}$.

**Success cases.** The top row shows two canonical scenarios where $\mathcal{S}$
operates as intended. In the first case, a static indoor background frame ---
with no inter-frame motion --- is correctly skipped ($s_t = 0$), avoiding an
unnecessary invocation of $\mathcal{M}$ and contributing directly to the 44.39\%
skip rate reported in Table~\ref{tb:val_results_accMotionDet}. In the second
case, a frame containing active fire (left) and smoke (right) exhibits
sufficient accumulated motion to trigger inference ($s_t = 1$), and
$\mathcal{M}$ correctly raises an alarm. These cases confirm that $\mathcal{S}$
reliably distinguishes scene-level activity from quiescence in the target indoor
surveillance setting.

```{=latex}
\input{./3.fig/fig_quality_success.tex}
```

**Failure Cases**

The bottom row exposes two structural limitations of the motion-based gating
approach. First, in a slow-onset smoke video, the initial frames of smoke
diffusion produce minimal inter-frame pixel change --- below the activation
threshold $\tau_m$ of the AccMotionDet accumulator --- causing $\mathcal{S}$ to
skip these frames and delay the first alarm. This represents the primary recall
risk identified in Section~\ref{sec:hyperparam}: the skip module is conservative
by design, but gradual diffusion events are the edge case where this
conservatism incurs a safety cost. Second, in a scene containing continuous
background motion (e.g., a person walking repeatedly through the frame),
$\mathcal{S}$ assigns $s_t = 1$ to nearly every frame, reducing the skip rate to
near zero and negating the computational efficiency gain for that video. Both
failure modes are consistent with the motion-only design of $\mathcal{S}$: they
arise not from classification errors in $\mathcal{M}$, but from the inherent
limitation of using inter-frame motion as a sole proxy for scene
informativeness.

```{=latex}
\input{./3.fig/fig_quality_failure.tex}
```

<!-- !END_SYNC_BLOCK -->

<!-- #### Eager Mode: Recovery of Missed Cases

**Fix for slow-onset smoke:** We intro the skip module with eager mode and
periodically check for specific scences

```{=latex}
\input{./6.algo/skip_module_period_check.tex}
```

We do hyper-parameter search on $D_\text{val}$ to find the optimal params for
this eager mode.
```{=latex}
\input{./4.table/tb_gridsearch_accMotionDetEager.tex}
```
we Compare again on test set
```{=latex}
\input{./4.table/tb_cmp_other_temp_eager.tex}
``` -->

# Conclusion {#sec:conclusion label="Conclusion"}

<!-- !START_SYNC_BLOCK -->
<!-- TARGET_PROJECT: G:\My Drive\1_PhD\Obsidian\Home\3. Writing\paperfire2 -->
<!-- SYNC_TARGET_FILE: 04_Conclusion.md-->
<!-- BLOCK_ID: conclusion -->

We propose a lightweight, plug-and-play skip module $\mathcal{S}$ designed to
accelerate real-time fire and smoke detection in indoor surveillance scenarios.
Operating at a negligible fraction (${\approx}4.5\%$) of the computational cost of the
DL classifier $\mathcal{M}$, the skip module $\mathcal{S}$ efficiently filters
out static background frames before they reach $\mathcal{M}$, reducing
unnecessary inference overhead. Experiments on a real-world video dataset
demonstrate that the proposed approach yields a throughput improvement of
approximately 30\% in frames per second while sustaining high detection recall
above 94%.

<!-- !END_SYNC_BLOCK -->

<!-- !START_SYNC_BLOCK -->
<!-- TARGET_PROJECT: G:\My Drive\1_PhD\Obsidian\Home\3. Writing\paperfire2 -->
<!-- SYNC_TARGET_FILE: 04_Conclusion.md-->
<!-- BLOCK_ID: discussion -->

Nevertheless, the proposed approach has certain limitations. First, the skip
module $\mathcal{S}$ relies solely on inter-frame motion cues, which may be
unsuitable for outdoor surveillance scenarios where persistent motion sources —
such as swaying vegetation, moving vehicles, etc. — are continuously present,
potentially causing $\mathcal{S}$ to pass the majority of frames to
$\mathcal{M}$ and negating the efficiency gain. Second, the hyperparameters of
$\mathcal{S}$ require dataset-specific tuning prior to deployment, incurring
additional optimization overhead. Third, the overall detection accuracy of the
system remains bounded by the capacity of $\mathcal{M}$, which is treated as a
fixed component and not optimized in this work.

Future work may explore alternative skip strategies — such as leveraging color,
texture, or learned features as gating cues — and extend the framework to
outdoor surveillance scenarios where motion-based filtering is less effective.
Additionally, designing a hyperparameter-free or self-adaptive skip module, as well as jointly optimizing the skip module and the classifier in an end-to-end training framework, represent promising directions for further improving system efficiency and detection accuracy.

<!-- !END_SYNC_BLOCK -->

# References {#sec:references label="bibliography"}
