---
# ---------------- START ABSTRACT SYNC BLOCK ----------------#
# <!-- !START_SYNC_BLOCK -->
<!-- TARGET_PROJECT: G:\My Drive\1_PhD\Obsidian\Home\3. Writing\paperfire2 -->
<!-- SYNC_TARGET_FILE: 00_Meta_Abstract.md-->
<!-- BLOCK_ID: abstract -->
date: 2026.06.02
title: "Efficient Real-Time Fire Surveillance: A Lightweight Motion-Heuristic
Skip Module for Accelerated Inference in Indoor Environments"

abstract: " Conventional deep learning (DL)-based fire and smoke detection systems perform inference on every video frame using high-complexity models to classify the presence of fire or smoke. However, in surveillance scenarios, particularly indoor environments, video streams frequently contain long periods of static background frames with little or no motion, rendering per-frame inference redundant and wasteful of computational resources. To address this limitation, this study proposes an efficient inference acceleration framework for fire and smoke surveillance systems that combines a motion-aware skip module with an Eager mode for recall recovery. The skip module leverages motion detection to selectively bypass unnecessary DL inference on non-informative frames, while the Eager mode periodically forces inference updates to mitigate missed detections from slow-developing or nearly stationary smoke events.

Evaluation on a large-scale dataset comprising 150 indoor videos demonstrates that the proposed framework achieves nearly 30% higher throughput than the baseline performing per-frame inference, with only a 0.05% reduction in recall. In contrast, using the skip module alone results in a 1.23% recall drop, highlighting the effectiveness of the Eager mode in recovering detection performance while preserving most of the throughput gain. These results demonstrate that the proposed framework provides an effective plug-and-play inference acceleration strategy for real-time indoor fire and smoke surveillance systems."

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
[@southkoreafire:online]. These incidents highlight the critical importance of
automated early-stage fire and smoke detection systems for enabling timely
intervention and minimizing damage.

<!-- !Existing DL approaches -->

In recent years, DL-based approaches have emerged as the prevailing methodology in fire and smoke detection systems [@cheng2024visual; @gragnaniello2024fire]. In practical deployments, such  systems are commonly integrated  into  CCTV video pipelines, where DL-based classifiers or object detectors perform inference on every individual frame. While this per-frame paradigm is straightforward to implement, it exhibits two major limitations. First, it relies solely on spatial information within isolated frames, neglecting temporal cues like motion video sequences. Second, in surveillance scenarios, particularly in **indoor** environments, video streams can exhibit minimal visual variation across consecutive frames due to static scene content. Indiscriminately applying computationally intensive DL models to every frame under these conditions introduces significant processing redundancy, increasing latency and computational overhead without improving detection performance. This inefficiency is further amplified by the well-known accuracy-efficiency trade-off in DL: high-accuracy models typically demand substantial computational resources, and indiscriminate frame processing amplifies resource demands, and indiscriminate per-frame inference magnifies these demands, posing a significant barrier to real-time deployment.

<!-- !What we propose -->

A common strategy for reducing inference cost is to substitute a high-complexity
model with a lightweight alternative; however, this approach typically
compromises detection reliability ­ an unacceptable trade-off in safety-critical
applications such as fire and smoke surveillance. This study adopts a
complementary approach: rather than replacing the classifier, we propose an
efficient inference acceleration framework that introduces a lightweight,
motion-aware skip module acting as a computational gate upstream of the
classifier. The skip module leverages frame differencing-based motion estimation
to selectively bypass DL inference on static, non-informative frames, while
forwarding only frames with significant scene activity to the high-capacity
classifier. To address a key failure mode of motion-based skipping — namely,
missed detections arising from slow-developing or nearly stationary smoke events
— the framework additionally incorporates an Eager mode that periodically forces
inference updates to recover detection recall. As illustrated in Fig. 1, the
conventional pipeline processes every frame through the classifier, whereas the
proposed pipeline adds the skip module early in the process to conditionally
suppress redundant inference calls.

<!-- !Main Contributions -->

In particular, our main contributions are summarized as follows:

- **Motion-Aware Skip Module:** A lightweight skip module is proposed that
  exploits frame differencing-based motion estimation to identify and bypass DL
  inference on static frames, thereby reducing computational cost without
  modifying the underlying classifier. The proposed module is designed as a
  plug-and-play component compatible with existing fire and smoke detection
  pipelines. A systematic hyperparameter optimization procedure is further
  developed to identify the optimal operating configuration for maximizing
  throughput while preserving detection performance.

- **Indoor Fire and Smoke Video Dataset:** We construct an annotated dataset
  comprising 150 indoor fire and smoke videos (including fire, smoke, and
  background-only classes) captured by static surveillance cameras at
  resolutions ranging from $814 \times 720$ to $3840 \times 2160$. The dataset
  covers diverse environments such as warehouses, parking areas, and offices
  under varying lighting conditions, providing a realistic benchmark for
  evaluating detection systems in static surveillance scenarios.

- **Comprehensive System Evaluation and Analysis:** The proposed framework is
  integrated with a high-capacity DL classifier (BIG model) and evaluated on the
  constructed dataset against a per-frame inference baseline and existing
  methods. Results demonstrate approximately 30% higher throughput than the
  baseline, with a recall reduction of roughly 0.05% when the Eager mode is
  active — compared to a 1.23% recall drop when the skip module is used alone —
  highlighting the effectiveness of the Eager mode in recovering detection
  performance while preserving most throughput gains. Beyond quantitative
  benchmarking, a detailed efficiency analysis of the proposed system is
  conducted, and success and failure cases.
<!-- !Paper Organization -->

The rest of this paper is organized as follows: [@sec:relatedWork] reviews
existing fire and smoke detection methods for video and relevant motion
detection techniques, contextualizing the need for efficient processing in
static surveillance scenarios. [@sec:method] describes the proposed system
architecture, the design of the skip-module, its integration with the
classifier, the hyperparameter optimization strategy, and the Eager mode
proposed to address missed detections arising from slow-developing or nearly
stationary smoke events. [@sec:results] presents the experimental setup,
evaluation metrics, and results on the large-scale indoor video dataset,
including comparative analyses against the baseline and an existing
temporally-aware method, and qualitative examination of success and failure
cases of the proposed skip module. Finally, [@sec:conclusion] summarizes the key
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

However, processing frames independently discards temporal information
inherent in video data that is potentially informative for detection.
[@khan2025beyond] survey video-based fire and smoke detection approaches and
highlight the benefit of modeling temporal dynamics. Representative methods
include the work of [@cao2019attention], who combine a CNN for spatial feature
extraction with a bidirectional LSTM for temporal modeling, and
[@ali2025toward], who employ 3D CNNs with attention mechanisms to jointly
capture spatio-temporal patterns. While these approaches demonstrate improved
detection performance over purely image-based methods, they introduce greater
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
which fire or smoke is generally unlikely. Motivated by this observation, we
propose a skip module that filters such motion-free frames prior to DL
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
initialization latency. As the skip module is designed primarily as a fast,
lightweight gate, we instead adopt motion estimation based on frame
differencing, which offers minimal computational overhead and requires no
background model initialization.

Existing efficient video inference methods, such as FrameExit
[@ghodrati2021frameexit], employ learned gating mechanisms to selectively
process frames based on prediction confidence, thereby reducing redundant
inference; however, they require joint training of the gating mechanism and the
underlying classifier, limiting plug-and-play applicability to pre-trained
models. In contrast, this study adopts a training-free approach by integrating
lightweight frame-differencing-based motion estimation as a plug-and-play gate,
requiring no retraining of the underlying classifier, as described in Section
\ref{sec:method}.

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
fire/smoke or not. The proposed framework augments the conventional frame-wise
inference pipeline with a lightweight skip module $\mathcal{S}$, which serves as
a gating mechanism that selectively suppresses unnecessary DL inference on
frames with low likelihood of containing fire or smoke — most commonly, static
background frames — thereby reducing redundant computation and improving overall
throughput. Formally, let $f_t$ denote the video frame at time step $t$,
$\mathcal{M}$ the DL classifier, and $\hat{y}_t \in \{0, 1\}$ the predicted label
for $f_t$.

**Baseline system.** In this baseline, every frame $f_t$ is unconditionally
forwarded to $\mathcal{M}$, which outputs a binary prediction $\hat{y}_t \in
\{0, 1\}$ indicating the presence or absence of fire or smoke, as follows:

$$\hat{y}_t = \mathcal{M}(f_t)$$

The classifier operates at the full frame rate without any pre-filtering,
processing each frame independently of scene content. This per-frame inference
pipeline serves as the performance reference when evaluating the proposed skip
module.

**Proposed system.** The skip module $\mathcal{S}$ first evaluates $f_t$ based on
inter-frame motion cues and outputs a binary gate decision $s_t$. While this
work instantiates $\mathcal{S}$ solely motion-based cues, the formulation is
general and can incorporate other types of cues such as color, texture, or
learned features. The resulting system output is given by:
$$
\hat{y}_t = \begin{cases} \mathcal{M}(f_t) & \text{if } s_t = 1 \quad
\text{(run inference)} \\ 0 & \text{if } s_t = 0 \quad \text{(skip, label as
negative)} \end{cases}
$$

In this work, skip module $\mathcal{S}$ derives $s_t$ from the motion estimated
between $f_t$ and the previous frame $f_{t-1}$. Frames with little or no motion
are unlikely to contain fire or smoke, and are therefore skipped without
invoking the classifier $\mathcal{M}$. The classifier $\mathcal{M}$ is typically a
high-capacity and, accurate classifier but it is computationally expensive. In
contrast, $\mathcal{S}$ is deliberately designed to be computationally lightweight,
introducing negligible overhead. The objective of $\mathcal{S}$ is to skip
as many non-event frames as possible while minimizing missed
fire and smoke detections, thereby improving overall system throughput (FPS) while
maintaining high detection accuracy. The architecture and operation of skip module $\mathcal{S}$ are described in the following subsection.

## Skip Module Design

```{=latex}
\input{./3.fig/fig_skipmodule.tex}
```

```{=latex}
\input{6.algo/skip_module.tex}
```
The proposed skip module $\mathcal{S}$ is designed to suppress redundant DL
inference by selectively identifying informative frames in the continuous video
stream. As illustrated in Figure \ref{fig:skipmodule}, each input frame $f_t$
is first downscaled by a factor of $\alpha$ and padded to ensure its dimensions
are divisible by the effective block size $B_s = B \cdot \alpha$, where $B$ is
the block size defined on the original frame. The motion detector $\mathcal{D}$
then computes a binary foreground mask from consecutive frames, which is
partitioned into non-overlapping blocks of size $B_s \times B_s$. If the
fraction of active pixels in any block exceeds the block active threshold
$\tau$, the skip module sets $s_t = 1$ and forwards $f_t$ to the classifier
$\mathcal{M}$; otherwise, the frame is skipped ($s_t = 0$). The full procedure
is formalized in Algorithm \ref{alg:skipmodule}.

As illustrated in Figure \ref{fig:skipmodule} and formalized in
Algorithm \ref{alg:skipmodule}, the skip module $\mathcal{S}$ can leverage any
available information — such as motion, color, or texture — to compute the skip
decision $s_t$. In this work, we focus on motion information derived from
consecutive frames as the primary cue. Specifically, we design and evaluate two
motion-based instantiations of $\mathcal{D}$: (1) \textit{FrameDiffDet} — a
lightweight method that estimates motion via direct inter-frame differencing, and
(2) \textit{AccMotionDet} — an extension that accumulates motion evidence over
consecutive frames for a more robust skip decision. Both approaches are selected
for their simplicity and low computational cost, making them well-suited for
real-time deployment. The following subsections describe each method in detail,
beginning with the simpler \textsc{FrameDiffDet} before introducing the accumulation-based
\textsc{AccMotionDet}.

### FrameDiffDet — Naive Motion Detection

```{=latex}
\input{6.algo/frame_diff_det.tex}
```

### AccMotionDet — Motion Detection with Accumulation

Inspired by [@yu2013real], we design \textsc{AccMotionDet} to address the
key limitation of \textsc{FrameDiffDet}: its reliance on a single inter-frame
difference makes it sensitive to noise and single-frame flicker, which may
trigger unnecessary inference calls. Instead, \textsc{AccMotionDet} accumulates
motion evidence over multiple consecutive frames, providing a more robust signal
for motion estimation and enabling more accurate skip decisions.

Let $F_t$ and $F_{\text{prev}}$ denote the grayscale frames converted from the
downsampled frames $\tilde{f}_t$ and $\tilde{f}_{\text{prev}}$, respectively. Let $\mathbf{K}_{t-1} \in [0, K_{\max}]$ denote the per-pixel
\textit{accumulated motion mask} from the previous frame. The detector takes two
consecutive downsampled frames and $\mathbf{K}_{t-1}$ as input, and produces a
binary foreground mask $\mathbf{M}_t \in \{0, 255\}$ and the updated mask
$\mathbf{K}_t$ as output. $\mathbf{K}_t$ is incremented by $\omega$ when the
inter-frame pixel difference exceeds $\tau_d$, and decays by $\delta$ each
frame, bounded at $K_{\max}$ to prevent unbounded growth under persistent
motion. A pixel is marked active (255) only when $\mathbf{K}_t \geq \tau_m$,
suppressing transient noise and single-frame flicker, and inactive (0)
otherwise. The procedure is formalized in Algorithm \ref{alg:acc_motion_det}.


```{=latex}
\input{6.algo/acc_motion_det.tex}
```

## Eager Mode Design {#sec:eagerMode}

```{=latex}
\input{./3.fig/fig_eager_mode.tex}
```

A preliminary analysis of the self-constructed indoor surveillance video dataset
revealed that slow-developing or nearly stationary smoke events constitute a key
challenge for the skip module. In such cases, the subtle inter-frame motion
associated with these events may fall below the activation threshold of the
motion detector, causing the skip module to incorrectly bypass frames containing
actual fire or smoke activity. To mitigate this limitation, the proposed
framework augments the skip module with an \textsc{Eager} mode that periodically forces
inference updates.

When the classifier $\mathcal{M}$ produces positive fire/ smoke predictions on
$W_{\text{fire}}$ consecutive non-skipped frames, the system transition from Normal
mode to \textsc{Eager} mode and suppresses all skip decisions by setting $s_t = 1$,
forcing inference on every frame. The system remains in \textsc{Eager} mode until the
scene is declared safe again, which occurs when the classifier $\mathcal{M}$ returns
negative predictions for $W_{\text{clr}}$ consecutive frames. At that point, the system
returns to \textsc{Normal} mode with the skip module re-enabled.

Additionally, while operating in Normal mode, the system periodically forces
inference on $W_{\text{fire}}$ consecutive frames after every $N_{\text{chk}}$
skipped frames. This mechanism ensures the classifier retains opportunities to
detect emerging fire or smoke activity and transition to \textsc{Eager} mode when
necessary. Consequently, the proposed design improves recall for slow-developing
or nearly stationary smoke events while preserving the throughput gains provided
by the skip module during \textsc{Normal} mode operation.

Figure \ref{fig:eager_mode} illustrates the state machine of \textsc{Eager} mode
with the transition conditions and corresponding actions for each state and how
states transition during system operation.

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
for real-life deployment. For fine-tuning, a dataset of 253,325
fire/smoke/normal images was compiled from internet sources and partitioned into
training (90%) and validation (10%) subsets. The model was trained for 100
epochs with a batch size of 64 using the Adam optimizer [@kingma2014adam],
configured with a learning rate of 0.003, weight decay of 0.05, and a warmup
over the first 5 epochs. The training was conducted on a single NVIDIA RTX 5090
GPU (32 GB VRAM) paired with an Intel Core Ultra 9 285K processor and 128 GB of
system RAM, running PyTorch 2.7.1 with CUDA Toolkit 12.8. All input images were
preprocessed by resizing to $3 \times 360 \times 640$ $(C \times H \times W)$. The
fine-tuned model achieved a classification accuracy of 98.34% and a high recall
of 98.12% on the validation set.

**Implementation Details**: Unless stated otherwise, all experiments were
conducted on a workstation equipped with an Intel Core i9-10900K CPU, 64 GB DDR4
system memory, and an NVIDIA GeForce RTX 3090 GPU (24 GB VRAM), running Windows
10 Pro (22H2, build 19045). Deep learning inference was performed using PyTorch
2.7.1 under CUDA 12.9, and video processing and motion estimation were carried
out using OpenCV 4.11 (CPU-only).

## Results

### Hyperparameter Optimization Results
In this work, the recall tolerance $\delta_R$ is set to $\delta_R = 0.015$, permitting a
maximum absolute recall drop of $1.5\%$ relative to the baseline. At the
baseline recall of approximately $95\%$, this ensures the worst-case deployed
system retains a recall of at least $93.5\%$, while affording the hyperparameter
search sufficient flexibility to identify configurations with meaningful
efficiency gains. The optimization weights are set to $w_S = 0.7$ and $w_R =
0.3$, prioritizing skip rate (the primary determinant of throughput gain) while
using recall retention as a secondary criterion to favor configurations closer
to baseline performance among feasible candidates.


<!-- **Frame Diff Parameter Grid Search:** -->

#### Hyperparameter Optimization Results for the FrameDiffDet-based Skip Module {#sec:frameDiffResults label="frameDiffResults"}

To identify the optimal configuration for the FrameDiffDet-based skip module,
we performed a systematic grid search over four hyperparameters: scale factor
$\alpha$, block size $B$, block active threshold $\tau$, and difference
sensitivity threshold $\tau_d$, yielding a total of
$2 \times 2 \times 3 \times 4 = 48$ candidate configurations. The search
space is as follows:

+ **Scale factor** $\alpha \in \{0.5, 1.0\}$: controls the downsampling ratio
  applied to input frames prior to motion computation. A smaller $\alpha$
  reduces computational cost but may lose subtle motion detail. For simplicity,
  we evaluate full resolution ($\alpha = 1.0$) and half resolution ($\alpha =
  0.5$).

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
($22.02$--$23.32$) falls below the no-skip baseline ($24.66$), indicating that
the skip module overhead outweighs the benefit of skipped inference calls.
Conversely, configurations that do achieve substantial skip rates (reaching
$47$--$60\%$ at higher $\tau_d$ values of 7 and 10) suffer severe recall
degradation of $5.5$--$10\%$, far exceeding the safety tolerance.

These results demonstrate that \textsc{FrameDiffDet}, without noise suppression
or robust sustained-motion confirmation, is unsuitable as a skip gate: it
either skips too few frames to yield any efficiency gain, or skips too
aggressively and suffer unacceptable recall loss. This motivates the development
of \textsc{AccMotionDet}, which accumulates motion evidence over multiple frames
to provide a more reliable signal for skip decisions, as described in the
following section.

#### Hyperparameter Optimization Results for the AccMotionDet-based Skip Module {#sec:accMotionResults label="accMotionResults"}

```{=latex}
\input{./4.table/tb_val_results_accMotionDet.tex}
```

The hyperparameter search for \textsc{AccMotionDet} is designed to mirror the
FrameDiffDet study while accounting for the temporal accumulation of motion
scores over consecutive frames. Details of the search space (of 96
configurations) are as follows:

+ **Scale factor ($\alpha \in \{0.5, 1.0\}$)** and **Block size ($B \in \{16,
  32\}$)**: Retained from FrameDiffDet to maintain consistency in the spatial
  resolution and grid density trade-offs.

+ **Block active threshold ($\tau \in \{0.05, 0.10\}$)**: Restricted to a
  narrower range than in FrameDiffDet, as temporal accumulation naturally
  suppresses false activations, rendering more aggressive thresholds
  unnecessary.

+ **Difference sensitivity threshold ($\tau_d \in \{3, 5\}$)**: Limited to the lower end of the
  FrameDiffDet search space, as higher values were found to cause excessive
  frame skipping and unacceptable recall degradation, as described in [@sec:frameDiffResults].

+ **Accumulation step ($\omega=5$)** and **Decay rate ($\delta=1$)**: We fix
these parameters to streamline the hyperparameter search, allowing us to focus
on the primary sensitivity controls: the activation threshold ($\tau_m$) and
accumulation cap ($K_{\max}$). Since the optimal tuning of $\tau_m$ and
$K_{\max}$ depends directly on by $\omega$, keeping $\omega$ constant provides a
stable baseline for optimization. Furthermore, setting $\delta=1$ ensures
consistent, linear decay behavior while reducing the dimensionality of the
search space.

+ **Activation threshold ($\tau_m \in \{5, 10\}$)**: This threshold serves as
  the decision gate for the motion mask, labeling pixels as active only when
  their accumulated motion score exceeds $\tau_m$. Given the fixed accumulation
  parameters ($\omega=5, \delta=1$), $\tau_m=5$ and $\tau_m=10$ are
  approximately equivalent to requiring motion signals over 2 and 3 consecutive
  frames, respectively.

+ **Accumulation cap ($K_{\max} \in \{15, 25, 35\}$)**: This parameter limits
  the maximum accumulated motion score to prevent runaway accumulation during
  prolonged activity. Without this cap, prolonged or intense motion could drive
the accumulated score to arbitrarily large values, such that the subsequent
temporal decay requires excessive time to clear the score once motion ceases. By
selecting a range of values, we diversify the system's saturation behavior,
allowing us to tune how quickly the module resets following high-activity
periods.

Table \ref{tb:val_results_accMotionDet} presents representative configurations
of the \textsc{AccMotionDet}-based skip module evaluated on the validation set.
The results show a marked improvement over \textsc{FrameDiffDet}, with several
configurations achieving skip rates of 39%--56% while maintaining high recall
(above 93%, relative to the baseline of 94.347%), remaining within the safety
constraint $\delta_R = 0.015$. The best-ranked configuration ($\alpha{=}0.5,\
B{=}32,\ \tau{=}0.05,\ \tau_d{=}5,\ \tau_m{=}5,\
K_{\max}{=}15$) achieves a skip rate of 44.39% with a recall drop of only 0.452%,
yielding a throughput of 36.32 FPS --- a substantial improvement over the 24.66
FPS baseline.

### Frame-Level Performance and Throughput: Baselines and Skip Module {#sec:e2e-perf}

```{=latex}
\input{./4.table/tb_perf_per_frame.tex}
```
Table \ref{tb:perf_per_frame} benchmarks the proposed pipeline
(\textsc{AccMotionDet} skip module with the optimal configuration paired with
the Big Model) against
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
to 32.32 FPS, a gain of approximately 29%, while the F1-score is almost fully preserved
at 0.969. Recall drops by only 1.23 percentage points (95.616% $\to$ 94.384%),
remaining within the safety constraint $\delta_R = 0.015$. A less obvious but
notable benefit is the reduction in FAR from 0.251% to 0.136%, accompanied by a
precision increase from 99.166% to 99.541%: by suppressing inference on
near-static background frames, the skip module also eliminates a class of
incorrect detections that the BIG model would otherwise generate.

### Comparison with Temporal Post-Processing {#sec:cmp-temporal}

Temporal Persistence Thresholding (TPT) [@de2023hybrid] augments the standard
per-frame inference pipeline with a lightweight post-processing stage designed
to suppress isolated false alarms. Unlike the proposed skip module, TPT does
not reduce the number of frames forwarded to the classifier --- every frame is
still processed by the classification model. Instead, a circular boolean buffer of
length $W$ records the raw per-frame predictions over a rolling window. A
detection is confirmed only if the fraction of positive predictions within the
buffer exceeds a persistence threshold $\tau_\text{persist}$; otherwise, the
prediction is suppressed and the frame is relabelled as negative.

```{=latex}
\input{./4.table/tb_val_results_Tpt.tex}
```
```{=latex}
\input{./4.table/tb_cmp_other_temp.tex}
```

We follow the hyperparameter selection protocol of [@de2023hybrid], evaluating
TPT on the validation set $\mathcal{D}_\text{val}$. Two representative
configurations are reported: a strict variant ($\text{TPT}_\text{strict}$: $W =
5,\ \tau_\text{persist} = 0.2$) and a balanced variant
($\text{TPT}_\text{balanced}$: $W = 10,\
\tau_\text{persist} = 0.5$), with the full grid search results on the validation
set shown in Table \ref{tb:gridsearch_tpt}. Both configurations are then
compared against the proposed pipeline on test set in Table
\ref{tb:cmp_other_temp}.

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
(95.616% $\to$ 94.384%) and remains within the safety constraint
$\delta_R = 0.015$.

### AccMotionDet Skip Module with Eager Mode

```{=latex}
\input{./4.table/tb_gridsearch_accMotionDetEager.tex}
```

As established in Section \ref{sec:e2e-perf}, the AccMotionDet-based skip module
achieved a throughput improvement of nearly 30% in FPS; however, this came at
the cost of a 1.23% reduction in recall, which is likely attributable to missed
detections in slow-developing or nearly stationary smoke scenarios. To address
this limitation, the Eager mode is introduced as a complementary mechanism to
augment the skip module and recover recall in such cases. In this section, the
Eager mode is evaluated in conjunction with the AccMotionDet-based skip module,
with its hyperparameters fixed to the optimal values identified in Section
\ref{sec:accMotionResults} (i.e.,
$\alpha{=}0.5,B{=}32,\tau{=}0.05,\tau_d{=}5,\tau_m{=}5,K_{\max}{=}15$). A grid
search is then conducted over the Eager mode parameters to identify the optimal
configuration, as follows:

+ **Confirmation window $W_{\mathrm{fire}} \in \{1, 2, 3\}$**: defines the
  number of consecutive positive predictions required to trigger a transition
  into \textsc{Eager} mode. A deliberately small range is chosen to ensure the
  system reacts swiftly to early signs of fire or smoke.

+ **Clearance window $W_{\mathrm{clr}} \in \{3, 5, 7\}$**: This parameter
  defines the number of consecutive negative predictions required to exit eager
  mode and resume normal skip operation. The range is deliberately set larger than that of $W_{\mathrm{fire}}$, reflecting an asymmetric cost structure: a premature exit risks missed detections, whereas a delayed exit only causes the system to run unnecessary inference on a few extra frames — a minor and acceptable efficiency loss.

- **Forced-check interval $N_{\mathrm{chk}} \in \{10, 20, 30, 50\}$**: defines
  the maximum number of consecutive frames that are skipped in \textsc{Normal}
  mode before the system forces an inference call, irrespective of the skip module
  decision. This serves as a safety net against false negatives --- for instance,
  slowly developing or settled smoke may produce insufficient inter-frame motion to
  trigger $\mathcal{S}$, causing the scene to be treated as static and inference to
  be suppressed indefinitely. If the forced inference yields a positive prediction
  and the $W_{\mathrm{fire}}$ criterion is satisfied, the system transitions into
  \textsc{Eager} mode. Smaller values increase check frequency at the cost of
  redundant inference on truly static scenes, whereas larger values reduce this
  overhead at the expense of a longer worst-case recovery delay. The selected range
  provides a systematic sweep from aggressive ($N_{\mathrm{chk}} = 10$) to
  infrequent ($N_{\mathrm{chk}} = 50$) checking.

Following the hyperparameter selection protocol described in
Section \ref{sec:hyperparam}, the optimal \textsc{Eager} mode configuration
is identified on the validation set $\mathcal{D}_{\mathrm{val}}$, with
results reported in Table \ref{tb:gridsearch_accMotionDetEager}.


The best-ranked configuration ($W_{\text{fire}} = 1, W_{\text{clr}} = 7, N_{\text{check}} = 50$) is then evaluated on the test set $\mathcal{D}_{\text{test}}$
and compared against the skip-only configuration in Table \ref{tb:cmp_other_temp_eager}.

```{=latex}
\input{./4.table/tb_cmp_other_temp_eager.tex}
```
Table \ref{tb:cmp_other_temp_eager} reports the effect of \textsc{Eager} mode
on system performance. \textsc{Eager} mode successfully recovers recall on
slow-developing or settled smoke scenarios, increasing from $94.384\%$
(\textsc{AccMotionDet} only) to $95.565\%$ (\textsc{AccMotionDet} +
\textsc{Eager}), nearly matching the baseline recall of $95.616\%$ (Big Model
without skip module) --- a recovery of critical importance in safety-critical
applications. However, this improvement comes with two costs: the skip rate slightly
decreases from $34.93\%$ to $34.31\%$, and the false alarm rate (FAR) increases
from $0.136\%$ to $0.251\%$ (the implications of this FAR increase are
discussed in detail in Section \ref{sec:analysis_eager}).

Despite the reduced skip rate, overall throughput marginally increases from
$32.32$ to $32.54$ FPS. This is explained by the reduced computational overhead
in \textsc{Eager} mode: rather than executing the full \textsc{AccMotionDet}
pipeline (Algorithm \ref{alg:acc_motion_det}), our implementation converts each
frame to grayscale, stores it for subsequent use in \textsc{Normal} mode, and
applies only temporal decay. Consequently, per-frame computation in
\textsc{Eager} mode is lighter than in \textsc{Normal} mode (where both the
full-step \textsc{AccMotionDet} skip module and Big Model inference are active),
yielding the observed FPS gain despite the lower skip rate.

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

To understand how the skip module operates in coordination with the Big Model, we visualize the inference results of the \textsc{AccMotionDet} skip module on the test set $\mathcal{D}_\text{test}$ in RGB images together with the corresponding foreground motion masks generated by the skip module. The qualitative examples are organized into two major categories: success cases and failure cases.

**Success Cases**

```{=latex}
\input{./3.fig/fig_quality_success.tex}
```

Figure \ref{fig:qualitative_success} shows four representative success cases in which the skip module $\mathcal{S}$ operates as intended. These cases can be grouped into two categories:

- **Correctly skipped static background frames** (Fig. \ref{fig:qualitative_success}a,b): \textsc{AccMotionDet} identifies static background frames and skips them without invoking the Big Model. The corresponding motion masks are empty, indicating no active motion regions.

- **Correctly inferred fire/smoke frames** (Fig. \ref{fig:qualitative_success}c,d): The module correctly detects active fire and smoke regions and forwards these frames for inference. The yellow boxes indicate motion-active blocks detected by \textsc{AccMotionDet}, which are concentrated in the fire/smoke regions, consistent with the continuous nature of these phenomena.

**Failure Cases**

```{=latex}
\input{./3.fig/fig_quality_failure.tex}
```

Figure \ref{fig:qualitative_failure} shows five representative failure cases, which can be grouped into three categories:

- **Wrongly skipped frames** (Fig. \ref{fig:qualitative_failure}a): Slowly developing smoke or smoke in its settling phase produces motion that is too subtle to trigger the skip module, leading to missed detections.

- **Wrongly passed frames / wasted inference** (Fig. \ref{fig:qualitative_failure}b,c): Noise or persistent non-fire motion prevents background frames from being skipped, resulting in unnecessary inference.

- **Incorrect Big Model predictions** (Fig. \ref{fig:qualitative_failure}d,e): Even when fire or smoke frames are correctly forwarded, the Big Model may still produce an incorrect label, especially when the target occupies only a small portion of the frame.


<!-- !END_SYNC_BLOCK -->
#### Analysis of Eager Mode: Recall-FAR Trade-off {#sec:analysis_eager}

```{=latex}
\input{./3.fig/fig_eager_analysis.tex}
```

A notable drawback of Eager mode shown in Table \ref{tb:cmp_other_temp_eager} is its tendency to increase false alarms (from 0.136% to 0.251%). Our analysis reveals that Eager mode cannot distinguish false-alarm-prone static scenes from genuine slow-developing or settled smoke events, as both exhibit near-zero inter-frame motion. In such static scenes, the classifier may produce fire/smoke predictions on several consecutive frames, causing the system to enter and remain in Eager mode with prolonged forced inference and elevated FAR. This highlights a fundamental limitation of heuristic, motion-based skip logic: it is insufficient to robustly handle all scenarios. We leave this as an open research question, with potential directions including skip modules that incorporate richer cues (such as color, texture, or learned features) to discriminate between these scene types.


# Conclusion {#sec:conclusion label="Conclusion"}

<!-- !START_SYNC_BLOCK -->
<!-- TARGET_PROJECT: G:\My Drive\1_PhD\Obsidian\Home\3. Writing\paperfire2 -->
<!-- SYNC_TARGET_FILE: 04_Conclusion.md-->
<!-- BLOCK_ID: conclusion -->

We propose a lightweight, plug-and-play skip module $\mathcal{S}$ designed to
accelerate real-time fire and smoke detection in indoor surveillance scenarios.
Operating at a negligible fraction (${\approx}4.5\%$) of the computational cost
of the DL classifier $\mathcal{M}$, the skip module $\mathcal{S}$ efficiently
filters out static background frames before they reach $\mathcal{M}$, reducing
unnecessary inference overhead. Experiments on a real-world video dataset
demonstrate that the proposed approach achieves approximately $30\%$ throughput
improvement while sustaining high detection recall, with a recall drop of only
$1.23\%$ relative to the baseline. When augmented with \textsc{Eager} mode, the
recall drop is further reduced to $0.05\%$, nearly fully recovering baseline
performance.
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
