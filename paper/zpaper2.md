---
# ---------------- START ABSTRACT SYNC BLOCK ----------------#
# <!-- !START_SYNC_BLOCK -->
<!-- TARGET_PROJECT: G:\My Drive\1_PhD\Obsidian\Home\3. Writing\paperfire2 -->
<!-- SYNC_TARGET_FILE: 00_Meta_Abstract.md-->
<!-- BLOCK_ID: abstract -->
date: 2026.04.16
title: "Efficient Real-Time Fire Surveillance: A Lightweight Motion-Heuristic
Skip Module for Accelerated Inference"
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
Specifically, the skip module enable the system a 30% increase in frames per second (FPS)
with a minimal reduction in recall of only 1% relative to the baseline (recall ≥ 95%),
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
quickly escalate --- especially under dry and windy conditions --- resulting in
catastrophic consequences including loss of life, destruction of property, and
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
information within individual frames, neglecting temporal cues --- such as
motion --- inherent in video data. Second, in surveillance scenarios,
particularly in indoor environments, consecutive frames often exhibit minimal
variation due to static scene content. Applying a DL model indiscriminately to
every frame under such conditions is computationally redundant, increasing
processing cost and latency without contributing to detection performance. This
inefficiency is further compounded by the well-known accuracy--efficiency
trade-off in DL: more accurate models are generally more computationally
intensive, and redundant frame processing amplifies these demands, rendering
real-time performance difficult to achieve.

<!-- !What we propose -->

A common strategy to reduce inference cost is to substitute a heavy,
high-accuracy DL model with a lighter alternative; however, this typically
degrades detection reliability --- an unacceptable compromise in safety-critical
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
  DL inference on non-informative frames, reducing computational cost without
  modifying the underlying classifier. The module is designed as a plug-and-play
  component compatible with existing fire and smoke detection pipelines. To
  identify the optimal operating configuration, we further develop a systematic
  hyperparameter optimization procedure that maximizes throughput while
  preserving detection performance.

- **Indoor Fire and Smoke Video Dataset:** We construct an annotated dataset
  comprising 150 indoor fire and smoke videos --- including fire, smoke, and
  background-only classes --- captured by static surveillance cameras at
  resolutions from 720p to 1080p. The dataset covers diverse environments such
  as warehouses, parking areas, and offices under varying lighting conditions,
  providing a realistic benchmark for evaluating detection systems in static
  surveillance scenarios.

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
emerging as the dominant paradigm. [@cheng2024visual] provide a comprehensive
survey of visual fire detection methods, highlighting the rapid adoption of
DL-based approaches in this domain. The majority of these methods formulate the
problem as image-level binary classification (fire/smoke vs. none) or object
detection, in which a model receives a single RGB image and outputs either a
class label with confidence score or localized bounding boxes [@geng2024yolofm;
@khan2025optimized]. Research efforts have primarily focused on improving model
accuracy and inference speed through architectural modifications and advanced
training strategies. In video-based deployments, these image classifiers are
applied directly to each frame, enabling integration with existing pipelines
without architectural changes.

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
given the frame content. In static surveillance scenarios, particularly indoors,
video streams frequently consist of purely background frames with no motion, in
which fire or smoke is a priori unlikely. Motivated by this observation, we
propose a skip-module that filters such low-activity frames prior to DL
inference, reducing computational cost while preserving detection reliability.
The module prioritizes retaining positive frames (fire/smoke present) while
skipping negative frames (background only), functioning as a complementary
accelerator for any existing frame-wise detection pipeline.

**Motion Detection and The proposed Skip Module**: The skip decision in our
module relies on motion estimation, which connects to the well-studied problem
of background subtraction. [@sobral2014comprehensive] and
[@garcia2020background] provide comprehensive reviews of background subtraction
methods, ranging from simple statistical models to deep learning-based
approaches. More robust methods such as MOG2 and KNN [@zivkovic2006efficient]
model the scene background statistically and are resilient to gradual
illumination changes; however, they introduce non-trivial memory overhead and
initialization latency. As the skip-module is designed primarily as a fast,
lightweight gate, we instead adopt frame differencing --- computing the absolute
per-pixel intensity change between consecutive frames --- which offers minimal
computational overhead and requires no background model initialization.

While classical motion detection provides an efficient means of identifying
scene activity, it has not been systematically exploited to gate DL inference in
fire and smoke detection pipelines. Existing efficient video inference methods,
such as FrameExit [@ghodrati2021frameexit], address per-frame redundancy but
require joint training of the gating mechanism and the underlying classifier,
limiting plug-and-play applicability to pre-trained models. This study bridges
this gap by integrating lightweight frame-differencing-based motion estimation
as a training-free gate for a high-capacity classifier, as described in Section
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
fire/smoke or not. The core idea is to extend the conventional pipeline with a
lightweight **skip module** $\mathcal{S}$, which acts as a gating mechanism to
suppress unnecessary inferences on non-informative frames, thereby improving
computational efficiency. Formally, let $f_t$ denote the video frame at time
step $t$, $\mathcal{M}$ the DL classifier, and $\hat{y}_t \in \{0, 1\}$ the
predicted label for $f_t$.

**Baseline system.** Each frame is directly passed to the classifier:

$$\hat{y}_t = \mathcal{M}(f_t)$$

**Proposed system.** The skip module $\mathcal{S}$ first evaluates $f_t$ using
inter-frame motion cues or other information and outputs a binary gate decision
$s_t$. The full system output becomes:

$$\hat{y}_t = \begin{cases} \mathcal{M}(f_t) & \text{if } s_t = 1 \quad
\text{(run inference)} \\ 0 & \text{if } s_t = 0 \quad \text{(skip, label as
negative)} \end{cases}$$

In this work, $\mathcal{S}$ derives $s_t$ from the motion estimated between
$f_t$ and the previous frame $f_{t-1}$. Frames with little or no motion are
unlikely to contain fire or smoke, and are therefore skipped without invoking
$\mathcal{M}$. The model $\mathcal{M}$ is typically a high-capacity, accurate
classifier but is computationally expensive. The skip module $\mathcal{S}$ is
intentionally lightweight, adding negligible overhead. The goal of $\mathcal{S}$
is to skip as many true-negative frames as possible while minimizing the risk of
skipping true-positive frames, thus improving overall system throughput (FPS)
while maintaining high accuracy for fire/smoke detection.


## Skip Module Design

```{=latex}
\input{./3.fig/fig_skipmodule.tex}
```
```{=latex}
\input{6.algo/skip_module.tex}
```

As noted above, $\mathcal{S}$ can leverage any available information — such as
motion, color, or texture — to compute the skip decision $s_t$. In this work, we
focus on motion information derived from consecutive frames as the primary cue.
Specifically, we design and evaluate two motion-based instantiations of
$\mathcal{S}$: (1) a frame differencing method, and (2) a motion accumulation
method. Both approaches are selected for their simplicity and low computational
cost, making them well-suited for real-time deployment. The overall workflow of
$\mathcal{S}$ is illustrated in Figure \ref{fig:skipmodule} and the detailed
algorithms are described in the Algorithm \ref{alg:skipmodule}.

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


Publicly available video datasets for fire and smoke detection using static
cameras remain scarce. While several existing benchmarks address this detection
task — including FireNet [@jadon2019firenet], Firesense [@Firesens4:online],
FiSmo [@cazzolato2017fismo], and FURG [@steffens2015unconstrained] — these were
recorded with moving cameras and are therefore unsuitable for static
surveillance scenarios. Static-camera datasets such as DFire [@dfiredataset],
VisiFire Bilkent [@VisiFireBilkent:online], KMU Fire and Smoke
[@KMUFireSmokeDataset], Mivia Fire and Smoke [@foggia2015real], and USTC Smoke
[@lin2017smoke] do exist; however, they collectively suffer from several
limitations: a predominance of outdoor scenes, a small number of video samples,
low spatial resolution, and insufficient diversity in fire/smoke appearances and
environmental conditions.

To address these deficiencies, we constructed a dedicated static indoor video
dataset by aggregating clips from multiple heterogeneous sources. Fire and smoke
samples were sourced from the Korea AI Fire Dataset [@AIHub87:online], the USTC
Smoke Dataset [@lin2017smoke], and the VSD3K Dataset [@huang2022fire],
supplemented by videos collected from open online platforms (Pexels, Pixabay,
and YouTube). Non-fire/smoke (negative) samples were compiled from self-recorded
footage captured by real CCTV cameras in various indoor environments (e.g.,
parking areas), along with the Safe & Unsafe Behavior in Workplaces dataset
[@onal2024video], the Indoor Action dataset [@deniz2024optimized], the MPII
Cooking 2 Dataset [@rohrbach2016recognizing], and the WiseNet dataset
[@marroquin2019wisenet]. Table \ref{tb:videodb} summarizes the properties of the
self-collected video dataset alongside a comparison with existing benchmark
datasets, and Figure \ref{fig:videodb} presents representative sample frames.

For the hyperparameter optimization described in Section \ref{sec:hyperparam},
the video dataset was further partitioned into a validation set,
$D_{\text{val}}$ (46 videos), used to select the optimal skip-module
hyperparameters, and a test set, $D_{\text{test}}$ (104 videos), used for the
final evaluation of the skip-enabled system against the baseline and existing
methods. The split followed a 30:70 ratio (validation:test), following the
protocol of  [@de2023hybrid], and was performed using stratified sampling to
preserve balanced class distributions in both subsets.

## Evaluation Metrics {#sec:metrics label="metrics"}

Performance is evaluated under both **frame-level** and **video-level**
protocols, which are commonly used in fire and smoke video analysis
[@steffens2016non; @dfiredataset]. Frame-level evaluation measures detection
performance for each individual frame, providing a strict assessment of
classification accuracy. Video-level evaluation aggregates predictions over
entire video sequences, offering a coarser but practically relevant measure for
continuous surveillance scenarios.

The following label definitions apply to both evaluation protocols:

- True Positive ($TP$): Correct detection of fire or smoke.
- True Negative ($TN$): Correct identification of the absence of fire or smoke.
- False Positive ($FP$): Incorrect detection of fire or smoke when none is
  present.
- False Negative ($FN$): Failure to detect fire or smoke when it is present.

Quantitative results are reported using standard classification metrics —
accuracy, recall, false alarm rate ($\mathrm{FAR}$), precision, F1-score, and
frames per second (FPS) — as defined below.

```{=latex}
\input{./5.eq/eq_metrics.tex}
```

## Models and Implementation Details

**Baseline Models**: To demonstrate both the superiority of the high-capacity
classifier over lightweight alternatives and the effectiveness of the proposed
skip module in accelerating its inference, we evaluate the following baseline
models alongside our BIG model:

- **MobileNet** [@mukhopadhyay2019fpga]: A modified MobileNet architecture
  fine-tuned for fire and smoke detection.

- **FireNet** [@jadon2019firenet]: A lightweight CNN-based image classifier
  comprising 14 layers, with a model size of 7.5 MB and approximately 650k
  trainable parameters.

- **YOLOv5s** [@de2023hybrid]: A lightweight object detector based on YOLOv5
  [@JocherYOLOv5byUltralytics2020], trained with hyperparameters selected via
  grid search on the D-Fire dataset [@dfiredataset].

- **YOLOv5l** [@de2023hybrid]: A heavier variant of the above, based on the
  larger YOLOv5l architecture, offering higher capacity at increased
  computational cost.

**Proposed Classifier (BIG Model - HGNetV2)**: The BIG model is obtained by
 fine-tuning a pretrained High-Performance GPU Network v2 (HGNetV2),
  specifically the \texttt{hgnetv2\_b5.ssld\_stage2\_ft\_in1k} checkpoint
  [@hgnetv2timm:online] from the Timm library [@rw2019timm], which was
  originally trained using SSLD knowledge distillation [@cui2021beyond]. HGNetV2
  [@hgnetv2PaddleCl7:online] is a high-capacity CNN architecture designed to
  achieve substantially higher accuracy than models of comparable inference
  speed on NVIDIA GPUs, making it well-suited for deployment in our target
  environment. \textcolor{red}{For fine-tuning, we compiled a dataset of
  1,000,000 fire and smoke images collected from internet sources, partitioned
  into training (80\%) and test (20\%) subsets. The model was optimized using
  Adam with an initial learning rate of $1 \times 10^{-4}$ and weight decay of
  $1 \times 10^{-5}$ for 100 epochs with a batch size of 32. On the held-out
  test set, the final model achieved a recall of xx.xx\% and a false alarm rate
  of xx.xx\%.}. We use input size INPUT_SIZE  = (3, 360, 640)

**Implementation Details**: Unless otherwise specified, all experiments were
conducted on a workstation equipped with an Intel Core i9-12900K CPU, 64 GB DDR5
system memory, and an NVIDIA GeForce RTX 3090 GPU (24 GB VRAM), running Windows
10 Pro (build 19044). Deep learning inference was performed using PyTorch 2.7.1
under CUDA 12.9, and video processing and motion estimation were carried out
using OpenCV 4.11 (CPU-only).

## Results

### Hyperparameter Optimization Results

The recall tolerance is set to $\delta_R = 0.015$, permitting an absolute
recall drop of at most $1.5\%$ relative to the baseline.
This value is chosen to reflect the safety requirement of the application:
at the baseline recall of approximately $95\%$, a tolerance of $1.5\%$
ensures that the worst-case deployed system retains a recall of at least
$93.5\%$ --- a level consistent with operational fire and smoke detection
standards --- while providing the hyperparameter search sufficient
flexibility to identify configurations with meaningful efficiency gains.

In this work, we set
\[
    w_S = 0.70, \qquad w_R = 0.30,
\]
reflecting that skip rate is the primary optimization objective ---
as it directly determines throughput gain --- while recall retention
serves as a secondary criterion that fine-tunes selection among
configurations of comparable efficiency.
The recall hard constraint already screens all unsafe candidates
prior to ranking; $w_R > 0$ ensures that within the feasible set,
configurations closer to baseline recall are preferred as a
conservative tie-breaking rule.


**Frame Diff Parameter Grid Search:**

#### Hyperparameter Search Space for the FrameDiffDet Skip Module

To identify an optimal configuration for the `motion_only_block_skip_proc`
module using `FrameDiffDet` as its motion estimator, we conducted a systematic
grid search over four parameters: `scale_factor`, `block_size_orig`,
`block_ratio_th`, and `diff_thresh`. The search space was defined as follows:

```{=latex}
\input{./4.table/tb_gridsearch_framediff.tex}
```

<!-- ! Add explanation for each hyperparameter space choice -->

To identify an optimal configuration for the FrameDiffDet skip module, we
conducted a systematic grid search over four hyperparameters: scale factor
$\alpha$, block size $B$, block active threshold $\tau$, and diff threshold
$\tau_d$. The search space is summarized in Table \ref{tb:grid_search_space},
yielding a total of $2 \times 2 \times 3 \times 4 = 48$ configurations.

**Scale factor** $\alpha \in \{0.5, 1.0\}$: The scale factor controls the
spatial resolution at which per-block frame differences are computed. Full
resolution ($\alpha = 1.0$) preserves fine-grained pixel detail, while half
resolution ($\alpha = 0.5$) reduces sensitivity to high-frequency pixel noise
that may generate spurious motion signals unrelated to actual scene changes. Two
levels are evaluated to quantify the effect of pre-computation downscaling on
both detection reliability and computational cost.

**Block size** $B \in \{16, 32\}$: Block size $B$ determines the spatial
granularity of the motion map, expressed in pixels of the original unscaled
frame. Fine blocks ($B = 16$ px) enable detection of localized motion from small
or nascent fire regions, whereas coarser blocks ($B = 32$ px) aggregate motion
evidence over a broader spatial context, offering greater robustness against
isolated pixel-level disturbances. This range spans a practically meaningful
fine-to-coarse spectrum without becoming so coarse that spatially small fire
events are missed entirely.

**Block active threshold** $\tau \in \{0.05, 0.10, 0.15\}$: The threshold $\tau$
defines the minimum fraction of motion-active blocks required to trigger a full
inference pass; frames below $\tau$ are skipped. A low value ($\tau = 0.05$)
corresponds to a conservative policy where even sparse motion activity triggers
inference, minimizing the risk of missed detections. A higher value ($\tau =
0.15$) reflects a more aggressive skip policy that demands broader scene-level
motion before committing to inference. The three values are spaced at a uniform
interval of 0.05 to enable a systematic and interpretable sweep across this
conservative-to-aggressive spectrum.

**Diff threshold** $\tau_d \in \{3, 5, 7, 10\}$: The per-pixel difference
threshold $\tau_d$ determines the minimum absolute intensity change required for
a pixel to be counted as a motion event within a block. A low value ($\tau_d =
3$) is highly sensitive and responds to subtle illumination changes, while a
high value ($\tau_d = 10$) responds only to strong, unambiguous motion. The four
values span the full sensitivity spectrum --- from near-noise-level detection to
robust large-motion detection --- with closer spacing at the lower end (3, 5, 7)
to provide finer resolution in the sensitivity range most relevant to fire
detection, where motion tends to be subtle and spatially confined.


Table \ref{tb:val_search} specifies the search space for the rule-based
skip-module parameters. The ranking and selection of the optimal configuration
($\theta^*$) based on the validation set results are detailed in Table
\ref{tb:val_results}. Table \ref{tab:skip-selection} shows an example of the
validation-time ranking. The selected configuration $\theta_1^*$ satisfies the


This formulation is systematic, interpretable, and aligned with the intended
role of the skip module in real-time fire/smoke detection: preserve recall
first, then prefer candidates that skip more negative frames while still
improving operational false alarm behavior.

```{=latex}
\input{./4.table/tb_val_results_frameDiff.tex}
```

Table~\ref{tb:val_results_frameDiff} reveals a fundamental limitation of the
\textsc{FrameDiffDet} skip module: it fails to deliver meaningful
throughput improvement under the safety constraint $\delta_R = 0.015$.
Among all 48 evaluated configurations, only four satisfy the hard recall
constraint, and the best feasible configuration
($\alpha{=}1.0,\ B{=}16,\ \tau{=}0.05,\ \tau_d{=}3$) achieves a skip
rate of merely $0.72\%$ --- meaning the module bypasses fewer than
$1$ in $100$ background frames.
As a direct consequence, the FPS of the top-ranked configuration
($22.4$ FPS) is actually \emph{lower} than the no-skip baseline
($24.1$ FPS), indicating that the skip module introduces measurable
overhead without delivering any compensating efficiency gain.
This behavior stems from the intrinsic sensitivity of naive frame
differencing: ambient illumination fluctuations, sensor noise, and
subtle background variations continuously produce non-zero inter-frame
pixel differences, causing the detector to classify nearly every frame
as motion-active and trigger inference regardless.
Configurations that do achieve substantial skip rates --- reaching
$47$--$60\%$ at higher $\tau_d$ values --- do so only at the cost of
severe recall degradation of $5$--$10\%$, a level entirely
incompatible with fire and smoke safety requirements.
The results collectively demonstrate that \textsc{FrameDiffDet}, without
temporal smoothing or accumulation, lacks the noise robustness required
to operate effectively as a skip gate in real indoor surveillance
conditions.

#### Hyperparameter Search Space for the AccMotionDet Skip Module

```{=latex}
\input{./4.table/tb_gridsearch_accMotionDet.tex}
```

<!-- ! Add explanation for each hyperparameter space choice -->

**Scale factor** $\alpha \in \{0.5, 1.0\}$  Identical rationale to FrameDiffDet:
full resolution preserves fine-grained pixel detail, while half resolution
reduces sensitivity to high-frequency noise. Since AccMotionDet accumulates
differences across multiple frames, downscaling also reduces the memory
footprint of the accumulated motion buffer, making this parameter particularly
relevant for real-time deployment.

**Block size** $B \in \{16, 32\}$: Same motivation as FrameDiffDet. Coarser
blocks (32 px) are relatively more important for AccMotionDet because
accumulation inherently smooths out transient single-pixel disturbances --- so
fine-grained 16 px blocks are less necessary but still evaluated to confirm this
expectation.

**Block active threshold** $\tau \in \{0.05, 0.10\}$: Narrower range than
FrameDiffDet (which goes to 0.15) because AccMotionDet already suppresses
spurious activations through temporal accumulation. A more aggressive threshold
of 0.15 would double-penalize noise that accumulation already handles, so the
upper bound is reduced to 0.10 to focus the search on the practically useful
range.

**Diff threshold** $\tau_d \in \{3, 5\}$: Narrower than FrameDiffDet's $\{3, 5,
7, 10\}$. Because accumulated motion sums intensity differences across $\omega$
consecutive frames, the effective sensitivity is already amplified by the window
length. Higher per-pixel thresholds (7, 10) would be redundant --- accumulation
raises the effective signal level so that lower raw thresholds become
sufficient.

**Motion increment / accumulation step** $\omega \in \{5\}$: Fixed at 5 rather
than searched. A step of 5 frames at standard surveillance frame rates (15--25
FPS) corresponds to a temporal window of 200--333 ms --- short enough to detect
rapid flame onset yet long enough to distinguish sustained motion from
single-frame illumination flicker. Fixing this reduces the search space while
retaining the most physically motivated value.

**Activation threshold** $\tau_m \in \{5, 10\}$: This threshold operates on the
accumulated motion score (sum of per-block differences over $\omega$ frames)
required to trigger inference. A low value (5) triggers on weak but persistent
motion --- appropriate for slowly spreading smoke. A higher value (10) requires
stronger sustained activity, suppressing background flicker. Two values are
sufficient because $\tau_d$ and $\tau$ jointly control sensitivity at the frame
level.

**Accumulation cap** $K_{\max} \in \{15, 25, 35\}$: Caps the maximum accumulated
motion score to prevent runaway accumulation during prolonged high-motion
sequences (e.g., a person walking continuously). Without a cap, sustained motion
would keep $s_t = 1$ indefinitely even after the scene returns to static. Three
values span a low-saturation (15) to high-saturation (35) range, controlling how
quickly the module resets after a high-activity period.

**Decay rate** $\delta \in \{1\}$: Fixed at 1, meaning the accumulation score
decreases by 1 per frame when no motion is detected. A decay of 1 provides
linear cooldown behavior that is easy to interpret and tune. Larger decay values
would reset the accumulator too aggressively, discarding genuine slow-onset
events such as early-stage smoke diffusion. This parameter is fixed to isolate
the effect of the remaining hyperparameters.


```{=latex}
\input{./4.table/tb_val_results_accMotionDet.tex}
```
<!-- ! Analysis the hyperparameters search results of AccMotionDet -->
## Hyperparameter Optimization Results: AccMotionDet

Table \ref{tb:val_results_accMotionDet} shows the top-10 ranked configurations for the AccMotionDet
skip module on the validation set $\mathcal{D}_\text{val}$, ordered by combined score $\Phi$.
The baseline system (no skip module) achieves a recall of 94.35\% at 24.1~FPS.

### Selected Configuration

The optimal configuration (Exp.~1) uses a half-resolution scale ($\alpha = 0.5$), coarse
block size ($B = 32$), conservative block-active threshold ($\tau = 0.05$), moderate pixel
sensitivity ($\tau_d = 5$), low activation threshold ($\tau_m = 5$), and a small accumulation
cap ($K_{\max} = 15$).
This configuration achieves a skip rate of **44.39\%** and a recall of **93.90\%**
($\Delta = -0.45\%$), well within the tolerance $\delta_R = 0.015$.
The resulting FPS improves from 24.1 to **36.6**, a **52\% throughput gain**, with no change
in FPR (0.74\% throughout).

### Hyperparameter Sensitivity Analysis

**Scale factor $\alpha$.**
All top-10 configurations consistently use $\alpha = 0.5$ (half resolution).
Full-resolution processing ($\alpha = 1.0$) does not appear in any feasible high-scoring
configuration, confirming that downscaling reduces sensitivity to high-frequency pixel noise
while lowering skip-module latency --- both effects are beneficial.

**Block size $B$.**
$B = 32$ dominates the top-10 rankings, with only one entry using $B = 16$ (Exp.~10,
score 0.4906).
Coarser blocks aggregate motion evidence spatially, providing greater robustness to isolated
pixel-level disturbances that could otherwise trigger unnecessary inference calls.
The result confirms the prior expectation in Section~\ref{secmethod} that temporal
accumulation already suppresses transient noise, making fine-grained $B = 16$ blocks
less necessary for AccMotionDet.

**Block active threshold $\tau$.**
Configurations with $\tau = 0.05$ yield the highest skip rates (42--44\%), while
$\tau = 0.10$ achieves slightly lower skip rates (39--40\%) but marginally better
recall retention.
The top-ranked configuration (score 0.5203) uses $\tau = 0.05$, confirming that a
conservative trigger policy --- requiring only sparse motion evidence before invoking
inference --- is preferred for maximizing the combined score under the recall constraint.

**Activation threshold $\tau_m$.**
Configurations with $\tau_m = 5$ (Exp.~1--6) consistently outscore those with
$\tau_m = 10$ (Exp.~7--8).
A higher activation threshold $\tau_m = 10$ requires stronger sustained motion before
triggering inference, which slightly improves the skip rate but incurs a larger recall
drop (up to $-0.53\%$), reducing the recall retention term in $\Phi$.

**Accumulation cap $K_{\max}$.**
Across all three values evaluated ($K_{\max} \in \{15, 25, 35\}$), the skip rate
decreases monotonically with increasing $K_{\max}$
(e.g., 44.39\% $\to$ 42.89\% $\to$ 42.38\% for Exp.~1--3).
A lower cap allows the accumulator to reset more readily after high-motion periods,
enabling faster recognition of subsequent static frames and thus higher skip rates.
The difference in combined score across the three values is small ($\leq 0.007$),
indicating low sensitivity to this parameter within the evaluated range.

### Comparison with FrameDiffDet

A fundamental contrast emerges when comparing AccMotionDet and FrameDiffDet on the
validation set (Table @tbl:frameDiff-val).
All feasible FrameDiffDet configurations --- those satisfying $\Delta R \leq \delta_R$
--- achieve skip rates below **1.4\%**, yielding no meaningful FPS improvement over
the baseline (22.4--23.5~FPS vs.\ 24.1~FPS baseline).
This reveals a structural limitation of naive frame differencing: in the static indoor
surveillance setting, the per-pixel sensitivity required to safely detect slow-onset
smoke events forces the threshold $\tau_d$ to remain low (i.e., $\tau_d = 3$), which
in turn flags even subtle illumination changes as motion, suppressing skip decisions
on nearly all frames.
By contrast, the temporal accumulation in AccMotionDet absorbs transient pixel
fluctuations over multiple frames, enabling confident skip decisions on genuinely
static frames while preserving sensitivity to sustained motion signatures
characteristic of fire and smoke.
This results in a **30$\times$ higher skip rate** (44.39\% vs.\ 0.72\%) and a
**63\% higher FPS** (36.6 vs.\ 22.4) for the respective best configurations, at a
comparable recall cost ($-0.45\%$ vs.\ $-0.17\%$).
FrameDiffDet achieves high skip rates only at the cost of severe recall degradation
($\geq 5.5\%$ for skip rates $\geq 47\%$), confirming that it lacks the temporal
smoothing necessary for safe operation in this domain.
AccMotionDet is therefore selected as the skip module for all subsequent
system-level evaluations.


### System-Level Performance: Frame-Based Efficiency {#sec:e2e-perf}

We subsequently integrated the skip modules into the full inference pipeline to
measure end-to-end efficiency. System latency for our method is calculated as
the inherent skip module overhead plus the conditional latency of the BIG MODEL
applied only to unskipped frames.

The overall impact of integrating the skip modules into the full detection
pipeline is quantified in Table \ref{tb:perf_per_frame} (frame-level
accuracy/latency), also comparing against the baseline system without skipping.

```{=latex}
\input{./4.table/tb_perf_per_frame.tex}
```
FLOPs (small s) static complexity of a model
FlOPs = Floating-Point Operations
MFLOPs = $10^6$ FLOPs, GFLOPs = $10^9$ FLOPs

Obviously, the baseline system (BIG MODEL only) achieves the best performance
compare to other lightweight alternatives like MobileNet, FireNet, and YOLOv5s/l
due to more powerful capacity of the network architecture and the much larger
dataset size used for training as it reflect the consequences of neural scaling
laws [@hestness2017deep; @alabdulmohsin2022revisiting; @bahri2024explaining].

<!-- ! Place holder -->
_Analysis:_ Simply replacing the BIG MODEL with lightweight alternatives (M1,
M2) results in an unacceptable 17-22% degradation in F1-Score. Our proposed
pipeline (Approach 2 + BIG MODEL) successfully bridges this gap. By filtering
72.1% of frames at a cost of only 2.5ms per frame, the average system latency
drops to 16.5ms. This achieves a 67% reduction in computational cost, tripling
the effective frame rate from 20 FPS to 60 FPS while perfectly matching the
Baseline's 98.5% F1-Score.

### Comparison with Temporal Methods (M3) {#sec:cmp-temporal}

<!-- A critical distinction in anomaly detection is between frame-level and
video-level processing. The M3 baseline reduces false alarms by executing
majority voting across a 30-frame window. While highly accurate, this
architectural choice introduces significant latency. -->
Temporal baseline MEthod: Temporal Persistence Thresholding [@de2023hybrid]: The temporal baseline method, referred to as Temporal Persistence Thresholding (TPT),
augments the standard per-frame inference pipeline with a lightweight post-processing
stage designed to suppress isolated false alarms.
Unlike the proposed skip module, TPT does not reduce the number of frames forwarded
to the classifier --- every frame is still processed by the full BIG model.
Instead, a circular boolean buffer of length $W$ records the raw per-frame predictions
over a rolling window.
When the classifier predicts fire or smoke on a given frame, the detection is
confirmed only if the fraction of positive predictions within the buffer exceeds a
persistence threshold $\tau_\text{persist}$; otherwise, the prediction is suppressed
and the frame is relabelled as background.
This mechanism filters out transient, single-frame activations caused by brief
illumination changes or camera artefacts, trading a fixed minimum detection latency
of $\lceil \tau_\text{persist} \times W \rceil$ frames for a reduction in false
alarm rate.

We follow the implementation and hyperparameter selection protocol of
[@de2023hybrid] on our validation set $\mathcal{D}_\text{val}$, and get two
representative configurations shown in the table below:

```{=latex}
\input{./4.table/tb_val_results_Tpt.tex}
```

Table \ref{tb:cmp_other_temp} compares our method against other temporal
processing techniques, evaluating both detection performance and computational
efficiency.

And we found two Configuration of TPT let call it $\text{TPT}_\text{strict}$ and
$\text{TPT}_\text{balanced}$. The strict ones ($W = 5, \tau_\text{persist} =
0.2$) and the balanced one ($W = 10, \tau_\text{persist} = 0.5$) achieve a
recall of 98.7\%.

```{=latex}
\input{./4.table/tb_cmp_other_temp.tex}
```

_Analysis:_ Because M3 requires a full temporal buffer before confirming an
event, the system inherently delays the alarm trigger by over 750ms. In
contrast, our frame-level approach triggers the BIG MODEL immediately upon
detecting heuristic indicators. This results in a time-to-first-alarm of
approximately 52.5ms, making our method over 14 times faster to react than
temporal aggregation methods. Even when evaluated strictly on video-level
metrics, our skip-module approach achieves higher recall and requires
significantly less aggregate computational time per second of video, proving
highly competitive in false alarm suppression.

### Quantitative and Qualitative Analysis

<!-- #### Component Analysis: Efficacy of the Skip Module {#sec:comp-perf}

First, we evaluate the skip modules in isolation to ensure they function as safe
gatekeepers. The primary objective is to maximize the Filter Rate without
compromising Recall. Because the ultimate goal of the system is simply to detect
whether _any_ hazard exists (regardless of whether it is fire or smoke), we
measure safety using a unified anomaly recall metric. We also compare this
against the recall capabilities of the lightweight standalone models (M1 and
M2).

_Analysis:_ As demonstrated in Table \ref{tb:tb_no_skip_perf}, lightweight
standalone models (M1 and M2) offer fast processing but miss between 11% and 15%
of critical anomaly events. Approach 1 (Naive Motion) operates extremely fast
(1.2ms) but struggles with the slow, diffusing nature of smoke. This deficiency
drags its overall combined Recall down to 97.2%—an unacceptable safety margin
for early-warning systems. Conversely, our proposed Approach 2 integrates color
and texture heuristics to successfully capture both rapid flames and
semi-transparent smoke. It achieves a near-perfect combined Recall of 99.1%
while actually improving the Filter Rate to 72.1% by effectively distinguishing
true anomaly indicators from environmental noise (e.g., swaying trees).

To isolate the impact of our specific heuristic rules, we conducted an ablation
study comparing the naive motion baseline (Approach 1) against the full
rule-based engine (Approach 2). While Approach 1 performed adequately for
dynamic, rapidly flickering fires, its recall dropped significantly on smoke
events. Because smoke diffuses slowly and lacks sharp edge transitions, naive
background subtraction thresholds frequently misclassified it as static
background lighting changes.

The integration of the rule-based engine in Approach 2—specifically the
grayish-color tracking and temporal texture consistency rules—corrected this
deficiency, bridging the gap in Recall. This quantitative improvement is
supported by qualitative reviews of edge cases (see Figure 4).

_(Insert Figure 4 here: 2x2 grid showing a frame with smoke missed by Appr 1 but
caught by Appr 2, and a safe frame with swaying trees flagged by Appr 1 but
skipped by Appr 2)._

As illustrated in Figure 4, Approach 2 successfully captures slow-diffusion
smoke events that lack the raw pixel displacement required to trigger Approach
1. Furthermore, in scenes featuring subtle twilight illumination shifts and
swaying foliage, Approach 1 frequently generated false positives (unnecessarily
passing safe frames to the BIG MODEL). Approach 2 successfully filtered these
frames by applying color-channel heuristics, verifying that the moving pixels
did not match the chromatic signatures of either fire or smoke. -->

#### Efficiency of Skip Module:

```{=latex}
\input{./3.fig/fig_fps_increase.tex}
```

<!--! GUIDE: #### Qualitative Analysis:
  + Success case analysis: Side-by-side frames: correctly skipped static
  background vs. correctly passed fire/smoke frame, one for each class -- Gives
  reviewers visual intuition of what the gating looks like in practice
  + Failure case analysis: Pick 2-3 failure videos from test set: (1) slow smoke
  onset over-skipped, (2) persistent background motion (zero skip), show
  representative frames with caption -- Scientific honesty; safety-critical
  papers are expected to show where the system fails -->


#### Qualitative Analysis

Figure~\ref{fig:qualitative} illustrates representative success and
failure cases of the AccMotionDet skip module on the test set
$\mathcal{D}_\text{test}$.

**Success cases.**
The top row shows two canonical scenarios where $\mathcal{S}$ operates
as intended.
In the first case, a static indoor background frame --- with no
inter-frame motion --- is correctly skipped ($s_t = 0$), avoiding an
unnecessary invocation of $\mathcal{M}$ and contributing directly to
the 44.39\% skip rate reported in
Table~\ref{tb:val_results_accMotionDet}.
In the second case, a frame containing active fire (left) and smoke
(right) exhibits sufficient accumulated motion to trigger inference
($s_t = 1$), and $\mathcal{M}$ correctly raises an alarm.
These cases confirm that $\mathcal{S}$ reliably distinguishes
scene-level activity from quiescence in the target indoor
surveillance setting.

```{=latex}
\input{./3.fig/fig_quality_success.tex}
```


The bottom row exposes two structural limitations of the motion-based
gating approach.
First, in a slow-onset smoke video, the initial frames of smoke
diffusion produce minimal inter-frame pixel change --- below the
activation threshold $\tau_m$ of the AccMotionDet accumulator ---
causing $\mathcal{S}$ to skip these frames and delay the first alarm.
This represents the primary recall risk identified in
Section~\ref{sec:hyperparam}: the skip module is conservative by
design, but gradual diffusion events are the edge case where this
conservatism incurs a safety cost.
Second, in a scene containing continuous background motion
(e.g., a person walking repeatedly through the frame), $\mathcal{S}$
assigns $s_t = 1$ to nearly every frame, reducing the skip rate to
near zero and negating the computational efficiency gain for that
video.
Both failure modes are consistent with the motion-only design of
$\mathcal{S}$: they arise not from classification errors in
$\mathcal{M}$, but from the inherent limitation of using inter-frame
motion as a sole proxy for scene informativeness.

```{=latex}
\input{./3.fig/fig_quality_failure.tex}
```
<!-- !END_SYNC_BLOCK -->

# Conclusion {#sec:conclusion label="Conclusion"}


<!-- !START_SYNC_BLOCK -->
<!-- TARGET_PROJECT: G:\My Drive\1_PhD\Obsidian\Home\3. Writing\paperfire2 -->
<!-- SYNC_TARGET_FILE: 04_Conclusion.md-->
<!-- BLOCK_ID: conclusion -->

We propose a lightweight, plug-and-play skip module $\mathcal{S}$ designed to
accelerate real-time fire and smoke detection in indoor surveillance scenarios.
Operating at a tiny fraction (${\approx}2.5\%$) of the computational cost of the
DL classifier $\mathcal{M}$, $\mathcal{S}$ efficiently filters out static
background frames before they reach $\mathcal{M}$, reducing unnecessary
inference overhead. Experiments on a real-world video dataset demonstrate that
the proposed approach yields a throughput improvement of over 30\% in frames per
second while maintaining detection reliability.

<!-- !END_SYNC_BLOCK -->


<!-- !START_SYNC_BLOCK -->
<!-- TARGET_PROJECT: G:\My Drive\1_PhD\Obsidian\Home\3. Writing\paperfire2 -->
<!-- SYNC_TARGET_FILE: 04_Conclusion.md-->
<!-- BLOCK_ID: discussion -->
Nevertheless, the proposed approach has several limitations. First,
$\mathcal{S}$ relies solely on inter-frame motion cues, which may be unsuitable
for outdoor surveillance scenarios where persistent motion sources — such as
wind, moving vehicles, etc. — are continuously present, potentially causing
$\mathcal{S}$ to pass the majority of frames to $\mathcal{M}$ and negating the
efficiency gain. Second, the hyperparameters of $\mathcal{S}$ require
dataset-specific tuning prior to deployment, incurring additional optimization
overhead. Third, the overall detection accuracy of the system remains bounded by
the capacity of $\mathcal{M}$, which is treated as a fixed component and not
optimized in this work.

Future work may explore alternative skip strategies — such as leveraging color,
texture, or learned features as gating cues — and extend the framework to
outdoor surveillance scenarios where motion-based filtering is less effective.


<!-- !END_SYNC_BLOCK -->


# References {#sec:references label="bibliography"}
