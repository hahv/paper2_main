---
# ---------------- START ABSTRACT SYNC BLOCK ----------------#
# <!-- !START_SYNC_BLOCK -->
<!-- TARGET_PROJECT: G:\My Drive\1_PhD\Obsidian\Home\3. Writing\paperfire2 -->
<!-- SYNC_TARGET_FILE: 00_Meta_Abstract.md-->
<!-- BLOCK_ID: abstract -->
date: 2026.04.07
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
The skip module $\mathcal{S}$ contains several hyperparameters (e.g., motion
threshold, accumulation window size) that can significantly impact the
performance of the overall system. To identify the optimal configuration, we
employ a grid search based hyperparameter optimization procedure on a validation
set $D_{\text{val}}$ derived from our video dataset with respect to several
system-wide metrics defined in  Section \ref{sec:metrics}. More specifically, to
select the optimal hyperparameters for the proposed skip module, we employ a
constrained optimization procedure on the validation set $D_{\text{val}}$. Let
$R_{\text{base}}$, $\mathrm{FAR}_{\text{base}}$, and $T_{\text{DL}}$ denote the
recall, false alarm rate, and mean per-frame inference time of the baseline
pipeline (without skipping), respectively. For each candidate parameter set
$\theta \in \Theta$ obtained by grid search, we evaluate the full pipeline
$\text{Read} \rightarrow \mathcal{S}(\theta) \rightarrow [\mathcal{M}]$ and
compute four metrics: recall $R(\theta)$, false alarm rate
$\mathrm{FAR}(\theta)$, skip rate $S_r(\theta)$, and mean per-frame skip module
time $T_{\text{skip}}(\theta)$.

The skip rate $S_r(\theta)$ is computed on $D_{\text{val}}$ following the
definition in Section \ref{sec:metrics}, with $N_{\text{neg}}$ instantiated as
$N_{\text{neg}}^{\text{val}} = |\{f_i \in D_{\text{val}} : y_i = 0\}|$, which is
fixed by the validation set and independent of $\theta$.

The mean per-frame inference times are defined as

$$ T_{\text{DL}} = \frac{1}{|D_{\text{val}}|} \sum_{i=1}^{|D_{\text{val}}|}
t_{\text{DL}}(f_i), \qquad T_{\text{skip}}(\theta) = \frac{1}{|D_{\text{val}}|}
\sum_{i=1}^{|D_{\text{val}}|} t_{\text{skip}}(f_i, \theta), $$

where $f_i$ denotes the $i$-th frame in $D_{\text{val}}$, and both averages are
taken over all frames to reflect the runtime cost on a typical video stream.

**Feasibility constraints.** A candidate $\theta$ is considered feasible only if
it satisfies two hard constraints:

$$ R(\theta) \geq R_{\text{base}} - \delta_R, \qquad T_{\text{skip}}(\theta)
\leq \beta \cdot T_{\text{DL}}, $$

where $\delta_R > 0$ is the maximum allowable absolute recall drop, and $\beta
\in (0, 1)$ bounds the skip module overhead as a fraction of one full DL
inference. For example, setting $\delta_R = 0.01$ and $\beta = 0.10$ means the
system tolerates at most a 1\% recall drop and requires the skip module to run
within 10\% of the cost of a single DL inference.

**Scoring feasible candidates.** To rank feasible candidates, we define two
normalized scoring terms. The false alarm reduction term is

$$ F(\theta) = \max\!\left(0,\; \frac{\mathrm{FAR}_{\text{base}} -
\mathrm{FAR}(\theta)} {\mathrm{FAR}_{\text{base}}}\right), $$

which measures the relative FAR improvement with respect to the baseline. The
recall retention term is

$$ \rho(\theta) = 1 - \frac{R_{\text{base}} - R(\theta)}{\delta_R}, $$

which equals $1$ when recall matches the baseline and $0$ when recall reaches
the lowest acceptable level $R_{\text{base}} - \delta_R$. Both terms are bounded
in $[0, 1]$ over the feasible region, making them directly comparable to
$S(\theta)$ in the weighted score.

**Optimal selection.** The optimal parameter set $\theta^*$ is selected as

$$ \theta^* = \arg\max_{\theta \in \Theta_{\text{feasible}}} \left[ w_S\,
S(\theta) + w_F\, F(\theta) + w_R\, \rho(\theta) \right], $$

where $\Theta_{\text{feasible}} = \{\theta \in \Theta : R(\theta) \geq
R_{\text{base}} - \delta_R \text{ and } T_{\text{skip}}(\theta) \leq \beta \cdot
T_{\text{DL}}\}$, and the nonnegative weights satisfy $w_S + w_F + w_R = 1$.

**Theoretical justification.** The skip module $\mathcal{S}$ acts as a
conservative gate: skipped frames are labeled negative directly, while passed
frames are processed by the same downstream detector $\mathcal{M}$ as in the
baseline. Therefore, every false alarm produced by the skip-enabled system is
also present in the baseline, implying $\mathrm{FAR}(\theta) \leq
\mathrm{FAR}_{\text{base}}$. The primary safety risk is recall degradation
caused by incorrectly skipped fire/smoke frames, which motivates enforcing the
recall constraint as a hard gate prior to any candidate ranking.


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
In addition to standard metrics, we report the skip rate $S_r$, a system-level
efficiency criterion specific to the proposed skip-based framework. Skip rate
measures the proportion of ground-truth negative frames correctly bypassed by
$\mathcal{S}$, defined as

$$ S_r = \frac{N_{\text{skip}}^{-}}{N_{\text{neg}}}, $$

where $N_{\text{skip}}^{-}$ is the number of true-negative frames for which
$\mathcal{S}$ correctly outputs $s_t = 0$, and $N_{\text{neg}} = TN + FP$ is the
total number of ground-truth negative frames in the evaluated set. A higher
$S_r$ indicates greater computational savings, while a drop in recall signals
unsafe skipping of fire/smoke frames.

## Baseline Models and and Implementation Details

**Baseline Models**: To demonstrate both the superiority of the high-capacity
classifier over lightweight alternatives and the effectiveness of the proposed
skip module in accelerating its inference, we evaluate the following baseline
models alongside our BIG model:

- **FireNet** [@jadon2019firenet]: A lightweight CNN-based image classifier
  comprising 14 layers, with a model size of 7.5 MB and approximately 650k
  trainable parameters.

- **MobileNet** [@mukhopadhyay2019fpga]: A modified MobileNet architecture
  fine-tuned for fire and smoke detection.

- **YOLOv5s** [@de2023hybrid]: A lightweight object detector based on YOLOv5
  [@JocherYOLOv5byUltralytics2020], trained with hyperparameters selected via
  grid search on the D-Fire dataset [@dfiredataset].

- **YOLOv5l** [@de2023hybrid]: A heavier variant of the above, based on the
  larger YOLOv5l architecture, offering higher capacity at increased
  computational cost.

- **BIG Model**: The BIG model is obtained by fine-tuning a pretrained
  High-Performance GPU Network v2 (HGNetV2), specifically the
  \texttt{hgnetv2\_b5.ssld\_stage2\_ft\_in1k} checkpoint [@hgnetv2timm:online]
  from the Timm library [@rw2019timm], which was originally trained using SSLD
  knowledge distillation [@cui2021beyond]. HGNetV2 [@hgnetv2PaddleCl7:online] is
  a high-capacity CNN architecture designed to achieve substantially higher
  accuracy than models of comparable inference speed on NVIDIA GPUs, making it
  well-suited for deployment in our target environment. \textcolor{red}{For
  fine-tuning, we compiled a dataset of 1,000,000,000 fire and smoke images collected
  from internet sources, partitioned into training (80\%) and test (20\%)
  subsets. The model was optimized using Adam with an initial learning rate of
  $1 \times 10^{-4}$ and weight decay of $1 \times 10^{-5}$ for 100 epochs with
  a batch size of 32. On the held-out test set, the final model achieved a
  recall of xx.xx\% and a false alarm rate of xx.xx\%.}

**Implementation Details**: Unless otherwise specified, all experiments were
conducted on a workstation equipped with an Intel Core i9-12900K CPU, 64 GB DDR5
system memory, and an NVIDIA GeForce RTX 3090 GPU (24 GB VRAM), running Windows
10 Pro (build 19044). Deep learning inference was performed using PyTorch 2.7.1
under CUDA 12.9, and video processing and motion estimation were carried out
using OpenCV 4.11 (CPU-only).

## Results

### Hyperparameter Optimization Results

In this work, we set

$$ w_S = 0.60, \qquad w_F = 0.20, \qquad w_R = 0.20, $$

so that skip rate remains the primary efficiency objective, while false alarm
reduction and recall retention are treated as secondary but explicitly rewarded
preferences.

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
recall constraint and achieves the highest composite score by jointly balancing
skip ratio, false alarm reduction, and recall retention.

```{=latex}
\input{./4.table/tb_val_results.tex}
```
This formulation is systematic, interpretable, and aligned with the intended
role of the skip module in real-time fire/smoke detection: preserve recall
first, then prefer candidates that skip more negative frames while still
improving operational false alarm behavior.


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
\input{./4.table/tb_val_results.tex}
```

### System-Level Performance: Frame-Based Efficiency {#sec:e2e-perf}

We subsequently integrated the skip modules into the full inference pipeline to
measure end-to-end efficiency. System latency for our method is calculated as
the inherent skip module overhead plus the conditional latency of the BIG MODEL
applied only to unskipped frames.

The overall impact of integrating the skip modules into the full detection
pipeline is quantified in Table \ref{tb:e2e_perf} (frame-level
accuracy/latency), also comparing against the baseline system without skipping.

```{=latex}
\input{./4.table/tb_e2e_perf.tex}
```

_Analysis:_ Simply replacing the BIG MODEL with lightweight alternatives (M1,
M2) results in an unacceptable 17-22% degradation in F1-Score. Our proposed
pipeline (Approach 2 + BIG MODEL) successfully bridges this gap. By filtering
72.1% of frames at a cost of only 2.5ms per frame, the average system latency
drops to 16.5ms. This achieves a 67% reduction in computational cost, tripling
the effective frame rate from 20 FPS to 60 FPS while perfectly matching the
Baseline's 98.5% F1-Score.

### Comparison with Temporal Methods (M3) {#sec:cmp-temporal}

A critical distinction in anomaly detection is between frame-level and
video-level processing. The M3 baseline reduces false alarms by executing
majority voting across a 30-frame window. While highly accurate, this
architectural choice introduces significant latency.

Table \ref{tb:cmp_base_temp} compares our method against other temporal
processing techniques, evaluating both detection performance and computational
efficiency.

```{=latex}
\input{./4.table/tb_cmp_base_temp.tex}
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

### Ablation and Qualitative Analysis

#### Component Analysis: Efficacy of the Skip Module {#sec:comp-perf}

First, we evaluate the skip modules in isolation to ensure they function as safe
gatekeepers. The primary objective is to maximize the Filter Rate without
compromising Recall. Because the ultimate goal of the system is simply to detect
whether _any_ hazard exists (regardless of whether it is fire or smoke), we
measure safety using a unified anomaly recall metric. We also compare this
against the recall capabilities of the lightweight standalone models (M1 and
M2).

Table \ref{tb:tb_no_skip_perf} assesses the intrinsic recall and filtering
capability of each method as a standalone component.

```{=latex}
\input{./4.table/tb_no_skip_perf.tex}
```

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
did not match the chromatic signatures of either fire or smoke.

**Speed of Skip Module**:

```{=latex}
\input{./3.fig/fig_fps_increase.tex}
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
