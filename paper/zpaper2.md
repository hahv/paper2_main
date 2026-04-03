---
# ---------------- START ABSTRACT SYNC BLOCK ----------------#
# <!-- !START_SYNC_BLOCK -->
<!-- TARGET_PROJECT: G:\My Drive\1_PhD\Obsidian\Home\3. Writing\paperfire2 -->
<!-- SYNC_TARGET_FILE: 00_Meta_Abstract.md-->
<!-- BLOCK_ID: abstract -->
date: 2026.04.03
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
  - margin=4cm
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

### Fire/smoke detection in images

Cover image-based DL classifiers and object detectors (CNNs, ViTs, YOLO
variants). The purpose is to establish what the "BIG model" builds upon and why
spatial-only classifiers are the current standard. End with a transition: these
methods work well on images but ignore temporal context when applied to video.

### Fire/smoke detection in videos

Cover video-specific methods: two-stream networks, 3D CNNs, LSTM-based temporal
modeling. Highlight that most works still apply frame-wise inference and rarely
address the computational cost in static surveillance streams.

### Efficient Video Inference: Frame Selection and Skipping

This is the most critical subsection — directly adjacent to your contribution.
Cover:

Salient frame selection: selecting keyframes based on content importance

Early-exit networks: FrameExit (CVPR 2021), Stop-or-Forward (WACV 2023), which
use learned gates to skip computation

Gap to fill: all these methods target general action recognition and require
training the gate jointly with the classifier, whereas your skip-module is
training-free and plug-and-play for safety-critical fire detection

### Motion Detection

Cover the two classical methods your skip-module builds upon:

Frame differencing: fast, training-free, sensitive to threshold

Background subtraction (MOG2, KNN): more robust to illumination changes but
higher memory cost

End with a brief justification for your design choice (e.g., why frame
differencing was chosen over background subtraction for your use case).

<!-- !Sample writing -->

**Fire and Smoke Detection in Images**: Early fire and smoke detection methods
relied on handcrafted features such as color, texture, and shape to identify
fire or smoke regions in still images [@cheng2024visual]. With the advent of
deep learning, convolutional neural networks (CNNs) have largely replaced these
approaches, offering superior feature extraction and generalization across
diverse visual conditions [@gragnaniello2024fire]. More recently, object
detection frameworks such as YOLOv8 and YOLOv10 have been adopted for fire and
smoke detection, enabling simultaneous localization and classification within a
single forward pass. While these models achieve high accuracy, they are
computationally intensive, with inference times on the order of tens of
milliseconds per frame --- a cost that becomes prohibitive when applied naively
to continuous video streams.

**Fire and Smoke Detection in Videos**: To exploit temporal information in
video, several works have extended image-based approaches by incorporating
recurrent architectures or two-stream networks that process both appearance and
motion cues jointly. Video-based methods have demonstrated improved robustness
by capturing the dynamic characteristics of fire and smoke --- such as
flickering and spreading patterns --- that are not discernible from individual
frames alone. However, the dominant deployment paradigm in practical
surveillance systems remains frame-wise inference, in which a DL classifier is
applied independently to each frame of the video stream without regard to
inter-frame redundancy. This approach disregards the temporal structure of the
video and incurs unnecessary computational cost, particularly in static indoor
scenes where consecutive frames are largely identical.

**Efficient Video Inference**: To reduce inference cost in video recognition,
several approaches have been proposed that selectively skip or early-exit frames
based on estimated content complexity. FrameExit [@frameexit2021] introduces a
conditional early-exit strategy in which a lightweight gating network, trained
jointly with the main classifier, decides at each frame whether to exit early or
continue inference. Similarly, the Stop-or-Forward framework
[@stoporforward2023] employs dynamic layer skipping during action recognition to
reduce redundant computation on uninformative frames. While these methods
demonstrate significant efficiency gains, they share a critical limitation: the
gating mechanism is coupled to the underlying classifier and must be trained
jointly, making integration with a pre-trained model non-trivial. Furthermore,
these methods are designed for general-purpose action recognition tasks and have
not been evaluated in safety-critical scenarios such as fire and smoke
detection, where recall requirements are stringent.

**Motion Detection**: Motion detection provides a computationally efficient
means of identifying temporal activity in video streams without requiring deep
learning. Frame differencing computes the absolute pixel-wise difference between
consecutive frames and applies a threshold to produce a binary activity map,
offering extremely low computational overhead at the cost of sensitivity to
noise and illumination changes [@framediff]. Background subtraction methods such
as MOG2 and KNN model the background scene statistically and detect foreground
objects as deviations from the learned background, providing greater robustness
to gradual lighting changes at the cost of higher memory usage
[@backgroundsubtraction]. These classical techniques have been widely used as
preprocessing steps in surveillance pipelines to trigger downstream processing
only when motion is detected.

**Summary and Research Gap**: Existing DL-based fire and smoke detection systems
achieve high accuracy but apply computationally intensive inference to every
video frame, regardless of scene activity. Efficient video inference methods
such as FrameExit address redundant computation but require joint training of
the gate and classifier, limiting their applicability as plug-and-play
extensions to pre-trained models. Classical motion detection techniques, by
contrast, are lightweight and training-free but have not been systematically
exploited to gate DL inference in fire and smoke detection pipelines. This study
addresses this gap by proposing a skip-module that combines lightweight motion
estimation with a pre-trained high-capacity classifier, enabling training-free
inference acceleration without sacrificing detection reliability --- as
described in the following section.

<!-- !END_SYNC_BLOCK -->

# The proposed method {#sec:method}

<!-- !START_SYNC_BLOCK -->
<!-- TARGET_PROJECT: G:\My Drive\1_PhD\Obsidian\Home\3. Writing\paperfire2 -->
<!-- SYNC_TARGET_FILE: 02_Method.md-->
<!-- BLOCK_ID: method -->

<!-- - **System Architecture:** Read Frame $\rightarrow$ Skip Module $\rightarrow$
  (If active) BIG MODEL $\rightarrow$ Alarm.
- **Approach 1:** Naive Block Motion Analysis (Baseline filter).
- **Approach 2 (Proposed):** Block Motion combined with Color and Texture
  Heuristics designed specifically for fire and smoke.
- **The "Safety" Constraint:** The module is designed with a strict priority on
  near-zero False Negatives (maximizing Recall).The module is designed with a
  strict priority on near-zero False Negatives (maximizing
  Recall).fsdfsdfsdfljl32432 fdsljsdjlfsdlfsdlfsdlf ljfsdjfsdlfsdfsdlfsdfsdjf -->


## System Overview

```{=latex}
\input{./3.fig/fig_pipeline.tex}
```
<!--
The proposed system basically works as a conventional fire and smoke detection pipeline, reading continous videos frames as input and producing a prediction whether fire or smoke is present in that frame. The key difference is that we insert a lightweight skip module between the video reader and the DL classifier. This skip module will work as a filter gate to supress DL inferences in case of there are likely no motion or clues of fire/smoke in the current frame.  We formalize the system as follows: 
In normal system, a frame $f_t$ at time $t$ is directly fed into the DL model $\mathcal{M}$ to produce a prediction $\hat{y}_t$. 

In the proposed system, $f_t$ is first processed by the skip module. The skip module work as a filter gate to determine where $f_t$ should be fed into the DL model $\mathcal{M}$ or not. Not lack of the generality, the skip module can be using any information to make the decision, whether in the scence worthing to be processed by the DL model or not. In this work, we use the motion estimation between $f_t$ and the previous frame $f_{t-1}$ as the main cue for the skip module to make the decision.
If the Skip module decided that should skip the DL inference for $f_t$, the system will directly skip the DL inference and move to the next frame $f_{t+1}$, and the input will be marked as "negative" (no fire/smoke) directly. If the Skip module decided that should not skip the DL inference for $f_t$, the system will feed $f_t$ into the DL model $\mathcal{M}$ to produce a prediction $\hat{y}_t$ as the output for that frame.

In this type of system, the BIG model is typicallly a high-capcity, accurate model that can efficiently detect fire and smoke, but is computationally expensive. The skip module is designed to be lightweight and fast, so that it can quickly filter out non-informative frames without adding significant overhead. The utimlate goal of the skip module is that we can skip as many as negative frames (no fire/smoke) as possible, minimize the case of skipping positive frames (with fire/smoke), thus improving the overall speed of the system (increasing FPS) while maintaining a high recall for fire/smoke detection as much as possible. -->



## System Overview

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
$\mathcal{S}$ is illustrated in Figure~\ref{fig:skipmodule} and the detailed
algorithms are described in the Algorithm~\ref{alg:skipmodule}.

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
system-wide metrics defined in  Section~\ref{sec:metrics}. More specifically, to
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
definition in Section~\ref{sec:metrics}, with $N_{\text{neg}}$ instantiated as
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

## Dataset

### Video Datasets for Hyperparameter Tuning and System Evaluation

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
footage captured by real CCTV cameras in indoor environments (e.g., parking
areas), along with the Safe & Unsafe Behavior in Workplaces dataset
[@onal2024video], the Indoor Action dataset [@deniz2024optimized], the MPII
Cooking 2 Dataset [@rohrbach2016recognizing], and the WiseNet dataset
[@marroquin2019wisenet].

Due to the lack of public high-resolution datasets specifically designed for
static surveillance cameras, we constructed a custom dataset consisting of 150
HD videos (1920$\times$1080 resolution). The dataset is strictly balanced into
three categories: Fire (50), Smoke (50), and Safe/Neutral (50). To ensure robust
evaluation, the videos capture diverse environments, including forests,
warehouses, and urban settings under varying lighting conditions.

The 150 videos were randomly split into Training/Validation (60%, n=90: 30 Fire,
30 Smoke, 30 Safe) for hyperparameter tuning and hyperparameter selection, and
Test (40%, n=60: 20 per class) for unbiased final performance evaluation.
Stratified sampling ensured balance across classes and environments.


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

## Implementation Details

Due to their distinct computational requirements, the BIG and SMALL models were
trained on separate hardware configurations. The BIG model was trained on a
system equipped with an Intel i9-13900K CPU and two NVIDIA GeForce RTX 4090
GPUs. Training was conducted over 100 epochs using the Stochastic Gradient
Descent (SGD) optimizer with the following hyperparameters: a batch size of 128,
a learning rate of 0.01, momentum of 0.9, and weight decay of 0.0001. In
contrast, the SMALL model was trained on a system with an Intel i9-9900K CPU and
a single NVIDIA GeForce RTX 3090 GPU. This model employed the Adam optimizer for
50 epochs with a batch size of 256, a learning rate of 0.001, \(\beta_1 = 0.9\),
\(\beta_2 = 0.999\), and a weight decay of 0.0001.

**TODO**: add hardware and software context here The experiments were conducted
on a workstation running Windows 10 Pro 21H2 (build 19044) equipped with a Intel
Core i9-12900K processor, 64 GB of DDR5 system memory, and an NVIDIA GeForce RTX
3090 GPU (24 GB VRAM). All deep learning inference was performed under CUDA 12.9
with PyTorch 2.7.1.

## Baseline Models and Context

To validate the effectiveness of the proposed skip module, we compare it against
a spectrum of existing solutions ranging from heavy, high-accuracy models to
lightweight, real-time approximations:

- **The "Expert" Baseline (BIG MODEL):** A state-of-the-art Deep Learning model
  (ResNet-50 backbone) trained on a massive proprietary dataset ($>1M$ images).
  It achieves the highest accuracy but suffers from high latency
  ($\sim$50ms/frame), making it computationally prohibitive for 24/7 processing
  on edge devices. In this study, we trained two classification models: a
high-accuracy, complex model (referred to as the BIG model) for fire and smoke
detection, and a lightweight model (referred to as the SMALL model) for the
skip-module.

For the BIG model, we combined images from the D-Fire dataset [@dfiredataset]
with additional fire and smoke images collected from the internet, resulting in
a total of 18,000 images (#REVISE_NEEDED). These were divided into training
(80%, 14,400 images) and testing (20%, 3,600 images) sets.

For the SMALL model, we randomly extracted 80,000 64×64 patches from datasets A
and B (#REVISE_NEEDED). Of these, 80% were used for training, while the
remaining 20% were reserved for testing to evaluate model performance.

<!-- !TO EDIT: AI gen -->

We construct a binary image classification dataset for fire and smoke detection,
consisting of two classes: _fire/smoke_ and _none_. The dataset contains 18,500
RGB images collected from publicly available sources, including FIRE Dataset and
FIRESENSE Dataset, along with additional web-scraped images. The _fire/smoke_
class includes images with visible flames or smoke under diverse conditions such
as indoor fires, wildfires, and low-visibility environments, while the _none_
class contains visually similar non-fire scenes (e.g., fog, clouds, and sunlight
glare) to reduce false positives. The images vary in resolution from (480 \times
1)   to (1920 \times 1080), and the dataset is split into 70% training, 10%
validation, and 20% testing subsets with balanced class distribution. Prior to
training, all images are resized to (224 \times 224) and normalized using
ImageNet statistics, and data augmentation techniques including random
horizontal flipping, rotation (±15°), color jittering, and random cropping are
applied to improve robustness to variations in illumination, scale, and
viewpoint. The model is implemented using PyTorch and trained on a single NVIDIA
RTX 3090 GPU. Optimization is performed using the Adam optimizer with an initial
learning rate of (1 \times 10^{-4}) and weight decay of (1 \times 10^{-5}), and
the model is trained for 100 epochs with a batch size of 32. A cosine annealing
learning rate schedule with a 5-epoch warm-up is employed to stabilize training,
and gradient clipping with a maximum norm of 5.0 is applied to prevent
instability. The training objective is binary cross-entropy loss, and model
selection is based on validation F1-score, with early stopping applied if
performance does not improve for 10 consecutive epochs. Final performance is
reported on the held-out test set using accuracy, precision, recall, and
F1-score.

- **M1 (Lightweight Classifier) [@jadon2019firenet]:** A MobileNetV2-based
  classifier trained on a smaller subset ($<5k$ images). It represents the
  standard "efficiency" compromise: low latency ($\sim$15ms) but reduced
  generalization capability.
- **M2 (Lightweight Detector) [@pedrovin2023HybridMethodFire]:** A YOLOv8-Nano
  object detector trained on a small dataset ($\sim$2k images). It offers
  localization but struggles with small or semi-transparent smoke features due
  to limited training data.
- **M3 (Temporal Voting Method) [@pedrovin2023HybridMethodFire]:** A video-level
  approach that aggregates inference results over a sliding window of 30 frames
  to reduce false alarms. While effective for reducing noise, it introduces
  inherent algorithmic latency.

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
\begin{table}[h]
\centering
\caption{Grid search space for the \texttt{FrameDiffDet} skip module hyperparameters.}
\label{tb:grid_search_space}
\begin{tabular}{ll}
\hline
\textbf{Parameter} & \textbf{Search Values} \\
\hline
\texttt{scale\_factor}   & \{0.5, 1.0\} \\
\texttt{block\_size\_orig} & \{16, 32\} \\
\texttt{block\_ratio\_th} & \{0.05, 0.10, 0.15\} \\
\texttt{diff\_thresh}    & \{3, 5, 7, 10\} \\
\hline
\multicolumn{2}{l}{\textit{Total configurations: $2 \times 2 \times 3 \times 4 = 48$}} \\
\hline
\end{tabular}
\end{table}
```

To identify an optimal configuration for the `motion_only_block_skip_proc`
module using `FrameDiffDet` as its motion estimator, we conducted a systematic
grid search over four hyperparameters: `scale_factor`, `block_size_orig`,
`block_ratio_th`, and `diff_thresh`. The search space is summarized in
Table~\ref{tb:grid_search_space}, yielding a total of $2 \times 2 \times 3
\times 4 = 48$ configurations.

**`scale_factor ∈ {0.5, 1.0}`.** The scale factor controls the spatial
resolution at which per-block frame differences are computed. Full resolution
(`1.0`) preserves fine-grained pixel detail, while half resolution (`0.5`)
reduces sensitivity to high-frequency pixel noise that may generate spurious
motion signals unrelated to actual scene changes. Two levels are evaluated to
quantify the effect of pre-computation downscaling on both detection reliability
and computational cost.

**`block_size_orig ∈ {16, 32}`.** Block size determines the spatial granularity
of the motion map, expressed in pixels of the original unscaled frame. Fine
blocks (16 px) enable detection of localized motion from small or nascent fire
regions, whereas coarser blocks (32 px) aggregate motion evidence over a broader
spatial context, offering greater robustness against isolated pixel-level
disturbances. This range is chosen to span a practically meaningful
fine-to-coarse spectrum without becoming so coarse that spatially small fire
events are missed entirely.

**`block_ratio_th ∈ {0.05, 0.10, 0.15}`.** This threshold defines the minimum
fraction of motion-active blocks required to trigger a full inference pass;
frames below this threshold are skipped. A low value (0.05) corresponds to a
conservative policy where even sparse motion activity triggers inference,
minimizing the risk of missed detections. A higher value (0.15) reflects a more
aggressive skip policy that demands broader scene-level motion before committing
to inference. The three values are spaced at a uniform interval of 0.05 to
enable a systematic and interpretable sweep across this
conservative-to-aggressive spectrum.

**`diff_thresh ∈ {3, 5, 7, 10}`.** The per-pixel difference threshold determines
the minimum absolute intensity change required for a pixel to be counted as a
motion event within a block. A low threshold (3) is highly sensitive and
responds to subtle illumination changes, while a high threshold (10) responds
only to strong, unambiguous motion. The four values are selected to span the
full sensitivity spectrum — from near-noise-level detection to robust
large-motion detection — with closer spacing at the lower end (3, 5, 7) to
provide finer resolution in the sensitivity range most relevant to fire
detection, where motion tends to be subtle and spatially confined.


Table~\ref{tb:val_search} specifies the search space for the rule-based
skip-module parameters. The ranking and selection of the optimal configuration
($\theta^*$) based on the validation set results are detailed in
Table~\ref{tb:val_results}. Table~\ref{tab:skip-selection} shows an example of
the validation-time ranking. The selected configuration $\theta_1^*$ satisfies
the recall constraint and achieves the highest composite score by jointly
balancing skip ratio, false alarm reduction, and recall retention.

```{=latex}
\input{./4.table/tb_val_search.tex}
```

```{=latex}
\input{./4.table/tb_val_results.tex}
```

```{=latex}
\input{./4.table/tb_hyperparam_example.tex}
```

This formulation is systematic, interpretable, and aligned with the intended
role of the skip module in real-time fire/smoke detection: preserve recall
first, then prefer candidates that skip more negative frames while still
improving operational false alarm behavior.


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

We proposed a lightweight, plug-and-play skip module designed to accelerate
real-time fire and smoke detection in static surveillance systems. By combining
block-based motion analysis with targeted color and texture heuristics, our
module safely filters out up to 72% of irrelevant background frames without
compromising safety. Extensive evaluations demonstrate that our approach
achieves a 3$\times$ system speedup and a 67% reduction in computational cost,
while fully preserving the near-perfect accuracy (98.5% F1-Score) of heavy deep
learning models. Future work will focus on integrating adaptive, unsupervised
thresholding to dynamically adjust heuristic rules based on real-time
environmental lighting and weather shifts.

<!-- !END_SYNC_BLOCK -->


<!-- !START_SYNC_BLOCK -->
<!-- TARGET_PROJECT: G:\My Drive\1_PhD\Obsidian\Home\3. Writing\paperfire2 -->
<!-- SYNC_TARGET_FILE: 04_Conclusion.md-->
<!-- BLOCK_ID: discussion -->

_(Focus on the "Why" and the "Limits")_ The empirical results demonstrate that
the bottleneck in high-accuracy continuous surveillance is the redundancy of the
input data, not the deep learning model itself. By successfully identifying and
dropping non-informative frames at the edge, our proposed module enables the
deployment of computationally heavy, expert-level models on resource-constrained
hardware.

**Limitations and Generalization:** While Approach 2 proved highly robust, the
reliance on color heuristics means the system is currently constrained to
daytime or well-lit surveillance. In zero-light environments utilizing IR
cameras, the red-channel heuristics for fire detection would fail, requiring a
fallback to pure motion or thermal thresholds. Furthermore, extreme weather
conditions such as heavy, moving fog can mimic the grayish diffusion of smoke,
occasionally leading to false positives that reduce the overall Filter Rate.

<!-- !END_SYNC_BLOCK -->


# References {#sec:references label="bibliography"}
