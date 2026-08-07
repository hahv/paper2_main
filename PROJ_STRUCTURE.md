================================================================================
                   PROJECT SYSTEM ARCHITECTURE OVERVIEW
================================================================================

CORE FLOW:
----------

    ┌──────────────┐
    │  Config.py   │  ◄─── Central Configuration Hub
    └──────┬───────┘
           │ contains:
           ├─ DatasetCfg (video paths, labels)
           ├─ MethodCfg (which method to use)
           ├─ MetricSetCfg (which metrics to compute)
           ├─ ModelConfig (CNN model info)
           └─ InferConfig (save_csv, save_video, etc.)
           │
           ▼
    ┌──────────────┐
    │  MyExp       │  ◄─── Experiment Orchestrator
    └──────┬───────┘
           │
           │ 1. Creates Method
           ▼
    ┌──────────────────────────────────────────────────────────┐
    │                   src/methods/                           │
    │                                                          │
    │  ┌─────────────────┐                                     │
    │  │  MethodFactory  │ ──creates──►  BaseMethod (ABC)      │
    │  └─────────────────┘                       │             │
    │                                            │             │
    │                              ┌─────────────┴─────────────┐
    │                              ▼                           ▼
    │                        NoTempMethod                 TempMethod
    │                      (frame-by-frame)               (with skip)
    │                                                          │
    │                                                          │ uses
    │                                                          ▼
    │    ┌──────────────────────────────────────────────────┐  │
    │    │          Skip Pipeline (Optional)                │  │
    │    │                                                  │  │
    │    │  BaseSkipProc ◄──creates── SkipProcFactory       │  │
    │    │       │                                          │  │
    │    │       ├──► motion_only_block_skip_proc           │  │
    │    │       ├──► motion_only_block_skip_proc_eager     │  │
    │    │       └──► etc...                                │  │
    │    └──────────────────────────────────────────────────┘  │
    └──────────────────────────────────────────────────────────┘
           │
           │ 2. Inference on Videos
           │    (frame-by-frame processing)
           │
           ▼
    ┌──────────────────────────────────────────────────────────┐
    │                   src/results/                           │
    │         Result Processors (Handle Output)                │
    │                                                          │
    │  BaseRsProc (ABC)                                        │
    │        │                                                 │
    │        ├──► CsvRsProc                                    │
    │        │      └─ Saves: video_name_results.csv           │
    │        │         Columns: frame_idx, pred_label,         │
    │        │                  probs, logits, elapsed_time    │
    │        │                                                 │
    │        └──► VideoInferRsProc                             │
    │               │                                          │
    │               ├─ Creates: VideoPipeline                  │
    │               │   └─ Uses Multiple Renderers:            │
    │               │       • InferRsRenderer (OSD: pred, fps) │
    │               │       • GridRenderer    (block grid)     │
    │               │       • .. other renderer                │
    │               │                                          │
    │               └─ Subclasses:                             │
    │                   • VideoBlockSkipRsProc                 │
    │                   • FgmaskBlockSkipRsProc                │
    │                     (visualize motion masks)             │
    └──────────────────────────────────────────────────────────┘
           │
           │ Output Files:
           ├─ video1_results.csv
           ├─ video2_results.csv
           └─ video1_out.mp4, video2_out.mp4
           │
           │ 3. Metric Calculation
           ▼
    ┌──────────────────────────────────────────────────────────┐
    │                   src/metrics/                           │
    │       Metric Computation (Loads CSV → Calculates)        │
    │                                                          │
    │  ┌────────────────┐                                      │
    │  │MetricSrcFactory│ ──creates──► BaseMetricSrc (ABC)     │
    │  └────────────────┘                      │               │
    │                                          │               │
    │                                          ▼               │
    │                                    CsvMetricSrc          │
    │                                          │               │
    │                          ┌───────────────┴────────┐      │
    │                          │ Uses CSV Loaders:      │      │
    │                          │  • BaseRawCsvLoader    │      │
    │                          │                        │      │
    │                          │ Process:               │      │
    │                          │  1. Load *_results.csv │      │
    │                          │  2. Load ground truth  │      │
    │                          │  3. Match pred vs GT   │      │
    │                          └───────────────┬────────┘      │
    │                                          │               │
    │                                          ▼               │
    │                          Return: (preds_tensor, gt_tensor)
    │                                          │               │
    │                                          ▼               │
    │  ┌────────────────┐                                      │
    │  │ MetricFactory  │ ──creates──► TorchMetrics            │
    │  └────────────────┘                    │                 │
    │                                        │                 │
    │                                        ├──► FPS          │
    │                                        ├──► FPR          │
    │                                        ├──► Accuracy     │
    │                                        ├──► F1Score      │
    │                                        ├──► Precision    │
    │                                        └──► Recall (TPR) │
    │                                                          │
    │  Final Output: *__perf*.csv                              │
    └──────────────────────────────────────────────────────────┘

---

KEY INTERACTIONS:
================================================================================

1. CONFIG → METHODS:
   - Config.methodCfg selects which method class to instantiate
   - Config.modelCfg provides model path and class names
   - Config.inferCfg controls inference behavior (skip, save options)

2. METHODS → RESULTS:
   - BaseMethod has list of result_handlers (CsvRsProc, VideoInferRsProc)
   - After each frame inference, calls handler.handle_frame_results()
   - Handlers independently save CSV and/or video

3. RESULTS → METRICS:
   - CsvRsProc saves predictions to CSV files
   - CsvMetricSrc loads these CSV files + ground truth labels
   - Compares predictions vs ground truth
   - Passes to MetricFactory → computes accuracy, F1, FPS, etc.

4. SKIP PIPELINE (Optional in TempMethod):
   - Motion detector analyzes frame for changes
   - Rule system checks if motion regions contain fire/smoke
   - If no fire/smoke detected → skip inference (return dummy result)
   - If detected → use full frame → run model

5. VISUALIZATION PIPELINE:
   - VideoPipeline orchestrates multiple renderers
   - Each renderer draws different info (OSD, grid, blocks, ...)
   - Composites all layers → writes to output video

---

DATA FLOW EXAMPLE
================================================================================

Input: video.mp4
       ↓
[TempMethod] (in `src/methods`)
  ├─ Frame 1 → MotionDet → No motion → Skip → dummy result
  ├─ Frame 2 → MotionDet → Motion detected → do inference → pred=fire, probs=[0.9, 0.1]
  ↓
[Result Handlers] (in `src/results/`)
  ├─ CsvRsProc → video_results.csv (frame_idx, pred, probs, time)
  └─ VideoInferRsProc
       └─ VideoPipeline
            ├─ InferRsRenderer (draw pred label + FPS)
            ├─ GridRenderer (draw block grid)
            └─ BlockRuleRenderer (highlight fire blocks)
       → video_out.mp4
  ↓
[Metrics] (in `src/metrics/`)
  ├─ CsvMetricSrc loads video_results.csv + ground_truth
  ├─ Compare: pred vs GT for each frame
  └─ MetricFactory computes:
       • Accuracy: 0.95
       • F1-Score: 0.92
       • FPS: 30.5
       • FPR: 0.03

