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
    │  Paper2Exp   │  ◄─── Experiment Orchestrator
    └──────┬───────┘
           │
           │ 1. Creates Method
           ▼
    ┌──────────────────────────────────────────────────────────┐
    │                   src/methods/                           │
    │                                                          │
    │  ┌─────────────────┐                                    │
    │  │  MethodFactory  │ ──creates──►  BaseMethod (ABC)    │
    │  └─────────────────┘                      │              │
    │                                            │              │
    │                        ┌───────────────────┼─────────────┐
    │                        ▼                   ▼             ▼
    │                  NoTempMethod       TempMethod   TempBaselineTptMethod
    │                  (frame-by-frame)   (with skip)  (temporal filtering)
    │                                           │
    │                                           │ uses
    │                                           ▼
    │  ┌──────────────────────────────────────────────────────┐
    │  │            Skip Pipeline (Optional)                  │
    │  │                                                      │
    │  │  BaseSkipProc ◄──creates── SkipProcFactory         │
    │  │       │                                              │
    │  │       ├──► BlockSkipProc   (rule-based detection)  │
    │  │       ├──► ProfSkipProc    (motion ROI cropping)   │
    │  │       ├──► NoSkipProc      (no optimization)       │
    │  │       └──► RandSkipProc    (random skip)           │
    │  │                                                      │
    │  │  Each uses:                                          │
    │  │    • Motion Detectors (FrameDiffDet, AccMotionDet)  │
    │  │    • Rules (FireBlockYCbCrRule, WaveletRule, etc.)  │
    │  └──────────────────────────────────────────────────────┘
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
    │       │                                                  │
    │       ├──► CsvRsProc                                     │
    │       │     └─ Saves: video_name_results.csv            │
    │       │        Columns: frame_idx, pred_label,           │
    │       │                 probs, logits, elapsed_time      │
    │       │                                                  │
    │       └──► VideoInferRsProc                              │
    │             │                                            │
    │             ├─ Creates: VideoPipeline                    │
    │             │   └─ Uses Multiple Renderers:              │
    │             │       • InferRsRenderer   (OSD: pred, fps) │
    │             │       • GridRenderer      (block grid)     │
    │             │       • BlockRuleRenderer (fire/smoke)     │
    │             │       • BlockProfRenderer (motion blocks)  │
    │             │                                            │
    │             └─ Subclasses:                               │
    │                 • VideoBlockSkipRsProc                   │
    │                 • FgmaskBlockSkipRsProc                  │
    │                   (visualize motion masks)               │
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
    │  │MetricSrcFactory│ ──creates──► BaseMetricSrc (ABC)   │
    │  └────────────────┘                      │               │
    │                                           │               │
    │                                           ▼               │
    │                                    CsvMetricSrc           │
    │                                           │               │
    │                          ┌────────────────┴───────┐      │
    │                          │                        │      │
    │                          │ Uses CSV Loaders:      │      │
    │                          │  • BaseCsvLoader (ABC) │      │
    │                          │     ├─ DFireCsvLoader  │      │
    │                          │     └─ UFireIndoorCsvLoader   │
    │                          │                        │      │
    │                          │ Process:               │      │
    │                          │  1. Load *_results.csv │      │
    │                          │  2. Load ground truth  │      │
    │                          │  3. Match pred vs GT   │      │
    │                          └────────────────────────┘      │
    │                                           │               │
    │                                           ▼               │
    │                          Return: (preds_tensor, gt_tensor)
    │                                           │               │
    │                                           ▼               │
    │  ┌────────────────┐                                      │
    │  │ MetricFactory  │ ──creates──► TorchMetrics           │
    │  └────────────────┘                                      │
    │           │                                              │
    │           ├──► FPS     (frames per second)              │
    │           ├──► FPR     (false positive rate)            │
    │           ├──► Accuracy                                 │
    │           ├──► F1Score                                  │
    │           ├──► Precision                                │
    │           └──► Recall (TPR)                             │
    │                                                          │
    │  Final Output: performance_metrics.csv                  │
    └──────────────────────────────────────────────────────────┘


================================================================================
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
   - If detected → crop ROI or use full frame → run model

5. VISUALIZATION PIPELINE:
   - VideoPipeline orchestrates multiple renderers
   - Each renderer draws different info (OSD, grid, blocks, rules)
   - Composites all layers → writes to output video


================================================================================
DATA FLOW EXAMPLE:
================================================================================

Input: video.mp4
       ↓
[TempMethod]
  ├─ Frame 1 → MotionDet → No motion → Skip → dummy result
  ├─ Frame 2 → MotionDet → Motion detected → Check rules
  │              ├─ FireBlockYCbCrRule → PASS
  │              └─ Crop ROI → Run CNN → pred="fire_smoke"
  ↓
[Result Handlers]
  ├─ CsvRsProc → video_results.csv (frame_idx, pred, probs, time)
  └─ VideoInferRsProc
       └─ VideoPipeline
            ├─ InferRsRenderer (draw pred label + FPS)
            ├─ GridRenderer (draw block grid)
            └─ BlockRuleRenderer (highlight fire blocks)
       → video_out.mp4
  ↓
[Metrics]
  ├─ CsvMetricSrc loads video_results.csv + ground_truth
  ├─ Compare: pred vs GT for each frame
  └─ MetricFactory computes:
       • Accuracy: 0.95
       • F1-Score: 0.92
       • FPS: 30.5
       • FPR: 0.03


================================================================================
DESIGN PATTERNS USED:
================================================================================

• Factory Pattern: MethodFactory, SkipProcFactory, MetricSrcFactory
• Strategy Pattern: BaseSkipProc (different skip strategies)
• Observer/Handler Pattern: BaseRsProc (multiple result handlers)
• Template Method: BaseMethod (defines inference workflow)
• Adapter Pattern: BaseCsvLoader (dataset-specific parsing)
• Composite Pattern: BaseRule (AnyRule, AllRule combining sub-rules)
• Pipeline Pattern: VideoPipeline + Renderers (sequential processing)


================================================================================
KEY FILES REFERENCE:
================================================================================

src/config.py              - Central configuration (Config, DatasetCfg, etc.)
src/exp.py                 - Paper2Exp (orchestrates entire experiment)

src/methods/
  base_method.py           - BaseMethod (inference template)
  no_temp_method.py        - Basic frame-by-frame inference
  temp_method.py           - Inference with skip optimization
  skip/base_skip_proc.py   - Skip strategy interface
  skip/block_skip_proc.py  - Rule-based skip (fire/smoke detection)
  skip/prof_skip_proc.py   - ROI-based skip (motion cropping)

src/results/
  base_rs_proc.py          - Result handler interface
  csv_rs_proc.py           - Save predictions to CSV
  video_infer_rs_proc.py   - Save visualization videos
  viz/base_renderer.py     - Renderer interface
  viz/video_pipeline.py    - Video writing pipeline

src/metrics/
  base_metric_src.py       - Metric data source interface
  csv_metric_src.py        - Load CSV predictions + ground truth
  custom_metrics.py        - Custom metrics (FPS, FPR)
  loaders/                 - Dataset-specific CSV parsers


================================================================================
END OF DIAGRAM
================================================================================
