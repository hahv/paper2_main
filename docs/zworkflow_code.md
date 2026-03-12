# Workflow Diagram for zbin/run_multi.py and related components
┌─────────────────────────────────────────────────────────────────────────────┐
│                          zbin/run_multi.py  (Entrypoint)                    │
│                                                                             │
│  RunOptimArgs (CLI)                                                         │
│   --base_yaml   --sweep_yaml   --clean_slack                                │
│         │               │                                                   │
│         └───────┬────────┘                                                  │
│                 ▼                                                            │
│       ParamGen.from_files()                                                 │
│       .expand()  ──► ls_run_dicts (list of merged YAML dicts)               │
│                 │                                                            │
│                 ▼                                                            │
│   Config.from_custom_yaml_file_or_str()  ──► initial_ls_run_cfgs            │
│                 │                                                            │
│   ┌─────────────▼──────────────────────────┐                               │
│   │  Optim Expansion Loop                   │                               │
│   │  get_opt_cfg(method_name)               │                               │
│   │    └─► if found: ParamGen expands       │                               │
│   │         optim params & merges into cfg  │                               │
│   └─────────────┬──────────────────────────┘                               │
│                 ▼                                                            │
│         all_optim_run_cfgs  (final list of Config objects)                  │
│                 │                                                            │
│   ┌─────────────▼──────────────────────────┐                               │
│   │  Main Run Loop  (for each Config)       │                               │
│   │  WandbLogger ◄─ Config.get_wandb_logger │                               │
│   │  Paper2Exp(config, wandb_logger)        │                               │
│   │  single_exp.run_exp()        ───────────┼─────────┐                    │
│   │  wandb_logger.log_hyperparams()         │         │                    │
│   │  wandb_logger.experiment.finish()       │         │                    │
│   └─────────────────────────────────────────┘         │                    │
└───────────────────────────────────────────────────────┼────────────────────┘
                                                         │
┌────────────────────────────────────────────────────────▼──────────────────┐
│                  src/exp.py  ─  Paper2Exp (extends BaseExp)                │
│                                                                            │
│  run_exp()                                                                 │
│   ├─► shouldSkipExp?  ──yes──► return  (skip if same cfg already run)      │
│   ├─► init_general()       seed_everything()                               │
│   ├─► prepare_dataset()    resolve video_dir_path from DatasetCfg          │
│   ├─► prepare_metrics()    MetricFactory ──► TorchMetricsBackend           │
│   ├─► config.save_to_outdir()  (save __config.yaml, __method_cfg.yaml)     │
│   ├─► exec_exp()  ──────────────────────────────────────┐                 │
│   │                                                      │                 │
│   └─► calc_perfs()  (for each mode)                      │                 │
│        ├─► compute accuracy, F1, FPS, FPR ...            │                 │
│        ├─► save to .csv  (PERF_FILE_PREFIX + cfg_name)   │                 │
│        └─► wandb_logger.log_metrics()                    │                 │
└──────────────────────────────────────────────────────────┼─────────────────┘
                                                           │
┌──────────────────────────────────────────────────────────▼─────────────────┐
│                src/exp.py  ─  exec_exp()                                   │
│                                                                            │
│  MethodFactory.create_method(config)                                       │
│   ├─► resolve method class from  src.methods.<name>                        │
│   ├─► build rs_handlers:                                                   │
│   │    CsvRsProc  (if save_csv_results)                                    │
│   │    VideoInferRsProc / FgmaskBlockSkipRsProc  (if save_video_results)   │
│   └─► method_cls(cfg, rs_handlers)                                         │
│                          │                                                  │
│  method.infer_video_dir(video_dir_path)  ───────────────┐                 │
│  method.prepare_metric_src()  ◄─── BaseMetricSrc ◄──────┘                 │
└────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│           src/methods/  ─  Method Hierarchy                                  │
│                                                                              │
│  BaseMethod (ABC)                                                            │
│   ├── infer_frame()  (abstract)                                              │
│   ├── infer_video_dir()  ──► parallel via ProcessPoolExecutor                │
│   │     for each video:                                                      │
│   │       rs_handler.before_video()                                          │
│   │       for each frame:                                                    │
│   │         frame_rs = infer_frame(frame, idx)                              │
│   │         rs_handler.handle_frame_results(frame, frame_rs)                │
│   │       rs_handler.after_video()                                           │
│   └── prepare_metric_src()  ──► MetricSrcFactory                            │
│                                                                              │
│  NoTempMethod(BaseMethod)          TempMethod(NoTempMethod)                 │
│   infer_frame():                    infer_frame():                           │
│    _pre_process_frame()              1. skip_proc.should_skip(idx, frame)   │
│    model(frame) ──► logits           │   ├─ yes → get_dummy_result()        │
│    softmax ──► probs                 │   └─ no  ──────────────────┐         │
│    argmax ──► predLabel              2. skip_proc.prepare_infer_input()      │
│    return dict                       3. super().infer_frame()  ◄──┘          │
│    {logits, probs,                   4. merge meta_data into result          │
│     predLabelIdx, predLabel}         return enriched result dict             │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│           src/methods/skip/  ─  Skip Processor Hierarchy                     │
│                                                                              │
│  SkipProcFactory.create_skip_proc(config)                                    │
│   └─► loads class from config.methodCfg.extra_cfgs["skip_proc"]["name"]     │
│                                                                              │
│  BaseSkipProc (ABC)                                                          │
│   ├── should_skip(frame_idx, frame) → (bool, meta_data)   (abstract)        │
│   ├── prepare_infer_input(frame, meta_data) → frame                         │
│   ├── get_dummy_result(class_names) → dict                                  │
│   └── motion_det: BaseMotionDet  (optional, from params["motion"])           │
│                                                                              │
│  Concrete Skip Procs:                                                        │
│   TrivialSkipProc          ─ never skips                                    │
│   MotionOnlyBlockSkipProc  ─ skip if block has no motion                    │
│   RuleBasedBlockSkipProc   ─ skip if block satisfies rule set               │
│                                                                              │
│  Motion Detectors (src/methods/skip/motion/):                               │
│   FrameDiffDet   ─ frame-difference threshold                               │
│   AccMotionDet   ─ accumulated motion with decay                            │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│           src/results/  ─  Result Handler Hierarchy                          │
│                                                                              │
│  BaseRsProc (ABC)                                                            │
│   before_video() / handle_frame_results() / after_video()                   │
│        │                                                                     │
│   ├─ CsvRsProc          ──► writes inference CSV per video                  │
│   ├─ VideoInferRsProc   ──► writes annotated output video                   │
│   ├─ FgmaskBlockSkipRsProc ─► renders block-level skip mask overlay         │
│   └─ VideoBlockSkipRsProc  ─► block skip visualization pipeline             │
│                                                                              │
│  timeline/TlReportGen  ──► generates timeline report from profiler data     │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│           src/config.py  ─  Configuration Data Model                         │
│                                                                              │
│  Config                                                                      │
│   ├── general: GeneralCfg       (seed, outdir, skip_exp_if_exists, WandB)   │
│   ├── inferCfg: InferConfig     (workers, save_csv, save_video, profiler)   │
│   ├── modelCfg: ModelConfig     (base_model, model_path, class_names)       │
│   ├── dbset_selector  ──► dbsetCfg: DatasetCfg   (dir_path, gt pattern)    │
│   ├── metric_selector ──► metricCfg: MetricSetCfg (metric_names)            │
│   └── method_selector ──► methodCfg: MethodCfg   (name, extra_cfgs)        │
│                                      └── extra_cfgs["skip_proc"]            │
│                                            ├── name  (skip proc class)      │
│                                            └── params (motion cfg, etc.)    │
└──────────────────────────────────────────────────────────────────────────────┘