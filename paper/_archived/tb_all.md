---
title: "Summary of Validation and Performance Experiments"
author: hahv
date: March 2026
documentclass: article
fontsize: 10pt
geometry:
  - a4paper
  - margin=0.5cm
---

## Component-Level Performance (Baselines) {#sec:comp-perf}

This section presents a comprehensive evaluation of the proposed
skip-module architectures ($A_1$, $A_2$). We compare our frame-level
skipping approach against standard lightweight alternatives ($M_1$,
$M_2$) and the heavy baseline detector.

Table \ref{tb:tb_no_skip_perf} assesses the intrinsic recall and
filtering capability of each method as a standalone component.

```{=latex}
\input{./4.table/tb_no_skip_perf.tex}
```

## Hyperparameter Optimization (Validation Set) {#sec:hyperparam}

Table \ref{tb:val_search} specifies the search space for the rule-based
skip-module parameters. The ranking and selection of the optimal
configuration ($\theta^*$) based on the validation set results are
detailed in Table \ref{tb:val_results}.

```{=latex}
\input{./4.table/tb_val_search.tex}
```

```{=latex}
\input{./4.table/tb_val_results.tex}
```

## End-to-End System Performance (Test Set) {#sec:e2e-perf}

The overall impact of integrating the skip modules into the full
detection pipeline is quantified in Table \ref{tb:e2e_perf}
(frame-level accuracy/latency), also comparing against the baseline
system without skipping.

```{=latex}
\input{./4.table/tb_e2e_perf.tex}
```

## Comparison with Other Temporal Methods {#sec:cmp-temporal}

Finally, Table \ref{tb:cmp_base_temp} compares our method against other
temporal processing techniques, evaluating both detection performance
and computational efficiency.

```{=latex}
\input{./4.table/tb_cmp_base_temp.tex}
```
