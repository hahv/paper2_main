# Skip Module Hyperparameter Selection

| Symbol | Meaning | Code name |
| --- | --- | --- |
| $R_{\text{base}}$ | Baseline recall measured with skip disabled | `context["baseline"]["recall"]` / `r_base` |
| $\mathrm{FAR}_{\text{base}}$ | Baseline false alarm rate | `context["baseline"]["far"]` / `far_base` |
| $\delta_R$ | Max tolerable absolute recall drop | `context["delta_r"]` |
| $S(\theta)$ | Negative-frame skip ratio for candidate $\theta$ | `df["skip_ratio"]` |
| $R(\theta)$ | Measured recall for candidate $\theta$ | `df["recall"]` |
| $\mathrm{FAR}(\theta)$ | False alarm rate for candidate $\theta$ | `df["far"]` |
| $\Delta \widetilde{\mathrm{FAR}}(\theta)$ | Normalized FAR reduction vs. baseline | `far_reduction_norm` |
| $\widetilde{R}(\theta)$ | Recall-retention term | `recall_retention` |
| $w_S, w_F, w_R$ | Objective weights for skip ratio, FAR reduction, recall retention | `w_s`, `w_f`, `w_r` |
| $\text{score}(\theta)$ | Weighted composite score | `Combined_Score` |

```
+-----------------------------+
|        START (VAL SET)      |
|  Dataset D_val, Detector M  |
+-------------+---------------+
              |
              v
   +--------------------------+
   | Evaluate baseline        |
   | skip=None, get R_base &  |
   | FAR_base                 |
   +-------------+------------+
                 |
                 v
   +--------------------------+
   | Initialize best_score=-∞ |
   | best_theta=None          |
   +-------------+------------+
                 |
                 v
   +--------------------------+
   | For each theta in Θ      |
   +-------------+------------+
                 |
                 v
   +--------------------------+
   | Build SkipModule(theta)  |
   +-------------+------------+
                 |
                 v
   +--------------------------+
   | Evaluate pipeline with   |
   | this skip module → get   |
   | R(theta), FAR(theta),    |
   | S(theta)                 |
   +-------------+------------+
                 |
                 v
   +--------------------------+
   | HARD CONSTRAINT CHECK    |
   | R(theta) >= R_base - δ_R?|
   +------+------+-----------+
          |      |
          |No    |Yes
          |      v
          |  +---------------------------+
          |  | Compute derived metrics:  |
          |  | ΔFAR~ = max(0, (FAR_base -|
          |  | FAR(theta))/FAR_base)     |
          |  | R~ = 1 - (R_base - R(theta)) / δ_R |
          |  +-------------+-------------+
          |                |
          |                v
          |     +-----------------------+
          |     | Weighted score        |
          |     | score = w_S * S +     |
          |     |          w_F * ΔFAR~ +|
          |     |          w_R * R~     |
          |     +-------------+---------+
          |                   |
          |                   v
          |     +-----------------------+
          |     | score > best_score?   |
          |     +------+----------------+
          |            |
          |            |Yes
          |            v
          |  +---------------------------+
          |  | best_score = score        |
          |  | best_theta = theta        |
          |  +---------------------------+
          |
          v
   +--------------------------+
   | NEXT theta in Θ          |
   +-------------+------------+
                 |
           (loop until done)
                 |
                 v
   +--------------------------+
   | RETURN best_theta        |
   +--------------------------+
```

- Gray arrows indicate the main loop across candidate parameter sets.
- The hard constraint ensures only recall-safe configurations are scored.
- Weighted scoring mirrors Algorithm 1 (docs/param_search.tex).
