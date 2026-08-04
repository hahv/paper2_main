<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# so you know that I try to solve is that:

+ skip the FAR-prone static scene.
+ use eager mode to recover the recall (but still have good skip rate).

I'm get loss right now. based on what we discuess so far. Summary the current situation, and predict what I'm trying to do right now?

Good time to step back and reorient. Let me summarise everything cleanly.[^1][^2]

***

## The Core Problem You Are Trying to Solve

```
Two goals that pull in opposite directions:

GOAL 1 — Skip FAR-prone static scenes
          → no motion = skip → FAR improves ✅
          → but slow smoke also has no motion → recall drops ❌

GOAL 2 — Recover recall on slow/settled smoke
          → probe more often → recall improves ✅
          → but probing also hits static FAR-prone scenes → FAR returns to baseline ❌
```

Every method you have tried hits this same wall.[^1][^2]

***

## What You Have Tried So Far

```
Attempt 1 — Skip Only (AccMotionDet)
  Result: FAR ↓ to 0.00136 ✅  Recall ↓ to 94.38% ❌  Skip ~35%
  Problem: slow smoke has no motion → wrongly skipped

Attempt 2 — Eager Mode (n_chk timer + state machine)
  Result: Recall ↑ back to ~95.5% ✅  FAR ↑ back to 0.00251 ❌
  Problem: eager probes static FAR-prone scenes too →
           cannot tell them apart from slow smoke at probe time

Attempt 3 — tau_dark (haze proxy)
  Result: rescued some smoke frames but net frame gain negative
  Problem: white smoke and clean bright scenes both give low tau →
           cannot separate them reliably
```

All three attempts confirm the same structural barrier: **the DL model and all lightweight signals cannot distinguish slow smoke from static FAR-prone scenes on a single frame or simple global metric.**[^2][^1]

***

## What You Are Trying to Do Right Now

Based on our full discussion, your current idea is:[^1]

> **Use temporal drift as a pre-condition for eager entry — not just DL confidence alone.**

Specifically:

```
Current eager entry condition (single):
    DL = Fire/Smoke → enter eager
    Problem: static FAR-prone scene also gives DL = Fire/Smoke

Your new idea (dual condition):
    DL = Fire/Smoke   AND   scene was actually drifting over time
         ↓                        ↓
    (model says yes)         (SlowMotionDet / EMA fgmask says yes)
         ↓
    THEN enter eager

Static FAR-prone scene:
    DL = Fire/Smoke ✅   BUT   no drift ❌ → do NOT enter eager → FAR stays low

Real slow smoke:
    DL = Fire/Smoke ✅   AND   drift detected ✅ → enter eager → recall recovered
```


***

## Honest Assessment of Where This Gets You

```
Best case:
  Slow smoke that spreads globally → drift detectable → dual condition works ✅
  FAR-prone static scene → no drift → blocked from eager ✅

Uncertain case:
  Very early faint smoke (thin veil, <5% scene coverage)
  → drift too small to detect
  → dual condition not met
  → still missed ⚠️

Worst case (edge):
  Lighting change (light turns on/off, sun moves)
  → drift detected falsely
  → dual condition met on non-smoke scene
  → enters eager unnecessarily ⚠️
```


***

## The Honest Bottom Line

You are trying to **find a temporal signal that is present in slow smoke but absent in static FAR-prone scenes**, so that eager mode entry becomes selective rather than blind.[^1][^2]

This is a genuinely good idea and worth implementing.
The most promising low-param version is **Option 4 — EMA of fgmask pixel ratio** — because it reuses existing computation, adds zero parameters, and has a natural meaning: sub-threshold motion accumulating over time.[^3]

But be honest with yourself about the scope: this is a **partial fix** for the most common settled-smoke case, not a complete solution to the structural tradeoff.[^4][^2]
Present it as "a targeted improvement over blind eager mode" — that is a valid and honest thesis contribution.[^4]

<div align="center">⁂</div>

[^1]: https://www.perplexity.ai/search/5f8a24c9-1ab1-4226-b457-2950fbb11317

[^2]: https://www.perplexity.ai/search/1da0e8d3-9aa9-4b2b-8b55-372ec3818b04

[^3]: Summary.md

[^4]: https://www.perplexity.ai/search/7b9906e0-e16c-4505-bcbb-e24835ed20bb

