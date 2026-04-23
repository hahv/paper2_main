<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Ok give me the reframe the problem text (in details. Initial context. what is the problem with current appraoch, what we want to archive, ...)

Here is the full problem reframing.

***

## Context

The system is a fire and smoke detection pipeline running on video streams. The core model is a DL classifier that takes a video frame and outputs a predicted label (Fire, SmokeOnly, None) with a confidence probability. Running DL on every frame is expensive and also wasteful — most frames in a surveillance video are uninformative.

To reduce unnecessary inference, a **skip module** is added as a front-gate. The skip module uses AccMotionDet, a temporal accumulation-based motion detector, to produce a **binary motion signal** (motion = 0 or 1) per frame. The skip decision is simple: if motion is detected, run DL; if no motion is detected, skip the frame entirely and return a dummy "skipped" result.[^1][^2]

The motion detector itself is not a naive frame-differencer — it uses pixel-level temporal accumulation with decay to suppress flicker and transient noise before thresholding to a binary output. So `motion = 1` already means "sustained meaningful pixel change has been detected," not just a one-frame flicker.[^1]

***

## Why the skip module helps in the baseline

In the **NoTemp baseline (Case X)**, DL runs on every frame. The DL model has weaknesses — on certain static scenes (e.g. bright sunlight, reflections, steam), it occasionally produces false alarms even with no fire present. Because DL runs on every frame, these false alarm frames contribute directly to a high FAR.

In the **skip-only case (Case A)**, the motion gate accidentally suppresses many of these false alarms. The reason is structural: false-alarm-prone static scenes have **no motion**, so the skip module never sends those frames to DL. FAR drops significantly because DL simply never sees the frames it would get wrong. However, **recall also drops** for the same structural reason: slow smoke or early-stage fire that has not yet produced visible motion also has `motion = 0`, so it gets skipped too.[^2][^1]

***

## The core structural problem

The motion signal is binary, and the skip module treats `motion = 0` as a single class. But in reality, `motion = 0` contains two fundamentally different types of frames:

```
motion = 0
  ├── Type A: truly static scene, no fire/smoke
  │           → CORRECT to skip (reduces wasted inference, suppresses false alarms)
  │
  └── Type B: slow fire or early smoke, visually present but sub-threshold for motion
              → WRONG to skip (misses real events, hurts recall)
```

The current skip module cannot distinguish between these two. It treats all `motion = 0` frames identically and skips all of them.[^2][^1]

This creates an **irresolvable tradeoff** in the current design:

- If you skip all no-motion frames → FAR reduces, but recall drops on slow smoke.
- If you run DL on all no-motion frames → recall recovers, but FAR climbs back toward baseline (because static false-alarm scenes are also no-motion).

Any policy that overrides the skip on a large fraction of no-motion frames will tend to restore FAR toward the NoTemp baseline, because it reintroduces exactly the frames the skip module was protecting against.[^2]

***

## What went wrong with the eager mode approach

The eager mode attempt tried to solve the recall problem by entering a "run DL on every frame" state after a fire detection, and using a periodic forced check to catch slow-smoke cases during no-motion stretches.[^2]

The problem is that both of these mechanisms violate the core principle: they run DL on **large numbers** of no-motion frames without evidence that those frames are Type B (real slow smoke) rather than Type A (static false-alarm scene).[^2]

Specifically:

- The **periodic forced check** fires on a no-motion frame at a fixed cadence regardless of scene content. On static false-alarm-prone scenes, this produces a false alarm → triggers eager mode → DL runs on every frame → the same static scene keeps producing false alarms → FAR climbs back to baseline. The system enters a cycle that never resolves.[^2]
- The **eager mode itself** amplifies the problem once triggered, because it applies globally to all no-motion frames with no selectivity.[^2]

The root cause is that neither mechanism has any way to distinguish Type A from Type B no-motion frames. They just blindly re-enable DL, which is structurally equivalent to removing the skip module for those periods.[^1][^2]

***

## What we want to achieve

The goal is a skip policy that achieves three properties simultaneously:

**1. Keep most no-motion frames skipped**
The default behavior for `motion = 0` should remain skip. This is what keeps FAR below baseline and what keeps compute savings meaningful. Skipping static no-motion frames is not a bug — it is the feature.[^2]

**2. Recover some recall on slow smoke / no-motion fire**
Allow a small, selective subset of no-motion frames to be run through DL. This should be just enough to catch some Type B frames (slow smoke events) that would otherwise be entirely invisible to the system. The recall recovery does not need to be full — it should be better than pure skip-only, but it is acceptable for it to remain below baseline.[^1][^2]

**3. Do not amplify false alarms from Type A frames**
Any mechanism that runs DL on no-motion frames must not open up large sustained windows of DL inference on those frames. A single probe that returns a false alarm should not unlock many subsequent DL calls, because that restores baseline FAR behavior. The escape hatch for Type B must be **bounded and not self-reinforcing**.[^2]

***

## Design principle

The fundamental principle that should guide any solution is:

> **No-motion frames are skip-by-default. Type B recovery should come from sparse, bounded, non-self-reinforcing sampling of no-motion frames — not from persistent override modes.**

In practice this means:

- Any probing of no-motion frames should have a fixed, short duration.
- A positive result from a no-motion probe should not trigger extended DL runs, because a positive result is equally consistent with Type A (false alarm) or Type B (real slow smoke).
- The expected fraction of no-motion frames sent to DL should remain small, so that even if some probes land on Type A scenes, the total FAR contribution stays well below baseline.[^1][^2]

***

## Summary of the design space

| Behavior | Current skip-only (A) | Eager mode (B) | Target |
| :-- | :-- | :-- | :-- |
| motion=1 frames | Run DL | Run DL | Run DL |
| motion=0 default | Always skip | Sometimes override | Usually skip |
| motion=0 exception | None | Periodic + persistent | Sparse + bounded |
| FAR vs baseline | ↓ | ≈ baseline | ↓ (but less than A) |
| Recall vs baseline | ↓↓ | ≈ baseline | Between A and baseline |
| False alarm amplification | None | Yes (eager loop) | None |

The right solution lives in the bottom-right of this tradeoff space: **mostly respect the skip module's motion gate, add only a small sparse probing budget for no-motion frames, and ensure that any probe result — positive or negative — does not permanently change the skip policy.**[^1][^2]

<div align="center">⁂</div>

[^1]: acc_motion_det-2.tex

[^2]: skip_module_period_check_full.tex

