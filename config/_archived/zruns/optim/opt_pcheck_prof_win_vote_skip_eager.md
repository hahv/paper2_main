# Justify `window_size >= fd_period` in `temp_method_pcheck_prof_win_vote_skip_eager`


To justify why `window_size >= fd_period` is the safer configuration, we need to look at how these two parameters represent **Time Blindness** versus **Time Patience**.

Here is the technical justification for why violating this rule breaks down video tracking:

### 1. The Concepts: Blindness vs. Patience

* **`fd_period` is your Blind Spot:** When the system is idle, this is the maximum number of frames you are completely blind. (e.g., if `fd_period=30`, you are blind for up to 30 frames).
* **`window_size` is your Patience:** When the system is tracking a fire, this is how many consecutive "empty" frames it must see before it decides the fire is completely gone.

### 2. The Danger Scenario (`window_size < fd_period`)

Imagine a real fire starts, but there is heavy smoke rolling in front of it, or the AI model's confidence just dips for a split second.

* **Setup:** `fd_period = 30` (1 second), `window_size = 10` (0.33 seconds).
* **Frame 30:** The system wakes up, detects the fire! It switches to Eager Mode.
* **Frames 31 to 35:** It successfully tracks the fire frame-by-frame.
* **Frames 36 to 45:** A person walks in front of the camera, blocking the fire for exactly 10 frames.
* **The Failure:** Because the `window_size` is only 10, the system's patience runs out instantly. It thinks, *"I haven't seen fire in 10 frames, it must be gone forever."* It drops out of Eager Mode and resets its sleep timer.
* **Frames 46 to 75:** The person has walked away, the fire is raging on screen, but **your system is now asleep for the next 30 frames** (`fd_period`).

By having patience shorter than your blind spot, a tiny 10-frame occlusion resulted in a massive 30-frame loss of tracking.

### 3. The Safe Scenario (`window_size >= fd_period`)

Now, let's make `window_size = 30` (Patience $\ge$ Blind Spot).

If that exact same occlusion happens (the person blocking the fire for 10 frames), the history window still contains positive detections from right before the person walked by. The system *waits* for the person to pass. It stays in Eager Mode.

By setting `window_size >= fd_period`, you enforce a logical guarantee: **The system will never go to sleep unless the fire has been entirely absent for a duration longer than your standard sleep cycle.** ### The Impact on Downstream Systems
If you allow the "Stuttering Trap" (`window_size < fd_period`), the consequences ripple through your entire pipeline:

1. **Broken Object Tracking:** Bounding boxes will constantly disappear and reappear with new Object IDs, ruining trajectory algorithms.
2. **Alert Spam:** If your system sends an API alert every time a "new" fire is detected, a single flickering fire could trigger 10 different alerts in 10 seconds because the system keeps falling asleep and re-waking.
3. **Bad User Experience:** If a human operator is watching the stream, the UI overlays will flash violently instead of providing a smooth, confident bounding box.

Therefore, keeping `window_size >= fd_period` acts as a crucial debouncing filter, ensuring temporal continuity in your predictions.