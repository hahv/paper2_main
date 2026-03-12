# The main document describing the project (goals, progress, etc.)

## Task
Design a lightweight skip module for real-time fire and smoke detection in static camera surveillance systems.

### Problem Statement
Current deep learning (DL) pipelines are computationally inefficient because they process every video frame through heavy models (~50ms per frame), even for static scenes with no fire or smoke activity. The goal is to implement a fast pre-check mechanism that filters out irrelevant frames before running expensive inference, ensuring that indicators for *both* fire and smoke are considered.

**System Context:**
- **Current Workflow:** Read frame → Preprocess → DL inference (detect fire/smoke) → Alarm if detected
- **Proposed Workflow:** Read frame → Fast pre-check → Skip if negative *or* DL inference if positive → Alarm if detected

### Data Specifications
- **Input:** Video stream from static cameras (no camera movement or additional sensors).
- **Expected Output:** The skip module must correctly bypass frames without fire/smoke and trigger DL inference for frames with potential indicators.

### Requirements & Constraints
- Methods must be computationally lightweight for real-time processing.
- Crucially, the system must minimize false negatives (missing actual events) while maximizing skipped frames.
- Must identify indicators for both fire and smoke.

### Current Proposed Approaches

**Approach 1: Naive Block-Based Motion Analysis**
1. **Grid Formulation:** Divide the motion mask into an *N x N* grid and calculate the percentage of motion pixels per block.
2. **Active Block Detection:** Mark a block as "active" if motion exceeds a defined threshold.
3. **Skip Decision:** Skip the frame if there are no active blocks; otherwise, run DL inference.
4. **Output:** "Skip" or "DL Inference Output."

**Approach 2: Rule-Based Block-Based Motion Analysis**
1. **Grid Formulation:** Same as Approach 1.
2. **Active Block Detection:** Same as Approach 1.
3. **Rule-Based Skip Decision:**
    - Skip if no active blocks are found.
    - If an active block is detected, apply heuristic rules to check for fire/smoke indicators. Examples include:
        - **Color Rules:** Check for high red channel intensity (fire) or grayish colors (smoke).
        - **Temporal Consistency:** Analyze texture and color changes over time (e.g., flickering for fire, growing/dissipating for smoke).
        - **Additional Rules:** Other domain-specific heuristics.
    - Run DL inference if rules suggest potential fire/smoke; otherwise, skip the frame.
4. **Output:** "Skip" or "DL Inference Output."

## Current Progress

### Progress Summary
*   **Dataset:** Created a high-quality custom dataset of 150 HD videos (balanced 50/50/50 split) from static cameras.
*   **Implementation:** Developed two lightweight skip modules:
    *   *Approach 1:* Naive Block-Based Motion Analysis.
    *   *Approach 2:* Rule-Based Motion Analysis (color/texture heuristics).
*   **The "Villain":** The "BIG MODEL"—a heavy, highly accurate, but slow (50ms/frame) deep learning model trained on millions of images.
*   **The Goal:** Prove that your skip module makes the *system* faster without compromising the BIG MODEL's accuracy.

***

### Strategy: Choosing Baselines & Defining the Story
You are worried about baselines because you are comparing a "pre-filter" against "models." This is a category error. **You are not competing with M1, M2, or M3 on accuracy.** You are competing on **Efficiency vs. Accuracy Trade-off.**

**The Narrative Arc:**
1.  **The Status Quo (Baseline 0):** Running "BIG MODEL" on *every single frame*. This is accurate but computationally wasteful (100% load).
2.  **The Weak Competitor (M1/M2/M3):** Using a lightweight model *instead* of the BIG MODEL. These are fast but have poor accuracy (high False Negatives) because they were trained on small datasets (2k-5k images vs millions).
3.  **The Solution (Your Method):** Running "Skip Module + BIG MODEL." This retains the high accuracy of the BIG MODEL but approaches the speed of the lightweight models.

**How to handle M3 (Temporal/Voting):**
Since M3 uses temporal voting to reduce false alarms *after* inference, it is actually complementary, not a direct competitor to a *pre-inference* skip module. However, you can frame M3 as a "Post-Processing Optimization" and your method as "Pre-Processing Optimization." *Recommendation: Keep M3 as a discussion point or secondary comparison, but focus on M1/M2 as the primary "Lightweight Alternatives."*
