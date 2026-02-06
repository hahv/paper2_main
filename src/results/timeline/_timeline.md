# Timeline Visualization & Analysis Workflow

This workflow transforms raw video inference data into frame-level timeline visualizations and statistical reports. The process is divided into three key stages: **Input**, **Parsing (Logic)**, and **Configuration (Style)**.

### 1. Input Data Structure

The system accepts a DataFrame where each row represents a single video frame.

* **Required Columns:**
* `gt_label`: Ground Truth class (e.g., "Fire", "None").
* `method_X_label`: Prediction/Status columns for one or more methods (e.g., "Baseline", "SkipLogic").



**Example Input:**

| video | frame | gt_label | method1_label | method2_label |
| --- | --- | --- | --- | --- |
| vid_01 | 0 | Fire | Fire | Processed |
| vid_01 | 1 | None | Fire | Skipped |

---

### 2. Parsing Logic (`DataParser`)

Each method column is assigned a specific `TLParser` subclass. The parser's role is to translate raw predictions into **Performance State Labels** (e.g., transforming "Fire" vs "None" into "False Alarm").

* **Process:** The `parse()` method performs a vectorized comparison (e.g., using `np.select`) between the `method_col` and `gt_label` for every frame.
* **Output:** A NumPy array of state labels that strictly match the keys defined in the configuration.

**Example Logic:**

```python
def parse(self, df: pd.DataFrame, method_col: str) -> np.ndarray:
    # Logic: Define conditions for False Positive (FP) and False Negative (FN)
    return np.select(
        [
            (~is_gt_fire) & (is_pred_fire),  # Condition: FP
            (is_gt_fire) & (~is_pred_fire),  # Condition: FN
        ],
        ["False Alarm (FP)", "Miss (FN)"],   # Output Labels
        default="Correct"                    # Default Label
    )

```

---

### 3. Configuration & Visualization

A YAML configuration file governs how these State Labels are rendered and reported. It links the parser's logical output to visual properties.

**Config Structure (`timeline_cfg.yaml`):**

* **`legend_title`**: The display title for the timeline section.
* **`include_col_in_table`**: (Boolean) If `true`, adds a statistical breakdown (percentages) of each label to the summary report table.
* **`labels_colors`**: A dictionary mapping the *exact* State Labels from step 2 to hex colors.

**Example Config:**

```yaml
no_skip:
  legend_title: "2. NO SKIP (Baseline)"
  include_col_in_table: true
  labels_colors:
    Correct: "#2ecc71"           # Green
    False Alarm (FP): "#f1c40f"  # Gold
    Miss (FN): "#e74c3c"         # Red

```

---

### 4. Final Output Generation

The report generator combines the parsed data and configuration to produce:

1. **Visual Timelines:** Color-coded strips visualizing the sequence of events (e.g., a Red block for a "Miss" event).
2. **Summary Statistics:** For methods with `include_col_in_table: true`, the system calculates the percentage of frames for each state (e.g., "Miss (FN): 2.5%") and appends these metrics to the results table.

---