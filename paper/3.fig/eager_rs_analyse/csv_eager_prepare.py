"""
csv_prepare.py

Generates two CSV files for eager_timeline visualization:

  eager_timeline_caseA.csv  — CASE A: False-prone scene (false-alarm failure)
    Video: aihub__lb_none__0175 (background-only, ground truth = None)
    Range: frames 1-50
    Story: Skip module correctly skips static frames -> periodic Nchk check forces
           inference on frame 34 -> DL falsely predicts SmokeOnly (Wfire=1) ->
           Eager mode immediately activates and stays stuck for the entire video.

  eager_timeline_caseB.csv  — CASE B: Eager mode rescues missed detections
    Video: aihub__lb_smoke__0208 (slow-developing smoke, ground truth = SmokeOnly)
    Range: frames 55-115
    Story: Slow smoke has minimal inter-frame motion -> skip module keeps skipping ->
           Skip-only would miss detections. Around frame 71 the skip module passes
           a frame -> DL detects smoke -> Eager activates -> all subsequent frames
           forwarded to DL -> smoke correctly detected throughout.

Output CSV format (semicolon-separated):
  frame;classifier;prob;skipped;eager_mode
    frame       : frame index (int)
    classifier  : 1 = positive (fire/smoke), 0 = negative or skipped
    prob        : SmokeOnly softmax probability (0.0-1.0); 0.0 when skipped
    skipped     : 1 = skip module blocked this frame, 0 = inference was run
    eager_mode  : 1 = Eager mode active this frame, 0 = Normal mode
"""

import pandas as pd
import ast

BASE = "./paper/3.fig/eager_rs_analyse/raw_csv/"


def parse_probs(probs_str):
    """Extract SmokeOnly probability (class index 2) from probs column string."""
    try:
        probs = ast.literal_eval(str(probs_str))
        return float(probs[2])
    except Exception:
        return 0.0


def build_csv(df_eager, df_skip, frame_start, frame_end):
    """
    Build timeline CSV for a given frame range.

    skip decision  -> from df_skip  (pred_label == 'skipped' -> skipped=1)
    eager_mode     -> from df_eager (eager_mode column)
    classifier/prob -> from df_eager when inference was run; 0 when skipped
    """
    sub_eager = df_eager[
        (df_eager["frame_idx"] >= frame_start) & (df_eager["frame_idx"] <= frame_end)
    ].set_index("frame_idx")
    sub_skip = df_skip[
        (df_skip["frame_idx"] >= frame_start) & (df_skip["frame_idx"] <= frame_end)
    ].set_index("frame_idx")

    rows = []
    for fi in range(frame_start, frame_end + 1):
        e_row = sub_eager.loc[fi] if fi in sub_eager.index else None
        s_row = sub_skip.loc[fi] if fi in sub_skip.index else None

        # Skip decision: was this frame blocked by the skip module?
        skipped = 1 if (s_row is not None and s_row["pred_label"] == "skipped") else 0

        # Eager mode active?
        eager = 1 if (e_row is not None and e_row["eager_mode"]) else 0

        # Classifier output - use eager-run result (the live system output)
        if (
            e_row is not None
            and pd.notna(e_row.get("pred_label"))
            and e_row["pred_label"] not in ("skipped",)
        ):
            classifier = 1 if e_row["pred_label"] in ("Fire", "SmokeOnly") else 0
            prob = parse_probs(e_row["probs"])
        else:
            classifier = 0
            prob = 0.0

        rows.append(
            {
                "frame": fi,
                "classifier": classifier,
                "prob": round(prob, 4),
                "skipped": skipped,
                "eager_mode": eager,
            }
        )

    return pd.DataFrame(rows)


# ── Load raw results ──────────────────────────────────────────────────────────

df_a_eager = pd.read_csv(BASE + "aihub__lb_none__0175_results_eager.csv", sep=";")
df_a_skip = pd.read_csv(BASE + "aihub__lb_none__0175_results_skipOnly.csv", sep=";")
df_b_eager = pd.read_csv(BASE + "aihub__lb_smoke__0208_results_eager.csv", sep=";")
df_b_skip = pd.read_csv(BASE + "aihub__lb_smoke__0208_results_skipOnly.csv", sep=";")

# ── CASE A : frames 1-50 ─────────────────────────────────────────────────────
df_case_a = build_csv(df_a_eager, df_a_skip, frame_start=1, frame_end=50)
df_case_a.to_csv(BASE + "_eager_timeline_failure.csv", sep=";", index=False)
print(f"[CASE A] Saved eager_timeline_failure.csv  ({len(df_case_a)} rows)")

# ── CASE B : frames 55-115 ───────────────────────────────────────────────────
df_case_b = build_csv(df_b_eager, df_b_skip, frame_start=55, frame_end=115)
df_case_b.to_csv(BASE + "_eager_timeline_success.csv", sep=";", index=False)
print(f"[CASE B] Saved eager_timeline_success.csv  ({len(df_case_b)} rows)")
