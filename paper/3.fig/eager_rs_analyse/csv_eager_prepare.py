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


import os
import cv2


def extract_frame(video_path, frame_idx, out_dir=BASE):
    """
    Extract frame index from video filename. Frame index start from 1
    The output frame will be named based on the video filename (e.g., video.mp4 -> video.png)
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Get the base filename without extension
    video_filename = os.path.basename(video_path)
    base_name, _ = os.path.splitext(video_filename)

    output_path = os.path.join(out_dir, f"{base_name}.png")

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None

    # OpenCV frame indices are 0-based, so subtract 1 from the 1-based index
    target_frame = frame_idx - 1

    # Set the video position to the target frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    # Read the frame
    ret, frame = cap.read()

    if ret:
        # Save the frame
        cv2.imwrite(output_path, frame)
        print(f"Successfully extracted frame {frame_idx} to {output_path}")
    else:
        print(f"Error: Could not read frame {frame_idx} from {video_path}")

    # Release the video capture object
    cap.release()

    return output_path


# ── Load raw results ──────────────────────────────────────────────────────────

df_a_eager = pd.read_csv(BASE + "aihub__lb_smoke__0208_results_eager.csv", sep=";")
df_a_skip = pd.read_csv(BASE + "aihub__lb_smoke__0208_results_skipOnly.csv", sep=";")

df_b_eager = pd.read_csv(BASE + "aihub__lb_none__0175_results_eager.csv", sep=";")
df_b_skip = pd.read_csv(BASE + "aihub__lb_none__0175_results_skipOnly.csv", sep=";")

# ── CASE Success : frames 10-55 ───────────────────────────────────────────────────
df_case_success = build_csv(df_a_eager, df_a_skip, frame_start=10, frame_end=55)
df_case_success.to_csv(BASE + "_eager_timeline_success.csv", sep=";", index=False)
print(f"[CASE B] Saved eager_timeline_success.csv  ({len(df_case_success)} rows)")

extract_frame(
    BASE + "aihub__lb_smoke__0208.mp4", frame_idx=175
)  # example frame from the slow-developing smoke video
os.rename(
    BASE + "aihub__lb_smoke__0208.png",
    BASE + "_eager_timeline_success_example.png",
)

# ── CASE Failure : frames 20-50 ─────────────────────────────────────────────────────
df_case_failure = build_csv(df_b_eager, df_b_skip, frame_start=20, frame_end=50)
df_case_failure.to_csv(BASE + "_eager_timeline_failure.csv", sep=";", index=False)
print(f"[CASE A] Saved eager_timeline_failure.csv  ({len(df_case_failure)} rows)")
extract_frame(BASE + "aihub__lb_none__0175.mp4", frame_idx=20)  # example frame
os.rename(
    BASE + "aihub__lb_none__0175.png",
    BASE + "_eager_timeline_failure_example.png",)