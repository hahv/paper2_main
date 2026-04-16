import sys

# +-----------------------------------------------------------------------+
# |                    fig_qualitative_failure.pdf                        |
# |                                                                       |
# |         (d) WRONGLY SKIPPED — Slow Smoke                             |
# |                    (e) FORCED RUN — Persistent Motion                |
# |                                                                       |
# |  Raw   +---------------------+    +---------------------+           |
# |        |                     |    |                     |           |
# |        | 💨 Barely visible   |    | 🚶 Walking person   |           |
# |        |    smoke onset      |    |    no fire/smoke    |           |
# |        |    early stage      |    |    continuous move  |           |
# |        |                     |    |                     |           |
# |        |   [RED border]      |    |  [ORANGE border]    |           |
# |        +---------------------+    +---------------------+           |
# |                                                                       |
# |  Mask  +---------------------+    +---------------------+           |
# |        |                     |    |                     |           |
# |        |  ░░░░░░░░░░░░░░░   |    | ████████████████    |           |
# |        |  (near dark —       |    | (saturated —        |           |
# |        |  accumulator        |    |  K_max reached,     |           |
# |        |  not triggered)     |    |  never resets)      |           |
# |        +---------------------+    +---------------------+           |
# |         s_t = 0  ✗ MISSED          s_t = 1  ✗ NO SAVINGS           |
# +-----------------------------------------------------------------------+

sys.path.append("E:/Dev/__halib")


from halib import *
from halib.system import filesys as fs
from halib.exp.viz.plot import PlotHelper as plth
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTFILE_NAME = "fig_qualitative_failure"

df = plth.get_img_grid_df(f"{CURRENT_DIR}/src")
df.columns = [f"sample_{i}" for i in range(len(df.columns))]
df.to_csv(f"{CURRENT_DIR}/plot_df.csv", index=False, encoding="utf-8", sep=";")


def fmt_row_label_func(x):
    # MAPPING_ROW_NAME = {
    #     "row_01_rgb": "<b>RGB</b>",
    #     "row_02_mask": "<b>ForeGround<br>Mask</b>",
    # }
    # return MAPPING_ROW_NAME[x]
    return ""


def fmt_col_label_func(x):
    # MAPPING_COL_NAME = {
    #     "sample_0": "SKIP — Static BG",
    #     "sample_1": "RUN — Fire",
    #     "sample_2": "RUN — Smoke",
    # }
    return ""


target_fmt = ["pdf"]
for fmt in tqdm(target_fmt):
    console.rule(f"Exporting {fmt} ...")
    outfile = f"{CURRENT_DIR}/{OUTFILE_NAME}.{fmt}"
    fig = plth.plot_image_grid(
        df,
        # save_path=outfile,
        img_width=300,
        img_height=300,
        img_stack_padding_px=5,
        img_stack_direction="horizontal",
        img_scale_mode="fit",
        fig_margin=dict(l=0, r=10, t=10, b=10),
        outline_color="#000000",
        outline_size=2,
        cell_margin_px=5,
        row_line_size=2,
        col_line_size=2,
        tickfont=dict(size=16, family="CMU Serif", color="black"),
        fig_extra_size=(0, 40),
        format_row_label_func=fmt_row_label_func,
        format_col_label_func=fmt_col_label_func,
        show=False,
    )

    # ! ROW LABELS
    fig = plth.add_canvas_padding(fig, pad_top=0, pad_bottom=70, pad_left=100, pad_right=0)

    ROW_NAMES = [
        r"<b><span style=\"color: black;\">Frame<br>Type</b>",
        "<b>RGB</b>",
        "<b>ForeGround<br>Mask</b>",
    ]
    # expand the figure's left margin to accommodate row labels
    row_label_x_pos_ls = [
        0.08,
        0.07,
        0.1,
    ]  # Keep x position constant for all row labels
    row_label_y_pos_ls = [0.95, 0.7, 0.35]  # Adjust y positions for row labels
    # add row labels on the left side of the grid
    for i, row_label in enumerate(ROW_NAMES):
        fig.add_annotation(
            x=row_label_x_pos_ls[i],
            y=row_label_y_pos_ls[i],
            xref="paper",
            yref="paper",
            text=row_label,
            showarrow=False,
            font=dict(size=16, family="CMU Serif"),
            xanchor="right",
            yanchor="middle",
        )

    # ! COLUMN LABELS
    annotations = [
        r"(a) WRONGLY SKIPPED <br> Slow Smoke (✗ MISSED)",
        r"(b) WASTED INFER <br> Persistent Motion (✗ NO SAVINGS)",
        r"(c) WRONG INFER <br> Properly Infer <br> But Wrongly Predicted <br> (✗ BIG MODEl ERROR)",
    ]
    BOLD_START = "<b>"
    BOLD_END = "</b>"
    CUSTOM_COLOR_START_TAG = '<span style="color: #a80319;">'
    CUSTOM_COLOR_END_TAG = "</span>"
    for i in range(len(annotations)):
        annotations[i] = (
            f"{CUSTOM_COLOR_START_TAG}{annotations[i]}{CUSTOM_COLOR_END_TAG}"
        )
    num_cols = len(annotations)
    x_pos_ls = [0.25, 0.55, 0.85]  # Adjust manually for 3 columns visually
    y_pos_ls = [0.04] * num_cols  # Keep y position constant for all annotations
    y_pos_ls[-1] = 0.01  # Adjust y position for the last annotation if needed

    for i, text in enumerate(annotations):
        pprint(f"Adding annotation: {text} at x={x_pos_ls[i]}, y={y_pos_ls[i]}")
        fig.add_annotation(
            x=x_pos_ls[i],
            y=y_pos_ls[i],
            xref="paper",
            yref="paper",
            text=text,
            showarrow=False,
            font=dict(size=16, family="CMU Serif"),
            xanchor="center",
            yanchor="bottom",
        )

    fig.write_image(outfile, scale=2)
    fs.open_file(outfile)
    # pprint_local_path(outfile, get_wins_path=True)
