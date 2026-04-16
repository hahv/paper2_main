import sys

# +-----------------------------------------------------------------------+
# |                    fig_qualitative_success.pdf                        |
# |                                                                       |
# |        (a) SKIP — Static BG    (b) INFER — Fire    (c) INFER — Smoke   |
# |                                                                       |
# |  Raw   +----------------+    +------------+    +------------+        |
# |        |                |    |            |    |            |        |
# |        |  Empty room    |    | 🔥 Flames  |    | 💨 Haze    |        |
# |        |  No change     |    | Flickering |    | Diffusing  |        |
# |        |                |    |            |    |            |        |
# |        | [GREEN border] |    |[BLUE border]    |[BLUE border]        |
# |        +----------------+    +------------+    +------------+        |
# |                                                                       |
# |  Mask  +----------------+    +------------+    +------------+        |
# |        |                |    |            |    |            |        |
# |        |   ░░░░░░░░░░   |    | ██████████ |    | ░░███░░░░  |        |
# |        |   (near dark)  |    | (fully lit)|    |(partially  |        |
# |        |   no activity  |    | high accum.|    | lit)       |        |
# |        +----------------+    +------------+    +------------+        |
# |         s_t = 0  ✓ SKIP       s_t = 1  ✓ RUN   s_t = 1  ✓ RUN      |
# +-----------------------------------------------------------------------+

sys.path.append("E:/Dev/__halib")


from halib import *
from halib.system import filesys as fs
from halib.exp.viz.plot import PlotHelper as plth
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
outfile = os.path.join(CURRENT_DIR, "fig_qualitative_success.pdf")
OUTFILE_NAME = "fig_qualitative_success"

df = plth.get_img_grid_df(f"{CURRENT_DIR}/src/success")
df.columns = [f"sample_{i}" for i in range(len(df.columns))]

py_fname = os.path.basename(__file__)
df.to_csv(f"{CURRENT_DIR}/{py_fname.replace('.py', '')}.csv", index=False, encoding="utf-8", sep=";")


def fmt_row_label_func(x):
    # MAPPING_ROW_NAME = {
    #     "row_01_rgb": "<b>RGB</b>",
    #     "row_02_mask": "<b>ForeGround<br>Mask</b>",
    # }
    # return MAPPING_ROW_NAME[x]
    return ""


def fmt_col_label_func(x):
    # MAPPING_COL_NAME = {
    #     "sample_0": "Type of Frame",
    #     "sample_1": "(a) SKIP — Static BG",
    #     "sample_2": "(b) INFER — Fire",
    #     "sample_3": "(c) INFER — Smoke",
    # }
    # return MAPPING_COL_NAME.get(x, x)]
    return ""

# target_fmt = ["png", "pdf"]
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
    fig = plth.add_canvas_padding(fig, pad_top=0, pad_bottom=20, pad_left=100, pad_right=0)

    ROW_NAMES = [
        r"<b><span style=\"color: black;\">Frame<br>Type</b>",
        "<b>RGB</b>",
        "<b>ForeGround<br>Mask</b>",
    ]
    # expand the figure's left margin to accommodate row labels
    # fig.update_layout(margin=dict(l=120, r=10, t=0, b=0))
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
        r"(a) SKIP <br> Static BG  ✓ SKIP",
        r"(b) INFER <br> Fire  ✓ RUN",
        r"(c) INFER <br> Smoke ✓ RUN",
    ]

    BOLD_START = "<b>"
    BOLD_END = "</b>"
    CUSTOM_COLOR_START_TAG = '<span style="color: #035922;">'  # dark green color
    CUSTOM_COLOR_END_TAG = "</span>"

    for i in range(1, len(annotations)):
        annotations[i] = (
            f"{CUSTOM_COLOR_START_TAG}{annotations[i]}{CUSTOM_COLOR_END_TAG}"
        )

    num_cols = len(annotations)
    x_pos_ls = [0.25, 0.55, 0.85]
    y_pos_ls = [0.01] * num_cols

    for i, text in enumerate(annotations):
        x_pos = x_pos_ls[i]
        fig.add_annotation(
            x=x_pos,
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
