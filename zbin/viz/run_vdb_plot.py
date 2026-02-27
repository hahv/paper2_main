import sys
sys.path.append("E:/Dev/__halib")

from halib import *
from halib.exp.viz.plot import PlotHelper as plth

df = plth.get_img_grid_df(
    r"E:/SyncData/paper2_main/zout/reports/viz/vdb_screenshots/src"
)
df.columns = [f'sample_{i}' for i in range(len(df.columns))]
df.to_csv(r"./vdb_grid_input.csv", index=False, encoding="utf-8", sep=";")

os.chdir(r"E:/SyncData/paper2_main/zout/reports/viz/vdb_screenshots")

def fmt_row_label_func(x):
    x = x.replace("row_", "").upper()
    if x.startswith("FIRE"):
        return "FIRE or SMOKE"
def fmt_col_label_func(x):
    return ""

plth.plot_image_grid(
    df,
    save_path=r"./vdb_grid_plot.pdf",
    img_width=300,
    img_height=300,
    img_stack_padding_px=5,
    img_stack_direction="horizontal",
    img_scale_mode="fit",
    title="Image Grid Test",
    fig_margin=dict(l=0, r=10, t=50, b=10),
    outline_color="#000000",
    outline_size=2,
    cell_margin_px=5,
    row_line_size=2,
    col_line_size=2,
    tickfont=dict(size=13, family="Arial", color="black"),
    fig_extra_size=(100, 150),
    format_row_label_func=fmt_row_label_func,
    format_col_label_func=fmt_col_label_func,
    show=False,
)

