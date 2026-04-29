from IPython.testing.plugin.simplevars import x
from ipykernel.pickleutil import can
import src
import yaml
import sys

sys.path.append("E:/Dev/__halib")


from halib import *
from halib.system import filesys as fs
from halib.exp.viz.plot import PlotHelper as plth
import os
from halib import *
from tap import *


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


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


class CustomArgs(Tap):
    yaml_file: str = "fig_qualitative_failure.yaml"

    def configure(self):
        self.add_argument(
            "-i", "--yaml_file", type=str, help="Path to the YAML configuration file."
        )

SUB_FIG_SIZE_W = 640
SUB_FIG_SIZE_H = 480

def main():
    # Parse arguments
    args = CustomArgs().parse_args()
    yaml_path = os.path.join(CURRENT_DIR, args.yaml_file)
    from halib.filetype import yamlfile

    vis_cfg = yamlfile.load_yaml(yaml_path, to_dict=True)
    src_dir = os.path.join(CURRENT_DIR, vis_cfg["src_dir"])
    outfile_name = vis_cfg.get("outfile_name")
    font_size = vis_cfg.get("font_size", 16)
    font_family = vis_cfg.get("font_family", "CMU Serif")
    # vis df
    df = plth.get_img_grid_df(src_dir)
    df.columns = [f"sample_{i}" for i in range(len(df.columns))]
    py_fname = os.path.basename(__file__)
    df.to_csv(
        f"{CURRENT_DIR}/{py_fname.replace('.py', '')}.csv",
        index=False,
        encoding="utf-8",
        sep=";",
    )
    target_fmt = vis_cfg.get("target_fmt", ["pdf"])
    BOLD_START_TAG = "<b>"
    BOLD_END_TAG = "</b>"
    CUSTOM_COLOR_START_TAG_FMT = '<span style="color: {};">'
    COLOR_END_TAG = "</span>"
    for fmt in tqdm(target_fmt):
        console.rule(f"Exporting {fmt} ...")
        outfile = f"{CURRENT_DIR}/{outfile_name}.{fmt}"
        fig = plth.plot_image_grid(
            df,
            # save_path=outfile,
            img_width=SUB_FIG_SIZE_W,
            img_height=SUB_FIG_SIZE_H,
            img_stack_padding_px=5,
            img_stack_direction="horizontal",
            img_scale_mode="fill",
            fig_margin=dict(l=0, r=10, t=10, b=10),
            outline_color="#000000",
            outline_size=2,
            cell_margin_px=5,
            row_line_size=2,
            col_line_size=2,
            tickfont=dict(size=font_size, family=font_family, color="black"),
            fig_extra_size=(0, 40),
            format_row_label_func=fmt_row_label_func,
            format_col_label_func=fmt_col_label_func,
            show=False,
        )
        canvas_padding = vis_cfg.get(
            "canvas_padding", dict(pad_top=0, pad_bottom=70, pad_left=100, pad_right=0)
        )
        fig = plth.add_canvas_padding(
            fig,
            pad_top=canvas_padding["top"],
            pad_bottom=canvas_padding["bottom"],
            pad_left=canvas_padding["left"],
            pad_right=canvas_padding["right"],
        )

        # ! Add LEFT MOST ANNOTATION -- Frame Type
        left_most_anno_ls = vis_cfg.get("left_most_anno", [])
        for i, anno_item in enumerate(left_most_anno_ls):
            x_pos = anno_item["x"]
            y_pos = anno_item["y"]
            is_bold = anno_item.get("bold", False)
            color = anno_item.get("color", "black")
            text = anno_item["text"]
            bold_tag_start, bold_tag_end = (
                (BOLD_START_TAG, BOLD_END_TAG) if is_bold else ("", "")
            )
            color_tag_start = CUSTOM_COLOR_START_TAG_FMT.format(color) if color else ""
            formatted_text = (
                f"{bold_tag_start}{color_tag_start}{text}{COLOR_END_TAG}{bold_tag_end}"
            )
            fig.add_annotation(
                x=x_pos,
                y=y_pos,
                xref="paper",
                yref="paper",
                text=formatted_text,
                showarrow=False,
                font=dict(size=font_size, family=font_family, color="black"),
                xanchor="right",
                yanchor="middle",
            )
        # ! Add BOTTOM MOST ANNOTATION -- Column Labels
        anno_text_file = os.path.join(src_dir, "col_anno.txt")
        # col_1---SKIP <br> Static BG  ✓ SKIP
        anno_text_ls = []
        with open(anno_text_file, "r", encoding="utf-8") as file:
            anno_text_ls = file.readlines()
            anno_text_ls = [line.rstrip() for line in anno_text_ls]

        anno_text_ls = [
            line.split("---")[1].strip() for line in anno_text_ls if line.strip()
        ]

        # get positions for bottom annotations
        bottom_anno_pos_ls = vis_cfg.get("bottom_most_anno", [])
        num_pos = len(bottom_anno_pos_ls)

        assert num_pos == len(anno_text_ls), (
            f"Number of annotation positions ({num_pos}) must match number of annotation texts ({len(anno_text_ls)})"
        )
        for i, anno_item in enumerate(anno_text_ls):
            bottom_anno_pos_ls[i]["text"] = anno_item

        bottom_anno_ls = bottom_anno_pos_ls.copy()

        for i, anno_item in enumerate(bottom_anno_ls):
            x_pos = anno_item["x"]
            y_pos = anno_item["y"]
            text = anno_item["text"]
            is_bold = anno_item.get("bold", False)
            color = anno_item.get("color", "black")
            bold_tag_start, bold_tag_end = (
                (BOLD_START_TAG, BOLD_END_TAG) if is_bold else ("", "")
            )
            color_tag_start = CUSTOM_COLOR_START_TAG_FMT.format(color) if color else ""
            formatted_text = (
                f"{bold_tag_start}{color_tag_start}{text}{COLOR_END_TAG}{bold_tag_end}"
            )
            fig.add_annotation(
                x=x_pos,
                y=y_pos,
                xref="paper",
                yref="paper",
                text=formatted_text,
                showarrow=False,
                font=dict(size=font_size, family=font_family, color="black"),
                xanchor="center",
                yanchor="bottom",
            )

        fig.write_image(outfile, scale=2)
        fs.open_file(outfile)


if __name__ == "__main__":
    main()
