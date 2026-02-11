import os
from src.config import *
from halib.filetype import csvfile
from collections import OrderedDict
from src.results.base_rs_proc import *
from src.common import GlobalConstants




class CsvRsProc(BaseRsProc):
    CSV_FIXED_COLUMNS = [
        GlobalConstants.COL_VIDEO,
        GlobalConstants.COL_NUM_FRAMES,
        GlobalConstants.COL_FRAME_IDX,
        GlobalConstants.COL_ELAPSED_TIME,
    ]

    def __init__(self, cfg: Config):
        self.cfg = cfg
        assert self.cfg.inferCfg.save_csv_results, (
            "CSV saving is disabled in the config"
        )
        self.dfmk = None
        self.table_name = None
        self.csv_rows = []
        self.out_csv_file = None
        self.outdir = os.path.abspath(cfg.get_outdir())
        self.extra_cols = self.cfg.inferCfg.csv_columns
        self.csv_columns = CsvRsProc.CSV_FIXED_COLUMNS + self.extra_cols
        self.outfile_exists = False

    def before_video(self, video_path: str, **kwargs):
        if not self.cfg.inferCfg.save_csv_results:
            return
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.out_csv_file = os.path.join(self.outdir, f"{video_name}_results.csv")

        skip_if_exists = self.cfg.inferCfg.skip_if_exists
        if skip_if_exists and os.path.exists(self.out_csv_file):
            self.outfile_exists = True
            print(f"CSV file already exists, skipping: {self.out_csv_file}")
            return  # skip creating dfmk and table

        self.dfmk = csvfile.DFCreator()
        self.dfmk.create_table(video_name, columns=self.csv_columns)
        self.table_name = video_name
        self.csv_rows = []

    # ! can be override
    def prepare_csv_row(self, frame_rs_dict: dict):
        """Prepare a CSV row dictionary from frame results."""
        row_dict = OrderedDict()
        for col in CsvRsProc.CSV_FIXED_COLUMNS:
            row_dict[col] = frame_rs_dict[col]

        row_dict["class_names"] = self.cfg.modelCfg.class_names
        infer_dict = frame_rs_dict["infer_rs"]
        row_dict["logits"] = infer_dict["logits"]
        row_dict["probs"] = infer_dict["probs"]
        row_dict["pred_label_idx"] = infer_dict["predLabelIdx"]
        row_dict["pred_label"] = infer_dict["predLabel"]
        return row_dict

    def handle_frame_results(self, frame_bgr, frame_rs_dict: dict):
        # Unpack data from the dictionary
        row_dict = self.prepare_csv_row(frame_rs_dict)
        # pprint(row_dict)
        row_array = list(row_dict.values())
        self.csv_rows.append(row_array)

    def after_video(self, video_path: str, **kwargs):
        if self.outfile_exists:
            self.outfile_exists = False  # reset for next video
            return
        if not self.cfg.inferCfg.save_csv_results or self.dfmk is None:
            return

        self.dfmk.insert_rows(self.table_name, self.csv_rows)
        self.dfmk.fill_table_from_row_pool(self.table_name)
        self.dfmk[self.table_name].to_csv(
            self.out_csv_file, index=False, sep=";", encoding="utf-8"
        )
        with ConsoleLog("Results saved to:"):
            pprint_local_path(self.out_csv_file, get_wins_path=True)  # ty:ignore[invalid-argument-type]
