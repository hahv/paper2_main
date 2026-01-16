import os
import pandas as pd
from halib import *  # Assuming fs and other utils are here
from src.metrics.loaders.base_csv_loader import BaseCsvLoader


class UFireIndoorCsvLoader(BaseCsvLoader):
    """
    Adapter for datasets that provide a separate CSV file for Ground Truth labels.
    Expected format: [video_name]__labels.csv
    """

    def load_pred_df(self, csv_file: str) -> pd.DataFrame:
        # Same loading logic as DFire, or customize if needed
        df = pd.read_csv(
            csv_file,
            sep=";",
            encoding="utf-8",
            dtype={"pred_label": str, "elapsed_time": float},
            keep_default_na=False,
        )
        video_name = fs.get_file_name(csv_file, split_file_ext=True)[0]
        video_name = video_name.replace("_results", "")
        df["video"] = video_name

        df["pred"] = (
            df["pred_label"]
            .str.lower()
            .apply(
                lambda x: self.POS_LABEL
                if ("fire" in x or "smoke" in x)
                else self.NEG_LABEL
            )
        )
        return df

    def get_gt(self, video_name: str, num_frames: int, pred_df: pd.DataFrame) -> list:
        # 1. Get Path Mapping from Config
        recursive = self.cfg.dbsetCfg.extra_cfgs.get("ds_recursive", False)
        vname2path_dict = self.cfg.dbsetCfg.get_vname2path(recursive=recursive)

        # 2. Construct Label File Path
        label_gt_file_name = f"{video_name}__labels.csv"
        # Assuming vname2path_dict gives the path to the video file, we want its dir
        video_path = vname2path_dict.get(video_name)
        assert video_path is not None, f"Video {video_name} not found in dataset map"

        video_dir = os.path.dirname(video_path)
        csv_path = os.path.join(video_dir, label_gt_file_name)

        assert os.path.exists(csv_path), f"GT Label file {csv_path} does not exist"

        # 3. Read GT CSV
        gt_df = pd.read_csv(
            csv_path,
            sep=";",
            encoding="utf-8",
            dtype={"label": str},
            keep_default_na=False,
        )

        # 4. Convert Labels
        raw_labels = gt_df["label"].tolist()[:num_frames]
        gt_list = []
        for label in raw_labels:
            if "fire" in label.lower() or "smoke" in label.lower():
                gt_list.append(self.POS_LABEL)
            else:
                gt_list.append(self.NEG_LABEL)

        return gt_list
