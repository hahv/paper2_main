from halib import *
from halib.filetype import yamlfile
from src.db_anno.video_labeler import VideoLabelerBase
from typing import Optional, Callable

class UFireIndoorReLabeler(VideoLabelerBase):
    """Label provider for NewDBConsole dataset, handling video metadata and frame labeling."""

    VALID_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]
    VALID_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]
    LABEL_FIRE = "fire" # having fire (with or without smoke)
    LABEL_SMOKE = "smoke" # having smoke only (no fire)
    LABEL_NONE = "none" # no fire, no smoke
    LABEL_MIXED = "mixed" # mixed labels across frames

    def __init__(
        self,
        cfg_yaml: str = "video_db/__db_cfg_ufire_relabel.yaml",
        video_path_to_label_func: Optional[Callable[[str, str], str]] = None,
    ):
        cfg_dict = yamlfile.load_yaml(cfg_yaml, to_dict=True)
        self.dataset_name = cfg_dict.get("dataset_name", "NewDB")
        self.dataset_path = cfg_dict.get("dataset_path", "./videos")
        self.search_recursive = cfg_dict.get("recursive", True)
        self.label_file_posfix = cfg_dict.get("label_file_posfix", "_labels")
        self.skip_existing = cfg_dict.get("skip_existing", True)

        # ! must call before _collect_videos
        self.get_all_frames_label_func = (
            video_path_to_label_func
            if video_path_to_label_func
            else self.map_video_to_label
        )
        self._collect_videos()
        self.list_csv_to_check = []

    @staticmethod
    def map_video_to_label(dataset_name: str, video_path: str) -> str:  # ty:ignore[invalid-return-type]
        fname = fs.get_file_name(video_path, split_file_ext=True)[0]
        if dataset_name == "UFireIndoorFullRelabel":
            """Map video file name to its label based on naming conventions."""
            if "_lb_none" in fname:
                return UFireIndoorReLabeler.LABEL_NONE
            elif "_lb_fire" in fname:
                return UFireIndoorReLabeler.LABEL_FIRE
            elif "_lb_smoke" in fname:
                return UFireIndoorReLabeler.LABEL_SMOKE
            elif "_lb_mixed" in fname:
                return UFireIndoorReLabeler.LABEL_MIXED
            else:
                raise ValueError(
                    f"Cannot determine label for video: {video_path}. "
                    f"Expected naming convention: '_lb_none', '_lb_fire', '_lb_smoke', or '_lb_mixed'."
                )
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    def _collect_videos(self) -> None:
        """Collect video files and their directories from the dataset path."""
        self.video_list = fs.filter_files_by_extension(
            directory=self.dataset_path,
            ext=[".mp4", ".avi", ".mov", ".mkv"],
            recursive=self.search_recursive,
        )
        self.video_priority_list = [0] * len(self.video_list)
        for idx, video in enumerate(self.video_list):
            all_video_lb = self.get_all_frames_label_func(
                dataset_name=self.dataset_name, video_path=video  # ty:ignore[unknown-argument]
            )  # ty:ignore[missing-argument]
            if all_video_lb != self.LABEL_MIXED:
                self.video_priority_list[idx] = 1
        # Sort videos by priority (non-mixed first)
        sorted_videos = [
            v
            for _, v in sorted(
                zip(self.video_priority_list, self.video_list), reverse=True
            )
        ]
        self.video_list = sorted_videos

    def get_labels(self, video_path: str) -> pd.DataFrame:
        
        """Extract labels and metadata for a given video."""
        type_label = self.get_all_frames_label_func(
            dataset_name=self.dataset_name, video_path=video_path  # ty:ignore[unknown-argument]
        )  # ty:ignore[missing-argument]
        outfile = self.get_label_outfile(video_path, self.label_file_posfix)
        fname = fs.get_file_name(video_path, split_file_ext=True)[0]
        print(f'proc {fname} with label {type_label}', end="\r")
        time.sleep(0.5)  # simulate processing time
        # the old anotation file should exist, otherwise raise an error
        assert os.path.exists(outfile), f"Label file does not exist: {outfile}"
        # columns=["frame_idx", "video_path", "label"]
        # read the old csv file as pandas
        df = pd.read_csv(
            outfile,
            sep=";",
            encoding="utf-8",
            dtype={"frame_idx": int, "video_path": str, "label": str},
            keep_default_na=False,
        )
        # make sure the label column is type of string to save to csv
        df["label"] = df["label"].astype(str)
        # save directly to the same file
        if type_label in [UFireIndoorReLabeler.LABEL_NONE, UFireIndoorReLabeler.LABEL_FIRE, UFireIndoorReLabeler.LABEL_SMOKE]:
            df["label"] = df["label"].map(
                {
                    "None": UFireIndoorReLabeler.LABEL_NONE,
                    "Fire_Smoke": type_label
                }
            )
            df["label"] = df["label"].astype(str)
            df.to_csv(outfile, index=False, sep=";")
            self.list_csv_to_check.append(outfile)
        elif type_label == UFireIndoorReLabeler.LABEL_MIXED:
            df["label"] = df["label"].map(
                {
                    "None": UFireIndoorReLabeler.LABEL_NONE,
                    "Fire_Smoke": "Fire_Smoke_ToCheck"
                }
            )
            df["label"] = df["label"].astype(str)
            outfile = outfile.replace(".csv", "_new.csv")
            df.to_csv(outfile, index=False, sep=";")
            self.list_csv_to_check.append(outfile)
        else:
            raise ValueError(f"Unknown label type: {type_label}")
        # return empty DataFrame
        return pd.DataFrame()  # just an empty DF

    def after_process_labeling(self) -> None:
        with ConsoleLog('List of CSV files to check:'):
            for csv_file in self.list_csv_to_check:
                pprint_local_path(csv_file, get_wins_path=True)
