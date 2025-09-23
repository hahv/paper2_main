from halib import *
from halib.filetype import yamlfile
from video_db.lb_provider import LabelProviderBase
from typing import List, Optional
import cv2


class NewDBLbProvider(LabelProviderBase):
    """Label provider for NewDBConsole dataset, handling video metadata and frame labeling."""

    VALID_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]
    VALID_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]
    LABEL_POSITIVE = "Fire_Smoke"
    LABEL_NEGATIVE = "None"
    LABEL_MIXED = "Mixed"

    def __init__(self, cfg_yaml: str = "video_db/__db_cfg.yaml"):
        cfg_dict = yamlfile.load_yaml(cfg_yaml, to_dict=True)
        self.dataset_name = cfg_dict.get("dataset_name", "NewDB")
        self.dataset_path = cfg_dict.get("dataset_path", "./videos")
        self.search_recursive = cfg_dict.get("recursive", True)
        self.label_file_posfix = cfg_dict.get("label_file_posfix", "_labels")

        self.meta_df_creator = csvfile.DFCreator()
        self.meta_df_creator.create_table(
            table_name="dataset_meta",
            columns=["video_path", "frame_count", "fps", "frame_width", "frame_height"],
        )
        self.meta_rows: List[List] = []
        self._collect_videos()

    def _collect_videos(self) -> None:
        """Collect video files and their directories from the dataset path."""
        self.video_list = fs.filter_files_by_extension(directory=
            self.dataset_path, ext=[".mp4", ".avi", ".mov", ".mkv"], recursive=self.search_recursive
        )
        self.video_priority_list = [0]*len(self.video_list)
        for idx, video in enumerate(self.video_list):
            all_video_lb = self.get_all_frames_label(video)
            if all_video_lb != self.LABEL_MIXED:
                self.video_priority_list[idx] = 1
        # Sort videos by priority (non-mixed first)
        sorted_videos = [v for _, v in sorted(zip(self.video_priority_list, self.video_list), reverse=True)]
        self.video_list = sorted_videos
    def get_all_frames_label(self, video_path):
        lb_all_frames = self.LABEL_MIXED
        fname = fs.get_file_name(video_path, split_file_ext=True)[0]
        if '__lb_none__' in fname:
            lb_all_frames = self.LABEL_NEGATIVE
        elif '__lb_fire_smoke__' and '__all_fires' in fname:
            lb_all_frames = self.LABEL_POSITIVE
        return lb_all_frames

    # def linux_to_wins_path(self, path: str) -> str:
    #     """Convert a Linux-style path to a Windows-style path."""
    #     # first remove /mnt
    #     if path.startswith("/mnt/"):
    #         path = path[5:]
    #     # then replace first char with drive letter and colon
    #     if len(path) > 1 and path[0].isalpha() and path[1] == '/':
    #         path = f"{path[0].upper()}:{path[1:]}"
    #     return path

    def get_labels(self, video_path: str) -> pd.DataFrame:
        """Extract labels and metadata for a given video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        self.meta_rows.append([video_path, frame_count, fps, frame_width, frame_height])

        df_creator = csvfile.DFCreator()
        table_name = fs.get_file_name(video_path, split_file_ext=True)[0]
        df_creator.create_table(
            table_name=table_name, columns=["frame_idx", "video_path", "label"]
        )
        console.rule(f"Labeling video:")
        all_frames_label = self.get_all_frames_label(video_path)
        pprint_local_path(video_path, get_wins_path=True)
        # pprint(f'-->>All frames label: <{all_frames_label}>')
        if all_frames_label != self.LABEL_MIXED:
            # All frames have the same label
            pprint(
                f"-->>Set <{all_frames_label}> for all frames in {fs.get_file_name(video_path)}"
            )
            rows = [
                [idx, video_path, all_frames_label] for idx in range(1, frame_count + 1)
            ]
            df_creator.insert_rows(table_name=table_name, singleRow_or_rowList=rows)
            df_creator.fill_table_from_row_pool(table_name=table_name)
        else: # Mixed labels, need manual input
            # Interactive input for fire_smoke ranges
            fire_ranges = []
            while True:
                start = 1 # !1-based indexing
                end = frame_count # default to full range
                console.print(f"Total frames: [red]{frame_count}[red]")
                c_firesmoke_range = ", ".join([f"[{s},{e}]" for s, e in fire_ranges]) if fire_ranges else "None"
                console.print(f"Current fire_smoke ranges: [green]{c_firesmoke_range}[/green]")
                entry = input(f"Enter fire_smoke range ({start=}, {end=}) or 'done': ").strip()
                if entry.lower() == "done":
                    break
                try:
                    start, end = map(int, entry.split(" "))
                    if 1 <= start <= end <= frame_count:
                        fire_ranges.append((start, end))
                    else:
                        print("⚠️ Invalid range. Must be within [1, total_frames]")
                except Exception as e:
                    print("⚠️ Format error. Use start <space> end (e.g., 10<space>50)")
            # Assign labels
            labels = ["none"] * frame_count
            labels.insert(0, "place_holder")  # 1-based indexing
            for start, end in fire_ranges:
                for i in range(start, end + 1):
                    labels[i] = "fire_smoke"
            rows = [[idx, video_path, labels[idx]] for idx in range(1, frame_count + 1)]
            df_creator.insert_rows(table_name=table_name, singleRow_or_rowList=rows)
            df_creator.fill_table_from_row_pool(table_name=table_name)
        return df_creator[table_name]

    def after_process_labeling(self) -> None:
        """Save dataset metadata after labeling process."""
        self.meta_df_creator.insert_rows(
            table_name="dataset_meta", singleRow_or_rowList=self.meta_rows
        )
        self.meta_df_creator.fill_table_from_row_pool(table_name="dataset_meta")
        output_dir = os.path.join(self.dataset_path, self.dataset_name)
        output_file = os.path.join(output_dir, "___dataset_meta.csv")
        console.print(f"Saving dataset metadata to [green]{output_file}[/green]")
        self.meta_df_creator["dataset_meta"].to_csv(output_file, index=False, sep=";")
