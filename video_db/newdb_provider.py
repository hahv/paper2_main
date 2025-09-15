from halib import *
from video_db.lb_provider import LabelProviderBase
from video_db.hrwen_provider import HpwrenLbProvider
from typing import List, Tuple, Optional
import cv2
from concurrent.futures import ThreadPoolExecutor

class NewDBLbProvider(HpwrenLbProvider):
    """Label provider for NewDBConsole dataset, handling video metadata and frame labeling."""

    VALID_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]
    VALID_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

    def __init__(
        self, dataset_path: str = "/mnt/e/NextCloud/paper2_main/datasets/__TestVideos"
    ):
        self.dataset_name = "NewDB"
        self.dataset_path = dataset_path
        self.meta_df_creator = csvfile.DFCreator()
        self.meta_df_creator.create_table(
            table_name="dataset_meta",
            columns=["video_path", "frame_count", "fps", "frame_width", "frame_height"],
        )
        self.meta_rows: List[List] = []
        self.output_dir = "datasets/___annotations"
        self._collect_videos()

    def _collect_videos(self) -> None:
        """Collect video files and their directories from the dataset path."""
        self.video_list = fs.filter_files_by_extension(directory=
            self.dataset_path, ext=[".mp4", ".avi", ".mov", ".mkv"], recursive=True
        )

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
        fire_ranges = []
        console.rule(f"Labeling video: [blue]{video_path}[/blue]")
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