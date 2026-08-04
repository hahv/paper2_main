from halib import *
from video_db.lb_provider import LabelProviderBase
from typing import List, Tuple, Optional
import cv2
from concurrent.futures import ThreadPoolExecutor


class HpwrenLbProvider(LabelProviderBase):
    """Label provider for HPWREN dataset, handling video metadata and frame labeling."""

    VALID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
    VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

    def __init__(
        self, dataset_path: str = "/mnt/d/ZDev/paper2_Prof_Video/HPWREN_ALL-2"
    ):
        super().__init__()
        self.dataset_name = "HPWREN"
        self.dataset_path = dataset_path
        self.dirs_list: List[str] = []
        self.meta_df_creator = csvfile.DFCreator()
        self.meta_df_creator.create_table(
            table_name="dataset_meta",
            columns=["video_path", "frame_count", "fps", "frame_width", "frame_height"],
        )
        self.meta_rows: List[List] = []
        self._collect_videos_and_dirs()

    def _collect_videos_and_dirs(self, max_workers: int = 8) -> None:
        """Collect video files and their directories from the dataset path."""

        def process_subdir(subdir: str) -> Optional[Tuple[str, str]]:
            full_path = os.path.join(self.dataset_path, subdir)
            if not os.path.isdir(full_path) or subdir == "__archived":
                return None

            videos = [
                f
                for f in os.listdir(full_path)
                if f.lower().endswith(self.VALID_EXTENSIONS)
            ]
            if len(videos) != 1:
                raise ValueError(
                    f"Expected exactly one video in {full_path}, found {len(videos)}"
                )
            return os.path.join(full_path, videos[0]), full_path

        subdirs = os.listdir(self.dataset_path)
        video_files, video_dirs = [], []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_subdir, subdir) for subdir in subdirs]
            for future in tqdm(futures, desc="Collecting videos"):
                result = future.result()
                if result:
                    video_file, video_dir = result
                    video_files.append(video_file)
                    video_dirs.append(video_dir)

        self.video_list = video_files
        self.dirs_list = video_dirs

    def _verify_extracted_frames(
        self, extracted_frames: List[str], frame_count: int
    ) -> bool:
        """Verify that the number of extracted frames matches the video frame count."""
        if len(extracted_frames) != frame_count:
            console.print(
                f"[red]Error: Number of extracted frames ({len(extracted_frames)}) "
                f"does not match frame count ({frame_count})[/red]"
            )
            return False
        return True

    def get_labels(self, video_path: str) -> pd.DataFrame:
        """Extract labels and metadata for a given video."""
        video_dir = os.path.dirname(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        self.meta_rows.append([video_path, frame_count, fps, frame_width, frame_height])

        extracted_frames = fs.filter_files_by_extension(
            video_dir, self.VALID_IMAGE_EXTENSIONS
        )
        if not self._verify_extracted_frames(extracted_frames, frame_count):
            raise ValueError(f"Invalid frame count for video: {video_path}")

        df_creator = csvfile.DFCreator()
        table_name = fs.get_file_name(video_path, split_file_ext=True)[0]
        df_creator.create_table(
            table_name=table_name, columns=["frame_idx", "frame_path", "label"]
        )
        rows = []

        for idx, frame_path in enumerate(sorted(extracted_frames)):
            frame_idx = idx + 1  # 1-based indexing
            frame_name = fs.get_file_name(frame_path, split_file_ext=False)
            parts = frame_name.split("_")
            if len(parts) != 2:
                raise ValueError(f"Unexpected frame name format: {frame_name}")

            label_part = parts[1]
            label = (
                "none"
                if label_part.startswith("-")
                else "fire_smoke"
                if label_part.startswith("+")
                else None
            )
            if label is None:
                raise ValueError(f"Unexpected label format in frame name: {frame_name}")

            rows.append([frame_idx, frame_path, label])

        df_creator.insert_rows(table_name=table_name, singleRow_or_rowList=rows)
        df_creator.fill_table_from_row_pool(table_name=table_name)
        return df_creator[table_name]

    def after_process_labeling(self) -> None:
        """Save dataset metadata after labeling process."""
        self.meta_df_creator.insert_rows(
            table_name="dataset_meta", singleRow_or_rowList=self.meta_rows
        )
        self.meta_df_creator.fill_table_from_row_pool(table_name="dataset_meta")
        output_dir = os.path.join(self.output_dir, self.dataset_name)
        output_file = os.path.join(output_dir, "___dataset_meta.csv")
        console.print(f"Saving dataset metadata to [green]{output_file}[/green]")
        self.meta_df_creator["dataset_meta"].to_csv(output_file, index=False, sep=";")
