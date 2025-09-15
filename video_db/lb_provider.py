import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional
import cv2
import yaml
import pandas as pd
from tqdm import tqdm
from halib import console
from halib.filetype import csvfile
from halib.system import filesys as fs


class LabelProviderBase(ABC):
    """Base class for label providers handling video processing and labeling."""

    def __init__(self, output_dir: str = "datasets/___annotations"):
        self.video_list: List[str] = []
        self.dataset_name: Optional[str] = None
        self.dataset_path: Optional[str] = None
        self.output_dir: str = output_dir

    def _generate_config(self) -> dict:
        """Generate configuration dictionary for the dataset."""
        return {
            "dataset_name": self.dataset_name,
            "dataset_path": self.dataset_path,
            "output_dir": self.output_dir,
            "num_videos": len(self.video_list),
        }

    def process_labeling(self, to_csv: bool = True, max_workers: int = 8) -> None:
        """Process labeling for all videos in the dataset."""
        if not self.video_list:
            raise ValueError("Video list is empty. Populate it before processing.")
        if not self.dataset_name:
            raise ValueError("Dataset name is not set.")
        if not self.output_dir:
            raise ValueError("Output directory is not set.")

        console.rule(f"Processing labeling for {self.dataset_name}")
        output_dir = os.path.join(self.output_dir, self.dataset_name)
        os.makedirs(output_dir, exist_ok=True)

        # Save configuration to YAML
        config_file = os.path.join(output_dir, "__cfg.yaml")
        with open(config_file, "w") as f:
            yaml.dump(self._generate_config(), f)

        def process_single_video(video_path: str) -> str:
            """Process a single video and save labels if required."""
            label_df = self.get_labels(video_path)
            if to_csv:
                output_file = os.path.join(
                    output_dir, f"{os.path.basename(video_path)}.csv"
                )
                console.print(f"Saving to [green]{output_file}[/green]")
                label_df.to_csv(output_file, index=False, sep=";")
            return video_path

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_single_video, video)
                for video in self.video_list
            ]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing videos"
            ):
                try:
                    future.result()
                except Exception as e:
                    console.print(f"[red]⚠️ Error processing video: {e}[/red]")

        self.after_process_labeling()

    @abstractmethod
    def get_labels(self, video_path: str) -> pd.DataFrame:
        """Retrieve labels for a given video.

        Args:
            video_path: Path to the video file.

        Returns:
            DataFrame containing frame indices, paths, and labels.
        """
        pass

    @abstractmethod
    def after_process_labeling(self) -> None:
        """Hook for post-processing actions after labeling."""
        pass


class HPWRENLabelProvider(LabelProviderBase):
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
                else "fire_smoke" if label_part.startswith("+") else None
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


def main() -> None:
    """Main function to initialize and run the HPWREN label provider."""
    provider = HPWRENLabelProvider()
    provider.process_labeling(to_csv=True)


if __name__ == "__main__":
    main()
