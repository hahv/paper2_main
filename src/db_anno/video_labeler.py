import yaml
from halib import *

from typing import List, Optional
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed


class VideoLabelerBase(ABC):
    """Base class for label providers handling video processing and labeling."""

    def __init__(self):
        self.video_list: List[str] = []
        self.dataset_name: Optional[str] = None
        self.dataset_path: Optional[str] = None
        self.label_file_posfix: str = "_labels.csv"
        self.skip_existing: bool = True

    def _generate_config(self) -> dict:
        """Generate configuration dictionary for the dataset."""
        return {
            "dataset_name": self.dataset_name,
            "dataset_path": self.dataset_path,
            "num_videos": len(self.video_list),
        }

    def process_labeling(self, to_csv: bool = True, max_workers: int = 8) -> None:
        """Process labeling for all videos in the dataset."""
        if not self.video_list:
            raise ValueError("Video list is empty. Populate it before processing.")
        if not self.dataset_name:
            raise ValueError("Dataset name is not set.")

        console.rule(f"Processing labeling for {self.dataset_name}")
        # Save configuration to YAML (directly in dataset path)
        config_file = os.path.join(self.dataset_path, "__cfg.yaml")  # ty:ignore[no-matching-overload]

        with open(config_file, "w") as f:
            yaml.dump(self._generate_config(), f)

        def process_single_video(video_path: str) -> str:
            """Process a single video and save labels if required."""
            # get parent dir of video_path
            parent_dir = os.path.dirname(video_path)
            fname = fs.get_file_name(video_path, split_file_ext=True)[0]
            fname_csv = f"{fname}{self.label_file_posfix}.csv"
            outfile = os.path.join(
                parent_dir,
                fname_csv,
            )
            if os.path.exists(outfile):
                console.print(f"Label file already exists, skipping: {outfile}")
                return video_path
            label_df = self.get_labels(video_path)
            if to_csv:
                console.print(f"Saving label to:")
                pprint_local_path(outfile, get_wins_path=True)
                print("\n\n")
                label_df.to_csv(outfile, index=False, sep=";")
            return video_path

        if max_workers < 1:  # do not use threading
            for video in tqdm(self.video_list, desc="Processing videos"):
                try:
                    process_single_video(video)
                except Exception as e:
                    console.print(f"[red]⚠️ Error processing video: {e}[/red]")
        else:  # use threading
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
