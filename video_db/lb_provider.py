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

        if max_workers < 1: # do not use threading
            for video in tqdm(self.video_list, desc="Processing videos"):
                try:
                    process_single_video(video)
                except Exception as e:
                    console.print(f"[red]⚠️ Error processing video: {e}[/red]")
        else: # use threading
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