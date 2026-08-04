from halib import *

from halib.common.common import seed_everything
import os
import cv2
import numpy as np
from tap import Tap
import fiftyone.zoo as foz
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.model_selection import train_test_split


class DBSplitArgs(Tap):
    video_dir: str = "/mnt/d/zdataset_paper2/build_video_dataset/my_firesmoke_indoor/none"  # Path to your video folder
    out_dir: str = "/mnt/e/zDatasets/paper2_datasets"  # Output directory
    val_size: float = 0.3  # Size of validation set (0.0 to 1.0)
    seed: int = 42  # Random seed for reproducibility
    num_clusters: int = 5  # Estimated number of "categories"
    add_time: bool = True  # Whether to add timestamp to output dir names


def extract_three_frames(video_path):
    """
    Extracts 3 frames (10%, 50%, 90%) from a video.
    Returns a list of numpy arrays (images).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        pprint(f"Error opening video: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate indices for 10%, 50%, 90%
    frame_indices = [
        int(total_frames * 0.1),
        int(total_frames * 0.5),
        int(total_frames * 0.9),
    ]

    extracted_images = []

    # Iterate through target indices
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR (OpenCV) to RGB (FiftyOne/PIL expectation)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            extracted_images.append(frame_rgb)

    cap.release()

    # If video is too short or failed to read enough frames, return empty
    if len(extracted_images) < 3:
        return []

    return extracted_images


def split_db_dir(args):
    """
    Main logic to extract features, cluster, and split videos.
    """
    # 1. Gather all video paths
    if not os.path.exists(args.video_dir):
        pprint(f"Directory not found: {args.video_dir}")
        return [], [], []

    video_paths = fs.filter_files_by_extension(
        directory=args.video_dir, ext=[".mp4", ".avi", ".mov", ".mkv"], recursive=False
    )
    assert len(video_paths) > 0, f"No video files found in {args.video_dir}"
    pprint(
        f"Found {len(video_paths)} videos. Extracting frames and computing embeddings..."
    )

    # 2. Load Embedding Model (CLIP)
    pprint("Loading CLIP model via FiftyOne Zoo...")
    model = foz.load_zoo_model("clip-vit-base32-torch")

    video_embeddings = []
    valid_video_paths = []

    # 3. Process videos: Extract frames -> Embed -> Average
    for v_path in tqdm(video_paths, desc="Calc video embedding"):
        frames = extract_three_frames(v_path)

        if not frames:
            continue

        # --- KEY FIX: Iterate and embed individually ---
        # The model.embed() method works on a single image.
        try:
            current_embeddings = [model.embed(frame) for frame in frames]
        except Exception as e:
            print(f"Error embedding frames for {v_path}: {e}")
            continue

        # Average the 3 vectors to get one vector representing the video
        # Shape: (3, 512) -> (512,)
        video_avg_embedding = np.mean(current_embeddings, axis=0)

        video_embeddings.append(video_avg_embedding)
        valid_video_paths.append(v_path)

    if not valid_video_paths:
        pprint("Could not extract valid frames from any video.")
        return [], [], []

    X = np.array(video_embeddings)

    # 4. Clustering (Simulate Categories)
    pprint(
        f"Clustering videos into {args.num_clusters} categories based on visual similarity..."
    )
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=args.seed, n_init=10)
    labels = kmeans.fit_predict(X)

    # Organize categories for output/debug
    categories = defaultdict(list)
    for path, label in zip(valid_video_paths, labels):
        categories[label].append(path)

    # 5. Stratified Split
    pprint(f"Splitting data with val_size={args.val_size}...")

    try:
        # Note: train_test_split returns (train, test).
        # Since we want a specific Val size (e.g. 0.3), we treat the 'test' part of the function as 'val'.
        # train_paths -> Test Set (70%)
        # val_paths   -> Val Set (30%)
        train_paths, val_paths = train_test_split(
            valid_video_paths,
            test_size=args.val_size,
            stratify=labels,
            random_state=args.seed,
        )
        val_list = val_paths
        test_list = train_paths

    except ValueError:
        pprint(
            f"\n⚠️ Warning: Stratified split failed (likely a cluster had too few videos)."
        )
        pprint("Falling back to random split.")
        train_paths, val_paths = train_test_split(
            valid_video_paths, test_size=args.val_size, random_state=args.seed
        )
        val_list = val_paths
        test_list = train_paths

    # --- FINAL OUTPUT ---

    with ConsoleLog("Dataset Split Summary"):
        pprint(f"Total Videos: {len(valid_video_paths)}")
        pprint(f"Test Set: {len(test_list)}")
        pprint(f"Val Set:  {len(val_list)}")

    console.rule("Generated Categories (Similarity Groups)")
    for label_id, paths in categories.items():
        pprint(f"Category {label_id}: {len(paths)} videos")

    return val_list, test_list, categories


def main():
    # Parse arguments
    args = DBSplitArgs().parse_args()
    # ! to ensure reproducibility
    seed_everything(args.seed)

    # Run logic
    val, test, cats = split_db_dir(args)

    # Example: Print first 3 of each list
    if val:
        pprint(f"Total Val Videos: {len(val)}")
    if test:
        pprint(f"Total Test Videos: {len(test)}")

    for cate_name, cate_videos in cats.items():
        with ConsoleLog(f"Category {cate_name} Videos"):
            for v in cate_videos:
                pprint_local_path(v)


if __name__ == "__main__":
    main()
