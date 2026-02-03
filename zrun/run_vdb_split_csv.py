from halib import *
from halib.common.common import seed_everything
from halib.filetype import csvfile
from sklearn.model_selection import train_test_split
from tap import Tap
from typing import Literal


class DBSplitCSVArgs(Tap):
    video_dir: str = "/mnt/d/zdataset_paper2/build_video_dataset/my_firesmoke_indoor/firesmoke"  # Path to your video folder
    task: Literal["mk_csv", "split_by_csv", "copy_files"] = (
        "split_by_csv"  # Set a default value if desired  # Set a default value if desired
    )
    out_dir: str = "/mnt/e/zDatasets/paper2_datasets"  # Output directory
    val_size: float = 0.3  # Size of validation set (0.0 to 1.0)
    seed: int = 42  # Random seed for reproducibility
    add_time: bool = True  # Whether to add timestamp to output dir names


META_DATA_CSV = "video_metadata"
FILE_NAME_SEP = "__"
TABLE_NAME = "video_meta"
SET_NAMES = ["test", "val"]


def get_meta_csv_file_path(video_dir, out_dir):
    video_dir_name = fs.get_dir_name(video_dir)
    return os.path.join(out_dir, f"{video_dir_name}_{META_DATA_CSV}.csv")


def get_split_csv_file_path(video_dir, out_dir, split_name):
    video_dir_name = fs.get_dir_name(video_dir)
    return os.path.join(out_dir, f"{video_dir_name}_{split_name}_split.csv")


def get_video_dir_meta_csv(video_dir, out_dir):
    """
    Main logic to extract features, cluster, and split videos.
    """
    # 1. Gather all video paths
    if not os.path.exists(video_dir):
        pprint(f"Directory not found: {video_dir}")
        return [], [], []

    video_paths = fs.filter_files_by_extension(
        directory=video_dir, ext=[".mp4", ".avi", ".mov", ".mkv"], recursive=False
    )
    assert len(video_paths) > 0, f"No video files found in {video_dir}"
    pprint(
        f"Found {len(video_paths)} videos. Extracting frames and computing embeddings..."
    )
    # the new csv will have columns: video_path, data_source, category
    dfmk = csvfile.DFCreator()

    dfmk.create_table(
        table_name=TABLE_NAME, columns=["video_path", "data_source", "category"]
    )
    rows = []
    for vp in tqdm(video_paths, desc="Proc video ..."):
        vpath = os.path.abspath(vp)
        vfile_name = fs.get_file_name(vpath, split_file_ext=True)[0]
        vfile_name_parts = vfile_name.split(FILE_NAME_SEP)
        data_source = vfile_name_parts[0]
        category = vfile_name_parts[1]
        rows.append([vpath, data_source, category])
    dfmk.insert_rows(TABLE_NAME, rows)
    dfmk.fill_table_from_row_pool(TABLE_NAME)
    outfile = get_meta_csv_file_path(video_dir, out_dir)
    df = dfmk[TABLE_NAME].copy()
    df = df[["data_source", "category", "video_path"]]
    df.sort_values(by=["data_source", "category"], inplace=True)
    df.to_csv(
        outfile,
        index=False,
        sep=";",
        encoding="utf-8",
    )
    pprint(f"Saved video metadata CSV to: ⏬")
    pprint_local_path(outfile, get_wins_path=True)
    return df


def split_by_csv(csv_path, video_dir, out_dir, val_size=0.3, seed=42, add_time=True):
    seed_everything(seed)
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8")
    df["stratify_key"] = df["data_source"] + "_" + df["category"]
    try:
        test_df, val_df = train_test_split(
            df,
            test_size=0.3,  # 30% for Val, 70% for Test
            stratify=df["stratify_key"],
            random_state=42,
        )
        print("✅ Stratified split successful.")
    except ValueError as e:
        # Fallback/Handling for small datasets with single-item classes
        print("⚠️ Error:", e)
        print("Falling back to standard random split for classes with only 1 item.")

        # Filter out single-item classes to split stratifiably,
        # then add the single items randomly to either set.
        counts = df["stratify_key"].value_counts()
        single_items = df[df["stratify_key"].isin(counts[counts < 2].index)]
        multi_items = df[df["stratify_key"].isin(counts[counts >= 2].index)]

        test_df, val_df = train_test_split(
            multi_items,
            test_size=0.3,
            stratify=multi_items["stratify_key"],
            random_state=42,
        )

        # Randomly distribute the single items
        if not single_items.empty:
            s_test, s_val = train_test_split(
                single_items, test_size=0.3, random_state=42
            )
            test_df = pd.concat([test_df, s_test])
            val_df = pd.concat([val_df, s_val])
    # 4. Drop the helper column and check results

    test_df = test_df.drop(columns=["stratify_key"])
    val_df = val_df.drop(columns=["stratify_key"])

    # --- OUTPUT RESULTS ---
    print("\n--- Test Set (70%) ---")
    print(test_df[["data_source", "category"]].value_counts())

    print("\n--- Validation Set (30%) ---")
    print(val_df[["data_source", "category"]].value_counts())

    df_ls = [test_df, val_df]

    for df, set_name in zip(df_ls, SET_NAMES):
        out_csv_file = get_split_csv_file_path(video_dir, out_dir, set_name)
        df.to_csv(
            out_csv_file,
            index=False,
            sep=";",
            encoding="utf-8",
        )
        pprint(f"Saved {set_name} split CSV to: ⏬")
        pprint_local_path(out_csv_file, get_wins_path=True)


def copy_files(video_dir, out_dir):
    csv_files = [
        get_split_csv_file_path(video_dir, out_dir, split) for split in SET_NAMES
    ]
    assert all([os.path.exists(csv_file) for csv_file in csv_files]), (
        "Split CSV files not found. Please run split_by_csv first."
    )
    video_dir_name = fs.get_dir_name(video_dir)
    for csv_file, set_name in zip(csv_files, SET_NAMES):
        df = pd.read_csv(csv_file, sep=";", encoding="utf-8")
        out_set_dir = os.path.join(out_dir, f"{video_dir_name}_{set_name}")
        os.makedirs(out_set_dir, exist_ok=True)
        for _, row in tqdm(
            df.iterrows(), total=len(df), desc=f"Copying files to {set_name}..."
        ):
            # ! for video file
            video_src_path = row["video_path"]
            assert os.path.exists(video_src_path), (
                f"Video file not found: {video_src_path}"
            )
            file_name = os.path.basename(video_src_path)
            video_dst_path = os.path.join(out_set_dir, file_name)

            # ! for annotation file
            parent_dir = os.path.dirname(video_src_path)
            anno_file_name = f"{fs.get_file_name(video_src_path, split_file_ext=True)[0]}__labels.csv"
            anno_src_path = os.path.join(parent_dir, anno_file_name)
            anno_dst_path = os.path.join(out_set_dir, anno_file_name)
            assert os.path.exists(anno_src_path), (
                f"Annotation file not found: {anno_src_path}"
            )
            # Do actualy copying if no assertion errors
            fs.copy_file(video_src_path, video_dst_path)
            fs.copy_file(anno_src_path, anno_dst_path)


def main():
    # Parse arguments
    args = DBSplitCSVArgs().parse_args()
    # ! to ensure reproducibility
    seed_everything(args.seed)
    console.rule(f"[bold red] Task: {args.task} [/bold red]")
    if args.task == "mk_csv":
        get_video_dir_meta_csv(args.video_dir, args.out_dir)
    elif args.task == "split_by_csv":
        meta_csv_file = get_meta_csv_file_path(args.video_dir, args.out_dir)
        split_by_csv(
            csv_path=meta_csv_file,
            video_dir=args.video_dir,
            out_dir=args.out_dir,
            val_size=args.val_size,
            seed=args.seed,
            add_time=args.add_time,
        )
    elif args.task == "copy_files":
        copy_files(args.video_dir, args.out_dir)
    else:
        raise ValueError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    main()
