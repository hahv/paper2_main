from halib import *
from halib.filetype import yamlfile
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description="desc text")
    parser.add_argument(
        "-indir",
        "--indir",
        type=str,
        help="input directory",
        default=r"/mnt/e/SyncData/paper2_baseline/zout/DFire_Val",
    )
    parser.add_argument(
        "-perf",
        "--perf",
        type=str,
        help="input directory",
        default=r"__perf_per_frame_results.csv",
    )
    # add bool flag to indicate whether to ignore cols with only one unique value
    parser.add_argument(
        "-i",
        "--ignore",
        action='store_true',
        help="whether to ignore columns with only one unique value",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    indir = args.indir
    perf_csv = args.perf
    ignore_one_value_cols = args.ignore

    all_dirs = fs.list_dirs(indir)
    all_dirs = sorted(all_dirs)  # asc order
    pprint(all_dirs)
    trial_dict_ls = []
    ignore_cols = ["tiny_model", "video_rs_handler", "frame_diff_cfg"]
    for idx, d in enumerate(tqdm(all_dirs)):
        cfg_file = os.path.join(indir, d, "__config.yaml")
        cfg_dict = yamlfile.load_yaml(cfg_file, to_dict=True)
        method_params = cfg_dict['method-cfg']['method-used']['extra-cfgs']
        trial_dict = {'trial_name': d}
        trial_dict.update(method_params)
        trial_dict_ls.append(trial_dict)
    df = pd.DataFrame(trial_dict_ls)
    df = df.drop(columns=ignore_cols, errors='ignore')
    # csvfile.fn_display_df(df)

    perf_csv = os.path.join(indir, perf_csv)
    perf_df = pd.read_csv(perf_csv, sep=";", index_col=False, encoding='utf-8')
    # merge
    tune_df = pd.merge(df, perf_df, left_on='trial_name', right_on='method', how='inner')
    if ignore_one_value_cols:
        col_have_only_one_value = []
        INDEX_OF_METHOD_COLS = tune_df.columns.to_list().index('method')
        for col in tune_df.columns.tolist()[:INDEX_OF_METHOD_COLS]:
            unique_vals = tune_df[col].unique()
            if len(unique_vals) == 1:
                col_have_only_one_value.append(col)
        col_have_only_one_value.append('method') # also drop the 'method' col
        tune_df = tune_df.drop(columns=col_have_only_one_value, errors='ignore')
    csvfile.fn_display_df(tune_df)
    tune_df.to_csv('./zreport/tuning_results.csv', index=False, sep=';', encoding='utf-8')

if __name__ == "__main__":
    main()
