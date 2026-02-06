from halib import *
# from src.results.timeline.report_helper import TimelineReportHelper
from src.results.timeline.data_parser import *
from src.results.timeline.report_helper import TimelineReportHelper


def rand_mock_data_by_timeline_type(
    total_frames=500, timeline_type="gt"
) -> pd.DataFrame:
    label_options = list(TLParser.get_labels_colors_by_type(timeline_type).keys())
    num_segments = np.random.randint(2, 6)
    split_indices = np.sort(
        np.random.choice(range(1, total_frames), num_segments - 1, replace=False)
    )
    boundaries = [0] + list(split_indices) + [total_frames]
    all_labels = []
    # 4. For each segment, pick one random label and repeat it
    for i in range(num_segments):
        seg_label = np.random.choice(label_options)
        seg_length = boundaries[i + 1] - boundaries[i]
        # Extend our list by repeating the chosen label for the length of the segment
        all_labels.extend([seg_label] * seg_length)
    return pd.DataFrame(
        {
            "gt_label": all_labels,
        }
    )


def gen_mock_timelines_data(
    total_frames=500, timeline_types=["gt", "noskip", "skip"]
) -> pd.DataFrame:
    """Generates the frame_id | gt_label | method_X structure."""
    df_src_dict = {
        "frame_id": range(total_frames),
    }
    for t_type in timeline_types:
        df_part = rand_mock_data_by_timeline_type(total_frames, t_type)
        df_src_dict[t_type] = df_part["gt_label"]
    return pd.DataFrame(df_src_dict)

def main():
    num_videos = 3
    timeline_types = ["gt", "no_skip", "skip"]
    for i in range(num_videos):
        total_frames = np.random.randint(300, 800)
        df = gen_mock_timelines_data(
            total_frames=total_frames, timeline_types=timeline_types
        )
        print(f"=== Video {i+1} Data ===")
        # csvfile.fn_display_df(df.head(10))
    # 2. Process Videos
    stats_list = []
    helper = TimelineReportHelper

    for i in range(3):
        df = gen_mock_timelines_data()
        stats = helper.calculate_video_stats(f"Scenario_{i + 1}.mp4", df)
        stats_list.append(stats)

    OUTDIR = "./zout/reports"
    os.makedirs(OUTDIR, exist_ok=True)
    outfile = os.path.join(OUTDIR, "flexible_report.html")

    # 3. Generate HTML
    helper.render_report(stats_list, outfile)
    print(f"Report generated: ⏬")
    pprint_local_path(outfile, get_wins_path=True)


if __name__ == "__main__":
    main()
