from temporal.methods.base_method import *
from torchvision import transforms
from PIL import Image
from temporal.config import Config
from temporal.metric_src.test_metric_src import TestDSMetricSrc
from temporal.methods.no_temp import NoTempMethod
import copy
import glob
from halib.system import filesys as fs
from halib import *

class TempTptMethod(NoTempMethod):

    def _run_tpt_method_on_df(self, df):
        """
        Run the temporal persistence thresholding (TPT) method on the DataFrame."""
        assert (
            self.methodUsed.method_name == "temp_tpt"
        ), f"Method {self.methodUsed.method_name} is not supported for this operation."
        persistence_thres = self.persistence_thres
        pos = 0
        temporal_buffer = np.zeros(self.window_size, dtype=bool)
        detected = False
        for idx, row in df.iterrows():
            # Get the current prediction
            # pprint(row)
            pred_lb = row["pred_label"]
            pred_lb_lower = pred_lb.lower()
            is_fire_or_smoke = ("fire" in pred_lb_lower) or ("smoke" in pred_lb_lower)
            if is_fire_or_smoke:
                temporal_buffer[pos] = True
            pos = (idx + 1) % self.window_size  # circular buffer
            # Check if the buffer has enough positive predictions
            num_det_frames = np.sum(temporal_buffer)
            if num_det_frames >= persistence_thres * self.window_size:
                detected = True
                break
        return self.POSITIVE_LABEL if detected else self.NEGATIVE_LABEL

    def before_infer_video(self, video_path: str):
        assert self.cfg.method_cfg.method_used.method_name == "temp_tpt", f"Method {self.cfg.method_cfg.method_used.method_name} is not supported for this operation"
        self.window_size = self.cfg.method_cfg.method_used.extra_cfgs['window_size']
        self.persist_thres = self.cfg.method_cfg.method_used.extra_cfgs["persist_thres"]
        self.temporal_buffer = np.zeros(self.window_size, dtype=bool)
        self.pos = 0

    def _check_no_temp_exists(self):
        no_temp_cfg = copy.deepcopy(self.cfg)
        no_temp_cfg.method_cfg.selected_method_name = "no_temp"
        no_temp_cfg.method_cfg.post_init() # for to rebuild method_used
        no_temp_cfg.post_init() # for to rebuild cfg_name
        cfg_name = no_temp_cfg.get_cfg_name()
        # find the last "__" in cfg_name
        last_underscore_idx = cfg_name.rfind("__")
        if last_underscore_idx != -1:
            cfg_name = cfg_name[:last_underscore_idx]
        pprint(f"{cfg_name=}")
        master_out_dir = self.cfg.general.outdir
        # find the subdir in master_out_dir that contains cfg_name** (glob)
        search_pattern = os.path.join(master_out_dir, f"{cfg_name}*")
        matched_dirs = glob.glob(search_pattern)
        if len(matched_dirs) == 0:
            pprint(f"No output directory found for NoTempMethod with pattern {search_pattern}")
            return None
        else:
            return matched_dirs[0]
    def infer_video_dir(self, video_dir: str, recursive: bool = True):
        # Check if NoTempMethod output exists, copy out csv files to current outdir
        no_temp_outdir = self._check_no_temp_exists()
        if no_temp_outdir is None:
            super().infer_video_dir(video_dir, recursive)
        else:
            no_temp_outdir = os.path.abspath(no_temp_outdir)
            csv_files = fs.filter_files_by_extension(no_temp_outdir, [".csv"], recursive=False)
            # ignore all performance csv files "__perf__"
            csv_files = [f for f in csv_files if "__perf__" not in os.path.basename(f)]
            assert len(csv_files) > 0, f"No CSV files found in {no_temp_outdir}"
            num_videos = self.cfg.dataset_cfg.dataset_used.get_num_videos()
            assert len(csv_files) == num_videos, f"Number of CSV files ({len(csv_files)}) does not match number of videos ({num_videos})"
            targetdir = self.cfg.get_outdir()
            for csv_file in tqdm(csv_files, desc="Cp from NoTemp..."):
                fs.copy_file(csv_file, targetdir)
