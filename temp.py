from halib import *
from src.utils import *

pkg_cls = {
    "src.methods": ["no_temp_method", "temp_Baseline_TPT_method", "temp_method"],
    "src.metrics": ["csv_metric_src"],
    "src.results": ["csv_rs_proc", "video_base_rs_proc", "video_rs_fgmask_proc"],
}

for pkg_name, file_names in pkg_cls.items():
    for file_name in file_names:
        cls = get_cls_in_pkg(pkg_name, fileName_or_fileNameAndClsName=file_name)
        pprint(cls)