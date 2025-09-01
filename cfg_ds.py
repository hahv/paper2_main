from dataclasses import dataclass, field
from typing import List, Dict, Any

from dataclass_wizard import YAMLWizard


@dataclass
class General(YAMLWizard):
    seed: int
    outdir: str


@dataclass
class Dataset_cfg(YAMLWizard):
    dataset_dir: str
    dataset_list: List[str]
    block_size: int
    num_blocks_required: int
    splits: List[Dict[str, float]]


@dataclass
class DsConfig(YAMLWizard):
    general: General
    dataset_cfg: Dataset_cfg