from tap import Tap
import os
import yaml
from dataclasses import dataclass, field


ROWS = ["row_01_rgb", "row_02_mask"]


# ── Args ─────────────────────────────────────────────────────────────────────


class Args(Tap):
    yaml_path: str = "./zz_fig_extract.yaml"  # Path to the fig-extract YAML config file


# ── Data model ───────────────────────────────────────────────────────────────


@dataclass
class Condition:
    exp: str
    gt_label: str | None = None
    raw_label: str | list[str] | None = None


@dataclass
class SubCase:
    case_name: str
    case_desc: str
    num_cases: int
    conditions: list[Condition] = field(default_factory=list)


@dataclass
class Case:
    name: str
    sub_cases: list[SubCase]


# ── Parsing ───────────────────────────────────────────────────────────────────


def parse_condition(raw: dict) -> Condition:
    return Condition(
        exp=raw["exp"],
        gt_label=raw.get("gt_label"),
        raw_label=raw.get("raw_label"),
    )


def parse_sub_case(raw: dict) -> SubCase:
    return SubCase(
        case_name=raw["case_name"],
        case_desc=raw["case_desc"],
        num_cases=raw["num_cases"],
        conditions=[parse_condition(c) for c in raw.get("conditions", [])],
    )


def load_config(yaml_path: str) -> tuple[str, list[Case]]:
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    outdir = cfg["outdir"]
    cases = [
        Case(
            name=c["name"],
            sub_cases=[parse_sub_case(s) for s in c["sub_cases"]],
        )
        for c in cfg["list_cases"]
    ]
    return outdir, cases


# ── Folder creation ───────────────────────────────────────────────────────────


def get_sub_case_folders(col_idx: int, sub_case: SubCase) -> list[str]:
    """Return the list of folder names for a sub_case, one per num_cases."""
    base = f"col_{col_idx:02d}_{sub_case.case_name}"
    return [f"{base}_{i:02d}" for i in range(1, sub_case.num_cases + 1)]


def create_case_dirs(outdir: str, case: Case) -> None:
    case_dir = os.path.join(outdir, case.name)
    for row in ROWS:
        for col_idx, sub_case in enumerate(case.sub_cases, start=1):
            for folder_name in get_sub_case_folders(col_idx, sub_case):
                folder = os.path.join(case_dir, row, folder_name)
                os.makedirs(folder, exist_ok=True)
                print(f"  created: {folder}")


def create_all_dirs(outdir: str, cases: list[Case]) -> None:
    for case in cases:
        create_case_dirs(outdir, case)


# ── Pretty-print ──────────────────────────────────────────────────────────────


def _fmt_condition(c: Condition) -> str:
    label = f"gt={c.gt_label}" if c.gt_label else f"raw={c.raw_label}"
    return f"{c.exp}[{label}]"


def print_tree(outdir: str, cases: list[Case]) -> None:
    print("\nFolder structure:")
    for case in cases:
        print(f"\n{os.path.join(outdir, case.name)}/")
        for row in ROWS:
            print(f"  {row}/")
            for col_idx, sub_case in enumerate(case.sub_cases, start=1):
                cond_str = "  &  ".join(_fmt_condition(c) for c in sub_case.conditions)
                for folder_name in get_sub_case_folders(col_idx, sub_case):
                    print(f"    {folder_name}/  ← {cond_str}")


# ── Entry point ───────────────────────────────────────────────────────────────


def main(args: Args) -> None:
    outdir, cases = load_config(args.yaml_path)
    os.makedirs(outdir, exist_ok=True)
    create_all_dirs(outdir, cases)
    print_tree(outdir, cases)


if __name__ == "__main__":
    main(Args().parse_args())
