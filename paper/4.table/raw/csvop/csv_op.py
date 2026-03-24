from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
import pandas as pd
from tap import Tap
from ..common.common import *

# ── CLI Args ──────────────────────────────────────────────────────────────────


class CsvOpArgs(Tap):
    cfg: str = ""
    input: Optional[str] = None
    output: Optional[str] = None

    def configure(self):
        self.add_argument("-cfg", "--cfg", help="Path to transform YAML")
        self.add_argument("-i", "--input", help="Override input CSV path")
        self.add_argument("-o", "--output", help="Override output CSV path")


# ── Base SubStep ──────────────────────────────────────────────────────────────


class BaseSubStep(ABC):
    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame: ...

    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict) -> "BaseSubStep": ...

    def __repr__(self):
        return f"<{self.__class__.__name__}>"


# ── Concrete SubSteps ─────────────────────────────────────────────────────────


@dataclass
class RenameStep(BaseSubStep):
    mapping: Dict[str, str]

    def apply(self, df):
        return df.rename(columns=self.mapping)

    @classmethod
    def from_dict(cls, d):
        return cls(mapping=d["mapping"])


@dataclass
class DropStep(BaseSubStep):
    columns: List[str]

    def apply(self, df):
        return df.drop(columns=self.columns, errors="ignore")

    @classmethod
    def from_dict(cls, d):
        return cls(columns=d["columns"])


@dataclass
class FillNaStep(BaseSubStep):
    mapping: Dict[str, Any]

    def apply(self, df):
        return df.fillna(self.mapping)

    @classmethod
    def from_dict(cls, d):
        return cls(mapping=d["mapping"])


@dataclass
class CastStep(BaseSubStep):
    _CAST_MAP = {"int": int, "float": float, "str": str, "bool": bool}
    mapping: Dict[str, str]

    def apply(self, df):
        for col, typ in self.mapping.items():
            if col not in df.columns:
                continue
            if typ == "datetime":
                df[col] = pd.to_datetime(df[col], errors="coerce")
            else:
                df[col] = df[col].astype(self._CAST_MAP[typ])
        return df

    @classmethod
    def from_dict(cls, d):
        return cls(mapping=d["mapping"])

@dataclass
class StrOpStep(BaseSubStep):
    """Groups multiple string ops in one step block."""

    @dataclass
    class Op:
        column: str
        op: str
        search: Optional[str] = None
        replace_with: Optional[str] = None
        new_col: Optional[str] = None  # ← new: write to different column

    ops: List[Op]

    def apply(self, df):
        for o in self.ops:
            if o.column not in df.columns:
                continue
            col = df[o.column]
            target = o.new_col if o.new_col else o.column  # ← new
            if o.op == "strip":
                df[target] = col.str.strip()
            elif o.op == "lower":
                df[target] = col.str.lower()
            elif o.op == "upper":
                df[target] = col.str.upper()
            elif o.op == "replace":
                df[target] = col.str.replace(o.search, o.replace_with, regex=False)
            elif o.op == "regex_replace":  # ← new
                df[target] = col.str.replace(o.search, o.replace_with or "", regex=True)
        return df

    @classmethod
    def from_dict(cls, d):
        ops = [
            cls.Op(
                column=o["column"],
                op=o["op"],
                search=o.get("search"),
                replace_with=o.get("replace_with") or o.get("replace-with"),
                new_col=o.get("new_col") or o.get("new-col"),  # ← new
            )

            for o in d["ops"]
        ]
        return cls(ops=ops)

@dataclass
class FilterStep(BaseSubStep):
    """Groups multiple filter conditions in one step block (all AND)."""

    @dataclass
    class Condition:
        column: str
        op: str
        value: Optional[Any] = None

    _OPS = {
        "==": lambda c, v: c == v,
        "!=": lambda c, v: c != v,
        ">": lambda c, v: c > v,
        ">=": lambda c, v: c >= v,
        "<": lambda c, v: c < v,
        "<=": lambda c, v: c <= v,
        "in": lambda c, v: c.isin(v),
        "not_in": lambda c, v: ~c.isin(v),
        "contains": lambda c, v: c.str.contains(v, case=False, na=False),
        "startswith": lambda c, v: c.str.startswith(v, na=False),
        "endswith": lambda c, v: c.str.endswith(v, na=False),
        "isnull": lambda c, v: c.isnull(),
        "notnull": lambda c, v: c.notnull(),
    }

    conditions: List[Condition]

    def apply(self, df):
        for cond in self.conditions:
            try:
                df = df[self._OPS[cond.op](df[cond.column], cond.value)]
            except KeyError as e:
                raise KeyError(
                    f"FilterStep: column {e} not found. Available: {list(df.columns)}"
                ) from None
        return df

    @classmethod
    def from_dict(cls, d):
        conditions = [
            cls.Condition(column=c["column"], op=c["op"], value=c.get("value"))
            for c in d["conditions"]
        ]
        return cls(conditions=conditions)


@dataclass
class DedupeStep(BaseSubStep):
    subset: Optional[List[str]] = None
    keep: str = "first"

    def apply(self, df):
        return df.drop_duplicates(subset=self.subset, keep=self.keep)  # ty:ignore[no-matching-overload]

    @classmethod
    def from_dict(cls, d):
        return cls(subset=d.get("subset"), keep=d.get("keep", "first"))


@dataclass
class AddColumnStep(BaseSubStep):
    """Groups multiple derived column definitions in one step block."""

    @dataclass
    class Col:
        name: str
        expr: str

    cols: List[Col]

    def apply(self, df):
        for col in self.cols:
            try:
                df[col.name] = eval(col.expr)
            except KeyError as e:
                raise KeyError(
                    f"AddColumnStep '{col.name}': column {e} not found. Available: {list(df.columns)}"
                ) from None
        return df

    @classmethod
    def from_dict(cls, d):
        cols = [cls.Col(name=c["name"], expr=c["expr"]) for c in d["cols"]]
        return cls(cols=cols)


@dataclass
class SelectStep(BaseSubStep):
    columns: List[str]

    def apply(self, df):
        return df[[c for c in self.columns if c in df.columns]]

    @classmethod
    def from_dict(cls, d):
        return cls(columns=d["columns"])


@dataclass
class SortStep(BaseSubStep):
    by: List[str]
    ascending: Any = True

    def apply(self, df):
        return df.sort_values(by=self.by, ascending=self.ascending)

    @classmethod
    def from_dict(cls, d):
        return cls(by=d["by"], ascending=d.get("ascending", True))


@dataclass
class LimitStep(BaseSubStep):
    n: int

    def apply(self, df):
        return df.head(self.n)

    @classmethod
    def from_dict(cls, d):
        return cls(n=d["n"])


@dataclass
class JoinStep(BaseSubStep):
    """Joins the current df with an external CSV file."""

    file: str
    how: str  # inner | left | right | outer
    on: Optional[List[str]] = None  # same key name on both sides
    left_on: Optional[List[str]] = None  # if key names differ
    right_on: Optional[List[str]] = None  # if key names differ
    sep: str = ";"
    suffixes: tuple = ("", "_right")

    def apply(self, df):
        # ✅ validate BEFORE try/except so ValueError is never caught
        if not self.on and not (self.left_on and self.right_on):
            raise ValueError(
                "JoinStep: provide either 'on' or both 'left_on' and 'right_on'"
            )

        right = pd.read_csv(self.file, sep=self.sep, encoding="utf-8")
        try:
            if self.on:
                return df.merge(right, on=self.on, how=self.how, suffixes=self.suffixes)
            else:
                return df.merge(
                    right,
                    left_on=self.left_on,
                    right_on=self.right_on,
                    how=self.how,
                    suffixes=self.suffixes,
                )
        except KeyError as e:
            raise KeyError(f"JoinStep: column {e} not found.") from None

    @classmethod
    def from_dict(cls, d):
        on = d.get("key")  # ← was d.get("on")
        if isinstance(on, str):
            on = [on]
        left_on = d.get("left_on")
        right_on = d.get("right_on")
        if isinstance(left_on, str):
            left_on = [left_on]
        if isinstance(right_on, str):
            right_on = [right_on]
        return cls(
            file=d["file"],
            how=d.get("how", "left"),
            on=on,
            left_on=left_on,
            right_on=right_on,
            sep=d.get("sep", ";"),
            suffixes=tuple(d.get("suffixes", ["", "_right"])),
        )


# ── Registry + Factory ────────────────────────────────────────────────────────

STEP_REGISTRY: Dict[str, type] = {
    "rename": RenameStep,
    "drop": DropStep,
    "fillna": FillNaStep,
    "cast": CastStep,
    "str_op": StrOpStep,
    "filter": FilterStep,
    "dedupe": DedupeStep,
    "add_column": AddColumnStep,
    "select": SelectStep,
    "sort": SortStep,
    "limit": LimitStep,
    "join": JoinStep,
}


def build_step(d: dict) -> BaseSubStep:
    typ = d.get("type", "").lower().replace("-", "_")
    cls = STEP_REGISTRY.get(typ)
    if cls is None:
        raise ValueError(
            f"Unknown step type: '{typ}'. Available: {list(STEP_REGISTRY)}"
        )
    return cls.from_dict(d)  # ty:ignore[unresolved-attribute]


# ── Root Config ───────────────────────────────────────────────────────────────

@dataclass
class CsvOpCfg:
    input: str
    output: str
    sep: str = ";"
    steps: List[BaseSubStep] = field(default_factory=list)

    @classmethod
    def from_yaml_file(cls, path: str) -> "CsvOpCfg":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(
            input=raw["input"],
            output=raw["output"],
            sep=raw.get("sep", ";"),
            steps=[build_step(s) for s in raw.get("steps", [])],
        )


# ── CsvOp ───────────────────────────────────────────────────────────────

class CsvOp:
    def __init__(self, cfg: CsvOpCfg):
        self.cfg = cfg

    @classmethod
    def from_yaml_file(cls, path: str) -> "CsvOp":
        return cls(CsvOpCfg.from_yaml_file(path))

    @staticmethod
    def _resolve_output(input_path: str, output_path: str) -> str:
        out = Path(output_path)
        if out.is_dir() or output_path.endswith("/") or output_path.endswith("\\"):
            out.mkdir(parents=True, exist_ok=True)
            return str(out / f"{Path(input_path).stem}_processed.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        return str(out)

    def run(self) -> tuple[str, pd.DataFrame]:
        df = pd.read_csv(self.cfg.input, sep=self.cfg.sep, encoding="utf-8")
        for step in self.cfg.steps:
            df = step.apply(df)
        outfile = self._resolve_output(self.cfg.input, self.cfg.output)
        df.to_csv(outfile, index=False, sep=self.cfg.sep, encoding="utf-8")
        return outfile, df


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = CsvOpArgs().parse_args()

    if not args.cfg:
        raise ValueError(
            "❌ --cfg is required. Usage: python csv_op.py -cfg transforms/users.yaml"
        )

    op = CsvOp.from_yaml_file(args.cfg)
    if args.input:
        op.cfg.input = args.input
    if args.output:
        op.cfg.output = args.output

    with ConsoleLog("Csv Processing..."):
        pprint_box(
            {
                "Input CSV": op.cfg.input,
                "Output CSV": op.cfg.output,
                "Cfg file": args.cfg,
            }
        )
    try:
        outfile, _ = op.run()
        pprint_local_path(
            outfile, tag_or_box_title="✅ Processed CSV saved to", get_wins_path=True
        )
    except Exception as e:
        pprint_stack_trace(msg="[CsvOp] Error occurred while processing CSV", e=e)
