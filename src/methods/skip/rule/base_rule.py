import numpy as np
from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List


class RuleStatus(Enum):
    PASS = "PASS"  # Condition Met
    FAIL = "FAIL"  # Condition Not Met
    SKIPPED = "SKIPPED"  # Logic short-circuited (e.g., first part of OR passed)


@dataclass
class RuleResult:
    rule_name: str
    status: RuleStatus
    details: Dict[str, Any] = field(default_factory=dict)
    sub_results: List["RuleResult"] = field(default_factory=list)

    def is_pass(self) -> bool:
        return self.status == RuleStatus.PASS


class BaseRule(ABC):
    def __init__(self, name: str = "", params: Optional[Dict[str, Any]] = None):
        self.name = name if len(name.strip()) > 0 else self.__class__.__name__
        self.params = params or {}
        self.block_id = self.params.get("block_idx", (0, 0))  # (row, col) of the block

    @abstractmethod
    def check(
        self, frame_or_roi: np.ndarray, extra_dict: Optional[Dict[str, Any]] = None
    ) -> RuleResult:
        pass


class BlockBasedRule(BaseRule):
    """
    Base class for rules that operate on image blocks.
    Inherits from BaseRule.
    """

    def __init__(self, name: str = "", params: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, params=params)
        self.block_id = self.params.get("block_idx", (0, 0))  # (row, col) of the block

    def prepare(self, extra_dict):
        if extra_dict and "block_id" in extra_dict:
            return {"block_id": extra_dict["block_id"]}
        return {}


# --- LOGIC RULES ---


class AnyRule(BaseRule):
    """(OR Logic) Returns PASS if ANY sub-rule passes."""

    def __init__(self, rules: List[BaseRule], name="AnyRule"):
        super().__init__(name=name)
        self.rules = rules

    def check(
        self, frame_or_roi: np.ndarray, extra_dict: Optional[Dict[str, Any]] = None
    ) -> RuleResult:
        sub_results = []
        final_status = RuleStatus.FAIL  # Default to FAIL if nothing passes

        for rule in self.rules:
            # If we already passed, we SKIP the remaining rules (Short-circuit optimization)
            if final_status == RuleStatus.PASS:
                sub_results.append(RuleResult(rule.name, RuleStatus.SKIPPED))
                continue

            # Run the check
            res = rule.check(frame_or_roi, extra_dict)
            sub_results.append(res)

            if res.is_pass():
                final_status = RuleStatus.PASS
                # We don't break immediately so we can log SKIPPED for others (optional)
                # If you want speed, break here.

        return RuleResult(self.name, final_status, sub_results=sub_results)


class AllRule(BaseRule):
    """(AND Logic) Returns PASS only if ALL sub-rules pass."""

    def __init__(self, rules: List[BaseRule], name="AllRule"):
        super().__init__(name=name)
        self.rules = rules

    def check(
        self, frame_or_roi: np.ndarray, extra_dict: Optional[Dict[str, Any]] = None
    ) -> RuleResult:
        sub_results = []
        final_status = RuleStatus.PASS  # Default to PASS, fail if any fails

        for rule in self.rules:
            # If we already failed, SKIP remaining (Short-circuit)
            if final_status == RuleStatus.FAIL:
                sub_results.append(RuleResult(rule.name, RuleStatus.SKIPPED))
                continue

            res = rule.check(frame_or_roi, extra_dict)
            sub_results.append(res)

            if not res.is_pass():
                final_status = RuleStatus.FAIL

        return RuleResult(self.name, final_status, sub_results=sub_results)
