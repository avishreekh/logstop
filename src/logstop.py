import math
from typing import Dict, Mapping, Optional, Tuple


class LTLFormula:
    def __init__(self, op, left=None, right=None):
        self.op = op
        self.left = left
        self.right = right

    def __str__(self):
        if self.op == "True":
            return "True"
        elif self.op == "False":
            return "False"
        elif self.op == "class":
            return str(self.left)
        elif self.op == "not":
            return f"not {self.left}"
        elif self.op == "and":
            return f"{self.left} and {self.right}"
        elif self.op == "or":
            return f"{self.left} or {self.right}"
        elif self.op == "next":
            return f"next {self.left}"
        elif self.op == "until":
            return f"{self.left} until {self.right}"
        elif self.op == "always":
            return f"always {self.left}"
        elif self.op == "eventually":
            return f"eventually {self.left}"
        raise ValueError(f"Unknown operator: {self.op}")


def log(x):
    if x <= 0.0:
        return float("-inf")
    return math.log(x)


def exp(x):
    if x == float("-inf"):
        return 0.0
    return math.exp(x)


def _local_properties(phi: LTLFormula) -> set:
    if phi.op == "class":
        return {phi.left}
    if phi.op in {"not", "next", "always", "eventually"}:
        return _local_properties(phi.left)
    if phi.op in {"and", "or", "until"}:
        return _local_properties(phi.left) | _local_properties(phi.right)
    return set()


def _log_not(score: float) -> float:
    if score >= 0.0:
        return float("-inf")
    if score == float("-inf"):
        return 0.0
    return math.log1p(-math.exp(score))


def _log_or(left: float, right: float) -> float:
    probability = exp(left) + exp(right) - exp(left + right)
    return log(min(1.0, max(0.0, probability)))


def logstop(
    processed_trace: Mapping[str, list],
    phi: LTLFormula,
    start_idx: int,
    end_idx: int,
    memo: Optional[Dict[Tuple[LTLFormula, int], float]] = None,
) -> float:
    """Score a preprocessed trace over inclusive interval ``[start_idx, end_idx]``.

    A memo may be shared by calls with different starts only when the processed
    trace, formula, and end index are unchanged.
    """
    if start_idx < 0:
        raise ValueError("start_idx must be non-negative")
    if start_idx > end_idx:
        return float("-inf")

    for prop in _local_properties(phi):
        if prop not in processed_trace:
            raise KeyError(f"trace has no predictions for local property {prop!r}")
        if end_idx >= len(processed_trace[prop]):
            raise IndexError(
                f"end_idx {end_idx} is outside the prediction trace for {prop!r}"
            )
    cache = memo if memo is not None else {}

    def score(formula: LTLFormula, timestep: int) -> float:
        key = (formula, timestep)
        if key in cache:
            return cache[key]

        if timestep > end_idx:
            result = float("-inf")
        elif formula.op == "True":
            result = 0.0
        elif formula.op == "False":
            result = float("-inf")
        elif formula.op == "class":
            result = log(processed_trace[formula.left][timestep])
        elif formula.op == "not":
            result = _log_not(score(formula.left, timestep))
        elif formula.op == "and":
            result = score(formula.left, timestep) + score(formula.right, timestep)
        elif formula.op == "or":
            result = _log_or(
                score(formula.left, timestep), score(formula.right, timestep)
            )
        elif formula.op == "next":
            result = score(formula.left, timestep + 1)
        elif formula.op == "until":
            right = score(formula.right, timestep)
            if timestep == end_idx:
                result = right
            else:
                continuation = score(formula.left, timestep) + score(
                    formula, timestep + 1
                )
                result = _log_or(right, continuation)
        elif formula.op == "always":
            result = score(formula.left, timestep)
            if timestep < end_idx:
                result += score(formula, timestep + 1)
        elif formula.op == "eventually":
            result = score(formula.left, timestep)
            if timestep < end_idx:
                result = _log_or(result, score(formula, timestep + 1))
        else:
            raise ValueError(f"Unknown operator: {formula.op}")

        cache[key] = result
        return result

    return score(phi, start_idx)
