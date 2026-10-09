"""Shared query matching over local-property prediction traces."""

import math

from src.logstop import logstop
from src.preprocess import preprocess_trace
from src.utils.ltl import extract_local_properties


def validate_thresholds(local_threshold):
    """Validate a scalar threshold or a mapping of property thresholds."""
    values = local_threshold.values() if isinstance(local_threshold, dict) else [local_threshold]
    if isinstance(local_threshold, dict) and (
        any(not isinstance(key, str) for key in local_threshold)
    ):
        raise ValueError("Threshold mapping must have property names as keys")
    if any(type(value) not in (int, float) or not math.isfinite(value)
           or not 0 <= value <= 1 for value in values):
        raise ValueError("local_threshold values must be between 0 and 1")


def match_query(trace, formula, local_threshold=0.5, smoothing_radii=0):
    """Return the LogSTOP score, adaptive threshold, and strict match decision."""
    validate_thresholds(local_threshold)
    properties = extract_local_properties(formula)
    if not properties:
        raise ValueError("Query matching requires at least one local property")
    if isinstance(local_threshold, dict):
        thresholds = {prop: local_threshold.get(prop, 0.5) for prop in properties}
    else:
        thresholds = {prop: local_threshold for prop in properties}
    selected = {}
    for prop in properties:
        if prop not in trace:
            raise ValueError(f"Missing prediction trace for {prop!r}")
        values = [float(value) for value in trace[prop]]
        if not values or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError(f"Invalid prediction trace for {prop!r}")
        selected[prop] = values
    lengths = {len(values) for values in selected.values()}
    if len(lengths) != 1:
        raise ValueError("Prediction traces have inconsistent lengths")
    length = lengths.pop()
    processed = preprocess_trace(selected, smoothing_radii)
    threshold_trace = preprocess_trace(
        {prop: [thresholds[prop]] * length for prop in properties}, smoothing_radii
    )
    score = logstop(processed, formula, 0, length - 1)
    threshold = logstop(threshold_trace, formula, 0, length - 1)
    return {"score": score, "threshold": threshold, "matched": score > threshold}
