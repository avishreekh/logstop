from numbers import Integral
from typing import Dict, Mapping, Union


SmoothingRadii = Union[int, Mapping[str, int]]


def _normalize_radii(properties: set, radii: SmoothingRadii) -> Dict[str, int]:
    if isinstance(radii, Integral):
        normalized = {prop: int(radii) for prop in properties}
    elif isinstance(radii, Mapping):
        normalized = {prop: radii.get(prop, 0) for prop in properties}
    else:
        raise TypeError("smoothing_radii must be an integer or a mapping")

    for prop, radius in normalized.items():
        if not isinstance(radius, Integral):
            raise TypeError(f"smoothing radius for {prop!r} must be an integer")
        if radius < 0:
            raise ValueError(f"smoothing radius for {prop!r} must be non-negative")
        normalized[prop] = int(radius)
    return normalized


def preprocess_trace(
    trace: Mapping[str, list], smoothing_radii: SmoothingRadii = 0
) -> Dict[str, list]:
    """Smooth a complete local-property trace using centered window averages.

    Preprocessing is independent of any subsequently scored subsequence. The
    returned lists retain the input trace's global timestep indexing.
    """
    properties = set(trace)
    radii = _normalize_radii(properties, smoothing_radii)
    processed = {}

    for prop, values in trace.items():
        prefix = [0.0]
        for value in values:
            prefix.append(prefix[-1] + value)

        radius = radii[prop]
        result = [0.0] * len(values)
        for timestep in range(len(values)):
            left = max(timestep - radius, 0)
            right = min(timestep + radius, len(values) - 1)
            total = prefix[right + 1] - prefix[left]
            result[timestep] = total / (right - left + 1)
        processed[prop] = result

    return processed
