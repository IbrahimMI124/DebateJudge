from __future__ import annotations

from typing import Any, Iterable, List, Sequence


def ensure_list(value: Any) -> List[Any]:
    """Return `value` as a list; treat None as empty list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    # Accept any iterable, but avoid treating strings as iterables here.
    if isinstance(value, (tuple, set)):
        return list(value)
    raise TypeError(f"Expected list (or None), got {type(value).__name__}")


def stable_sorted_by_id(statements: Sequence[dict]) -> List[dict]:
    # Sorting is stable in Python; this preserves relative order for ties.
    return sorted(statements, key=lambda x: x["id"])
