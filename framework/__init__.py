"""Proactive AssetOpsBench v3.0 universal benchmark infrastructure."""

from __future__ import annotations

from datetime import datetime
from typing import Any

SCHEMA_VERSION = "3.0"
SUPPORTED_SCHEMA_VERSIONS = {"2.0", "3.0"}
VALID_MODALITIES = {"telemetry", "alarm", "maintenance_log", "workorder_history"}


def parse_iso_utc(ts: str) -> datetime:
    """Parse an ISO-8601 UTC timestamp string (Python 3.10 compatible)."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def get_asset_map(scenario: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build asset_id -> asset metadata mapping, falling back to asset_info."""
    assets = scenario.get("assets") or []
    mapping = {a["asset_id"]: a for a in assets if isinstance(a, dict) and a.get("asset_id")}
    primary = scenario.get("asset_info") or {}
    if primary.get("asset_id") and primary["asset_id"] not in mapping:
        mapping[primary["asset_id"]] = primary
    return mapping


def recursive_subset_match(expected: Any, actual: Any) -> bool:
    """Recursively verify that `expected` is a structural subset of `actual`."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return all(k in actual and recursive_subset_match(v, actual[k]) for k, v in expected.items())
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False
        return all(any(recursive_subset_match(exp_i, act_i) for act_i in actual) for exp_i in expected)
    if isinstance(expected, bool) != isinstance(actual, bool):
        return False
    return bool(expected == actual)
