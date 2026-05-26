"""EPD-based median baseline values per AIda category.

Boverket's climate database is organized by material composition (~200 generic
products), not by building component. For component categories that Boverket
lacks (notably golv and sanitet), the baseline agent falls back to LLM
estimation — which is often unreliable.

This module provides a middle tier: the median GWP-fossil A1-A3 value across
real Environdec EPDs in each AIda category. It's a category-aggregated typical
value — better than free-form LLM estimation, less precise than a Boverket
material proxy.

Source labels in the pipeline:
- "Boverkets klimatdatabas" → Tier 1: Boverket material proxy (existing)
- "Environdec EPD-medel"     → Tier 2: this module (new)
- "Uppskattning"             → Tier 3: LLM fallback when nothing else works

We only publish a median for (category, unit) pairs where the sample is
large enough AND reasonably homogeneous. Heterogeneous categories like
sanitet (a toilet, a cistern and a kitchen sink in the same bucket) are
better left to "Uppskattning" until subcategory-aware medians exist.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from statistics import median

logger = logging.getLogger(__name__)

EPD_DATA_PATH = Path(__file__).parent / "epd_alternatives.json"

# Minimum samples to publish a median. Below this, the value is too noisy
# to be a useful default — fall back to LLM estimation.
_MIN_SAMPLES = 5

# Categories that mix structurally different product types in the same
# bucket (e.g. sanitet covers toilets, sinks, taps — wildly different
# CO2e profiles). A category-aggregated median is misleading here, so we
# exclude them until subcategory-aware medians are built.
_HETEROGENEOUS_CATEGORIES = {"sanitet", "belysning", "vitvaror", "storköksutrustning"}


def _load_epd_data() -> list[dict]:
    if not EPD_DATA_PATH.exists():
        return []
    try:
        with open(EPD_DATA_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load EPD data: %s", e)
        return []


def _compute_medians() -> dict[tuple[str, str], dict]:
    """Compute median GWP per (category, unit). Returns lookup dict.

    Each entry has: median_co2e_per_unit, sample_size, min, max.
    """
    epds = _load_epd_data()
    if not epds:
        return {}

    grouped: dict[tuple[str, str], list[float]] = {}
    for e in epds:
        cat = e.get("category", "")
        unit = e.get("unit", "")
        gwp = e.get("gwp_a1a3")
        if not cat or not unit or not isinstance(gwp, (int, float)) or gwp <= 0:
            continue
        if cat in _HETEROGENEOUS_CATEGORIES:
            continue
        grouped.setdefault((cat, unit), []).append(float(gwp))

    result: dict[tuple[str, str], dict] = {}
    for key, values in grouped.items():
        if len(values) < _MIN_SAMPLES:
            continue
        result[key] = {
            "median_co2e_per_unit": round(median(values), 2),
            "sample_size": len(values),
            "min": round(min(values), 2),
            "max": round(max(values), 2),
        }
    return result


_MEDIANS: dict[tuple[str, str], dict] | None = None


def get_baseline_median(category: str, unit: str) -> dict | None:
    """Look up EPD-median baseline value for a category + unit combination.

    Returns a dict with median_co2e_per_unit, sample_size, min, max — or None
    if no usable median exists for this (category, unit) pair.

    Cached lazily on first call. Cheap enough to compute every time but the
    EPD file is bundled and immutable per deploy.
    """
    global _MEDIANS
    if _MEDIANS is None:
        _MEDIANS = _compute_medians()
    return _MEDIANS.get((category, unit))


def list_available_categories() -> list[tuple[str, str, int]]:
    """List all (category, unit, sample_size) tuples that have a published
    median. Useful for diagnostics and documentation.
    """
    global _MEDIANS
    if _MEDIANS is None:
        _MEDIANS = _compute_medians()
    return sorted(
        [(cat, unit, data["sample_size"]) for (cat, unit), data in _MEDIANS.items()],
        key=lambda x: (x[0], x[1]),
    )


def main():
    """CLI: print the median table for inspection."""
    print(f"{'Kategori':<25} {'Unit':<6} {'n':>3} {'min':>8} {'median':>8} {'max':>8}")
    print("-" * 70)
    if _MEDIANS is None:
        _compute_medians()
    for (cat, unit), data in sorted(_compute_medians().items()):
        print(
            f"{cat:<25} {unit:<6} {data['sample_size']:>3} "
            f"{data['min']:>8.2f} {data['median_co2e_per_unit']:>8.2f} {data['max']:>8.2f}"
        )


if __name__ == "__main__":
    main()
