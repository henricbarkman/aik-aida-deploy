"""EPD-based baseline typvärden per AIda category.

Boverket's climate database is organized by material composition (~200 generic
products), not by building component. For component categories that Boverket
lacks (notably golv and sanitet), the baseline agent falls back to LLM
estimation — which is often unreliable.

This module provides a middle tier: a category-aggregated "typvärde" derived
from the upper half of Environdec EPDs (by climate impact) in each AIda
category. It approximates "what conventional standard materials cost
climate-wise" — matching the NollCO2 methodology's "Typical" framing.

Why upper-half, not all-EPD median?
EPD databases skew toward climate-conscious producers — getting an EPD is
voluntary, and product manufacturers who care about climate document their
products. The median across ALL EPDs therefore underestimates what a user
who isn't actively climate-optimizing would actually choose. The upper half
(by GWP) is a better proxy for "default conventional choice".

Statistically: median of the upper 50% of values (sorted by GWP). For large
samples this approximates the 75th percentile but with less sensitivity to
single outliers in small samples — important since our category sample
sizes are often 5-15.

Source labels in the pipeline:
- "Boverkets klimatdatabas" → Tier 1: Boverket material proxy (existing)
- "Environdec EPD-typvärde" → Tier 2: this module
- "Uppskattning"            → Tier 3: LLM fallback when nothing else works

We only publish a typvärde for (category, unit) pairs where the sample is
large enough AND reasonably homogeneous. Heterogeneous categories like
sanitet (a toilet, a cistern and a kitchen sink in the same bucket) are
better left to "Uppskattning" until subcategory-aware typvärden exist.
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


def _upper_half_median(values: list[float]) -> float:
    """Median of the upper 50% of values (sorted ascending).

    Approximates the 75th percentile but more robust to single outliers in
    small samples. For odd N, includes the middle element in the upper half
    (slicing v[N//2:] gives ceil(N/2) elements).
    """
    s = sorted(values)
    upper = s[len(s) // 2:]
    return float(median(upper))


def _compute_typvärden() -> dict[tuple[str, str], dict]:
    """Compute upper-half median GWP per (category, unit). Returns lookup dict.

    Each entry has: baseline_co2e_per_unit, sample_size, full_median, min, max.
    full_median is kept for diagnostic purposes (compare typvärde against
    naive median).
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
            "baseline_co2e_per_unit": round(_upper_half_median(values), 2),
            "sample_size": len(values),
            "full_median": round(median(values), 2),
            "min": round(min(values), 2),
            "max": round(max(values), 2),
        }
    return result


_TYPVÄRDEN: dict[tuple[str, str], dict] | None = None


def get_baseline_typvärde(category: str, unit: str) -> dict | None:
    """Look up EPD-baseline typvärde for a category + unit combination.

    Returns a dict with baseline_co2e_per_unit, sample_size, full_median,
    min, max — or None if no usable typvärde exists for this (category, unit)
    pair.

    Cached lazily on first call.
    """
    global _TYPVÄRDEN
    if _TYPVÄRDEN is None:
        _TYPVÄRDEN = _compute_typvärden()
    return _TYPVÄRDEN.get((category, unit))


# Back-compat alias — old call sites used "median" terminology before we
# switched to upper-half methodology. Same value, clearer name.
def get_baseline_median(category: str, unit: str) -> dict | None:
    """Deprecated — use get_baseline_typvärde. Kept for back-compat."""
    data = get_baseline_typvärde(category, unit)
    if data is None:
        return None
    # Synthesize the old key name from the new structure
    return {
        **data,
        "median_co2e_per_unit": data["baseline_co2e_per_unit"],
    }


def list_available_categories() -> list[tuple[str, str, int]]:
    """List all (category, unit, sample_size) tuples with a published typvärde."""
    global _TYPVÄRDEN
    if _TYPVÄRDEN is None:
        _TYPVÄRDEN = _compute_typvärden()
    return sorted(
        [(cat, unit, data["sample_size"]) for (cat, unit), data in _TYPVÄRDEN.items()],
        key=lambda x: (x[0], x[1]),
    )


def main():
    """CLI: print the typvärde table for inspection."""
    print(f"{'Kategori':<25} {'Unit':<6} {'n':>3} {'min':>8} {'med':>8} {'typvärde':>10} {'max':>8}")
    print("-" * 80)
    for (cat, unit), data in sorted(_compute_typvärden().items()):
        print(
            f"{cat:<25} {unit:<6} {data['sample_size']:>3} "
            f"{data['min']:>8.2f} {data['full_median']:>8.2f} "
            f"{data['baseline_co2e_per_unit']:>10.2f} {data['max']:>8.2f}"
        )


if __name__ == "__main__":
    main()
