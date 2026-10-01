"""Price and CO2 reasonableness validation for Aida.

Validates prices and CO2 values against expected ranges per component category.
Mild outliers get flagged ("verifiera"). Extreme outliers get clamped to
the range midpoint — showing a wrong number with confidence is worse than
showing a reasonable estimate with a caveat.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

logger = logging.getLogger(__name__)

# Reasonable price ranges per category in SEK per unit.
# Keys match normalize_component_name() output from climate_data.py.
# Format: (min_sek_per_unit, max_sek_per_unit, unit)
PRICE_RANGES: dict[str, tuple[float, float, str]] = {
    "fönster":       (5_000, 25_000, "st"),
    "tak":           (300, 2_000, "m2"),
    "yttervägg":     (500, 3_000, "m2"),    # facade insulation / cladding
    "isolering":     (50, 1_500, "m2"),
    "golv":          (200, 1_500, "m2"),
    "innervägg":     (300, 2_000, "m2"),
    "betongvägg":    (500, 3_000, "m2"),
    "belysning":     (500, 15_000, "st"),
    "ventilation":   (200, 5_000, "m2"),
    "dörr":          (3_000, 30_000, "st"),
    "hiss":          (300_000, 2_000_000, "st"),
    "diskmaskin":    (15_000, 150_000, "st"),
    "kylanläggning": (30_000, 500_000, "st"),
    "sanitet":       (500, 15_000, "st"),
    "vitvaror":      (3_000, 50_000, "st"),
    "storköksutrustning": (10_000, 200_000, "st"),
}

# Catch-all for categories not listed above.
FALLBACK_MIN = 10       # SEK per unit — below this is nonsense
FALLBACK_MAX = 50_000   # SEK per unit — above this needs verification (except known big items)

# Factor: if price is more than CLAMP_FACTOR × range max (or less than
# range min / CLAMP_FACTOR), the value is almost certainly garbage from
# web search and gets clamped to the range midpoint.
CLAMP_FACTOR = 3.0

# Reasonable CO2e ranges per category in kg CO2e per unit (A1-A3).
#
# yttervägg, betongvägg and tak re-derived 2026-09-30 from the published
# EPD typvärde table (epd_baseline_medians): at 40, 80 and 25 kg/m2 their own
# typvärden (55.0, 110 and 35.1) sat above the range, so every such baseline
# was stamped "Oväntat CO2e-värde" by construction. Each max now holds its
# typvärde, and three times it holds the heaviest catalog row (175, 400, 56.3),
# so a real declared product is at most flagged, never clamped.
# test_baseline_clamp checks every published typvärde against its range.
CO2_RANGES: dict[str, tuple[float, float, str]] = {
    "golv":          (2, 20, "m2"),
    "innervägg":     (1, 15, "m2"),
    "yttervägg":     (5, 70, "m2"),
    "betongvägg":    (10, 150, "m2"),
    "fönster":       (20, 150, "st"),
    "tak":           (2, 45, "m2"),
    "isolering":     (1, 15, "m2"),
    "dörr":          (15, 100, "st"),
    "belysning":     (1, 30, "st"),
    "ventilation":   (1, 20, "lm"),
    "hiss":          (2_000, 30_000, "st"),
    "sanitet":       (10, 80, "st"),
    "vitvaror":      (50, 500, "st"),
    "kylanläggning": (100, 5_000, "st"),
    "storköksutrustning": (100, 2_000, "st"),
}


def _bounds(table: dict[str, tuple[float, float, str]], category: str,
            unit: str = "") -> tuple[float, float, str] | None:
    """The range for `category`, or None when the component is counted in a
    different unit than the range is written in.

    Every range carries its unit and until 2026-09-28 nothing read it. The
    ventilation CO2 range is 1-20 kg per lm, a duct; applied per piece it
    clamped an air handling unit's 4 500 kg estimate to 10.5 kg, and the
    innervägg price range (SEK per m2) lifted studs counted in lm from 90 to
    1 150 SEK per metre. A range in another unit says nothing about the value,
    so it is not applied. `unit` empty keeps the old behaviour for callers
    that do not know the component's unit.
    """
    bounds = table.get(category.lower().strip())
    if not bounds or not unit:
        return bounds
    from aida.data.pricing_provider import normalize_price_unit
    if normalize_price_unit(unit) != normalize_price_unit(bounds[2]):
        return None
    return bounds


def validate_unit_price(
    price_per_unit: float,
    category: str,
    *,
    is_estimate: bool = False,
    unit: str = "",
) -> tuple[float, str]:
    """Validate a per-unit price and return (price, note).

    Mild outliers (outside range but within CLAMP_FACTOR) get flagged.
    Extreme outliers (beyond CLAMP_FACTOR × range) get clamped to midpoint
    — a wrong number shown with confidence is worse than a reasonable estimate.

    Args:
        price_per_unit: Cost in SEK per unit (m2, st, etc.).
        category: Normalized component category key.
        is_estimate: True if the price came from LLM estimation rather than
                     a verified source (web search, EPD, database).

    Returns:
        (price, note) where note is empty if OK, or a flag string to append.
        Price may be adjusted if it was an extreme outlier.
    """
    if price_per_unit <= 0:
        return 0, "Pris ej tillgängligt"

    note = ""

    # Tag LLM estimates regardless of range
    if is_estimate:
        note = "Approximerat pris"

    # Check category-specific range
    cat_key = category.lower().strip()
    bounds = _bounds(PRICE_RANGES, cat_key, unit)

    if bounds:
        range_min, range_max, _unit = bounds
        midpoint = (range_min + range_max) / 2

        if price_per_unit > range_max * CLAMP_FACTOR:
            logger.warning(
                "Price CLAMPED for %s: %.0f SEK → %.0f SEK (extreme outlier, "
                "expected %s–%s)",
                cat_key, price_per_unit, midpoint, range_min, range_max,
            )
            return midpoint, "Justerat pris — sökvärdet var orimligt högt"
        elif price_per_unit < range_min / CLAMP_FACTOR:
            logger.warning(
                "Price CLAMPED for %s: %.0f SEK → %.0f SEK (extreme outlier, "
                "expected %s–%s)",
                cat_key, price_per_unit, midpoint, range_min, range_max,
            )
            return midpoint, "Justerat pris — sökvärdet var orimligt lågt"
        elif price_per_unit < range_min or price_per_unit > range_max:
            note = "Oväntat pris — verifiera"
            logger.info(
                "Price outside range for %s: %.0f SEK (expected %s–%s)",
                cat_key, price_per_unit, range_min, range_max,
            )
    else:
        # Fallback range for unknown categories
        if price_per_unit > FALLBACK_MAX * CLAMP_FACTOR:
            fallback_mid = (FALLBACK_MIN + FALLBACK_MAX) / 2
            logger.warning(
                "Price CLAMPED (fallback) for '%s': %.0f → %.0f SEK",
                cat_key, price_per_unit, fallback_mid,
            )
            return fallback_mid, "Justerat pris — sökvärdet var orimligt högt"
        elif price_per_unit < FALLBACK_MIN or price_per_unit > FALLBACK_MAX:
            note = "Oväntat pris — verifiera"
            logger.info(
                "Price outside fallback range: %.0f SEK for '%s'",
                price_per_unit, cat_key,
            )

    return price_per_unit, note


def coerce_per_unit_as_total(
    cost_sek: float,
    quantity: float,
    category: str,
    unit: str = "",
) -> tuple[float, str]:
    """Detect "per-unit price stored as total" and correct it.

    Guards against a recurring bug class: an upstream lookup returns a
    price per m² / per st, and the caller forgets to multiply by quantity
    before storing it in cost_sek. Symptom: a 45 m² floor showing 725 kr
    instead of 45 × 725 = 32 625 kr.

    Heuristic (intentionally strict — only triggers when the mix-up is
    virtually certain):
      - cost_sek falls inside the per-unit PRICE_RANGE for this category
      - AND cost_sek / quantity is absurdly low (below per_unit_min / CLAMP_FACTOR)

    In that window, the chance that cost_sek is a legitimate total is
    near zero — totals are always ≥ quantity × per_unit_min.

    Returns (corrected_cost, note). Note is empty if no correction needed.
    """
    if cost_sek <= 0 or quantity <= 0:
        return cost_sek, ""

    bounds = _bounds(PRICE_RANGES, category, unit)
    if not bounds:
        return cost_sek, ""

    per_unit_min, per_unit_max, _unit = bounds

    looks_like_per_unit = per_unit_min <= cost_sek <= per_unit_max
    derived_per_unit_absurd = (cost_sek / quantity) < per_unit_min / CLAMP_FACTOR

    if looks_like_per_unit and derived_per_unit_absurd:
        corrected = round(cost_sek * quantity)
        logger.warning(
            "Per-unit-as-total detected for %s: %.0f SEK × %g %s → %d SEK "
            "(cost_sek looked like per-unit price, not total)",
            category, cost_sek, quantity, _unit, corrected,
        )
        return corrected, "Korrigerat: priset tolkades som per enhet, inte total"

    return cost_sek, ""


class PriceCheck(NamedTuple):
    """validate_total_price's answer, with the one fact its note only implies.

    ``replaced`` is True when the price was an extreme outlier and the total is
    now the category range midpoint times the quantity: a number nobody found.
    The caller must then stop attributing it to wherever the original came
    from (a web search, a URL). A per-unit-as-total correction is not a
    replacement: that total is still the found price, multiplied properly.
    """
    cost: float
    note: str
    replaced: bool


def check_total_price(
    total_cost: float,
    quantity: float,
    category: str,
    *,
    is_estimate: bool = False,
    unit: str = "",
) -> PriceCheck:
    """Validate a total price by deriving per-unit and checking range.

    If the per-unit price was clamped (extreme outlier), the returned total
    is recalculated from the clamped per-unit × quantity.

    First pass: detect per-unit-as-total bug and correct it before range
    validation, so the corrected value gets validated cleanly.
    """
    if total_cost <= 0 or quantity <= 0:
        return PriceCheck(total_cost, "Pris ej tillgängligt" if total_cost <= 0 else "",
                          False)

    total_cost, coerce_note = coerce_per_unit_as_total(total_cost, quantity, category, unit)

    per_unit = total_cost / quantity
    validated_per_unit, note = validate_unit_price(
        per_unit, category, is_estimate=is_estimate, unit=unit)

    replaced = validated_per_unit != per_unit
    if replaced:
        # Price was clamped — recalculate total
        total_cost = round(validated_per_unit * quantity)

    # Merge notes — correction note takes precedence if present
    if coerce_note and note:
        note = f"{coerce_note}. {note}"
    elif coerce_note:
        note = coerce_note

    return PriceCheck(total_cost, note, replaced)


def validate_total_price(
    total_cost: float,
    quantity: float,
    category: str,
    *,
    is_estimate: bool = False,
    unit: str = "",
) -> tuple[float, str]:
    """(total_cost, note) from check_total_price, for callers that do not
    carry a price's provenance. One that does should call check_total_price
    and relabel the source when ``replaced`` is set."""
    checked = check_total_price(total_cost, quantity, category,
                                is_estimate=is_estimate, unit=unit)
    return checked.cost, checked.note


class CO2Check(NamedTuple):
    """validate_co2e's answer, with what a caller needs to relabel a clamp.

    When ``clamped`` is set the total is ``per_unit`` (the range midpoint)
    times the quantity, and ``bounds`` is the (min, max, unit) range it came
    from. The row's per-unit figure, basis and source then describe a number
    that is no longer there, and have to be rewritten with it."""
    total: float
    note: str
    clamped: bool
    per_unit: float
    bounds: tuple[float, float, str] | None


def check_co2e(
    co2e_per_unit: float,
    quantity: float,
    category: str,
    unit: str = "",
) -> CO2Check:
    """Validate CO2e value against expected range for the category.

    Extreme outliers (beyond CLAMP_FACTOR × range) get clamped to the range
    midpoint.
    """
    if co2e_per_unit <= 0 or quantity <= 0:
        return CO2Check(co2e_per_unit * quantity, "", False, co2e_per_unit, None)

    cat_key = category.lower().strip()
    bounds = _bounds(CO2_RANGES, cat_key, unit)
    if not bounds:
        return CO2Check(co2e_per_unit * quantity, "", False, co2e_per_unit, None)

    range_min, range_max, _unit = bounds
    midpoint = (range_min + range_max) / 2

    if co2e_per_unit > range_max * CLAMP_FACTOR:
        logger.warning(
            "CO2e CLAMPED for %s: %.1f → %.1f kg CO2e/unit (extreme outlier, "
            "expected %s–%s)",
            cat_key, co2e_per_unit, midpoint, range_min, range_max,
        )
        return CO2Check(round(midpoint * quantity, 1),
                        "Justerat CO2e — beräknat värde var orimligt högt",
                        True, midpoint, bounds)
    elif co2e_per_unit < range_min / CLAMP_FACTOR:
        logger.warning(
            "CO2e CLAMPED for %s: %.1f → %.1f kg CO2e/unit (extreme outlier, "
            "expected %s–%s)",
            cat_key, co2e_per_unit, midpoint, range_min, range_max,
        )
        return CO2Check(round(midpoint * quantity, 1),
                        "Justerat CO2e — beräknat värde var orimligt lågt",
                        True, midpoint, bounds)
    elif co2e_per_unit < range_min or co2e_per_unit > range_max:
        return CO2Check(co2e_per_unit * quantity, "Oväntat CO2e-värde — verifiera",
                        False, co2e_per_unit, bounds)

    return CO2Check(co2e_per_unit * quantity, "", False, co2e_per_unit, bounds)


def validate_co2e(
    co2e_per_unit: float,
    quantity: float,
    category: str,
    unit: str = "",
) -> tuple[float, str]:
    """(total_co2e, note) from check_co2e."""
    checked = check_co2e(co2e_per_unit, quantity, category, unit)
    return checked.total, checked.note
