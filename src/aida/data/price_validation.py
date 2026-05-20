"""Price and CO2 reasonableness validation for AIda.

Validates prices and CO2 values against expected ranges per component category.
Mild outliers get flagged ("verifiera"). Extreme outliers get clamped to
the range midpoint — showing a wrong number with confidence is worse than
showing a reasonable estimate with a caveat.
"""

from __future__ import annotations

import logging

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
CO2_RANGES: dict[str, tuple[float, float, str]] = {
    "golv":          (2, 20, "m2"),
    "innervägg":     (1, 15, "m2"),
    "yttervägg":     (5, 40, "m2"),
    "betongvägg":    (10, 80, "m2"),
    "fönster":       (20, 150, "st"),
    "tak":           (2, 25, "m2"),
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


def validate_unit_price(
    price_per_unit: float,
    category: str,
    *,
    is_estimate: bool = False,
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
    bounds = PRICE_RANGES.get(cat_key)

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

    bounds = PRICE_RANGES.get(category.lower().strip())
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


def validate_total_price(
    total_cost: float,
    quantity: float,
    category: str,
    *,
    is_estimate: bool = False,
) -> tuple[float, str]:
    """Validate a total price by deriving per-unit and checking range.

    If the per-unit price was clamped (extreme outlier), the returned total
    is recalculated from the clamped per-unit × quantity.

    First pass: detect per-unit-as-total bug and correct it before range
    validation, so the corrected value gets validated cleanly.

    Returns (total_cost, note).
    """
    if total_cost <= 0 or quantity <= 0:
        return total_cost, "Pris ej tillgängligt" if total_cost <= 0 else ""

    total_cost, coerce_note = coerce_per_unit_as_total(total_cost, quantity, category)

    per_unit = total_cost / quantity
    validated_per_unit, note = validate_unit_price(per_unit, category, is_estimate=is_estimate)

    if validated_per_unit != per_unit:
        # Price was clamped — recalculate total
        total_cost = round(validated_per_unit * quantity)

    # Merge notes — correction note takes precedence if present
    if coerce_note and note:
        note = f"{coerce_note}. {note}"
    elif coerce_note:
        note = coerce_note

    return total_cost, note


def validate_co2e(
    co2e_per_unit: float,
    quantity: float,
    category: str,
) -> tuple[float, str]:
    """Validate CO2e value against expected range for the category.

    Extreme outliers (beyond CLAMP_FACTOR × range) get clamped.
    Returns (total_co2e, note).
    """
    if co2e_per_unit <= 0 or quantity <= 0:
        return co2e_per_unit * quantity, ""

    cat_key = category.lower().strip()
    bounds = CO2_RANGES.get(cat_key)
    if not bounds:
        return co2e_per_unit * quantity, ""

    range_min, range_max, _unit = bounds
    midpoint = (range_min + range_max) / 2

    if co2e_per_unit > range_max * CLAMP_FACTOR:
        logger.warning(
            "CO2e CLAMPED for %s: %.1f → %.1f kg CO2e/unit (extreme outlier, "
            "expected %s–%s)",
            cat_key, co2e_per_unit, midpoint, range_min, range_max,
        )
        return round(midpoint * quantity, 1), "Justerat CO2e — beräknat värde var orimligt högt"
    elif co2e_per_unit < range_min / CLAMP_FACTOR:
        logger.warning(
            "CO2e CLAMPED for %s: %.1f → %.1f kg CO2e/unit (extreme outlier, "
            "expected %s–%s)",
            cat_key, co2e_per_unit, midpoint, range_min, range_max,
        )
        return round(midpoint * quantity, 1), "Justerat CO2e — beräknat värde var orimligt lågt"
    elif co2e_per_unit < range_min or co2e_per_unit > range_max:
        return co2e_per_unit * quantity, "Oväntat CO2e-värde — verifiera"

    return co2e_per_unit * quantity, ""
