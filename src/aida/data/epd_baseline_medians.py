"""EPD-based baseline typvärden per Aida category.

Boverket's climate database is organized by material composition (~200 generic
products), not by building component. For component categories that Boverket
lacks (notably golv and sanitet), the baseline agent falls back to LLM
estimation — which is often unreliable.

This module provides a middle tier: a category-aggregated "typvärde" derived
from the upper half of Environdec EPDs (by climate impact) in each Aida
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
- "Boverkets klimatdatabas" → Tier 1: genuine same-material Boverket hit
- "Environdec EPD-typvärde" → Tier 2: this module
- "Uppskattning"            → Tier 3: LLM fallback when nothing else works

We publish a typvärde for (category[, subcategory], unit) keys with enough
samples. Heterogeneous categories (sanitet, belysning, vitvaror) — a toilet,
a cistern and a sink in one bucket — are split PER SUBCATEGORY using the
Palats subcategory taxonomy, so each gets its own typvärde; items that don't
classify into a subcategory stay "Uppskattning".
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

# Per-(category, subcategory) floor overrides. Some genuine product types are
# sparsely represented in Environdec (toilets: the index has only 4 real
# building toilets — the rest are portable bajamajor, rejected at build time).
# For those, 4 verified EPDs beat an LLM guess, so we publish at n=4 rather
# than falling back to "Uppskattning". Kept targeted (not a global drop to 4)
# so each sparse category is opted in only after its rows are inspected — a
# global drop would also publish e.g. betongvägg/m2, whose 4-row bucket still
# holds a misclassified steel sheet. Default stays 5 for everything else.
_MIN_SAMPLES_OVERRIDE: dict[tuple[str, str], int] = {
    ("sanitet", "toalett"): 4,
}

# Categories that mix structurally different product types in the same bucket
# (sanitet covers toilets, sinks, taps — wildly different CO2e). A flat
# category aggregate is misleading, so we aggregate PER SUBCATEGORY instead
# (toalett, handfat, blandare...), reusing the Palats subcategory taxonomy.
# Each (category, subcategory, unit) gets its own typvärde; items that don't
# classify into a subcategory get no typvärde (stay LLM-uppskattning).
_SUBCATEGORIZED_CATEGORIES = {"sanitet", "belysning", "vitvaror", "fast_inredning"}

# Subcategorized categories whose typvärde is published per piece only. Fixed
# interior is specified and bought per piece (a front, a cabinet), so a st value
# is the one a component can use without a mass assumption. The kg rows are a
# different population as well as a different unit: measured 2026-09-14,
# badrumsinredning/kg was 9 of 12 rows Dahl Sverige AB (0.75, over the dominance
# ceiling), where badrumsinredning/st spreads over five makers. The kg and m2 rows
# stay in the catalog and are still offered as alternatives.
_ST_ONLY_CATEGORIES = {"fast_inredning"}

# Keys that clear the sample floor but are not published, each with its reason.
# Only for a key that would be NEW; the keys already published while dominated
# are recorded in test_typvarde_dominans.KNOWN_DOMINANCE instead.
#
# badrumsinredning/st, 2026-09-14: 8 rows, 5 of them Sonas bathrooms (0.62) --
# one product family in five sizes. Whether a typvärde that is mostly one
# supplier's line may be published at all is a decision Henric has open (golv/st,
# Kingspan), and a new key should not pre-empt it. A bathroom-cabinet component
# gets an LLM estimate until more makers declare one, or until that is decided.
_WITHHELD_KEYS: dict[tuple[str, str, str], str] = {
    ("fast_inredning", "badrumsinredning", "st"): "Sonas bathrooms 5 av 8 rader",
}

# Categories where a subtype is PREFERRED but the category aggregate is still a
# legitimate answer. Different from _SUBCATEGORIZED_CATEGORIES above: there, a
# flat aggregate over toilets and taps is meaningless, so an unclassified item
# gets nothing. Here the category mean is coherent enough to fall back on.
#
# The reason golv is split is not incoherence, it is namability. NollCO2 models
# a typical building part for a given building type ("ett idag byggnadstypiskt
# sätt"), and §5.3 puts a proxy for a merely similar product at the LOWEST data
# priority. A baseline that says "11.3 kg CO2e/m2, the average of linoleum,
# carpet, parquet, terrazzo and vinyl" cannot answer "which floor did you
# cost?", because the answer is "none that exists". The subtypes span 5.4 to
# 17.4, a factor of three.
#
# Fallback is mandatory, not a nicety: only textil (30), vinyl (29) and trä (18)
# clear _MIN_SAMPLES. linoleum (3), laminat (3), keramik (4) and gummi (1) do
# not, and a thin bucket is a worse answer than an honest category mean. Which
# level was used is reported back in the result so the UI can say so.
_SUBTYPE_PREFERRED_CATEGORIES = {"golv"}

# Still excluded wholesale: no subcategory taxonomy defined, too heterogeneous
# to aggregate meaningfully.
_HETEROGENEOUS_CATEGORIES = {"storköksutrustning"}

# Geographic scope of the published typvärde.
#
# The 2026-05-29 spec asked for a hard filter, `geo in (SE, NORD, EU)`. Two
# measurements say that cannot be the default, and both are about what `geo`
# IS: the region the declaration's electricity mix and transport scenarios were
# modelled for, not where the product is sold (nordic_supply.py has the long
# form; build_epd_alternatives.py removed geo from the catalog ordering on
# 2026-09-06 for the same reason).
#
#   - Swedish wholesalers declare GLO. Ahlsell AB, 9 GLO rows of 18. Dahl
#     Sverige AB, 23 of 41. A hard filter drops those and keeps a Turkish tile
#     declared RER.
#   - On the 1428-row catalog, 2026-09-14: a hard filter deletes 20 of the 47
#     published keys outright. kakel/m2 39 -> 0, golv/keramik/m2 45 -> 0,
#     ventilation/st 29 -> 0, because whole product families declare GLO.
#
# So the scope is a preference with a floor, the same shape as the subtype
# fallback above. The European bucket is published when it clears the sample
# floor on its own; the full bucket otherwise; and the payload says which
# (`geo_scope`), with both counts, so the report can state what the number
# rests on instead of implying a Swedish context it does not have.
#
# Codes as the two hubs actually write them. RER is ILCD "Europe"; NORD and
# SCAND are Environdec's Nordic regions; EPD Norge writes ISO countries. The
# EEA/EFTA states and the UK are one market for a Swedish buyer, and Norway is
# where the second registry lives, so they are in.
EUROPE_GEO_CODES: frozenset[str] = frozenset({
    "RER", "EU", "EUR", "EU-27", "EU-28", "NORD", "SCAND",
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR",
    "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK",
    "SI", "ES", "SE",
    "NO", "IS", "LI", "CH", "GB", "UK",
})
GEO_SCOPE_EUROPE = "europa"
GEO_SCOPE_GLOBAL = "global"
DEFAULT_GEO_SCOPE: frozenset[str] | None = EUROPE_GEO_CODES

# The European bucket must clear THIS on its own before it replaces the full
# bucket, not just the category floor. With the category floor (5) alone, six
# European el/lm rows replaced twenty global ones and seven European tak/kg rows
# replaced twenty-two: a narrower sample traded for a thinner one, which is the
# fragility the 2026-05-29 spec set out to remove. 15 is the lower end of the
# robust size that spec proposed.
_MIN_SAMPLES_GEO_SCOPE = 15


def _scope_floor(min_required: int) -> int:
    """Rows the European bucket needs before it is preferred over the full one."""
    return max(min_required, _MIN_SAMPLES_GEO_SCOPE)


# A European bucket where one supplier holds this share or more is that
# supplier's product line, not a category value, so the full bucket is used.
# Same ceiling as test_typvarde_dominans. With the scope on, fasadskikt/m2 was
# 65% Saint-Gobain Sweden (Weber renders) among its 49 European rows; the full
# bucket of 55 stays under 0.6.
_SCOPE_DOMINANCE_CEILING = 0.6


def _top_owner_share(rows: Rows) -> float:
    """Largest single owner's share, owners grouped on their first word
    (lowercased), so "Saint-Gobain Sweden AB" and "Saint-Gobain Finland Oy"
    count as one supplier."""
    if not rows:
        return 0.0
    counts: dict[str, int] = {}
    for _, e in rows:
        words = (e.get("owner") or "?").lower().split()
        k = words[0] if words else "?"
        counts[k] = counts.get(k, 0) + 1
    return max(counts.values()) / len(rows)


def _scope_codes_for(cat: str, scope_codes: frozenset[str] | None) -> frozenset[str] | None:
    """The geo scope a category's keys are published with.

    Subtype-preferred categories (golv) stay scope-free. Their category key and
    their subtype keys must rest on the same population: with the scope on,
    golv/m2 published from 36 European rows at 8.64 while vinyl (15.5) and trä
    (9.07) stayed global for lack of 15 European rows each, so the category sat
    below both of its own subtypes (test_baseline_material, 2026-09-14).
    """
    return None if cat in _SUBTYPE_PREFERRED_CATEGORIES else scope_codes


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


def _epd_subcategory(cat: str, e: dict) -> str:
    """Subcategory for an EPD row. Reads the stored field, falling back to
    deriving it from the name so an older catalog without the field still works.
    """
    sub = e.get("subcategory")
    if sub:
        return sub
    if cat in _SUBCATEGORIZED_CATEGORIES:
        from aida.data.palats_client import _normalize_to_aida_subcategory
        return _normalize_to_aida_subcategory(cat, e.get("name", ""))
    return ""


Rows = list[tuple[float, dict]]


def _group_rows(epds: list[dict]) -> dict[tuple[str, str, str], Rows]:
    """(category, subcategory, unit) -> [(gwp, row), ...].

    The one grouping every reader of the catalog shares. The row travels with
    its value so a caller can read geo or owner for exactly the population a
    median is built from: test_typvarde_dominans regroups through this
    function for that reason, so its owner counts cannot drift from the
    published sample by re-implementing the rules below.

    For most categories subcategory is "" (flat aggregate). For the
    heterogeneous-but-subcategorized ones (sanitet, belysning, vitvaror) the
    key is per subcategory; rows that don't classify are skipped.
    """
    grouped: dict[tuple[str, str, str], Rows] = {}
    for e in epds:
        cat = e.get("category", "")
        unit = e.get("unit", "")
        gwp = e.get("gwp_a1a3")
        # Volume-declared products count in the unit they convert to. Timber
        # cladding is declared per m3 while the component is measured in m2, so
        # without this the eighteen wood entries sat in an m3 bucket nobody
        # queries and the m2 typvärde was decided by six entries, two of them
        # aluminium.
        #
        # Two bridges are accepted, and a third is still refused.
        #
        # The m3 one is geometric: a facade panel IS 22 mm thick, so converting
        # it is a fact about the product.
        #
        # `fu_basis == "areal_density"` is an åtgång, the rate at which paint or
        # render is applied. It is what the trade specifies those products by,
        # it is restricted to an allow-list of product families that have such a
        # rate, and it converts the only way a painted or rendered surface can
        # honestly be counted -- per square metre. Before it, farg/m2 was five
        # rows and every Swedish paint sat in a kg bucket no component queries.
        #
        # What stays refused is the general kg bridge on a category-wide
        # density. Folding that in moved yttervägg/m2 from 39.4 to 207.9 and its
        # ceiling to 6548 kg/m2. The difference is not the arithmetic, which is
        # identical, but whether the number multiplied in describes the product
        # or the category it happens to sit in.
        if unit == "m3" or e.get("fu_basis") == "areal_density":
            fu_gwp = e.get("gwp_per_functional_unit")
            fu_unit = e.get("functional_unit")
            if isinstance(fu_gwp, (int, float)) and fu_unit:
                gwp, unit = fu_gwp, fu_unit
        if not cat or not unit or not isinstance(gwp, (int, float)) or gwp <= 0:
            continue
        if cat in _HETEROGENEOUS_CATEGORIES:
            continue
        if cat in _SUBCATEGORIZED_CATEGORIES:
            # Fixtures (toilets, taps, luminaires, appliances) are counted or
            # weighed — never area/volume. An m2/m3-declared EPD here is a
            # misclassification (produced absurd phantoms like blandare/m2 =
            # 1660), so only keep st and kg.
            if unit not in ("st", "kg"):
                continue
            if cat in _ST_ONLY_CATEGORIES and unit != "st":
                continue
            sub = _epd_subcategory(cat, e)
            if not sub:
                continue  # unclassified item in a heterogeneous category
        elif cat in _SUBTYPE_PREFERRED_CATEGORIES:
            # Counted TWICE on purpose, once in its own subtype bucket and once
            # in the category bucket. The category aggregate has to stay
            # complete, because it is the fallback for every subtype too thin to
            # publish — building it from the leftovers instead would make it the
            # mean of exactly the floors nobody could name.
            sub = _epd_subcategory(cat, e)
            if sub:
                grouped.setdefault((cat, sub, unit), []).append((float(gwp), e))
            sub = ""
        else:
            sub = ""
        grouped.setdefault((cat, sub, unit), []).append((float(gwp), e))
    return grouped


def _in_scope(row: dict, scope_codes: frozenset[str]) -> bool:
    return (row.get("geo") or "").strip() in scope_codes


def _select_scope(
    rows: Rows, min_required: int, scope_codes: frozenset[str] | None,
    scope_floor: int | None = None,
) -> tuple[Rows | None, str]:
    """The population a key is published from, and its label.

    The scoped bucket when it clears the floor by itself; the full bucket when
    it does not; None when even the full bucket is thin. The label is a fact
    about which population was used, never about which was asked for, for the
    same reason `level` exists: a fallback that reads as the thing it fell back
    from is the claim this module is here not to make.
    """
    if scope_codes is not None:
        scoped = [r for r in rows if _in_scope(r[1], scope_codes)]
        if (len(scoped) >= (min_required if scope_floor is None else scope_floor)
                and _top_owner_share(scoped) < _SCOPE_DOMINANCE_CEILING):
            return scoped, GEO_SCOPE_EUROPE
    if len(rows) >= min_required:
        return rows, GEO_SCOPE_GLOBAL
    return None, GEO_SCOPE_GLOBAL


def _compute_typvärden(
    scope_codes: frozenset[str] | None = DEFAULT_GEO_SCOPE,
) -> dict[tuple[str, str, str], dict]:
    """Compute upper-half median GWP per (category, subcategory, unit).

    `scope_codes` is the geographic preference (see EUROPE_GEO_CODES); None
    publishes every key from its full bucket, which is what this function did
    before 2026-09-14 and what the payload then labels "global".

    Each entry has: baseline_co2e_per_unit, sample_size, full_median, min, max,
    subcategory, level, geo_scope, sample_size_global, sample_size_europe,
    min_samples.
    """
    epds = _load_epd_data()
    if not epds:
        return {}

    result: dict[tuple[str, str, str], dict] = {}
    for key, rows in _group_rows(epds).items():
        if key in _WITHHELD_KEYS:
            continue
        cat, sub, _unit = key
        min_required = _MIN_SAMPLES_OVERRIDE.get((cat, sub), _MIN_SAMPLES)
        used, scope = _select_scope(rows, min_required, _scope_codes_for(cat, scope_codes),
                                    _scope_floor(min_required))
        if used is None:
            continue
        values = [gwp for gwp, _ in used]
        result[key] = {
            "baseline_co2e_per_unit": round(_upper_half_median(values), 2),
            "sample_size": len(values),
            "full_median": round(median(values), 2),
            "min": round(min(values), 2),
            "max": round(max(values), 2),
            "subcategory": sub,
            # "subtype" | "category" — which level the number actually came
            # from. Carried in the payload rather than inferred by the caller
            # from a non-empty subcategory, because a subtype request that fell
            # back to the category must not read as a subtype answer.
            "level": "subtype" if sub else "category",
            # "europa" | "global" — which population the number came from,
            # same argument as `level`. The two counts let the report say
            # "16 europeiska av 20" or "bara 3 europeiska, alla 39 används"
            # instead of a bare n that hides which of the two it is.
            "geo_scope": scope,
            "sample_size_global": len(rows),
            "sample_size_europe": sum(1 for _, e in rows if _in_scope(e, EUROPE_GEO_CODES)),
            "min_samples": min_required,
            "min_samples_europe": _scope_floor(min_required),
        }
    return result


_TYPVÄRDEN: dict[tuple[str, str, str], dict] | None = None


# Swedish material names -> EPD subtype key, for the subtype-preferred
# categories. Kept separate from palats_client's SUBCATEGORY_KEYWORDS on
# purpose: that taxonomy classifies second-hand LISTINGS and is tuned against
# 701 live ads, this one reads the standard material the baseline agent named
# ("Homogen vinylmatta (PVC)"). Same words, different job, and coupling them
# would mean a baseline tweak could silently move the reuse search.
#
# Order matters, substring match, most specific first. "linoleum" before
# "matta" because a linoleum floor is often written "linoleummatta", and
# "plastmatta" before the bare "matta" that textilgolv shares.
_MATERIAL_SUBTYPE_KEYWORDS: dict[str, list[tuple[str, list[str]]]] = {
    "golv": [
        ("linoleum", ["linoleum", "marmoleum"]),
        ("laminat", ["laminat"]),
        ("keramik", ["klinker", "kakel", "keramik", "terrazzo", "granitkeramik"]),
        ("trä", ["parkett", "trägolv", "massivt trä", "ekgolv", "furugolv",
                 "bambu", "kork"]),
        ("gummi", ["gummi"]),
        ("epoxi", ["epoxi", "härdplast", "akrylat", "polyuretangolv"]),
        ("vinyl", ["vinyl", "plastmatta", "pvc", "homogen matta",
                   "heterogen matta", "plastgolv"]),
        ("textil", ["textil", "heltäckningsmatta", "nålfilt", "mattplatt",
                    "matta"]),
    ],
}


def subtype_from_material(category: str, material: str) -> str:
    """Infer the EPD subtype key from a named standard material.

    Returns "" when the category has no subtype taxonomy or nothing matched —
    the caller then gets the category aggregate, which is a valid answer here
    (unlike the sanitet/belysning/vitvaror split, where a miss means no number).
    """
    subs = _MATERIAL_SUBTYPE_KEYWORDS.get(category)
    if not subs or not material:
        return ""
    text = material.lower()
    for subtype, keywords in subs:
        if any(kw in text for kw in keywords):
            return subtype
    return ""


def get_baseline_typvärde(category: str, unit: str, subcategory: str = "") -> dict | None:
    """Look up EPD-baseline typvärde for a (category, unit[, subcategory]).

    Three behaviours, by category:

    - subcategorized (sanitet, belysning, vitvaror): a subcategory is REQUIRED.
      A flat mean over toilets and taps is meaningless, so a miss returns None
      and the caller falls through to an LLM estimate.
    - subtype-preferred (golv): the subtype is tried first and the category
      aggregate is the fallback, so a floor the catalog has too few of still
      gets a number. Read ``level`` to tell the two apart.
    - everything else: subcategory is ignored.

    Returns a dict with baseline_co2e_per_unit, sample_size, full_median, min,
    max, subcategory and level — or None if no usable typvärde exists. Cached
    lazily on first call.
    """
    global _TYPVÄRDEN
    if _TYPVÄRDEN is None:
        _TYPVÄRDEN = _compute_typvärden()

    if category in _SUBTYPE_PREFERRED_CATEGORIES and subcategory:
        hit = _TYPVÄRDEN.get((category, subcategory, unit))
        if hit:
            return hit
        # Fall through to the category aggregate below. Deliberately not
        # returning None: an unfamiliar or thin floor subtype should still get a
        # baseline, just an honestly labelled one.

    sub = subcategory if category in _SUBCATEGORIZED_CATEGORIES else ""
    return _TYPVÄRDEN.get((category, sub, unit))


# Back-compat alias — old call sites used "median" terminology before we
# switched to upper-half methodology. Same value, clearer name.
def get_baseline_median(category: str, unit: str, subcategory: str = "") -> dict | None:
    """Deprecated — use get_baseline_typvärde. Kept for back-compat."""
    data = get_baseline_typvärde(category, unit, subcategory)
    if data is None:
        return None
    # Synthesize the old key name from the new structure
    return {
        **data,
        "median_co2e_per_unit": data["baseline_co2e_per_unit"],
    }


def list_available_categories() -> list[tuple[str, str, str, int]]:
    """List all (category, subcategory, unit, sample_size) with a typvärde."""
    global _TYPVÄRDEN
    if _TYPVÄRDEN is None:
        _TYPVÄRDEN = _compute_typvärden()
    return sorted(
        [(cat, sub, unit, data["sample_size"])
         for (cat, sub, unit), data in _TYPVÄRDEN.items()],
        key=lambda x: (x[0], x[1], x[2]),
    )


def main():
    """CLI: print the typvärde table for inspection.

    `--no-geo` prints the pre-scope table (every key from its full bucket), so
    the two can be diffed: that diff is the documented before/after the
    2026-05-29 spec asks for.
    """
    import sys
    scope = None if "--no-geo" in sys.argv[1:] else DEFAULT_GEO_SCOPE
    print(f"{'Kategori':<18} {'Subkat':<14} {'Unit':<5} {'n':>3} {'nEU':>4} {'nAll':>5} "
          f"{'scope':<7} {'min':>8} {'med':>8} {'typvärde':>10} {'max':>8}")
    print("-" * 104)
    for (cat, sub, unit), data in sorted(_compute_typvärden(scope).items()):
        print(
            f"{cat:<18} {sub:<14} {unit:<5} {data['sample_size']:>3} "
            f"{data['sample_size_europe']:>4} {data['sample_size_global']:>5} "
            f"{data['geo_scope']:<7} "
            f"{data['min']:>8.2f} {data['full_median']:>8.2f} "
            f"{data['baseline_co2e_per_unit']:>10.2f} {data['max']:>8.2f}"
        )


if __name__ == "__main__":
    main()
