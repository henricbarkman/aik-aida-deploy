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

from aida.data.aggregat import CLASS_SUBCATEGORY, FLOW_UNIT
from aida.data.koncern import (
    DOMINANCE_CEILING,
    collapse_plants,
    concentration,
    dominance_reason,
    owner_group,
)

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
    # Precast concrete frame, 2026-10-01 (HENRIC-3362): the other way, a
    # higher floor. One key per unit across hollow-core, solid slabs and beams
    # (their per-kg values overlap: 0.06-0.19, 0.08-0.36, 0.14-0.64), which is
    # only a typical value with a sample to match. kg had 17 products at the
    # first build; m3 and m2 one or two each, and five would not be enough.
    ("stomme", "betong"): 15,
}

# Categories that mix structurally different product types in the same bucket
# (sanitet covers toilets, sinks, taps — wildly different CO2e). A flat
# category aggregate is misleading, so we aggregate PER SUBCATEGORY instead
# (toalett, handfat, blandare...), reusing the Palats subcategory taxonomy.
# Each (category, subcategory, unit) gets its own typvärde; items that don't
# classify into a subcategory get no typvärde (stay LLM-uppskattning).
_SUBCATEGORIZED_CATEGORIES = {"sanitet", "belysning", "vitvaror", "fast_inredning",
                              "los_inredning"}

# Subcategorized categories whose typvärde is published per piece only. Fixed
# interior is specified and bought per piece (a front, a cabinet), so a st value
# is the one a component can use without a mass assumption. The kg rows are a
# different population as well as a different unit: measured 2026-09-14,
# badrumsinredning/kg was 9 of 12 rows Dahl Sverige AB (0.75, over the dominance
# ceiling), where badrumsinredning/st spreads over five makers. The kg and m2 rows
# stay in the catalog and are still offered as alternatives.
_ST_ONLY_CATEGORIES = {"fast_inredning", "los_inredning"}

# The exception: loose textiles are bought per piece and declared per m² (carpet
# by the roll, curtain fabric), and a rug's size varies more than its make. So
# their typvärde is per m², and a piece is counted by the size its name states
# (textile_typvärde; HENRIC-3366).
_PER_M2_SUBCATEGORIES = frozenset({("los_inredning", "matta"),
                                   ("los_inredning", "gardin")})

# Keys that clear the sample floor but are not published for a reason the data
# cannot state by itself, each with its reason. A key that is one company
# group's range is NOT listed here: the dominance rule in _compute_with_withheld
# finds those and words the reason (koncern.dominance_reason). Until 2026-10-01
# five such keys were listed by hand (badrumsinredning, förvaring, the apartment
# aggregat class, and HENRIC-3290's avjämning, bafflar and bärverk), and the
# rule now withholds every one of them on its own count.
#
# ventilation/aggregat/st, 2026-09-28: 24 rows, but they span 270 to 16 700 kg
# CO2e/st, a factor of sixty, and the spread is unit SIZE, not maker: Flexit's
# apartment units (Nordic S2-S7, 270-560) against Kampmann's 6 000-20 000 m3/h
# school units (5 000-16 700). The upper-half median, 6 195, would put a school
# unit's figure on every "FTX per lägenhet" component, times the number of
# flats. No single per-piece value describes both, and the catalog has no
# airflow field to scale by (Flexit's names do not state it). The rows stay in
# the catalog as alternatives; the baseline gets an estimate that can read the
# size from the description, and the row says why (split_subcategory_miss).
_WITHHELD_KEYS: dict[tuple[str, str, str], str] = {
    ("ventilation", "aggregat", "st"): (
        "EPD:erna spänner 270 till 24 500 kg CO2e/st beroende på aggregatets "
        "storlek (luftflöde), så inget enskilt typvärde per styck stämmer"
    ),
    # The kg key is empty today (one row). Withheld in advance anyway, because
    # the baseline and the reroute both bridge st -> kg when st has no value,
    # and a kg median over units of every size would bring the same spread
    # back by that door once five kg rows exist.
    ("ventilation", "aggregat", "kg"): (
        "aggregat-EPD:er per kg spänner över alla storlekar, och ett "
        "kg-värde ger inget styckvärde utan aggregatets vikt"
    ),
    # hiss/st, 2026-09-30 (handover review C5). Six whole-elevator EPDs
    # (Schindler 5500 3.33, FUJITEC ELSIA 9.09, TK EOX 13.4, TK endura MRL
    # 174) tagged per piece, and none of them can be one elevator: a lift is
    # tonnes of steel, and the baseline's own range starts at 2 000 kg. The
    # declared unit is something else (per kg, per trip, per year) that the
    # catalog build read as "st". The 13.4 median was then clamped to the
    # range midpoint, 16 000, under a label that still said "13,4 kg/st x 1
    # st". Withheld, so a lift gets the agent's estimate and the row says why.
    # The rows are unchanged in the catalog.
    ("hiss", "", "st"): (
        "katalogens EPD:er för hela hissar anger 3 till 174 kg CO2e per "
        "styck, vilket inte kan vara en hel hiss, så deras enhet är fel "
        "angiven"
    ),
}


def withheld_reason(category: str, subcategory: str, unit: str) -> str:
    """Why a key that clears the sample floor is not published, or ''.

    Either a reason above, or the dominance rule's (_compute_with_withheld)."""
    key = (category, subcategory, unit)
    if key in _WITHHELD_KEYS:
        return _WITHHELD_KEYS[key]
    _ensure_cache()
    return (_DOMINATED or {}).get(key, "")


def lookup_withheld_reason(category: str, unit: str, subcategory: str = "") -> str:
    """withheld_reason for the key get_baseline_typvärde would have answered
    from, following the same fallbacks: a golv subtype too thin to publish
    falls back to the category key, and when THAT is withheld (golv/st, one
    supplier's raised access floors) the row should say so."""
    if category == "ventilation" and subcategory in CLASS_SUBCATEGORY.values():
        return withheld_reason(category, subcategory, unit)
    split_units = _SPLIT_SUBCATEGORIES.get(category, {}).get(subcategory)
    if split_units:
        return withheld_reason(category, subcategory if unit in split_units else "", unit)
    if category in _SUBTYPE_PREFERRED_CATEGORIES and subcategory:
        if get_baseline_typvärde(category, unit, subcategory):
            return ""
        return (withheld_reason(category, subcategory, unit)
                or withheld_reason(category, "", unit))
    keyed = _SUBCATEGORIZED_CATEGORIES | _MATERIAL_SUBCATEGORIZED_CATEGORIES
    return withheld_reason(category, subcategory if category in keyed else "", unit)

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

# Subtypes split OFF a category that is otherwise flat. The third shape, beside
# the two above: the category bucket is coherent without them, and it has to be
# protected FROM them. An air handling unit is 1 700 to 16 700 kg CO2e/st; the
# ducts, dampers and diffusers in ventilation/st are tens. Until 2026-09-28 the
# units were kept out only by ventilation's 300 kg ceiling in the catalog build,
# so a "Ventilationsaggregat" component got the duct typvärde (143 kg/st) and
# nothing in the report said so.
#
# So a split row is counted in its own key and never in the category's, and a
# component that names the subtype gets the subtype's number or none: falling
# back to the duct value is exactly the error this exists to stop. The units
# listed are the ones the subtype is counted in. A component in another unit
# ("Ventilation, aggregat och kanaler", per m2 BTA) describes a whole system,
# not a unit, and keeps the category key it had before.
#
# golv/avjämning and innervägg/glasparti, 2026-09-28 (HENRIC-3290 del 3), the
# same shape. A levelling compound (0.2-1 kg CO2e/kg) lies under a floor
# covering and is never one; a glazed partition system (27-292 kg CO2e/m2) is
# ten to fifty times the plasterboard wall innervägg is otherwise made of.
# Both list every unit a component can be given in, unlike aggregat: an
# "Avjämningsmassa" in m2 or in säckar is still a levelling compound, and the
# floor covering's m2 or st value is not its number. With no value of its own
# it gets none, and the row says to give the quantity in kg.
_SPLIT_SUBCATEGORIES: dict[str, dict[str, tuple[str, ...]]] = {
    "ventilation": {"aggregat": ("st", "kg")},
    "golv": {"avjämning": ("kg", "m2", "st")},
    "innervägg": {"glasparti": ("m2", "st", "kg")},
}

# The two del 3 subtypes their category says nothing about, where the category
# is the wrong reference in two more places: the router may not move them (a
# glazed partition filed under fönster met used windows), and the baseline's
# CO2e range for the category does not apply to their typvärde (177.4 kg/m2
# was clamped to a plasterboard wall's 8). Not aggregat, whose range check
# caught a unit-tag fault in the catalog and stays. Since 2026-09-30 the
# baseline range-checks no typvärde at runtime (handover review C5); the range
# half of this set is now read by test_baseline_clamp, which checks every
# other published typvärde against its category's range.
UNLIKE_THEIR_CATEGORY = frozenset({("innervägg", "glasparti"), ("golv", "avjämning")})


def split_subcategory_miss(category: str, unit: str, subcategory: str) -> bool:
    """True when a component names a split subtype that has no typvärde.

    The caller then has an honest "no number" to report instead of the
    category value, and can say why in the row's text."""
    units = _SPLIT_SUBCATEGORIES.get(category, {}).get(subcategory)
    if not units or unit not in units:
        return False
    return get_baseline_typvärde(category, unit, subcategory) is None

# Frame materials, 2026-09-28 (HENRIC-3290). A fourth shape: per subcategory
# like sanitet, because a median over a stud, a glulam beam and a steel section
# describes none of them, but in ANY unit, because the families are declared
# the way they are traded: timber, glulam and board per m3, steel per kg. There
# is no flat category key, and a row without a family is skipped (the catalog
# build drops those anyway, see CATEGORY_SUBCATEGORY_REQUIRED).
#
# The m3 keys are not what a component is counted in. A stud is bought per
# löpmeter and a board per m2, and member_typvärde below bridges the m3 value
# with the member's own section or thickness, read from its name.
#
# undertak, 2026-09-28 (HENRIC-3290 del 3), for the same two reasons: a ceiling
# tile per m2, a wall absorber, a free-hanging baffle and the T24 grid per m2
# of ceiling or per kg are four products, and each is declared in the unit it
# is sold in. A ceiling of no stated kind ("Ljudabsorbent") gets no typvärde.
_MATERIAL_SUBCATEGORIZED_CATEGORIES = {"stomme", "undertak"}

# The stomme families whose m3 value may be bridged by geometry. Solid material
# only: a steel stud or section is a thin-walled profile, and its weight per
# metre is a property of the profile that no name like "Stålregel 70" states.
_GEOMETRY_BRIDGE_SUBCATEGORIES = {"virke", "limträ", "konstruktionsskiva"}

# ...but a STANDARD profile's weight per metre is a property no name has to
# state beyond the designation: EN 10365 fixes an HEA 200 at 42,3 kg/m, and a
# hollow section's or a thin-sheet stud's weight is in its tables. Those
# families' kg values are bridged by that weight (_profile_typvärde,
# steel_profiles; HENRIC-3363).
_PROFILE_BRIDGE_SUBCATEGORIES = {"konstruktionsstål", "stålregel"}

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
# Same ceiling as the publication rule below (koncern.DOMINANCE_CEILING). With
# the scope on, fasadskikt/m2 was 65% Saint-Gobain Sweden (Weber renders) among
# its 49 European rows.
_SCOPE_DOMINANCE_CEILING = DOMINANCE_CEILING


def _top_owner_share(rows: Rows) -> float:
    """Largest company group's share (koncern.owner_group). Until 2026-10-01
    owners were grouped on their first word, which joined "Saint-Gobain
    Sweden AB" and "Saint-Gobain Ecophon AB" but not Gyproc or Weber-Sodamco."""
    if not rows:
        return 0.0
    counts: dict[str, int] = {}
    for _, e in rows:
        k = owner_group(e.get("owner"))
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
            sub = _epd_subcategory(cat, e)
            if (cat, sub) in _PER_M2_SUBCATEGORIES:
                # Rugs and curtains (HENRIC-3366): per m², the unit every one
                # of their EPDs is declared in. A piece has no size until the
                # component's name gives one (textile_typvärde).
                if unit != "m2":
                    continue
            else:
                # Fixtures (toilets, taps, luminaires, appliances) are counted
                # or weighed — never area/volume. An m2/m3-declared EPD here is
                # a misclassification (produced absurd phantoms like
                # blandare/m2 = 1660), so only keep st and kg.
                if unit not in ("st", "kg"):
                    continue
                if cat in _ST_ONLY_CATEGORIES and unit != "st":
                    continue
            if not sub:
                continue  # unclassified item in a heterogeneous category
        elif cat in _SUBTYPE_PREFERRED_CATEGORIES:
            # Counted TWICE on purpose, once in its own subtype bucket and once
            # in the category bucket. The category aggregate has to stay
            # complete, because it is the fallback for every subtype too thin to
            # publish — building it from the leftovers instead would make it the
            # mean of exactly the floors nobody could name.
            #
            # Except a split subtype (golv/avjämning): that one is not a floor
            # the aggregate should describe, so it is counted once, in its own
            # key, like aggregat below.
            sub = _epd_subcategory(cat, e)
            if sub in _SPLIT_SUBCATEGORIES.get(cat, {}):
                grouped.setdefault((cat, sub, unit), []).append((float(gwp), e))
                continue
            if sub:
                grouped.setdefault((cat, sub, unit), []).append((float(gwp), e))
            sub = ""
        elif cat in _SPLIT_SUBCATEGORIES:
            # Counted ONCE, in its own key when it is a split subtype and in
            # the category key otherwise. The opposite of golv above, on
            # purpose: here the category value must not contain the subtype.
            sub = e.get("subcategory") or ""
            if sub not in _SPLIT_SUBCATEGORIES[cat]:
                sub = ""
        elif cat in _MATERIAL_SUBCATEGORIZED_CATEGORIES:
            sub = e.get("subcategory") or ""
            if not sub:
                continue
        else:
            sub = ""
        grouped.setdefault((cat, sub, unit), []).append((float(gwp), e))
        if (cat, sub, unit) == ("ventilation", "aggregat", "st"):
            _add_aggregat_keys(grouped, float(gwp), e)
    return grouped


def _add_aggregat_keys(grouped: dict[tuple[str, str, str], Rows], gwp: float,
                       e: dict) -> None:
    """Count an air handling unit a second time, in its size class, and a third,
    per m3/h of its airflow, when the catalog row carries them (aggregat.py
    says where each came from). The per-piece key it is also in stays withheld;
    these two are what the baseline uses instead."""
    klass = e.get("aggregat_class")
    if klass in CLASS_SUBCATEGORY:
        grouped.setdefault(("ventilation", CLASS_SUBCATEGORY[klass], "st"), []).append((gwp, e))
    flow = e.get("airflow_m3h")
    if isinstance(flow, (int, float)) and flow > 0:
        grouped.setdefault(("ventilation", "aggregat", FLOW_UNIT), []).append((gwp / flow, e))


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

    Each bucket has its plant duplicates folded first (koncern.collapse_plants),
    so floors and shares count products, not factories. Folded after the geo
    filter, so a product's European bucket holds only its European plants.
    """
    if scope_codes is not None:
        scoped = collapse_plants([r for r in rows if _in_scope(r[1], scope_codes)])
        if (len(scoped) >= (min_required if scope_floor is None else scope_floor)
                and _top_owner_share(scoped) < _SCOPE_DOMINANCE_CEILING):
            return scoped, GEO_SCOPE_EUROPE
    rows = collapse_plants(rows)
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
    min_samples, concentration (koncern.concentration of the population used).
    """
    return _compute_with_withheld(scope_codes)[0]


def _compute_with_withheld(
    scope_codes: frozenset[str] | None = DEFAULT_GEO_SCOPE,
) -> tuple[dict[tuple[str, str, str], dict], dict[tuple[str, str, str], str]]:
    """The typvärden, and the keys withheld because one company group holds
    DOMINANCE_CEILING or more of the products behind them, with the reason.

    The rule replaced a hand list (HENRIC-3368, 2026-10-01). Before it, a
    dominated key was withheld only if someone noticed: five were listed in
    _WITHHELD_KEYS one at a time, and four more (golv/st 100% Kingspan, among
    them) were published while a test recorded them as known.
    """
    epds = _load_epd_data()
    if not epds:
        return {}, {}

    result: dict[tuple[str, str, str], dict] = {}
    dominated: dict[tuple[str, str, str], str] = {}
    grouped = _group_rows(epds)
    for key, rows in grouped.items():
        if key in _WITHHELD_KEYS:
            continue
        cat, sub, _unit = key
        min_required = _MIN_SAMPLES_OVERRIDE.get((cat, sub), _MIN_SAMPLES)
        used, scope = _select_scope(rows, min_required, _scope_codes_for(cat, scope_codes),
                                    _scope_floor(min_required))
        if used is None:
            continue
        conc = concentration(used)
        why = dominance_reason(conc)
        if why:
            dominated[key] = why
            continue
        values = [gwp for gwp, _ in used]
        all_products = collapse_plants(rows)
        europe_products = collapse_plants(
            [r for r in rows if _in_scope(r[1], EUROPE_GEO_CODES)])
        # Why a global key is global, so the row can say the true reason:
        # "thin" (too few European products), "dominated" (enough, but one
        # group's range, see _select_scope) or "scope_free" (golv, by design).
        scope_reason = ""
        if scope == GEO_SCOPE_GLOBAL and scope_codes is not None:
            if _scope_codes_for(cat, scope_codes) is None:
                scope_reason = "scope_free"
            elif len(europe_products) < _scope_floor(min_required):
                scope_reason = "thin"
            else:
                scope_reason = "dominated"
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
            # Counted in products, plant duplicates folded, like sample_size.
            "sample_size_global": len(all_products),
            "sample_size_europe": len(europe_products),
            "scope_reason": scope_reason,
            "min_samples": min_required,
            "min_samples_europe": _scope_floor(min_required),
            # Who is behind the number: the largest group and the second,
            # with counts, so every row can say how concentrated it is.
            "concentration": conc,
            # EPD rows folded into another plant's product (0 when none).
            "plant_rows_folded": sum(e.get("plants", 1) - 1 for _, e in used),
        }
        if key == ("ventilation", "aggregat", FLOW_UNIT):
            # The flows the per-m3/h value was measured over, so a component
            # outside them can be told its number is an extrapolation.
            flows = [e["airflow_m3h"] for _, e in used]
            result[key]["airflow_min"] = min(flows)
            result[key]["airflow_max"] = max(flows)
            # All per-piece aggregat EPDs, so the row can say how many state no
            # flow and are left out.
            result[key]["sample_size_per_piece"] = len(
                grouped.get(("ventilation", "aggregat", "st"), []))
        elif cat == "ventilation" and sub in CLASS_SUBCATEGORY.values():
            flowed = [e for _, e in used
                      if isinstance(e.get("airflow_m3h"), (int, float))]
            if len(flowed) >= min_required:
                result[key]["airflow_median"] = _median_per_group(flowed)
                result[key]["airflow_median_groups"] = len(
                    {owner_group(e.get("owner")) for e in flowed})
    _class_values_from_flow(result)
    return result, dominated


def _median_per_group(rows: list[dict]) -> float:
    """The class's middle size, each company group's series counted once: the
    median of each group's stated flows, then the median of those.

    A plain median over the EPDs lets the maker that declares the most sizes
    set the class's size. Swegon declared nine GOLD/SILVER sizes (#800) and
    the building class's median moved from 3 700 to 6 000 m3/h, 4 588 to 7 380
    kg, without any building getting bigger. Same idea as folding a product's
    factories into one (koncern.collapse_plants, #795): a typical unit is a
    choice between makers, not between one maker's catalogue pages. Demi's
    decision, HENRIC-3369; Henric can ask for the plain median back.
    """
    by_group: dict[str, list[float]] = {}
    for e in rows:
        by_group.setdefault(owner_group(e.get("owner")), []).append(float(e["airflow_m3h"]))
    return float(median(median(flows) for flows in by_group.values()))


def _class_values_from_flow(result: dict[tuple[str, str, str], dict]) -> None:
    """A size class's per-piece value is a unit of the class's middle size,
    counted with the per-airflow typvärde. In place.

    The upper-half median is this module's answer to one bias: EPDs come from
    climate-conscious makers, so the upper half approximates the conventional
    choice. Inside an aggregat class the spread is not that. It is size: the
    building class runs from Flexit's 1 000 m3/h ProNordic (1 210 kg) to
    Kampmann's 20 000 m3/h unit, and its upper half is the big units, 11 050
    kg/st against a class median of 2 783. That number would describe a school's
    main unit and nothing smaller. So size takes the median of the flows the
    class's EPDs state (not the upper half: size is not a climate choice), and
    the conventional-choice uplift comes in through the per-m3/h value, which
    is an upper-half median over the same EPDs and where the spread is product
    choice (rotary against plate exchanger, 0.49 to 1.33). The per-piece
    upper-half median stays in the payload as `upper_half_per_piece`.

    A class whose EPDs state too few flows keeps its per-piece upper-half
    median (the apartment class, were it ever published: Flexit's Nordic EPDs
    state none, and their spread, 270 to 564 kg, is narrow).
    """
    per_flow = result.get(("ventilation", "aggregat", FLOW_UNIT))
    if not per_flow:
        return
    for sub in CLASS_SUBCATEGORY.values():
        entry = result.get(("ventilation", sub, "st"))
        if not entry or "airflow_median" not in entry:
            continue
        entry["upper_half_per_piece"] = entry["baseline_co2e_per_unit"]
        entry["per_m3h"] = per_flow["baseline_co2e_per_unit"]
        entry["baseline_co2e_per_unit"] = round(
            per_flow["baseline_co2e_per_unit"] * entry["airflow_median"], 1)


_TYPVÄRDEN: dict[tuple[str, str, str], dict] | None = None
# Keys withheld by the dominance rule, with the reason; filled with _TYPVÄRDEN.
_DOMINATED: dict[tuple[str, str, str], str] | None = None


def _ensure_cache() -> None:
    global _TYPVÄRDEN, _DOMINATED
    if _TYPVÄRDEN is None or _DOMINATED is None:
        _TYPVÄRDEN, _DOMINATED = _compute_with_withheld()


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
    _ensure_cache()

    # An aggregat size class is its own key and nothing else: a miss must not
    # fall through to ventilation/st, which is ducts and terminals.
    if category == "ventilation" and subcategory in CLASS_SUBCATEGORY.values():
        return _TYPVÄRDEN.get((category, subcategory, unit))

    # Split subtypes first, so a golv/avjämning miss never reaches the
    # subtype-preferred fallback below (the floor covering's aggregate).
    split_units = _SPLIT_SUBCATEGORIES.get(category, {}).get(subcategory)
    if split_units:
        if unit in split_units:
            # No fallback to the category key: see _SPLIT_SUBCATEGORIES.
            return _TYPVÄRDEN.get((category, subcategory, unit))
        return _TYPVÄRDEN.get((category, "", unit))

    if category in _SUBTYPE_PREFERRED_CATEGORIES and subcategory:
        hit = _TYPVÄRDEN.get((category, subcategory, unit))
        if hit:
            return hit
        # Fall through to the category aggregate below. Deliberately not
        # returning None: an unfamiliar or thin floor subtype should still get a
        # baseline, just an honestly labelled one.

    keyed = _SUBCATEGORIZED_CATEGORIES | _MATERIAL_SUBCATEGORIZED_CATEGORIES
    sub = subcategory if category in keyed else ""
    return _TYPVÄRDEN.get((category, sub, unit))


def member_typvärde(category: str, name: str, unit: str,
                    subcategory: str) -> dict | None:
    """The m3 typvärde of a solid frame family, per löpmeter or m2 of THIS member.

    "Reglar 45x95" in lm is 0.045 x 0.095 m of sawn timber per metre, so the
    virke/m3 value times that area is its value per metre. The dimension comes
    from the name and nowhere else (unit_conversion.member_volume_per_unit); a
    name without one gets None, and the caller asks for it rather than guess.

    Returns the m3 payload rescaled, with `geometry` naming the section or
    thickness used, or None.
    """
    from aida.data.unit_conversion import member_volume_per_unit

    if category not in _MATERIAL_SUBCATEGORIZED_CATEGORIES:
        return None
    if subcategory in _PROFILE_BRIDGE_SUBCATEGORIES:
        return _profile_typvärde(category, name, unit, subcategory)
    if subcategory not in _GEOMETRY_BRIDGE_SUBCATEGORIES:
        return None
    geometry = member_volume_per_unit(name, unit)
    if not geometry:
        return None
    m3_data = get_baseline_typvärde(category, "m3", subcategory)
    if not m3_data:
        return None
    factor, label = geometry
    bridged = dict(m3_data)
    for key in ("baseline_co2e_per_unit", "full_median", "min", "max"):
        if isinstance(m3_data.get(key), (int, float)):
            bridged[key] = round(m3_data[key] * factor, 4)
    bridged["geometry"] = label
    bridged["per_m3"] = m3_data["baseline_co2e_per_unit"]
    return bridged


def _profile_typvärde(category: str, name: str, unit: str,
                      subcategory: str) -> dict | None:
    """The kg typvärde of a steel family, per löpmeter of THIS profile.

    The steel half of member_typvärde (HENRIC-3363): "Stålbalk HEA 200" is
    42,3 kg per metre by EN 10365, so the konstruktionsstål/kg value times
    42,3 is its value per metre. The weight comes from steel_profiles' tables,
    the designation from the name, and a profile of the other family (a stud's
    weight against the beam mills' median) gives None.

    Returns the kg payload rescaled, with `geometry` naming the profile and its
    weight and source, `per_kg` the unscaled value, or None.
    """
    from aida.data.steel_profiles import profile_mass

    if (unit or "").strip().lower() not in ("lm", "m", "meter", "löpmeter"):
        return None
    profile, _ = profile_mass(name)
    if not profile or profile.family != subcategory:
        return None
    kg_data = get_baseline_typvärde(category, "kg", subcategory)
    if not kg_data:
        return None
    bridged = dict(kg_data)
    for key in ("baseline_co2e_per_unit", "full_median", "min", "max"):
        if isinstance(kg_data.get(key), (int, float)):
            bridged[key] = round(kg_data[key] * profile.kg_per_m, 4)
    bridged["geometry"] = f"profil {profile.label}"
    bridged["per_kg"] = kg_data["baseline_co2e_per_unit"]
    return bridged


def textile_typvärde(category: str, name: str, unit: str,
                     subcategory: str) -> dict | None:
    """The m² typvärde of a rug or a curtain, per piece of THIS size.

    "Matta 2x3 m" in st is 6 m² a piece (aida.data.textil.piece_area_m2), so
    the matta/m2 value times 6 is its value per piece (HENRIC-3366). The size
    comes from the name only; no size, or no published m² value, gives None.

    Returns the m² payload rescaled, with `geometry` naming the size and
    `per_m2` the unscaled value, or None.
    """
    from aida.data.textil import piece_area_m2

    if (category, subcategory) not in _PER_M2_SUBCATEGORIES:
        return None
    if (unit or "").strip().lower() not in ("st", "styck"):
        return None
    area, label = piece_area_m2(name)
    if not area:
        return None
    m2_data = get_baseline_typvärde(category, "m2", subcategory)
    if not m2_data:
        return None
    bridged = dict(m2_data)
    for key in ("baseline_co2e_per_unit", "full_median", "min", "max"):
        if isinstance(m2_data.get(key), (int, float)):
            bridged[key] = round(m2_data[key] * area, 4)
    bridged["geometry"] = label
    bridged["per_m2"] = m2_data["baseline_co2e_per_unit"]
    return bridged


def textile_typvärde_unit(category: str, unit: str, subcategory: str) -> str:
    """The unit a rug's or a curtain's typvärde is kept in ("m2"), whatever unit
    the component is given in; `unit` for every other product. So a withheld
    reason is looked up on the key that would have answered."""
    if (category, subcategory) in _PER_M2_SUBCATEGORIES:
        return "m2"
    return unit


def aggregat_typvärde(name: str, usage_context: str = "", quantity: float = 1) -> dict:
    """A, B or C for one air handling unit counted per piece (aggregat.py).

    Returns a dict with:
      method  "luftflöde" | "klass" | "uppskattning"
      payload the typvärde per PIECE for this unit (the usual typvärde keys,
              min/max/full_median in kg per piece too), or None under C
      and, per method: airflow_m3h, airflow_source, per_m3h, extrapolated
      (A); klass, class_why and flow_note (B: why a flow the text mentions
      was not used, or ''); reason, klass and airflow_m3h (C), the reason a
      sentence saying what is missing.

    The alternatives step sizes its rows from this same answer, so the two
    steps cannot read one component two ways.

    `quantity` is the component's count: a flow read from the usage_context is
    only taken for a single unit (aggregat.component_airflow).
    """
    from aida.data.aggregat import component_airflow, component_class

    _ensure_cache()

    flow, flow_where = component_airflow(name, usage_context, quantity)
    per_flow = _TYPVÄRDEN.get(("ventilation", "aggregat", FLOW_UNIT))
    if flow and per_flow:
        payload = dict(per_flow)
        for k in ("baseline_co2e_per_unit", "full_median", "min", "max"):
            payload[k] = round(per_flow[k] * flow, 1)
        return {
            "method": "luftflöde",
            "payload": payload,
            "airflow_m3h": flow,
            "airflow_source": flow_where,
            "per_m3h": per_flow["baseline_co2e_per_unit"],
            "extrapolated": not (per_flow["airflow_min"] <= flow <= per_flow["airflow_max"]),
        }

    klass, class_why = component_class(name, usage_context)
    if klass:
        sub = CLASS_SUBCATEGORY[klass]
        hit = _TYPVÄRDEN.get(("ventilation", sub, "st"))
        if hit:
            no_flow = flow is None and flow_where != "inget luftflöde angivet"
            return {"method": "klass", "payload": hit, "klass": klass,
                    "class_why": class_why,
                    "flow_note": flow_where if no_flow else ""}

    if flow:
        flow_part = "Katalogen saknar ett typvärde per luftflöde"
    else:
        flow_part = flow_where[:1].upper() + flow_where[1:]
    if klass:
        why = (withheld_reason("ventilation", CLASS_SUBCATEGORY[klass], "st")
               or "katalogen har för få EPD:er för klassen")
        reason = f"{flow_part}. Inget typvärde för {klass}saggregat: {why}"
    else:
        reason = f"{flow_part}, och {class_why}"
    return {"method": "uppskattning", "payload": None, "reason": reason, "klass": klass,
            "airflow_m3h": flow,
            # Whether stating a flow would give a number: the row only asks for
            # it when it would.
            "flow_available": per_flow is not None}


def list_available_categories() -> list[tuple[str, str, str, int]]:
    """List all (category, subcategory, unit, sample_size) with a typvärde."""
    _ensure_cache()
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
