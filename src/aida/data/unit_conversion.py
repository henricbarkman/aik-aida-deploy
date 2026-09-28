"""Unit conversion for climate data: kg CO2e/kg → kg CO2e per functional unit.

Boverket provides climate data in kg CO2e/kg. Aida needs values per functional
unit (m2, st, lm) to match how renovation projects are described.

Conversion formula for area-based products:
    CO2e/m2 = CO2e/kg × density (kg/m3) × thickness (m)

For count-based products (windows, doors, elevators):
    CO2e/st = CO2e/kg × typical_weight_kg

For linear products (ventilation ducts, pipes):
    CO2e/lm = CO2e/kg × weight_per_meter (kg/m)

Typical values below are defaults for common renovation scenarios.
JJ/project team can adjust these based on actual project specifications.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass
class ConversionSpec:
    """Specification for converting kg CO2e/kg to a functional unit."""
    target_unit: str          # "m2", "st", "lm"
    method: str               # "area", "count", "linear", "direct"
    typical_thickness_m: float = 0.0    # for area-based (m)
    typical_weight_kg: float = 0.0      # for count-based (kg/st) or linear (kg/m)
    description: str = ""


# Typical thicknesses and weights for renovation scenarios in Swedish buildings.
# Sources: Boverkets klimatdatabas, NollCO2 Manual 1.2, LFM30 beräkningsanvisning.
#
# These are DEFAULTS. Actual projects should specify their own values.
# JJ: Review and adjust based on fastighetsavdelningens typiska val.

COMPONENT_CONVERSIONS: dict[str, ConversionSpec] = {
    # --- Area-based (CO2e/m2 = CO2e/kg × density × thickness) ---

    "golv": ConversionSpec(
        target_unit="m2",
        method="area",
        typical_thickness_m=0.015,  # 15mm vinyl/linoleum/parkett
        description="Golvbeläggning, 15mm typisk tjocklek",
    ),
    "innervägg": ConversionSpec(
        target_unit="m2",
        method="area",
        typical_thickness_m=0.013,  # 13mm gipsskiva (en sida)
        description="Gipsskiva innervägg, 13mm standard",
    ),
    "yttervägg": ConversionSpec(
        target_unit="m2",
        method="area",
        typical_thickness_m=0.200,  # 200mm komplett vägguppbyggnad
        description="Yttervägg komplett, ~200mm",
    ),
    # Only the cladding layer, not the wall behind it. Swedish facade panel is
    # typically 22mm (finsågad eller hyvlad panel, 22x120 to 22x170); fibre
    # cement boards sit at 8-12mm, so 22 is the generous end for wood and the
    # right order of magnitude for the category. Running a wood panel through
    # yttervägg's 200mm instead overstates it roughly ninefold.
    "fasadskikt": ConversionSpec(
        target_unit="m2",
        method="area",
        typical_thickness_m=0.022,
        description="Fasadskikt (panel eller skiva), 22mm typisk",
    ),
    "betongvägg": ConversionSpec(
        target_unit="m2",
        method="area",
        typical_thickness_m=0.200,  # 200mm betongvägg
        description="Betongvägg, 200mm typisk",
    ),
    "tak": ConversionSpec(
        target_unit="m2",
        method="area",
        typical_thickness_m=0.025,  # 25mm takpannor
        description="Takbeläggning, 25mm",
    ),
    "isolering": ConversionSpec(
        target_unit="m2",
        method="area",
        typical_thickness_m=0.200,  # 200mm isolering
        description="Tilläggsisolering, 200mm",
    ),

    # Paint has no meaningful "thickness x density". What decides how much
    # product a square metre costs is åtgång (spread rate), so `farg` converts
    # only through AREAL_DENSITY_KG_M2 below. typical_thickness_m is left at 0
    # deliberately: it keeps the density path in convert_to_functional_unit and
    # the whole of convert_m3_to_m2 from firing on paint, so a row we cannot
    # give a spread rate keeps its native unit instead of getting an invented
    # one.
    "farg": ConversionSpec(
        target_unit="m2",
        method="area",
        typical_thickness_m=0.0,
        description="Färg, omräknad via åtgång (kg/m2), inte tjocklek",
    ),

    # --- Count-based (CO2e/st = CO2e/kg × typisk vikt) ---

    "fönster": ConversionSpec(
        target_unit="st",
        method="count",
        typical_weight_kg=40.0,  # ~40 kg per standardfönster 1.2x1.2m
        description="Standardfönster ~1.2x1.2m, ~40 kg",
    ),
    "dörr": ConversionSpec(
        target_unit="st",
        method="count",
        typical_weight_kg=25.0,  # ~25 kg innerdörr
        description="Innerdörr standard, ~25 kg",
    ),
    "storköksutrustning": ConversionSpec(
        target_unit="st",
        method="count",
        typical_weight_kg=80.0,
        description="Storköksutrustning, ~80 kg",
    ),
    "sanitet": ConversionSpec(
        target_unit="st",
        method="count",
        typical_weight_kg=15.0,
        description="Sanitetsprodukt (toalett/handfat), ~15 kg",
    ),
    "vitvaror": ConversionSpec(
        target_unit="st",
        method="count",
        typical_weight_kg=50.0,
        description="Vitvara (tvättmaskin/köksfläkt), ~50 kg",
    ),
    "kylanläggning": ConversionSpec(
        target_unit="st",
        method="count",
        typical_weight_kg=200.0,
        description="Kylsystem, ~200 kg",
    ),
    "hiss": ConversionSpec(
        target_unit="st",
        method="count",
        typical_weight_kg=3000.0,
        description="Personhiss komplett, ~3000 kg",
    ),
    "belysning": ConversionSpec(
        target_unit="st",
        method="count",
        typical_weight_kg=3.0,
        description="LED-armatur, ~3 kg",
    ),

    # --- Linear (CO2e/lm = CO2e/kg × kg/m) ---

    "ventilation": ConversionSpec(
        target_unit="lm",
        method="linear",
        typical_weight_kg=5.0,  # ~5 kg/m stålkanal
        description="Ventilationskanal stål, ~5 kg/m",
    ),
}


def convert_to_functional_unit(
    co2e_per_kg: float,
    component_key: str,
    density_kg_m3: float | None = None,
    areal_density_kg_m2: float | None = None,
) -> tuple[float, str]:
    """Convert kg CO2e/kg to CO2e per functional unit.

    Args:
        co2e_per_kg: Climate impact in kg CO2e per kg of material
        component_key: Aida component category (e.g. "golv", "fönster")
        density_kg_m3: Material density from Boverket (optional, used for area method)
        areal_density_kg_m2: Areal weight from Boverket (optional). When present
            it is used directly for area products (no thickness assumption), which
            is both more accurate and avoids the density/areal-weight confusion.

    Returns:
        Tuple of (co2e_per_unit, unit_string)
        Falls back to (co2e_per_kg, "kg") if no conversion is defined.
    """
    spec = COMPONENT_CONVERSIONS.get(component_key)
    if not spec:
        return co2e_per_kg, "kg"

    if spec.method == "area":
        # Prefer a measured kg/m2 (areal weight) when Boverket provides one:
        # CO2e/m2 = CO2e/kg x kg/m2, no thickness guess needed.
        if areal_density_kg_m2 and areal_density_kg_m2 > 0:
            return round(co2e_per_kg * areal_density_kg_m2, 2), "m2"
        if density_kg_m3 and density_kg_m3 > 0 and spec.typical_thickness_m > 0:
            co2e_per_m2 = co2e_per_kg * density_kg_m3 * spec.typical_thickness_m
            return round(co2e_per_m2, 2), "m2"
        return co2e_per_kg, "kg"  # can't convert without density

    elif spec.method == "count":
        if spec.typical_weight_kg > 0:
            co2e_per_st = co2e_per_kg * spec.typical_weight_kg
            return round(co2e_per_st, 1), "st"
        return co2e_per_kg, "kg"

    elif spec.method == "linear":
        if spec.typical_weight_kg > 0:
            co2e_per_lm = co2e_per_kg * spec.typical_weight_kg
            return round(co2e_per_lm, 2), "lm"
        return co2e_per_kg, "kg"

    return co2e_per_kg, "kg"


def convert_m3_to_m2(co2e_per_m3: float, component_key: str) -> tuple[float, str] | None:
    """Convert a volume-declared figure to per square metre, using the layer
    thickness the category is defined by.

    Timber products are declared per m3 while a facade component is measured in
    m2, and without this bridge those EPDs never join the m2 bucket. That left
    fasadskikt with six m2 entries, two of them aluminium, so the "conventional"
    reference for a wood facade landed on anodised aluminium at 36.7 kg/m2.

    Returns None when the category has no area-based thickness to divide by, so
    callers keep the native unit rather than inventing a number.
    """
    spec = COMPONENT_CONVERSIONS.get(component_key)
    if not spec or spec.method != "area" or spec.typical_thickness_m <= 0:
        return None
    return round(co2e_per_m3 * spec.typical_thickness_m, 2), "m2"


# --- Thermal resistance -----------------------------------------------------
#
# An m2-declared insulation figure is per square metre AT THAT PRODUCT'S OWN
# THICKNESS, and the thickness is in neither the product name (0 of 137 rows)
# nor any structured field. So 0.46 and 21.41 kg CO2e/m2 can sit in the same
# bucket describing the same material at different depths, and ranking them puts
# the thinnest declaration first. That is what decision #22 in the mission file
# flagged.
#
# Some declarations state a thermal resistance in the reference flow text, and
# many state R = 1 exactly, which is the industry's own way of making products
# comparable. Where it is stated we can read it; where it is not, nothing in the
# data recovers it.
#
# Deliberately NOT normalised into the ranking. Measured 2026-09-07: only 23 of
# 113 m2 rows declare R, and restricting the queue to those would drop 35 of the
# 40 Nordic rows -- all of ROCKWOOL, both Ekolution hemp rows and Isover
# Building Insulation -- leaving a Glava/Isover glass wool monoculture. Scaling
# the 23 to a reference R while leaving 90 rows unscaled would be worse still,
# since the normalised rows would read about five times higher than the rest for
# no physical reason. So the value is recorded and surfaced as a caveat rather
# than silently folded into an order it cannot support.
_R_DECLARED = re.compile(
    r"(?:r[-\s]?value|thermal resistance|resistenza termica)"
    r"[^0-9]{0,25}(\d+(?:[.,]\d+)?)",
    re.IGNORECASE,
)


def declared_thermal_resistance(flow_description: str) -> float | None:
    """The R-value (m2K/W) an EPD states for its declared unit, or None.

    Reads only what the declaration says. No thickness-times-lambda fallback:
    that would trade a manufacturer's figure for a guess at the material's
    conductivity, and it would apply to eight rows, several of them composites
    (a gypsum board with 20 mm of EPS bonded to it) or products for tunnels
    rather than buildings.
    """
    if not flow_description:
        return None
    match = _R_DECLARED.search(flow_description)
    if not match:
        return None
    value = float(match.group(1).replace(",", "."))
    # Building insulation sits between R=0.5 and R=10. Outside that the regex
    # has caught a product code, a year, or a lambda.
    return value if 0.3 <= value <= 12 else None


def get_density_from_extra(extra_json: str) -> float | None:
    """Extract density from a cache entry's extra_json field."""
    if not extra_json:
        return None
    try:
        extra = json.loads(extra_json)
        return extra.get("density_kg_m3")
    except (json.JSONDecodeError, TypeError):
        return None


def get_areal_density_from_extra(extra_json: str) -> float | None:
    """Extract areal weight (kg/m2) from a cache entry's extra_json field."""
    if not extra_json:
        return None
    try:
        extra = json.loads(extra_json)
        val = extra.get("areal_density_kg_m2")
        return val if val and val > 0 else None
    except (json.JSONDecodeError, TypeError):
        return None


# Typical densities for common building materials (kg/m3).
# Used as fallback when EPD/Boverket doesn't provide density.
# Sources: Boverkets klimatdatabas, IVL, materialhandböcker.
TYPICAL_DENSITIES: dict[str, float] = {
    "golv": 1400,       # vinyl/linoleum ~1300-1500
    "innervägg": 800,   # gipsskiva ~700-900
    "yttervägg": 1800,  # tegel+puts ~1600-2000
    "betongvägg": 2400, # betong C30/37
    "tak": 2100,        # betongpannor ~2000-2200
    "isolering": 30,    # mineralull/glasull ~20-40
}

# Applied weight per square metre (kg/m2) for products whose EPD is declared per
# kg but whose component is measured in m2, and where "density x thickness" is
# the wrong model. Paint and render are applied at a rate (åtgång), not built to
# a thickness, so this is the parameter the trade actually specifies.
#
# Keyed by category, then by product-name keywords, because a category can hold
# several application rates. A row matching nothing gets no bridge and keeps its
# kg unit, which is the point: this table is an allow-list, not a default.
#
# Why this is a smaller assumption than it looks. The bridge multiplies every
# matched row in a category by the SAME constant, so the ranking among bridged
# rows is completely unaffected by the value chosen. What the value decides is
# the level, i.e. how bridged rows compare against rows already declared per m2,
# and what the baseline typvärde comes out at. Getting the order of magnitude
# right is therefore what matters; the third digit is not doing any work.
#
# farg 0.45 kg/m2: two coats, paint density ~1.4 kg/l. Swedish facade paint is
#   commonly stated at 4-6 m2/l per coat and interior paint at 7-10, which puts
#   exterior work near 0.5-0.7 kg/m2 and interior near 0.28-0.40. The bucket
#   holds both in roughly equal numbers, so the midpoint is the honest single
#   value.
#
#   It was first set at 0.55, the exterior end, and that was wrong for a bucket
#   that is mostly interior wall paint. The tripwire in test_epd_bucket_purity
#   caught it: painting 8 m2 came out at 9.9 kg CO2e against 15.4 kg for the
#   gypsum board behind it, which is not a credible ratio for two coats of
#   paint. That test exists because Henric flagged the same ratio at 85.6 kg in
#   August, and it earned its place a second time here.
#
#   An interior/exterior split by product name was measured and rejected: it
#   classifies only 26 of 43 rows, and the coverage that actually applies is a
#   property of the job (how many coats, what substrate), not of what the tin is
#   marketed as. The component being repainted decides that, not the EPD.
#
# fasadskikt 20 kg/m2 for render: ~12 mm at ~1650 kg/m3. Weber's own data sheets
#   put putsbruk at 1.5-1.8 kg/m2 per mm, and Swedish render runs 10 mm for a
#   thin coat to 25 mm-plus for traditional three-coat work, so the range is
#   roughly 15-40 kg/m2. Only render matches here; the sandwich panels and WPC
#   profiles in the same category are built to a thickness and must not borrow
#   a mortar's spread rate.
AREAL_DENSITY_KG_M2: dict[str, list[tuple[list[str], float]]] = {
    "farg": [
        (["paint", "färg", "farg", "coating", "coatings", "enamel", "lasyr",
          "rödfärg", "sludgepaint", "husmaling", "emulsion", "primer"], 0.45),
    ],
    "fasadskikt": [
        (["puts", "putsbruk", "murbruk", "render", "renders", "mortar",
          "kalkspritputs", "stänkputs", "spritputs", "rivputs",
          "slamningsputs", "lerputs", "finputsbruk", "designputs"], 20.0),
    ],
}


def areal_density_for_product(category: str, product_name: str) -> float | None:
    """Applied kg/m2 for a product, or None when no rate is known for it.

    None is the common answer and the safe one: the caller then leaves the row
    in its declared unit rather than promoting it into an m2 queue on a spread
    rate borrowed from a different kind of product.
    """
    rules = AREAL_DENSITY_KG_M2.get(category)
    if not rules or not product_name:
        return None
    name_lower = product_name.lower()
    for keywords, kg_per_m2 in rules:
        if any(kw in name_lower for kw in keywords):
            return kg_per_m2
    return None


# Material-specific density overrides based on product name keywords.
# Used when the generic TYPICAL_DENSITIES is too broad (e.g. "golv" covers
# vinyl at 1400 AND wood at 600 — very different).
_MATERIAL_DENSITY_HINTS: list[tuple[list[str], float]] = [
    # Wood-based flooring
    (["parkett", "parquet", "trä", "wood", "timber", "oak", "ash", "birch",
      "bamboo", "bambú", "hardwood", "lightwood", "maxwood"], 600),
    # Laminate flooring
    (["laminat", "laminate"], 850),
    # Ceramic / stone tiles
    (["klinker", "ceramic", "porcelain", "kakel", "tile", "stone", "marble",
      "granite", "terrazzo", "slate"], 2200),
    # Cork
    (["kork", "cork"], 200),
    # Rubber
    (["gummi", "rubber"], 1200),
    # Resilient flooring (linoleum, vinyl, PVC) — listed BEFORE carpet so that
    # "linoleummatta"/"plastmatta"/"vinylmatta" match here, not the "matta"
    # carpet rule. These are ~3x denser than carpet (1200-1400 vs 400).
    (["linoleum"], 1200),
    (["vinyl", "pvc", "plastmatta", "plastgolv"], 1400),
    # Carpet — "matta" alone is broad, so keep it last among flooring rules
    (["matta", "carpet", "textile"], 400),
    # Epoxy / polyurethane coatings
    (["epoxy", "polyuretan", "polyurethane"], 1100),
]


def get_density_for_component(
    component_key: str,
    extra_json: str = "",
    product_name: str = "",
) -> float | None:
    """Get density for a component, trying extra_json first, then material-specific hints.

    Checks:
    1. density_kg_m3 in extra_json (from Boverket API)
    2. Material-specific density from product name keywords
    3. Typical density lookup by component key
    """
    # Try explicit density from data source
    density = get_density_from_extra(extra_json)
    if density and density > 0:
        return density

    # Try material-specific density from product name
    if product_name:
        name_lower = product_name.lower()
        for keywords, mat_density in _MATERIAL_DENSITY_HINTS:
            if any(kw in name_lower for kw in keywords):
                return mat_density

    # Fall back to typical density
    return TYPICAL_DENSITIES.get(component_key)


# Typical mass per item (kg/st) for count-denominated components whose EPDs are
# declared per kg. Lets baseline tier 2 bridge a kg typvärde to a per-st value
# (st typvärde = kg typvärde × mass). Keyed by (category, subcategory); use
# subcategory "" for flat categories (radiator). These are deliberate
# approximations — flagged as such in the baseline description — since EPD
# reference mass is "per 1 kg" and gives no product mass.
# Sources: manufacturer spec sheets, general product knowledge.
TYPICAL_ITEM_MASS_KG: dict[tuple[str, str], float] = {
    ("sanitet", "toalett"): 25.0,
    ("sanitet", "toalettsits"): 3.0,
    ("sanitet", "handfat"): 18.0,
    ("sanitet", "blandare"): 2.5,
    ("sanitet", "dusch"): 30.0,
    ("sanitet", "badkar"): 45.0,
    ("sanitet", "urinal"): 20.0,
    ("belysning", "armatur"): 3.0,
    ("belysning", "taklampa"): 3.0,
    ("belysning", "vägglampa"): 2.0,
    ("vitvaror", "tvättmaskin"): 65.0,
    ("vitvaror", "torktumlare"): 38.0,
    ("vitvaror", "spis"): 50.0,
    ("vitvaror", "köksfläkt"): 9.0,
    ("vitvaror", "mikro"): 14.0,
    ("radiator", ""): 30.0,
}


def typical_item_mass(category: str, subcategory: str = "") -> float | None:
    """Typical kg per item for a (category, subcategory), or None if unknown."""
    return TYPICAL_ITEM_MASS_KG.get((category, subcategory))


# --- Member geometry from the component name ---------------------------------
#
# A stud or a beam is counted in löpmeter and a structural board in m2, while
# the timber and board EPDs are declared per m3 (HENRIC-3290). The bridge is the
# member's own geometry: a 45x95 stud IS 0.045 x 0.095 m in section, so one m3
# of that timber is 234 m of stud. That is a fact the person naming the
# component stated, not a thickness or density borrowed from the category,
# which is the line the rest of this module holds: convert_m3_to_m2 uses a
# category thickness only where the category is a layer of that thickness, and
# the kg bridge on a category-wide density stays refused.
#
# So the dimension is read from the component name and nowhere else. No
# dimension in the name, no bridge; the caller then says what is missing
# instead of guessing, and the chat asks for it.
_CROSS_SECTION_RE = re.compile(r"(?<![\d.,])(\d{2,4})\s*[x×*]\s*(\d{2,4})(?![\d.,])")
_THICKNESS_RE = re.compile(r"(?<![\d.,x×*])(\d{1,3}(?:[.,]\d)?)\s*mm\b", re.IGNORECASE)
# A board named by its full format, sheet face then thickness: "OSB-skiva
# 1200x2400x12 mm", "Plywood 1250x2500x9mm". The thickness is the smallest of
# the three. Without this the two-number pattern took "1200x2400", left "x12
# mm" behind, and the thickness pattern refused the number after the "x".
_BOARD_FORMAT_RE = re.compile(
    r"(?<![\d.,])(\d{2,4})\s*[x×*]\s*(\d{2,4})\s*[x×*]\s*(\d{1,3}(?:[.,]\d)?)(?![\d.,])")
# The narrow side of a member section. The widest glulam in Swedish standard
# widths is 215 mm and a deep beam is still under 400, while the narrow side of
# a sheet format ("600x1200", "1200x2400") is 600 or more. Found in review
# 2026-09-28: "Konstruktionsskiva 600x1200 mm" in lm read as 0.72 m3 per metre.
_MAX_SECTION_NARROW_SIDE_MM = 400


def cross_section_mm(name: str) -> tuple[float, float] | None:
    """(width, height) in mm from "45x95", "45×70" or "90 x 315" in a name.

    Outside 10-2000 mm is not a member section: "2x4" (inches) and panel
    formats written in metres are left alone rather than read as millimetres.
    Nor is a pair whose narrow side is wider than any beam: that is the face of
    a sheet.
    """
    m = _CROSS_SECTION_RE.search(name or "")
    if not m:
        return None
    a, b = float(m.group(1)), float(m.group(2))
    if not (10 <= a <= 2000 and 10 <= b <= 2000):
        return None
    if min(a, b) > _MAX_SECTION_NARROW_SIDE_MM:
        return None
    return a, b


def thickness_mm(name: str) -> float | None:
    """A stated thickness ("12 mm", "22mm", "200 mm") in a name, or None.

    A full board format ("1200x2400x12 mm") gives its smallest number. A
    cross-section is removed first, so "45 x 95 mm" is a section and not a 95 mm
    board. Outside 3-500 mm is not a board or a slab.
    """
    fmt = _BOARD_FORMAT_RE.search(name or "")
    if fmt:
        value = min(float(g.replace(",", ".")) for g in fmt.groups())
        return value if 3 <= value <= 500 else None
    text = _CROSS_SECTION_RE.sub(" ", name or "")
    m = _THICKNESS_RE.search(text)
    if not m:
        return None
    value = float(m.group(1).replace(",", "."))
    return value if 3 <= value <= 500 else None


def member_volume_per_unit(name: str, unit: str) -> tuple[float, str] | None:
    """(m3 per component unit, label) from the name's own dimension, or None.

    lm needs a cross-section, m2 a thickness. Any other unit, or a name without
    the dimension its unit needs, returns None: a stud in st has no length, and
    a board without a thickness has no volume.
    """
    unit = (unit or "").strip().lower()
    if unit in ("lm", "m", "meter", "löpmeter"):
        section = cross_section_mm(name)
        if section:
            w, h = section
            return (w / 1000) * (h / 1000), f"tvärsnitt {w:g}×{h:g} mm"
    elif unit in ("m2", "m²", "kvm"):
        t = thickness_mm(name)
        if t:
            return t / 1000, f"tjocklek {t:g} mm"
    return None
