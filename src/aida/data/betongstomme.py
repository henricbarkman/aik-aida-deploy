"""Precast concrete frame: hollow-core slabs, solid slabs and beams (HENRIC-3362).

The sixth stomme family, `betong`. It differs from the timber and steel
families in two ways, and both are kept here so the catalog build, the
alternatives step and the tests read one answer.

1. Within the family a slab is not a beam. A hollow-core slab (håldäck), a
   solid slab (massivbjälklag, plattbärlag) and a beam or column are three
   products with three functions, so a component only meets rows of its own
   element kind (`element_kind`), the way a steel HEA beam only meets open
   sections. A name that states no kind, or two ("half-slab, sandwich wall,
   concrete beam"), meets nothing.

2. The EPDs are declared per tonne of element, and a slab is bought per m².
   How many kilograms one m² is depends on the element:

   - Solid concrete is solid: 200 mm of it is 500 kg/m² (Svensk Betong's
     element table, RD/F 120/20; PL 240/5 is 50 mm and 125 kg/m², the same
     2 500 kg/m³). So a solid slab's or a beam's kg rows are restated per m²
     or per metre from the thickness or section in the component's name.
   - A hollow-core slab is not. Its voids differ between makers, and the same
     table gives HD/F 120/20 (200 mm) as 255 to 330 kg/m². No single weight
     per m² follows from the thickness, so it is never derived from it. A
     hollow-core row reaches an m² component only through a weight per m² and
     a thickness the EPD itself states (`EPD_FACTS`), or through a weight per
     m² the component's own name states ("Håldäck 200 mm 290 kg/m2"), which
     is the supplier's figure for the slab actually bought.

Sources, read 2026-10-01:
  Svensk Betong, "Däckelement", svenskbetong.se/om-betong/prefab/
  produktredovisning/komponenter-till-hus-och-anlaggning/dackelement:
  HD/F 120/19 185 mm 275 kg/m², 120/20 200 mm 255-330, 120/27 265 mm 320-440,
  120/32 320 mm 385-400, 120/40 400 mm 415-500, 120/50 500 mm 610; RD/F
  120/20 and 240/20 200 mm 500 kg/m²; PL 240/5 50 mm 125, PL/F 240/7 70 mm 160.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Solid reinforced concrete, kg per m3. Svensk Betong's table above: RD/F
# 120/20 is 200 mm and 500 kg/m², PL 240/5 is 50 mm and 125 kg/m².
SOLID_DENSITY_KG_M3 = 2500.0
SOLID_DENSITY_SOURCE = ("Svensk Betong, Däckelement: massivplatta RD/F 200 mm "
                        "väger 500 kg/m², alltså 2 500 kg/m³")

# HD/F height in mm -> (lightest, heaviest) kg/m² in the same table. A range
# because "antalet hål varierar": each maker has its own void profile.
HDF_WEIGHT_RANGE_KG_M2: dict[float, tuple[float, float]] = {
    185: (275, 275), 200: (255, 330), 265: (320, 440), 320: (385, 400),
    400: (415, 500), 500: (610, 610),
}


def hdf_weight_text(t_mm: float) -> str:
    """What Svensk Betong says a hollow-core slab of this height weighs, as a
    clause for a reason. The 200 mm range when the height is not in the table."""
    height = min(HDF_WEIGHT_RANGE_KG_M2, key=lambda h: abs(h - t_mm))
    if abs(height - t_mm) > THICKNESS_TOLERANCE_MM:
        height = 200
    lo, hi = HDF_WEIGHT_RANGE_KG_M2[height]
    weight = f"{lo:g} kg/m²" if lo == hi else f"{lo:g}-{hi:g} kg/m²"
    return (f"Svensk Betong anger {weight} för ett {height:g} mm håldäck (HD/F), "
            f"beroende på tillverkarens hålprofil")

# HD/F designation -> height in mm, from the same table. "HD/F 120/27" is a
# 265 mm slab, not 270, which is why the number after the slash is looked up
# rather than multiplied by ten.
HDF_HEIGHT_MM: dict[int, float] = {19: 185, 20: 200, 27: 265, 32: 320, 40: 400, 50: 500}

# Two hollow-core slabs are the same product size when their heights differ by
# at most this. The Mineral Products Association's UK average is 201.24 mm;
# a 220 mm slab is a different, stronger slab and not an alternative to 200.
THICKNESS_TOLERANCE_MM = 5.0

HÅLDÄCK, MASSIV, BALK = "håldäck", "massiv", "balk"

KIND_LABELS = {
    HÅLDÄCK: "håldäck",
    MASSIV: "massiva bjälklag och plattbärlag",
    BALK: "betongbalkar och pelare",
}

# Words that name each kind, in the EPD languages (English, Swedish,
# Norwegian, Danish, Finnish) and in a Swedish component name. Matched on the
# lowercased name; a name hitting two kinds is mixed and gets no kind.
_KIND_PATTERNS: dict[str, tuple[re.Pattern, ...]] = {
    HÅLDÄCK: (
        re.compile(r"hollow[\s-]?core"),
        re.compile(r"håldäck|hålbjälklag|huldæk|hulldekk|kanalplatt|ontelolaat"),
        # The Swedish product designation, "HD/F 120/20", "HDF 120/27". Not
        # "HDF" alone, nor "HDF 3 mm": that is high-density fibreboard.
        re.compile(r"\bhd\s*/?\s*f\s*\d{2,3}\s*/\s*\d{2}\b"),
        re.compile(r"\bhd\s*/\s*f\b"),
    ),
    MASSIV: (
        re.compile(r"massivbjälklag|massivplatt|massivdekk|plattendekk"),
        re.compile(r"massive concrete slab|solid slab|solid precast concrete, pre-stressed slab"),
        re.compile(r"plattbärlag|filigr|half[\s-]?slab|d-platta|\brd\s*/\s*f\b"),
        re.compile(r"floor plates?\b"),
    ),
    BALK: (
        re.compile(r"\bbeams?\b|\bcolumns?\b|\bpurlins?\b|\bpillars?\b"),
        # "balk" but not "balkong" (a balcony is not frame).
        re.compile(r"balk(?!ong)|pelare|bjelke|søyle|bjælke|søjle"),
    ),
}

# Lightweight aggregate (lättklinker, Leca) is the betong family on the
# component side (palats_client), but no catalog row is lightweight concrete,
# and its weight per m3 is a third of a dense element's.
_LIGHTWEIGHT = re.compile(r"lättklinker|lettklinker|\bleca|klinkerbalk")
_REINFORCEMENT = re.compile(r"armering|armeringsjärn|rebar")
_CAST_IN_PLACE = re.compile(r"platsgjut|plattsgjut|in[\s-]situ|cast[\s-]in[\s-]place")


# A slab of no stated kind. Not a kind of its own (a "prestressed slab" can be
# hollow-core or solid), but it makes a beam name mixed: "Precast Concrete
# Products (Beams, planks, double tee slabs)" is not a beam EPD.
_ANY_SLAB = re.compile(r"\bslabs?\b|\bplanks?\b|double[\s-]?tee|dekke\b|\bdæk")


def element_kind(name: str) -> str:
    """HÅLDÄCK, MASSIV, BALK, or "" for a name stating no kind or two."""
    text = (name or "").lower()
    kinds = {kind for kind, patterns in _KIND_PATTERNS.items()
             if any(p.search(text) for p in patterns)}
    if kinds == {BALK} and _ANY_SLAB.search(text):
        return ""
    return kinds.pop() if len(kinds) == 1 else ""


def is_lightweight(name: str) -> bool:
    return bool(_LIGHTWEIGHT.search((name or "").lower()))


def is_reinforcement(name: str) -> bool:
    return bool(_REINFORCEMENT.search((name or "").lower()))


def is_cast_in_place(name: str) -> bool:
    return bool(_CAST_IN_PLACE.search((name or "").lower()))


_HDF_RE = re.compile(r"\bhd\s*/?\s*f\s*\d{2,3}\s*/\s*(\d{2})\b", re.IGNORECASE)
_KG_PER_M2_RE = re.compile(r"(\d{2,4}(?:[.,]\d+)?)\s*kg\s*/\s*(?:m2|m²|kvm)\b", re.IGNORECASE)
# A concrete section. unit_conversion.cross_section_mm caps the narrow side at
# 400 mm, which is right for timber and wrong for a 500x800 concrete beam.
_SECTION_RE = re.compile(r"(?<![\d.,/])(\d{2,4})\s*[x×*]\s*(\d{2,4})(?![\d.,])")


def slab_thickness_mm(name: str) -> tuple[float, str] | None:
    """(height in mm, how it was read) for a slab component, or None.

    "200 mm" in the name, or an HD/F designation looked up in Svensk Betong's
    table ("HD/F 120/27" is 265 mm)."""
    from aida.data.unit_conversion import thickness_mm

    m = _HDF_RE.search(name or "")
    if m and int(m.group(1)) in HDF_HEIGHT_MM:
        h = HDF_HEIGHT_MM[int(m.group(1))]
        return h, f"{m.group(0).upper()}, {h:g} mm enligt Svensk Betong"
    t = thickness_mm(name or "")
    if t and 30 <= t <= 600:
        return t, f"tjocklek {t:g} mm"
    return None


def stated_kg_per_m2(name: str) -> float | None:
    """A weight per m² the component's own name states ("290 kg/m2")."""
    m = _KG_PER_M2_RE.search(name or "")
    if not m:
        return None
    value = float(m.group(1).replace(",", "."))
    return value if 50 <= value <= 1500 else None


def beam_section_mm(name: str) -> tuple[float, float] | None:
    """(width, height) in mm of a concrete beam or column, or None."""
    m = _SECTION_RE.search(name or "")
    if not m:
        return None
    a, b = float(m.group(1)), float(m.group(2))
    if not (100 <= a <= 2000 and 100 <= b <= 2000):
        return None
    return a, b


@dataclass(frozen=True)
class ElementFacts:
    """What one EPD itself says about the element it declares, read by hand."""

    thickness_mm: float
    kg_per_m2: float
    source: str


# Keyed by registration number (the part before the colon, which survives a
# re-issue). Only facts the declaration states; nothing here is a typical value.
EPD_FACTS: dict[str, ElementFacts] = {
    # Mineral Products Association (UK), "Hollowcore slabs", declared per m².
    # Reference flow material properties: layer thickness 0.20124 m, grammage
    # 305.37 kg/m² (data.environdec.com, read 2026-10-01).
    "EPD-IES-0022253": ElementFacts(
        201.24, 305.37,
        "EPD-IES-0022253: tjocklek 201 mm och 305 kg/m² enligt EPD:n"),
    # INHUS Prefab OÜ (Estonia), declared per tonne. The name states the weight
    # per m², the reference flow a gross density; height = weight / density:
    # 302 / 1 373 = 0.220 m and 454 / 1 135 = 0.400 m (read 2026-10-01).
    "EPD-IES-0033332": ElementFacts(
        220.0, 302.0,
        "EPD-IES-0033332: 302 kg/m² enligt EPD:n, 220 mm härlett ur EPD:ns "
        "densitet 1 373 kg/m³"),
    "EPD-IES-0033333": ElementFacts(
        400.0, 454.0,
        "EPD-IES-0033333: 454 kg/m² enligt EPD:n, 400 mm härlett ur EPD:ns "
        "densitet 1 135 kg/m³"),
    "EPD-IES-0033334": ElementFacts(
        400.0, 454.0,
        "EPD-IES-0033334: 454 kg/m² enligt EPD:n, 400 mm härlett ur EPD:ns "
        "densitet 1 135 kg/m³"),
}


def facts_for(reg_no: str) -> ElementFacts | None:
    return EPD_FACTS.get((reg_no or "").split(":")[0].split(" ")[0].strip())


def apply_element_facts(e: dict) -> None:
    """Write a hollow-core row's stated thickness and weight per m² onto it, or
    strip them. Called by both catalog paths, like apply_aggregat_facts: a row
    that leaves the family or a fact removed from EPD_FACTS keeps nothing."""
    for k in ("element_thickness_mm", "element_kg_per_m2", "element_facts_source"):
        e.pop(k, None)
    if e.get("category") != "stomme" or e.get("subcategory") != "betong":
        return
    facts = facts_for(e.get("reg_no", ""))
    if not facts:
        return
    e["element_thickness_mm"] = facts.thickness_mm
    e["element_kg_per_m2"] = facts.kg_per_m2
    e["element_facts_source"] = facts.source
