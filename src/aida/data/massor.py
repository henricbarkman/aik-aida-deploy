"""The weight of a building part and the figures that go per kilo.

Two features need the same two things: how much one unit of a component weighs,
and which Boverket record stands for its material. Reuse (DECISIONS 18) is that
weight times Boverket's transport for a reused product. The life cycle stages
(DECISIONS 19) are that weight times transport, waste and demolition per kg.
One table, so a window cannot weigh one thing as reused and another as demolished.

Three kinds of weight, and the result says which it is:

- from the name: a thickness in the name times Boverket's density ("Gips 13 mm").
- from Boverket: a record's own kg/m², times the area of one piece when the
  component is counted in st ("Fönster, trä, sidhängt": 38,2 kg/m²).
- estimated: a typical value, marked as such everywhere it is shown. The area of
  a window in st, the weight of a toilet, a spiral duct's sheet steel.

No weight at all gives None and a reason, never a default and never zero: a lift
or a chiller differs by an order of magnitude from one to the next, and the
caller says so instead of showing a number with no basis. Frame members have
their own, finer rules in stomme_reuse, and this module defers to them.

Sources, read 2026-11-02:
- Boverkets klimatdatabas, version 02.07.000 (api.boverket.se/klimatdatabas,
  GetAllResources/senaste): A1-A3 Typical, A4, A5.1 and Conversions per record.
- IVL, Anvisningar för LCA-beräkning av byggprojekt, version 2025-02, tabell 17
  (anvisning S12): GWP-GHG for C3-C4 per product group, from CO2data.fi.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

BOVERKET_VERSION = "02.07.000"
IVL_SOURCE = "IVL, Anvisningar för LCA-beräkning av byggprojekt, version 2025-02"

# "Återanvänd byggprodukt", ResourceId 6000000174. A1-A3 is 0 and the A4 value
# covers 40 km to a cold store. Boverket: if the product is upgraded for its new
# use, that impact must be added. So the figure is a floor on what the reuse
# costs in practice, and the row says the upgrading is not in it.
REUSE_A4_PER_KG = 0.00539
REUSE_RECORD = "Återanvänd byggprodukt"

_LENGTH_UNITS = {"lm", "m", "meter", "löpmeter"}
_AREA_UNITS = {"m2", "m²", "kvm"}
_COUNT_UNITS = {"st", "styck", "stk", "pcs"}

# A window in st, when nothing else says how big it is: 1,2 x 1,2 m. Same size
# the count conversion for windows in unit_conversion has always assumed.
WINDOW_AREA_M2 = 1.44
# A door leaf in st: 9M x 21M, the standard Swedish opening.
DOOR_AREA_M2 = 1.89


@dataclass(frozen=True)
class Material:
    """One Boverket record's figures per kg, and its demolition group at IVL."""

    record: str
    a1a3: float
    a4: float
    a5: float | None  # module A5.1: waste on site; None when no record states it
    c34: float  # IVL table 17, kg CO2e per kg
    c34_group: str
    density_kg_m3: float | None = None
    kg_m2: float | None = None  # Boverket's own kg/m² for sheet goods and windows
    analog: bool = False  # Boverket has no record for the product itself; nearest one


# IVL table 17, kg CO2e per kg. The group names are IVL's.
_C34 = {
    "trä": (0.02, "Träprodukter"),
    "träskivor": (0.02, "Byggskivor trä"),
    "gips": (0.00077, "Gipsskivor"),
    "betong": (0.006, "Fabriksbetong och prefabbetong"),
    "murverk": (0.005, "Byggblock"),
    "bruk": (0.005, "Bindemedel, bruk, ballast"),
    "stål": (0.002, "Stål- och plåtprodukter"),
    "glasull": (0.005, "Isolering, glasull"),
    "stenull": (0.005, "Isolering, stenull"),
    "cellplast": (0.02, "Isolering, cellplast"),
    "plast": (0.02, "Plastprodukter"),
    "glas": (0.02, "Glas"),
    "övrigt": (0.02, "Övrigt, till exempel fönster, dörrar och kemiska produkter"),
}


def _m(record, a1a3, a4, a5, group, density=None, kg_m2=None) -> Material:
    c34, label = _C34[group]
    return Material(record, a1a3, a4, a5, c34, label, density, kg_m2)


# Boverket 02.07.000, as printed in the database. Only the records the table
# below points to.
MATERIALS: dict[str, Material] = {
    "gips": _m("Gipsskiva, standardskiva", 0.227, 0.0315, 0.0378, "gips", 710.0),
    "gips_våtrum": _m("Gipsskiva, våtrumsskiva", 0.26, 0.0315, 0.0428, "gips", 760.0),
    "gips_golv": _m("Gipsskiva, golvskiva", 0.237, 0.0315, 0.0393, "gips", 1120.0),
    "sågat": _m("Sågat virke, u 16 %, barrträ", 0.064, 0.0225, 0.0103, "trä", 455.0),
    "hyvlat": _m("Hyvlat virke, u 16 %, barrträ", 0.0735, 0.0225, 0.0114, "trä", 455.0),
    "plywood": _m("Plywood", 0.358, 0.0539, 0.0501, "träskivor", 460.0),
    "osb": _m("OSB", 0.358, 0.0809, 0.0528, "träskivor", 607.0),
    "spånskiva": _m("Spånskiva", 0.39, 0.0629, 0.055, "träskivor", 700.0),
    "glasull": _m("Glasull, skivor och rullar", 0.89, 0.045, 0.081, "glasull", 18.7),
    "stenull": _m("Stenull, skivor och rullar", 1.28, 0.045, 0.115, "stenull", 29.0),
    "eps": _m("EPS, expanderad polystyren", 3.2, 0.045, 0.283, "cellplast"),
    "tegel": _m("Tegelsten", 0.251, 0.0629, 0.0188, "murverk", 1800.0),
    "lättklinker": _m("Lättklinkerblock, 10-14 % cement (700-770 kg/m³)",
                      0.194, 0.036, 0.0139, "murverk", 750.0),
    "betong": _m("Fabriksbetong, husbyggnad C25/30", 0.103, 0.00472, 0.004,
                 "betong", 2350.0),
    "betongelement": _m("Vägg/Ytterskiva", 0.183, 0.0539, 0.0, "betong", 2400.0),
    "betongpanna": _m("Betongtakpannor", 0.18, 0.0629, 0.0144, "betong", 2400.0),
    "tegelpanna": _m("Tegeltakpannor", 0.216, 0.0629, 0.0166, "murverk", 1800.0),
    "murbruk": _m("Murbruk A (CS IV)", 0.199, 0.045, 0.0147, "bruk"),
    "fogmassa": _m("Fogmassa, övrigt", 3.25, 0.045, 0.164, "övrigt"),
    "spik": _m("Spikar, fästdon och beslag, primär stål", 2.59, 0.0989, 0.167, "stål"),
    "puts": _m("Mur- och putsbruk B (CS III)", 0.167, 0.045, 0.0127, "bruk", 1600.0),
    "avjämning": _m("Avjämningsmassor < 22 % cement", 0.176, 0.045, 0.0265, "bruk",
                    1750.0),
    "takpapp": _m("Takpapp, enskiktstätning", 0.65, 0.045, 0.0429, "övrigt",
                  kg_m2=5.5),
    "underlagspapp": _m("Underlagspapp", 0.68, 0.045, 0.0448, "övrigt", kg_m2=4.0),
    "takduk": _m("Takduk, PVC", 4.55, 0.0539, 0.287, "plast", kg_m2=1.8),
    "takshingle": _m("Takshingle", 0.4, 0.045, 0.0273, "övrigt", kg_m2=4.0),
    "plåt": _m("Tunnplåt för beklädnad, 80 % primär", 2.59, 0.0989, 0.167, "stål",
               7850.0),
    "stål": _m("Konstruktionsstål, alla sorter, 80 % primär råvara", 2.52, 0.0989,
               0.162, "stål", 7850.0),
    "aluminium": _m("Aluminiumprofiler, primär", 6.0, 0.0629, 0.378, "övrigt", 2700.0),
    "färg": _m("Utomhusfärg, vattenburen akryl", 2.5, 0.045, 0.127, "övrigt"),
    "slamfärg": _m("Slamfärg", 0.6, 0.045, 0.0318, "övrigt", kg_m2=0.4),
    "fibercement": _m("Fibercementskivor, fasadskiva, ≤ 85 vikt-% cement", 1.35,
                      0.0989, 0.179, "övrigt", 1950.0),
    "plast": _m("Plastprodukter typ termoplaster polyolefin, ospecificerad", 2.22,
                0.0629, 0.284, "plast", 900.0),
    "fönster_trä": _m("Fönster, trä, sidhängt, 3-glas", 2.0, 0.0539, 0.0, "övrigt",
                      kg_m2=38.2),
    "fönster_alu": _m("Fönster, trä/aluminium, sidhängt, 3-glas", 2.3, 0.0539, 0.0,
                      "övrigt", kg_m2=39.2),
    "fönster_pvc": _m("Fönster, PVC, alla typer, 3-glas", 2.1, 0.0539, 0.0, "övrigt",
                      kg_m2=39.0),
    "fönster_rena_alu": _m("Fönster, aluminium, alla typer, 3-glas", 3.06, 0.0809, 0.0,
                           "övrigt", kg_m2=35.0),
    "innerdörr": _m("Innerdörr utan glas, slät eller spegeldörr, oklassad", 1.43,
                    0.0539, 0.0, "övrigt", kg_m2=16.5),
    "innerdörr_ljud": _m("Innerdörr utan glas, laminerat trä, ljud och brand", 1.51,
                         0.0539, 0.0, "övrigt", kg_m2=27.1),
    "tamburdörr": _m("Tamburdörr utan glas, trä", 1.03, 0.0539, 0.0, "övrigt",
                     kg_m2=26.1),
    "ytterdörr_trä": _m("Ytterdörr utan glas, trä", 1.5, 0.0539, 0.0, "övrigt",
                        kg_m2=27.7),
    "ytterdörr_stål": _m("Ytterdörr, stål, massiv", 2.79, 0.0539, 0.0, "övrigt",
                         kg_m2=29.4),
}

# Boverket has no record for ceramics, vinyl, parquet, linoleum, porcelain or
# fittings. Their transport and waste per kg come from the nearest record, marked
# `analog` so the life cycle stages can say so. Reuse needs the weight only.
_ANALOG: dict[str, str] = {
    "klinker": "tegel",
    "trägolv": "hyvlat",
    "laminatgolv": "spånskiva",
    "linoleum": "plast",
    "plastmatta": "plast",
    "textilgolv": "plast",
    "korkgolv": "hyvlat",
    "radiator": "stål",
    "belysning": "aluminium",
    "sanitet": "tegel",
    "vitvaror": "plåt",
    "los_inredning": "spånskiva",
    "fast_inredning": "spånskiva",
    "vvs": "stål",
    "el": "plast",
}


def _analog(key: str) -> Material | None:
    name = _ANALOG.get(key)
    return replace(MATERIALS[name], analog=True) if name else None


@dataclass(frozen=True)
class Mass:
    """kg per component unit, or None and why."""

    kg: float | None
    how: str  # the arithmetic, or the Swedish reason there is no weight
    estimated: bool = False  # a typical value rather than one the name or Boverket gives
    material: Material | None = None  # for the per-kg figures; None when it has no record


def _sv(x: float) -> str:
    return f"{round(x, 3):g}".replace(".", ",")


def _thickness_mass(name: str, unit: str, material: Material, label: str,
                    typical_mm: float | None, typical_source: str = "") -> Mass:
    """Thickness times density, from the name when it says, else a typical one."""
    from aida.data.unit_conversion import thickness_mm

    density = material.density_kg_m3
    if density is None:
        return Mass(None, f"Boverket anger ingen densitet för {label}", False, material)
    t = thickness_mm(name)
    if t:
        kg = t / 1000 * density
        return Mass(kg, f"{_sv(t)} mm enligt namnet × {_sv(density)} kg/m³ (Boverket, "
                        f"{material.record}) = {_sv(kg)} kg/{unit}", False, material)
    if typical_mm is None:
        return Mass(None, f"vikten per {unit} kräver tjockleken i namnet", False, material)
    kg = typical_mm / 1000 * density
    why = f", {typical_source}" if typical_source else ""
    return Mass(kg, f"antagen tjocklek {_sv(typical_mm)} mm{why} × {_sv(density)} kg/m³ "
                    f"(Boverket, {material.record}) = {_sv(kg)} kg/{unit}", True, material)


def _has(text: str, *words: str) -> bool:
    return any(w in text for w in words)


def _floor_mass(text: str, name: str, unit: str) -> Mass:
    """A floor covering per m². Parquet and laminate by thickness, the rest typical."""
    from aida.data.unit_conversion import thickness_mm

    t = thickness_mm(name)
    # (thickness mm, density kg/m³): the density is unit_conversion's own table for
    # floor materials, and the thickness a common product. Estimated unless the
    # name states the thickness.
    if _has(text, "klinker", "keram", "tile"):
        mm, density, what = 8.0, 2200.0, "klinker"
    elif _has(text, "parkett", "trägolv", "massivgolv", "plank"):
        mm, density, what = 14.0, 600.0, "trägolv"
    elif "laminat" in text:
        mm, density, what = 8.0, 850.0, "laminatgolv"
    elif "linoleum" in text:
        mm, density, what = 2.5, 1200.0, "linoleum"
    elif _has(text, "vinyl", "plastmatta", "plastgolv", "pvc"):
        mm, density, what = 2.0, 1400.0, "plastmatta"
    elif _has(text, "textil", "heltäck", "matta"):
        mm, density, what = 6.0, 400.0, "textilgolv"
    elif "kork" in text:
        mm, density, what = 10.0, 200.0, "korkgolv"
    else:
        # 15 mm at 1400 kg/m³, unit_conversion's figures for a floor of unknown
        # material. No record for the per-kg figures: nothing says which it is.
        return Mass(21.0, f"golvmaterialet framgår inte, typvärde 21 kg/{unit}", True)
    material = _analog(what)
    kg = (t or mm) / 1000 * density
    if t:
        return Mass(kg, f"{what} {_sv(t)} mm enligt namnet × {_sv(density)} kg/m³ = "
                        f"{_sv(kg)} kg/{unit}", False, material)
    return Mass(kg, f"{what}, antagen tjocklek {_sv(mm)} mm × {_sv(density)} kg/m³ = "
                    f"{_sv(kg)} kg/{unit}", True, material)


def _window_material(text: str) -> Material:
    if _has(text, "pvc", "plastfönster"):
        return MATERIALS["fönster_pvc"]
    if _has(text, "aluminium", "alu"):
        if _has(text, "trä"):
            return MATERIALS["fönster_alu"]
        return MATERIALS["fönster_rena_alu"]
    return MATERIALS["fönster_trä"]


def _door_material(text: str) -> Material:
    if "tambur" in text or "lägenhetsdörr" in text:
        return MATERIALS["tamburdörr"]
    if "ytterdörr" in text or "entré" in text:
        if "stål" in text:
            return MATERIALS["ytterdörr_stål"]
        return MATERIALS["ytterdörr_trä"]
    if _has(text, "ljud", "brand"):
        return MATERIALS["innerdörr_ljud"]
    return MATERIALS["innerdörr"]


_MODULE_RE = re.compile(r"(?<![\d.,])(\d{1,2})\s*[x×*]\s*(\d{1,2})(?![\d.,])")


def _window_area(name: str) -> tuple[float, bool, str]:
    """(m² per window, estimated, label). A Swedish module size in the name counts:
    "12x14" is 1,2 x 1,4 m."""
    m = _MODULE_RE.search(name or "")
    if m:
        w, h = int(m.group(1)), int(m.group(2))
        if 4 <= w <= 30 and 4 <= h <= 30:
            area = w * h / 100
            return area, False, f"{w}x{h}M ({_sv(area)} m²) enligt namnet"
    return WINDOW_AREA_M2, True, f"antagen storlek {_sv(WINDOW_AREA_M2)} m² (1,2 x 1,2 m)"


def _windows_mass(text: str, name: str, unit: str) -> Mass:
    material = _window_material(text)
    if unit in _AREA_UNITS:
        return Mass(material.kg_m2, f"{_sv(material.kg_m2)} kg/m² (Boverket, "
                                    f"{material.record})", False, material)
    if unit not in _COUNT_UNITS:
        return Mass(None, f"ett fönster i {unit} har ingen känd vikt", False, material)
    area, estimated, label = _window_area(name)
    kg = area * material.kg_m2
    return Mass(kg, f"{label} × {_sv(material.kg_m2)} kg/m² (Boverket, {material.record}) "
                    f"= {_sv(kg)} kg/st", estimated, material)


def _doors_mass(text: str, unit: str) -> Mass:
    material = _door_material(text)
    if unit in _AREA_UNITS:
        return Mass(material.kg_m2, f"{_sv(material.kg_m2)} kg/m² (Boverket, "
                                    f"{material.record})", False, material)
    if unit not in _COUNT_UNITS:
        return Mass(None, f"en dörr i {unit} har ingen känd vikt", False, material)
    kg = DOOR_AREA_M2 * material.kg_m2
    return Mass(kg, f"antagen storlek {_sv(DOOR_AREA_M2)} m² (9x21M) × "
                    f"{_sv(material.kg_m2)} kg/m² (Boverket, {material.record}) = "
                    f"{_sv(kg)} kg/st", True, material)


def _roof_mass(text: str, name: str, unit: str) -> Mass:
    if _has(text, "tegel"):
        # About 40 kg/m²: a roofing tile's weight per m² is its supplier's, and
        # concrete and clay land close together. Estimated.
        return Mass(40.0, f"tegeltak, typvärde 40 kg/{unit}", True, MATERIALS["tegelpanna"])
    if _has(text, "betongpann", "takpann", "panna", "pannor"):
        return Mass(45.0, f"betongtakpannor, typvärde 45 kg/{unit}", True,
                    MATERIALS["betongpanna"])
    if _has(text, "plåt", "falsat"):
        kg = 0.0006 * 7850
        return Mass(kg, f"tunnplåt 0,6 mm × 7850 kg/m³ (Boverket) = {_sv(kg)} kg/{unit}",
                    True, MATERIALS["plåt"])
    if _has(text, "underlagspapp"):
        m = MATERIALS["underlagspapp"]
        return Mass(m.kg_m2, f"{_sv(m.kg_m2)} kg/m² (Boverket, {m.record})", False, m)
    if _has(text, "takpapp", "papp"):
        m = MATERIALS["takpapp"]
        return Mass(m.kg_m2, f"{_sv(m.kg_m2)} kg/m² (Boverket, {m.record})", False, m)
    if "shingel" in text or "shingle" in text:
        m = MATERIALS["takshingle"]
        return Mass(m.kg_m2, f"{_sv(m.kg_m2)} kg/m² (Boverket, {m.record})", False, m)
    if "takduk" in text or "duk" in text:
        m = MATERIALS["takduk"]
        return Mass(m.kg_m2, f"{_sv(m.kg_m2)} kg/m² (Boverket, {m.record})", False, m)
    return Mass(45.0, f"takmaterialet framgår inte, räknat som takpannor, typvärde 45 "
                      f"kg/{unit}", True)


def _insulation_mass(text: str, name: str, unit: str) -> Mass:
    if _has(text, "sten", "stone", "rock", "mineralull"):
        return _thickness_mass(name, unit, MATERIALS["stenull"], "stenull", 200.0,
                               "tilläggsisolering")
    if _has(text, "glas", "glass"):
        return _thickness_mass(name, unit, MATERIALS["glasull"], "glasull", 200.0,
                               "tilläggsisolering")
    if _has(text, "eps", "xps", "cellplast", "frigolit"):
        kg = 0.2 * 30.0
        return Mass(kg, f"cellplast, antagen tjocklek 200 mm × 30 kg/m³ = {_sv(kg)} kg/{unit}",
                    True, MATERIALS["eps"])
    # 200 mm at 30 kg/m³, unit_conversion's figures for insulation of unknown kind.
    return Mass(6.0, f"isoleringen framgår inte, typvärde 6 kg/{unit}", True)


# (kg per st, how the estimate is meant) for what is counted in st and has no
# Boverket weight. The numbers in unit_conversion.TYPICAL_ITEM_MASS_KG hold the
# items the baseline already bridges on; these are the same kind of approximation.
_ITEM_MASS: dict[str, tuple[float, str]] = {
    "diskmaskin": (45.0, "diskmaskin, typvikt"),
    "kylskåp": (65.0, "kylskåp, typvikt"),
    "frys": (65.0, "frysskåp, typvikt"),
    "tvättmaskin": (65.0, "tvättmaskin, typvikt"),
    "torktumlare": (38.0, "torktumlare, typvikt"),
    "spis": (50.0, "spis, typvikt"),
    "köksfläkt": (9.0, "köksfläkt, typvikt"),
    "toalett": (25.0, "toalettstol, typvikt"),
    "handfat": (18.0, "handfat, typvikt"),
    "tvättställ": (18.0, "tvättställ, typvikt"),
    "blandar": (2.5, "blandare, typvikt"),
    "badkar": (45.0, "badkar, typvikt"),
    "dusch": (30.0, "duschhörna, typvikt"),
    "urinal": (20.0, "urinal, typvikt"),
    "radiator": (30.0, "radiator, typvikt"),
    "armatur": (3.0, "armatur, typvikt"),
    "lampa": (3.0, "armatur, typvikt"),
    "spotlight": (0.5, "spotlight, typvikt"),
    "stol": (6.0, "stol, typvikt"),
    "skrivbord": (40.0, "skrivbord, typvikt"),
    "bänkskiva": (25.0, "bänkskiva, typvikt"),
    "bord": (25.0, "bord, typvikt"),
    "köksskåp": (20.0, "köksskåp, typvikt"),
    "skåp": (30.0, "skåp, typvikt"),
    "soffa": (50.0, "soffa, typvikt"),
    "hylla": (15.0, "hylla, typvikt"),
}

_CATEGORY_ITEM: dict[str, tuple[float, str]] = {
    "radiator": (30.0, "radiator, typvikt"),
    "belysning": (3.0, "armatur, typvikt"),
    "sanitet": (18.0, "sanitetsporslin, typvikt"),
    "vitvaror": (50.0, "vitvara, typvikt"),
    "los_inredning": (30.0, "möbel, typvikt"),
    "fast_inredning": (30.0, "fast inredning, typvikt"),
}


def _item_mass(category: str, text: str, unit: str) -> Mass:
    material = _analog(category)
    if unit not in _COUNT_UNITS:
        return Mass(None, f"{category or 'komponenten'} i {unit} har ingen känd vikt",
                    False, material)
    for key, (kg, how) in _ITEM_MASS.items():
        if key in text:
            return Mass(kg, f"{how} {_sv(kg)} kg/st", True, material)
    fallback = _CATEGORY_ITEM.get(category)
    if fallback:
        kg, how = fallback
        return Mass(kg, f"{how} {_sv(kg)} kg/st", True, material)
    return Mass(None, "vikten varierar för mycket mellan modeller för att ett typvärde "
                      "ska säga något. Ange mängden i kg", False, material)


_DIAMETER_RE = re.compile(r"(?:ø|Ø|dn|diameter\s*)\s*(\d{2,4})", re.IGNORECASE)


def _duct_mass(name: str, unit: str) -> Mass:
    if unit not in _LENGTH_UNITS:
        return Mass(None, f"ventilation i {unit} har ingen känd vikt")
    m = _DIAMETER_RE.search(name or "")
    diameter = float(m.group(1)) if m else 160.0
    material = MATERIALS["plåt"]
    # Spiral duct in 0,5 mm galvanized sheet: circumference x thickness x density.
    kg = 3.1416 * diameter / 1000 * 0.0005 * 7850
    label = f"Ø{_sv(diameter)} enligt namnet" if m else f"antagen Ø{_sv(diameter)}"
    return Mass(kg, f"spirokanal {label}, 0,5 mm plåt × 7850 kg/m³ (Boverket, "
                    f"{material.record}) = {_sv(kg)} kg/{unit}", True, material)


def _facade_mass(text: str, name: str, unit: str) -> Mass:
    if _has(text, "puts"):
        m = MATERIALS["puts"]
        return Mass(20.0, f"puts, typisk åtgång 20 kg/{unit} (12 mm)", True, m)
    if _has(text, "tegel"):
        return _thickness_mass(name, unit, MATERIALS["tegel"], "tegel", 120.0,
                               "fasadtegel")
    if _has(text, "fibercement"):
        return _thickness_mass(name, unit, MATERIALS["fibercement"], "fibercement", 8.0)
    if _has(text, "plåt"):
        kg = 0.0006 * 7850
        return Mass(kg, f"tunnplåt 0,6 mm × 7850 kg/m³ (Boverket) = {_sv(kg)} kg/{unit}",
                    True, MATERIALS["plåt"])
    if _has(text, "panel", "trä"):
        return _thickness_mass(name, unit, MATERIALS["hyvlat"], "träpanel", 22.0,
                               "fasadpanel")
    return Mass(None, "namnet säger inte vilket fasadmaterial det är, och vikten beror "
                      "på det")


def _paint_mass(text: str, unit: str) -> Mass:
    if unit not in _AREA_UNITS:
        return Mass(None, f"färg i {unit} har ingen känd vikt")
    from aida.data.unit_conversion import AREAL_DENSITY_KG_M2

    kg = AREAL_DENSITY_KG_M2["farg"][0][1]
    material = MATERIALS["slamfärg"] if _has(text, "slam", "rödfärg") else MATERIALS["färg"]
    return Mass(kg, f"två strykningar, {_sv(kg)} kg/{unit}", True, material)


def _wall_mass(text: str, name: str, unit: str, category: str) -> Mass:
    if _has(text, "tegel"):
        return _thickness_mass(name, unit, MATERIALS["tegel"], "tegel", 120.0)
    if _has(text, "lättklinker", "leca"):
        return _thickness_mass(name, unit, MATERIALS["lättklinker"], "lättklinker", 200.0)
    if _has(text, "betong"):
        return _thickness_mass(name, unit, MATERIALS["betongelement"], "betong", 200.0)
    if _has(text, "våtrum"):
        return _thickness_mass(name, unit, MATERIALS["gips_våtrum"], "våtrumsgips", 13.0,
                               "en skiva")
    if category == "innervägg" or _has(text, "gips"):
        # A partition counted per m² is two boards on a stud when the name does
        # not say otherwise: 13 mm on each side. The studs weigh about as little
        # as the numbers here are rough, and are left out.
        material = MATERIALS["gips"]
        from aida.data.unit_conversion import thickness_mm

        t = thickness_mm(name)
        if t:
            return _thickness_mass(name, unit, material, "gips", 13.0)
        kg = 2 * 0.013 * material.density_kg_m3
        return Mass(kg, f"antagen vägg med en 13 mm skiva på varje sida × "
                        f"{_sv(material.density_kg_m3)} kg/m³ (Boverket, {material.record}) "
                        f"= {_sv(kg)} kg/{unit}", True, material)
    return Mass(None, "namnet säger inte vilket väggmaterial det är, och vikten beror "
                      "på det")


def _stomme_mass(name: str, unit: str) -> Mass:
    from aida.data.palats_client import component_subcategory
    from aida.data.stomme_reuse import _mass_per_unit

    family = component_subcategory(name, "stomme")
    kg, res, how = _mass_per_unit(family, name, unit)
    if kg is None or res is None:
        return Mass(None, how)
    # Keep the frame's own record for the per-kg figures. Its waste is read from
    # Boverket and its demolition from IVL's group for the family.
    if family in ("konstruktionsstål", "stålregel"):
        c34, group = _C34["stål"]
    elif family == "betong":
        c34, group = _C34["betong"]
    elif family == "konstruktionsskiva":
        c34, group = _C34["träskivor"]
    else:
        c34, group = _C34["trä"]
    material = Material(res.name, res.a1a3_per_kg, res.a4_per_kg, STOMME_A5.get(res.name),
                        c34, group, res.density_kg_m3)
    return Mass(kg, how, False, material)


# A5.1 for the frame records stomme_reuse names, Boverket 02.07.000.
STOMME_A5: dict[str, float] = {
    "Sågat virke, u 16 %, barrträ": 0.0103,
    "Hyvlat virke, u 16 %, barrträ": 0.0114,
    "Limträ, u 12 %, gran": 0.00888,
    "Korslimmat trä, u 12 %, barrträ": 0.00825,
    "Fanerträ (LVL)": 0.0241,
    "Lättbalk av trä": 0.0244,
    "Plywood": 0.0501,
    "OSB": 0.0528,
    "Spånskiva": 0.055,
    "Konstruktionsstål, alla sorter, 80 % primär råvara": 0.162,
    "Lättreglar av stål, primär": 0.061,
    "Hålbjälklag, HD/F": 0.0,
    "Massivplattor, RD, RD/F": 0.0,
    "Balkar B, slakarmerad": 0.0,
}


def mass_per_unit(category: str, name: str, unit: str) -> Mass:
    """What one component unit weighs, and the material it stands for.

    `name` is the component's or listing's name, with the baseline's assumed
    material appended when the caller has it: both can name the material.
    """
    unit = (unit or "").strip().lower()
    text = (name or "").lower()
    category = (category or "").strip()

    if unit == "kg":
        probe = mass_per_unit(category, name, "st" if category in ("fönster", "dörr") else "m2")
        return Mass(1.0, "mängden är redan angiven i kg", False, probe.material)
    if category == "stomme":
        return _stomme_mass(name, unit)
    if category == "fönster":
        return _windows_mass(text, name, unit)
    if category == "dörr":
        return _doors_mass(text, unit)
    if category == "ventilation":
        return _duct_mass(name, unit)
    if category in ("hiss", "kylanläggning", "storköksutrustning"):
        return Mass(None, "vikten varierar för mycket mellan anläggningar för att ett "
                          "typvärde ska säga något. Ange mängden i kg")
    if category in ("radiator", "belysning", "sanitet", "vitvaror", "los_inredning",
                    "fast_inredning", "vvs", "el"):
        return _item_mass(category, text, unit)
    if unit not in _AREA_UNITS:
        return Mass(None, f"{category or 'komponenten'} i {unit} har ingen känd vikt")
    if category == "golv":
        return _floor_mass(text, name, unit)
    if category == "kakel":
        return _floor_mass(text + " klinker", name, unit)
    if category == "tak":
        return _roof_mass(text, name, unit)
    if category == "isolering":
        return _insulation_mass(text, name, unit)
    if category == "fasadskikt":
        return _facade_mass(text, name, unit)
    if category == "farg":
        return _paint_mass(text, unit)
    if category == "undertak":
        if _has(text, "gips"):
            return _thickness_mass(name, unit, MATERIALS["gips"], "gips", 13.0)
        return Mass(4.0, f"undertaksplatta, typvikt 4 kg/{unit}", True, MATERIALS["stenull"])
    if category == "tätskikt":
        return Mass(1.5, f"tätskikt, typvikt 1,5 kg/{unit}", True, MATERIALS["plast"])
    if category == "betongvägg":
        return _thickness_mass(name, unit, MATERIALS["betongelement"], "betong", 200.0)
    if category in ("innervägg", "yttervägg"):
        return _wall_mass(text, name, unit, category)
    return Mass(None, "namnet och kategorin säger inte vad komponenten väger")


@dataclass(frozen=True)
class ReuseFigure:
    """kg CO2e per component unit for a reused product, or None and why."""

    per_unit: float | None
    note: str
    estimated: bool = False


def _figure_from_mass(mass: Mass, unit: str) -> ReuseFigure:
    if mass.kg is None:
        return ReuseFigure(None, mass.how[:1].upper() + mass.how[1:])
    per_unit = round(mass.kg * REUSE_A4_PER_KG, 4)
    marked = ", vikten är en uppskattning" if mass.estimated else ""
    return ReuseFigure(per_unit, (
        f"Klimatvärdet för återbruket är transporten: {mass.how} × "
        f"{_sv(REUSE_A4_PER_KG * 1000)} g CO2e/kg (Boverkets klimatdatabas "
        f"{BOVERKET_VERSION}, {REUSE_RECORD}, tillverkning 0 och 40 km transport) = "
        f"{_sv(per_unit)} kg CO2e/{unit}{marked}. Nedmontering, reparation och "
        f"upprustning ingår inte"), mass.estimated)


_NOT_REUSED = ("Tätskikt läggs inte om: ett begagnat tätskikt har ingen klimatsiffra "
               "här, för det går inte att räkna det som samma produkt igen")


def reuse_figure(category: str, name: str, unit: str) -> ReuseFigure:
    """The climate figure per unit for a reused product (DECISIONS 18).

    Weight times Boverket's own record for a reused product, for every category,
    the frame included. Nothing in it for taking down, repairing or upgrading:
    there is no source, and Boverket says upgrading comes on top.
    """
    if category == "tätskikt":
        return ReuseFigure(None, _NOT_REUSED)
    if category == "stomme":
        from aida.data import stomme_reuse

        figure = stomme_reuse.reuse_figure(name, unit)
        return ReuseFigure(figure.per_unit, figure.note)
    unit_n = (unit or "").strip().lower()
    return _figure_from_mass(mass_per_unit(category, name, unit_n), unit_n)


def listing_reuse_figure(category: str, component_name: str, title: str,
                         unit: str) -> ReuseFigure:
    """The same for one listing: its own name first, then with the component's.

    A listing "Ekparkett 14 mm" on a component "Golv klinker" weighs what the
    listing says. A listing "Fina dörrar" on "Innerdörr" needs the component's.
    """
    if category == "tätskikt":
        return ReuseFigure(None, _NOT_REUSED)
    if category == "stomme":
        from aida.data import stomme_reuse

        figure = stomme_reuse.listing_reuse_figure(component_name, title, unit)
        return ReuseFigure(figure.per_unit, figure.note)
    unit_n = (unit or "").strip().lower()
    own = mass_per_unit(category, title or "", unit_n)
    if own.kg is not None and not own.estimated:
        return _figure_from_mass(own, unit_n)
    return _figure_from_mass(
        mass_per_unit(category, f"{title or ''} {component_name or ''}".strip(), unit_n),
        unit_n)
