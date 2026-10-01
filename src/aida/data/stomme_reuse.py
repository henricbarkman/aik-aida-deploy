"""The climate figure for a reused frame member: its transport (HENRIC-3364).

Every other reuse category has a set figure per unit in
palats_client.REUSE_CO2E_PER_UNIT. Stomme had none and took the 2 kg CO2e
default, which for a stud counted in löpmeter is far more than a new one
(45x95 sawn timber is 0,12 kg CO2e per metre in A1-A3), so a used stud would
have looked worse than a new one.

A reused member is not manufactured again. What it costs is the transport to
the site, and that is derived here rather than set:

    kg per unit of the member  x  kg CO2e per kg for transport (A4)

- The weight per unit is the one the baseline and the alternatives use for
  the same name: the section or thickness in the name times Boverket's
  density for timber and boards (unit_conversion.member_volume_per_unit), the
  standard weight per metre of a named steel profile (steel_profiles), and for
  precast concrete 2 500 kg/m3 times the thickness or section (betongstomme),
  or a weight per m2 the name states for a hollow-core slab, never one derived
  from its thickness.
- The transport is Boverket's generic A4 value for the same material: the
  climate database's typical transport for a new product from the factory via
  a warehouse to the building site, per kg. The distance for a given listing
  is unknown, and AIda does not invent one; Boverket's scenario is the sourced
  figure there is. A listing from the region likely travels less, so the
  figure leans high, never towards a saving that is not there.

Dismantling and refurbishing are not included: there is no source for them.
When the name does not give the weight (no section, no profile, a count in
st, a hollow-core slab without its weight per m2), there is no figure, and
the caller says so instead of showing the default.

Source, read 2026-10-01: Boverkets klimatdatabas, version 02.07.000
(api.boverket.se/klimatdatabas, GetAllResources/senaste), field DataItems,
module "A4" and "A1-A3 Typical", and Conversions (kg/m3).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

BOVERKET_VERSION = "02.07.000"


@dataclass(frozen=True)
class Resource:
    """One Boverket record: its name, A4 and A1-A3 per kg, density."""

    name: str
    a4_per_kg: float
    a1a3_per_kg: float  # "A1-A3 Typical"; kept to show reuse is below new
    density_kg_m3: float | None  # None where Boverket gives no conversion


# Boverket records, version 02.07.000, read 2026-10-01. Only what the record
# states; a record without a kg/m3 conversion has density None.
RESOURCES: dict[str, Resource] = {
    "sågat": Resource("Sågat virke, u 16 %, barrträ", 0.0225, 0.064, 455.0),
    "hyvlat": Resource("Hyvlat virke, u 16 %, barrträ", 0.0225, 0.0735, 455.0),
    "limträ": Resource("Limträ, u 12 %, gran", 0.045, 0.106, 434.0),
    "kl": Resource("Korslimmat trä, u 12 %, barrträ", 0.045, 0.096, 465.0),
    "lvl": Resource("Fanerträ (LVL)", 0.0989, 0.306, 510.0),
    "lättbalk": Resource("Lättbalk av trä", 0.0809, 0.325, None),
    "plywood": Resource("Plywood", 0.0539, 0.358, 460.0),
    "osb": Resource("OSB", 0.0809, 0.358, 607.0),
    "spånskiva": Resource("Spånskiva", 0.0629, 0.39, 700.0),
    "stål": Resource("Konstruktionsstål, alla sorter, 80 % primär råvara",
                     0.0989, 2.52, 7850.0),
    "stålregel": Resource("Lättreglar av stål, primär", 0.036, 2.41, 7850.0),
    "håldäck": Resource("Hålbjälklag, HD/F", 0.0539, 0.15, None),
    "massiv": Resource("Massivplattor, RD, RD/F", 0.0539, 0.183, None),
    "balk": Resource("Balkar B, slakarmerad", 0.0539, 0.198, None),
}

_LENGTH = {"lm", "m", "meter", "löpmeter"}
_AREA = {"m2", "m²", "kvm"}

_FAMILY_LABELS = {
    "virke": "virke", "limträ": "limträ, KL-trä eller fanerträ",
    "konstruktionsskiva": "konstruktionsskivor", "konstruktionsstål":
    "konstruktionsstål", "stålregel": "stålreglar", "betong": "betongelement",
}


@dataclass(frozen=True)
class ReuseFigure:
    """kg CO2e per component unit for a reused member, or None and why."""

    per_unit: float | None
    note: str  # the arithmetic with its sources, or the Swedish reason


def family_label(family: str) -> str:
    return _FAMILY_LABELS.get(family, family or "stomme")


def _sv(x: float) -> str:
    return f"{x:g}".replace(".", ",")


def _timber_resource(family: str, text: str) -> Resource | None:
    if family == "virke":
        return RESOURCES["sågat"]
    if family == "limträ":
        if re.search(r"lättbalk|masonitebalk|i-balk", text):
            return RESOURCES["lättbalk"]
        if re.search(r"lvl|kerto|fanerträ", text):
            return RESOURCES["lvl"]
        if re.search(r"kl-trä|klträ|korslimmat|massivträ", text):
            return RESOURCES["kl"]
        return RESOURCES["limträ"]
    if family == "konstruktionsskiva":
        if "osb" in text:
            return RESOURCES["osb"]
        if re.search(r"plywood|kryssfan", text):
            return RESOURCES["plywood"]
        if "spånskiv" in text:
            return RESOURCES["spånskiva"]
        if "råspont" in text:
            return RESOURCES["hyvlat"]
    return None


def _mass_per_unit(family: str, name: str, unit: str
                   ) -> tuple[float | None, Resource | None, str]:
    """(kg per component unit, Boverket record, how) or (None, record, why)."""
    text = (name or "").lower()
    if family in ("virke", "limträ", "konstruktionsskiva"):
        from aida.data.unit_conversion import member_volume_per_unit

        res = _timber_resource(family, text)
        if res is None:
            return None, None, ("namnet säger inte vilken sorts skiva det är (till "
                                "exempel OSB, plywood eller spånskiva), och vikten "
                                "beror på det")
        if unit == "kg":
            return 1.0, res, "1 kg"
        if res.density_kg_m3 is None:
            return None, res, (f"Boverket anger ingen densitet för {res.name.lower()}, "
                               f"så vikten per {unit} är okänd. Ange mängden i kg")
        if unit == "m3":
            return res.density_kg_m3, res, f"{_sv(res.density_kg_m3)} kg/m³ (Boverket)"
        geometry = member_volume_per_unit(name, unit)
        if not geometry:
            what = {"lm": "tvärsnittet (till exempel 45x95)",
                    "m2": "tjockleken (till exempel 12 mm)"}.get(
                "lm" if unit in _LENGTH else "m2" if unit in _AREA else "",
                "en längd eller en yta")
            return None, res, f"vikten per {unit} kräver {what} i namnet"
        factor, label = geometry
        kg = factor * res.density_kg_m3
        return kg, res, (f"{label} × {_sv(res.density_kg_m3)} kg/m³ (Boverket) = "
                         f"{_sv(round(kg, 3))} kg/{unit}")

    if family in ("konstruktionsstål", "stålregel"):
        from aida.data.steel_profiles import STUD, profile_mass

        res = RESOURCES["stålregel" if family == "stålregel" else "stål"]
        if unit == "kg":
            return 1.0, res, "1 kg"
        if unit not in _LENGTH:
            return None, res, (f"stål räknas i löpmeter med profilen i namnet eller i "
                               f"kg, inte i {unit}")
        profile, why = profile_mass(name)
        if not profile:
            return None, res, (why.rstrip(".") if why else
                               "namnet anger ingen standardprofil, så vikten per meter "
                               "är okänd")
        res = RESOURCES["stålregel" if profile.family == STUD else "stål"]
        return profile.kg_per_m, res, profile.label

    if family == "betong":
        from aida.data import betongstomme as bs

        if bs.is_reinforcement(name) or bs.is_lightweight(name) or bs.is_cast_in_place(name):
            return None, None, ("transportvärdet räknas bara för prefabricerade "
                                "betongelement (håldäck, massiva bjälklag, balkar och "
                                "pelare), inte för platsgjuten betong, armering eller "
                                "lättklinker")
        kind = bs.element_kind(name)
        if not kind:
            return None, None, ("namnet säger inte vilket sorts betongelement det är "
                                "(håldäck, massivt bjälklag eller balk)")
        res = RESOURCES[{bs.HÅLDÄCK: "håldäck", bs.MASSIV: "massiv",
                         bs.BALK: "balk"}[kind]]
        if unit == "kg":
            return 1.0, res, "1 kg"
        density = f"{_sv(bs.SOLID_DENSITY_KG_M3)} kg/m³ ({bs.SOLID_DENSITY_SOURCE})"
        if kind == bs.HÅLDÄCK:
            stated = bs.stated_kg_per_m2(name)
            if unit in _AREA and stated:
                return stated, res, f"vikt {_sv(stated)} kg/m² enligt namnet"
            return None, res, (f"ett håldäcks vikt per m² följer inte av tjockleken: "
                               f"{bs.hdf_weight_text((bs.slab_thickness_mm(name) or (200,))[0])}. "
                               f"Ange leverantörens vikt i namnet, till exempel "
                               f"\"Håldäck 200 mm, 290 kg/m2\", eller mängden i kg")
        if unit == "m3":
            return bs.SOLID_DENSITY_KG_M3, res, density
        if kind == bs.MASSIV and unit in _AREA:
            t = bs.slab_thickness_mm(name)
            if not t:
                return None, res, "vikten per m² kräver bjälklagets tjocklek i namnet"
            kg = t[0] / 1000 * bs.SOLID_DENSITY_KG_M3
            return kg, res, f"{t[1]} × {density} = {_sv(kg)} kg/m²"
        if kind == bs.BALK and unit in _LENGTH:
            section = bs.beam_section_mm(name)
            if not section:
                return None, res, "vikten per meter kräver balkens tvärsnitt i namnet"
            w, h = section
            kg = w / 1000 * h / 1000 * bs.SOLID_DENSITY_KG_M3
            return kg, res, f"tvärsnitt {w:g}×{h:g} mm × {density} = {_sv(kg)} kg/lm"
        return None, res, f"ett {bs.KIND_LABELS[kind]}-element i {unit} har ingen känd vikt"

    return None, None, ("namnet säger inte vilket stommaterial det är, och vikten "
                        "beror på det")


def reuse_figure(component_name: str, unit: str) -> ReuseFigure:
    """The transport figure per unit for a reused frame member, or why not."""
    from aida.data.palats_client import component_subcategory

    unit = (unit or "").strip().lower()
    family = component_subcategory(component_name, "stomme")
    if unit in ("st", "styck", "pcs", "stk"):
        return ReuseFigure(None, (
            "Komponenten är angiven i styck, som inte säger hur mycket material "
            "det är. Ange dimension och antal löpmeter, m² eller kg"))
    kg, res, how = _mass_per_unit(family, component_name, unit)
    if kg is None or res is None:
        return ReuseFigure(None, how[:1].upper() + how[1:])
    per_unit = round(kg * res.a4_per_kg, 4)
    return ReuseFigure(per_unit, (
        f"Klimatvärdet för återbruket är transporten till bygget: "
        f"{how} × {_sv(res.a4_per_kg)} kg CO2e/kg (Boverkets klimatdatabas "
        f"{BOVERKET_VERSION}, {res.name}, modul A4: typisk transport från fabrik "
        f"via lager till byggplatsen) = {_sv(per_unit)} kg CO2e/{unit}. Sträckan "
        f"för just den här annonsen är okänd, och nedmontering och upprustning "
        f"ingår inte"))
