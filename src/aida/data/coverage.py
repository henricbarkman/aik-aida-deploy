"""How much product a square metre takes, from each product's own datasheet.

Levelling compounds and liquid waterproofing are declared per kg (85 and 31
catalog rows), but a förvaltare renovating a bathroom thinks in m². Until
HENRIC-3371 a component in m² got "Ange mängden i kg" and nothing to compare.
The bridge is the product's application rate as its maker's technical
datasheet states it, read by hand and kept here with the URL and the sentence
it came from, the same way aggregat.EPD_FACTS keeps what the EPD PDFs say.

  - Levelling: kg per m² and mm of layer, times the thickness the component's
    name states ("Avjämning 10 mm"). Without a thickness there is no figure.
  - Waterproofing: kg per m² as the datasheet gives it for the wet-room
    build-up, floor and wall apart where the sheet sets them apart (FB 7: 1,6
    and 1,0). Where a sheet gives a rate per mm and a minimum total thickness
    (Mapelastic: 1,7 kg/m²/mm, at least 2 mm), the product of the two. Where it
    gives a range (FB 3: 2,2-2,5 kg/m² at two coats), the upper value, said so.

A row with no datasheet here is left out of the m² comparison, never given a
category-wide rate: a paste filler at 1,3 kg/m²/mm and a heavy screed at 1,9
are both "avjämning", and borrowing one's rate for the other would move the
ranking by a third. Keys are the registration number without its version
suffix (row_key). Read 2026-10-01; Weber's own site refuses scripts, so its
Swedish datasheets are the copies Beijer and Derome host.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PerMm:
    """A levelling compound: kg per m² and mm of layer."""
    kg_per_m2_mm: float
    product: str
    url: str
    quote: str


@dataclass(frozen=True)
class PerM2:
    """A waterproofing membrane: kg per m² on a floor and on a wall."""
    floor_kg_per_m2: float
    wall_kg_per_m2: float
    product: str
    url: str
    quote: str


LEVELLING = {
    "NEPD-4752-4007-EN": PerMm(1.7, "weberfloor 110 fine",
        "https://media-prod.beijerflow.com/media/medias/docus/118/$v2/PDS-SE-weberfloor_110_fine.pdf",
        "Åtgångstal 1,7 kg/m²/mm (enligt GBR mätmetod)"),
    "NEPD-4897-4245-EN": PerMm(1.75, "weberfloor 120 reno DR",
        "https://www.beijerbygg.se/wcsstore/BeijerCAS/HPMAssets/d120001/medias/docus/182/006238998_11585089_weberfloor_120_Reno_DR.pdf",
        "Åtgångstal 1,75 kg/m²/mm (enligt GBR mätmetod)"),
    "NEPD-4759-4015-EN": PerMm(1.85, "weberfloor 130 core",
        "https://media-prod.beijerflow.com/media/medias/docus/118/$v2/PDS-SE-weberfloor_130_core.pdf",
        "Åtgångstal 1,85 kg/m²/mm (enligt GBR mätmetod)"),
    "NEPD-4760-4014-EN": PerMm(1.7, "weberfloor 140 nova",
        "https://media-prod.beijerflow.com/media/medias/docus/118/$v2/PDS-SE-weberfloor_140_nova.pdf",
        "Åtgångstal 1,7 kg/m²/mm (enligt GBR mätmetod)"),
    "EPD-IES-0031754": PerMm(1.6, "weberfloor 4032 super flow rapid DR",
        "https://media.derome.se/medias/docus/65/$v3/PDS-SE-weberfloor-4032-super-flow-rapid-DR.pdf",
        "Åtgångstal 1,6 kg/m2/mm (enligt GBR mätmetod)"),
    "EPD-IES-0025048": PerMm(1.55, "weberfloor 4040 combi rapid DR",
        "https://media-prod.beijerflow.com/media/medias/docus/206/$v13/PDS-SE-weberfloor_4040_combi_rapid_DR.pdf",
        "Åtgångstal 1,55 kg/m2/mm (enligt GBR mätmetod)"),
    "NEPD-4027-3063-EN": PerMm(1.3, "weberfloor 4042 paste fine DR",
        "https://media.derome.se/medias/docus/65/$v3/PDS-SE-weberfloor-4042-paste-fine-DR.pdf",
        "Åtgångstal 1,3 kg/m2/mm (enligt GBR mätmetod)"),
    "NEPD-4052-3086-EN": PerMm(1.7, "weberfloor 4160 fine flow rapid",
        "https://media-prod.beijerflow.com/media/medias/docus/118/$v2/PDS-SE-weberfloor_4160_fine_flow_rapid.pdf",
        "Åtgångstal 1,7 kg/m2/mm (enligt GBR mätmetod)"),
    "NEPD-4051-3082-EN": PerMm(1.7, "weberfloor 4600 industry base",
        "https://media-prod.beijerflow.com/media/medias/docus/118/PDS-SE-weberfloor_4600_industry_base.pdf",
        "Åtgångstal 1,7 kg/m2/mm (enligt GBR mätmetod)"),
    "EPD-IES-0025049": PerMm(1.7, "weberfloor 4655 industry flow rapid",
        "https://media-prod.beijerflow.com/media/medias/docus/118/PDS-SE-weberfloor_4655_industry_flow_rapid.pdf",
        "Åtgångstal 1,7 kg/m2/mm (enligt GBR mätmetod)"),
    "NEPD-4034-3068-EN": PerMm(1.75, "weberfloor 644 värmegolvspackel DR",
        "https://media.derome.se/medias/docus/65/$v3/PDS-SE-weberfloor-644-varmegolvspackel-DR.pdf",
        "Åtgångstal 1,75 kg/m2/mm (enligt GBR mätmetod)"),
    "HUB-0430": PerMm(1.6, "Kiilto Tasoflex (self-levelling compound)",
        "https://pim.kiilto.com/kiilto-pim-api/api/pdf/download/17cb3604-cca4-41fb-b776-60122b57d443?name=T2017_TDS_Kiilto_Tasoflex_US.pdf",
        "coverage 1.6 kg/m²/mm"),
    "HUB-0228": PerMm(1.9, "Kiilto 60 Plus and Kiilto 60 (floor screed)",
        "https://pim.kiilto.com/kiilto-pim-api/api/pdf/download/2f5d1a5b-d321-46bd-aa41-f629de81f4d6?name=T2023_TDS_Kiilto_60_Plus_US.pdf",
        "coverage 1.9 kg/m²/mm (Kiilto 60 Plus och Kiilto 60 lika)"),
    "HUB-0229": PerMm(1.9, "Kiilto 70 (floor screed)",
        "https://pim.kiilto.com/kiilto-pim-api/api/pdf/download/17fb5844-cac2-46da-8adc-bcdc0228490c?name=T2327_TDS_Kiilto_70_US.pdf",
        "coverage 1.9 kg/m²/mm"),
    "HUB-1106": PerMm(1.7, "Kiilto 80 (levelling screed)",
        "https://pim.kiilto.com/kiilto-pim-api/api/pdf/download/e4b3f10c-9c78-4ccd-951f-1ab83ebd616a?name=T2118_TDS_Kiilto_80_US.pdf",
        "1 mm layer/m² solid matter weighs approx. 1.7 kg"),
    "HUB-0429": PerMm(1.7, "Kiilto Floor Heat DF (levelling screed)",
        "https://pim.kiilto.com/kiilto-pim-api/api/pdf/download/db66d05b-d59a-46d3-ba53-010e2122bc49?name=T2001_TDS_Kiilto_Floor_Heat_DF_US.pdf",
        "coverage 1.7 kg/m²/mm"),
    "HUB-1103": PerMm(1.6, "Kiilto HardPlan (smoothing compound)",
        "https://pim.kiilto.com/kiilto-pim-api/api/pdf/download/c79c55f8-2369-45ce-9ce4-b9c93c79ef91?name=T2183_TDS_Kiilto_HardPlan_US.pdf",
        "coverage 1.6 kg/m²/mm"),
    "HUB-6250": PerMm(1.7, "Kiilto Pro Plan Fiber (floor screed)",
        "https://pim.kiilto.com/kiilto-pim-api/api/pdf/download/c94a37a3-0b21-4fd7-bf5a-141e21a63559?name=T5507_TDS_Kiilto_Pro_Plan_Fiber_US.pdf",
        "COVERAGE approx. 1,7 kg powder/m²/mm"),
    "HUB-0461": PerMm(1.6, "Kiilto TopPlan DF (smoothing compound)",
        "https://pim.kiilto.com/kiilto-pim-api/api/pdf/download/03121f1f-814d-4c53-ac47-9a4b043a7333?name=T2292_TDS_Kiilto_TopPlan_DF_US.pdf",
        "coverage 1.6 kg/m²/mm"),
    "NEPD-14935-15704": PerMm(1.7, "FB 6000 Golvspackel",
        "https://vpp.bastaonline.se/Documents/90900/PB/Produktdatablad%20FB%206000.pdf",
        "1,7 kg/m [m², stavfel i bladet] och millimeter skikttjocklek"),
    "NEPD-14920-15667": PerMm(1.9, "FB 5000 Golvspackel",
        "https://vpp.bastaonline.se/Documents/90903/PB/Produktdatablad%20FB%205000.pdf",
        "1,9 kg/m2 och millimeter skikttjocklek"),
    "NEPD-14937-15702": PerMm(1.7, "FB 6500 Golvspackel",
        "https://www.hoganaskakel.se/app/uploads/2023/07/Produktdatablad-FB6500-2023.pdf",
        "1,7kg/m2 och millimeter skikttjocklek"),
    "EPD-IES-0016479": PerMm(1.7, "Ultraplan Trade",
        "https://cdnmedia.mapei.com/docs/librariesprovider2/products-documents/1_04079_ultraplan-trade_it-it_c7479b29ee0f47ef8c788298a3bc7c96.pdf?sfvrsn=afe5e5cf_0",
        "Il consumo di Ultraplan Trade è di 1,7 kg/m² per mm di spessore."),
    "EPD-IES-0023162": PerMm(1.6, "Ultraplan Eco 20 (Greek production)",
        "https://cdnmedia.mapei.com/docs/librariesprovider38/products-documents/1_4005_ultraplan_eco_20_gb_b293b9d349c74d12b8e8247b12e561e5.pdf?sfvrsn=74e549d9_0",
        "CONSUMPTION 1.6 kg/m² per mm of thickness."),
    "NEPD-5226-4506-EN": PerMm(1.7, "Robust Avretting Inne 5-60",
        "https://www.norebo.no/wp-content/uploads/2026/01/Produktdatablad-5-60mm.pdf",
        "Forbruk: ca. 1,7 kg/mm/m2"),
    "NEPD-9378-8972": PerMm(1.6, "Robust Tynnavretting inne 0-30",
        "https://www.norebo.no/wp-content/uploads/2026/01/Produktdatablad-Tynnavretting-Inne-0-30mm.pdf",
        "Forbruk: ca. 1,6 kg/mm/m2"),
    "EPD-IES-0013002": PerMm(1.8, "BOSTIK SL C600 EVOLUTION",
        "https://www.bostik.com/files/live/sites/shared_bostik/files/documents-brochures/Norway/Documents/TDS/bostik-no-tds-sl-c600-evolution.pdf",
        "Materialforbruk 1.8 kg/mm/m²"),
}
MEMBRANE = {
    "NEPD-14880-15604": PerM2(1.6, 1.0, "FB 7 Tätskikt",
        "https://vpp.bastaonline.se/Documents/90934/PB/Produktdatablad%20FB%207.pdf",
        "Skall vara min. 1,6 kg/m2 för golv och min. 1,0 kg/m2 för vägg."),
    "NEPD-14885-15623": PerM2(2.5, 2.5, "FB 2K Flex Tätskikt",
        "https://www.xlbygg.se/media/page_attachments/Produktblad-FB-2K-Flex.pdf",
        "Appliceras 2-3 skikt till total 2,5 kg/m²"),
    "NEPD-15005-15843": PerM2(2.5, 2.5, "FB 3 Tätskikt",
        "https://www.hoganaskakel.se/app/uploads/2023/07/Produktdatablad-FB3-23.pdf",
        "Vid två rollningar 2,2-2,5 kg/m2 (övre värdet)"),
    "EPD-IES-0011371": PerM2(3.4, 3.4, "Mapelastic (Nordic & Baltic market)",
        "https://www.betomur.no/file/filer-fra-uni-okonomi/pdb-mapelastic-no.pdf",
        "Forbruk manuell påføring ca. 1,7 kg/m²/mm; to strøk til total tykkelse ca. 2 mm (1,7 × 2)"),
    "EPD-IES-0017280": PerM2(3.4, 3.4, "Mapelastic Zero",
        "https://cdnmedia.mapei.com/docs/librariesprovider56/products-documents/1_mapelastic-zero_en_tds_0286f8249274435296eef6ebd9c50b83.pdf?sfvrsn=97df2a04_0",
        "CONSUMPTION approx. 1.7 kg/m² per mm of thickness (manual); second coat to form a final thickness at least 2 mm (1,7 × 2)"),
    "EPD-IES-0009967": PerM2(3.4, 3.4, "Mapelastic Zero",
        "https://cdnmedia.mapei.com/docs/librariesprovider56/products-documents/1_mapelastic-zero_en_tds_0286f8249274435296eef6ebd9c50b83.pdf?sfvrsn=97df2a04_0",
        "CONSUMPTION approx. 1.7 kg/m² per mm of thickness (manual); second coat to form a final thickness at least 2 mm (1,7 × 2)"),
    "EPD-IES-0016398": PerM2(3.4, 3.4, "Mapelastic Zero",
        "https://cdnmedia.mapei.com/docs/librariesprovider56/products-documents/1_mapelastic-zero_en_tds_0286f8249274435296eef6ebd9c50b83.pdf?sfvrsn=97df2a04_0",
        "CONSUMPTION approx. 1.7 kg/m² per mm of thickness (manual); second coat to form a final thickness at least 2 mm (1,7 × 2)"),
    "EPD-IES-0015848": PerM2(3.4, 3.4, "Mapelastic Zero (China Production)",
        "https://cdnmedia.mapei.com/docs/librariesprovider56/products-documents/1_mapelastic-zero_en_tds_0286f8249274435296eef6ebd9c50b83.pdf?sfvrsn=97df2a04_0",
        "CONSUMPTION approx. 1.7 kg/m² per mm of thickness (manual); second coat to form a final thickness at least 2 mm (1,7 × 2)"),
    "EPD-IES-0016402": PerM2(3.4, 3.4, "Mapelastic Zero (Malaysia Production)",
        "https://cdnmedia.mapei.com/docs/librariesprovider56/products-documents/1_mapelastic-zero_en_tds_0286f8249274435296eef6ebd9c50b83.pdf?sfvrsn=97df2a04_0",
        "CONSUMPTION approx. 1.7 kg/m² per mm of thickness (manual); second coat to form a final thickness at least 2 mm (1,7 × 2)"),
    "EPD-IES-0013868": PerM2(3.4, 3.4, "Mapelastic Zero (PL)",
        "https://cdnmedia.mapei.com/docs/librariesprovider56/products-documents/1_mapelastic-zero_en_tds_0286f8249274435296eef6ebd9c50b83.pdf?sfvrsn=97df2a04_0",
        "CONSUMPTION approx. 1.7 kg/m² per mm of thickness (manual); second coat to form a final thickness at least 2 mm (1,7 × 2)"),
    "EPD-IES-0013885": PerM2(3.4, 3.4, "Mapelastic Zero (PT)",
        "https://cdnmedia.mapei.com/docs/librariesprovider56/products-documents/1_mapelastic-zero_en_tds_0286f8249274435296eef6ebd9c50b83.pdf?sfvrsn=97df2a04_0",
        "CONSUMPTION approx. 1.7 kg/m² per mm of thickness (manual); second coat to form a final thickness at least 2 mm (1,7 × 2)"),
    "EPD-IES-0016400": PerM2(3.4, 3.4, "Mapelastic Zero (Singapore production)",
        "https://cdnmedia.mapei.com/docs/librariesprovider56/products-documents/1_mapelastic-zero_en_tds_0286f8249274435296eef6ebd9c50b83.pdf?sfvrsn=97df2a04_0",
        "CONSUMPTION approx. 1.7 kg/m² per mm of thickness (manual); second coat to form a final thickness at least 2 mm (1,7 × 2)"),
    "EPD-IES-0016399": PerM2(3.4, 3.4, "Mapelastic Zero (Vietnam Production)",
        "https://cdnmedia.mapei.com/docs/librariesprovider56/products-documents/1_mapelastic-zero_en_tds_0286f8249274435296eef6ebd9c50b83.pdf?sfvrsn=97df2a04_0",
        "CONSUMPTION approx. 1.7 kg/m² per mm of thickness (manual); second coat to form a final thickness at least 2 mm (1,7 × 2)"),
}


def row_key(reg_no: str | None) -> str:
    """"EPD-IES-0013002:001 (S-P-13002)" -> "EPD-IES-0013002"."""
    return str(reg_no or "").split(" ")[0].split(":")[0]


# "Avjämning 10 mm", "avjämning 10,5 mm". A range ("5-10 mm") names no one
# thickness and gives None, as does more than one figure.
_MM_RE = re.compile(r"(?<![\d,.\-–])(\d+(?:[.,]\d+)?)\s*mm\b", re.IGNORECASE)
_RANGE_RE = re.compile(r"\d\s*[-–]\s*\d+(?:[.,]\d+)?\s*mm", re.IGNORECASE)


def thickness_mm(name: str) -> float | None:
    """The layer thickness the component's name states, or None."""
    text = str(name or "")
    if _RANGE_RE.search(text):
        return None
    found = {float(m.group(1).replace(",", ".")) for m in _MM_RE.finditer(text)}
    if len(found) != 1:
        return None
    mm = found.pop()
    return mm if mm > 0 else None


def surface(name: str) -> str | None:
    """"golv", "vägg" or None when the name says neither or both."""
    text = str(name or "").lower()
    floor = bool(re.search(r"golv", text))
    wall = bool(re.search(r"vägg", text))
    if floor == wall:
        return None
    return "golv" if floor else "vägg"


def levelling(reg_no: str | None) -> PerMm | None:
    return LEVELLING.get(row_key(reg_no))


def membrane(reg_no: str | None) -> PerM2 | None:
    return MEMBRANE.get(row_key(reg_no))
