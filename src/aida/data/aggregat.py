"""Air handling units: airflow and size class, per EPD and per component.

The 35 aggregat EPDs in the catalog span 270 to 24 500 kg CO2e/st, and the
spread is unit size (PR #752). A per-piece typvärde therefore describes no unit,
and ventilation/aggregat/st stays withheld. Decided 2026-09-30 (Demi, Till dig
d-20260928-082148-e12ed7), in falling order:

  A. Per airflow, kg CO2e per m3/h, when the component's flow is known: read
     from its name or its usage_context, never guessed.
  B. Per size class when it is not: a lägenhetsaggregat (one per dwelling) or a
     byggnadsaggregat (one serving a building), each with its own per-piece
     typvärde, the class read from the component's own words. The per-piece
     value is a unit of the class's median stated flow counted with A's
     per-airflow value (epd_baseline_medians._class_values_from_flow), since
     the class's own upper half is its biggest units.
  C. The baseline agent's estimate, as before, when neither can be told.

Henric can say "behåll som förut" and C comes back for every aggregat.

This module holds the facts both halves rest on. EPD_FACTS below was read by
hand from each EPD document (the PDF, and the ILCD dataset where it states
more), 2026-09-30. The catalog build copies it onto the catalog rows as
`airflow_m3h` and `aggregat_class`, so the typvärde code reads the catalog and
never this table; test_aggregat_flow checks that the two agree.

How the facts were read
-----------------------
Airflow. Only a flow the EPD itself states for the declared unit counts.
  - Kampmann states it in the product name and again in the product table,
    "Airflow (m3/hr)".
  - Flexit ProNordic states two capacities, one at SFP 1.5 and one at SFP 2
    kW/(m3/s). The SFP 1.5 figure is used: it is the SFPv that BBR 29 table
    9:95 says a unit replacing another should not exceed (från- och tilluft
    med värmeåtervinning, 1,5 kW/(m3/s)), and replacing units is what AIda's
    renovations do. Taking the SFP 2 figure instead moves the flow typvärde
    from 1.24 to 0.99 kg CO2e per m3/h.
  - Salda's two EPDs are multiple-product EPDs whose results are for a
    representative unit, AmberAir Compact S-R-3000, stated at 3000 m3/h.
  - Swegon's GOLD/SILVER C RX EPDs (2024) state two flows for the
    representative size: "Airflow max." in the product table and a "Design
    airflow rate" (100 %) in the B6 use scenario, run at the annual average SFP
    of all RX units sold, 1.6 kW/(m3/s). The design flow is used: it is the
    flow the declaration itself sizes the unit for, and the maximum is the
    capacity at the highest SFP, the figure ProNordic's choice above already
    turns down. The maximum would move Swegon's eight rows from 0.76-1.33 to
    0.48-0.83 kg CO2e per m3/h. The older 011/012 EPD (S-P-05063, 2022) has no
    use scenario and states no flow.
  - S&P's PURECLASS 800 CL states a "constant representative operating point
    of 700 m3/h" for its school scenario, the same kind of figure as Swegon's
    design flow. NASHIRA S (residential, "airflow rates of up to 150 m3/h")
    states only a maximum and is not in the catalog (HENRIC-3369). SABIK
    states no flow ("its reference flow rate", no number).
  - Flexit's eight Nordic units state no flow ("please visit our webpage"),
    Acetec's EPD is a weighted average of a series from 36 to 3 960 m3/h, and
    Zehnder's is declared per kg. None of them enters the flow typvärde.

Declared unit. Swegon's ILCD datasets give the reference flow as a Mass
property equal to the unit's weight (266 to 3 920 kg) while the PDF declares
"1 finished product", and the results are per unit. Read as a mass, the build
divided 24 500 kg CO2e by 3 920 kg and filed a per-kg row. `piece_mass_kg`
records the declared unit's weight as the PDF states it; the build keeps such
a row per piece only when the dataset's mass equals it, so a new version with
another weight is not taken on trust (HENRIC-3386).

Class. From the EPD's own statement of where the unit is used, not from a
threshold: a boundary nobody drew in the data would be invented here.
  - lägenhet: Flexit Nordic, "suitable for apartments, (detached) houses,
    villas and small commercial buildings"; Nordic CL2 does not use the words
    but has the series' kitchen-hood connection, which only a dwelling unit has.
  - byggnad: Flexit ProNordic, "for commercial properties/buildings"; Kampmann,
    "office buildings, commercial and industrial buildings, hotels, retail
    chains, sales buildings and multi-functional halls".
  - Salda states no application. It is placed by its stated flow: 3 000 m3/h
    is three times the smallest unit an EPD calls commercial (ProNordic L110R,
    1 000 m3/h at SFP 1.5) and eight times the only unit an EPD calls
    residential with a stated flow (Zehnder, "Residential ventilation",
    maximum 374 m3/h). Not a boundary, a position between two stated points.
  - Swegon GOLD/SILVER C RX states no application either ("designed for
    comfort ventilation") and is placed the same way: its smallest design flow,
    1 476 m3/h (004/005), is above ProNordic L110R's 1 000. The 011/012 EPD
    states no flow; it is a size of the same series, between 007/008 (2 340)
    and 014/020 (5 040).
  - S&P SABIK: lägenhet, "Range of domestic MVHR units".
  - S&P PURECLASS 800 CL is neither: a non-ducted unit for one room, "in
    schools, offices, hotels". A building unit's alternatives must not be a
    classroom's, and a flat's must not be a school's, so it counts only where
    its flow does.
  - Acetec is left out of both: the declared unit is the series average, and
    the series is sold for both.

All lägenhet units but SABIK are Flexit (8 of 9), so that class is computed and
withheld by the dominance rule (epd_baseline_medians._compute_with_withheld),
like badrumsinredning and förvaring. A lägenhetsaggregat without a flow falls
to C and the row says why.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

AGGREGAT_CLASSES = ("lägenhet", "byggnad")

# The keys in epd_baseline_medians that the classes and the flow are counted in.
CLASS_SUBCATEGORY = {"lägenhet": "aggregat_lägenhet", "byggnad": "aggregat_byggnad"}
FLOW_UNIT = "m3/h"


@dataclass(frozen=True)
class EpdFact:
    airflow_m3h: float | None
    klass: str | None
    evidence: str
    # The declared unit is one finished unit of this weight, as the EPD's PDF
    # states it, although the ILCD dataset gives the reference flow as a Mass.
    # None for every EPD whose dataset already says pieces.
    piece_mass_kg: float | None = None


# Keyed by registration number without the version suffix (":002"), so a new
# version of the same declaration keeps its facts and a new declaration does
# not silently inherit someone else's. test_aggregat_flow fails when an aggregat
# row in the catalog has no entry here: a new EPD has to be read before it
# counts.
EPD_FACTS: dict[str, EpdFact] = {
    # Kampmann GmbH & Co. KG. Flow in the name and in the product table.
    "EPD-IES-0025633": EpdFact(6000, "byggnad", "Airflow (m3/hr) 6,000; office and commercial buildings"),
    "EPD-IES-0025579": EpdFact(9000, "byggnad", "Airflow (m3/hr) 9,000; office and commercial buildings"),
    "EPD-IES-0025554": EpdFact(10000, "byggnad", "Airflow (m3/hr) 10,000; office and commercial buildings"),
    "EPD-IES-0025669": EpdFact(10000, "byggnad", "Airflow (m3/hr) 10,000; office and commercial buildings"),
    "EPD-IES-0025660": EpdFact(15012, "byggnad", "Airflow (m3/hr) 15,012; office and commercial buildings"),
    "EPD-IES-0025570": EpdFact(16000, "byggnad", "Airflow (m3/hr) 16,000; office and commercial buildings"),
    "EPD-IES-0025664": EpdFact(20000, "byggnad", "Airflow (m3/hr) 20,000; office and commercial buildings"),
    # Flexit AS, ProNordic. "Capacity ... SFP 1,5: <flow> m3/h", commercial.
    "NEPD-6948-6341": EpdFact(1000, "byggnad", "Capacity 85 %/SFP 1,5: 1000 m3/h @ 200 Pa; commercial properties"),
    "NEPD-6949-6340": EpdFact(1950, "byggnad", "Capacity 86 %/SFP 1,5: 1950 m3/h @ 250 Pa; commercial properties"),
    "NEPD-7221-6617": EpdFact(1550, "byggnad", "Capacity 86 %/SFP 1,5: 1550 m3/h @ 200 Pa; commercial buildings"),
    "NEPD-7223-6615": EpdFact(2500, "byggnad", "Capacity 85 %/SFP 1,5: 2500 m3/h @ 250 Pa; commercial buildings"),
    "NEPD-7222-6616": EpdFact(2400, "byggnad", "Capacity 86 %/SFP 1,5: 2400 m3/h @ 250 Pa; commercial properties"),
    "NEPD-6954-6330": EpdFact(3700, "byggnad", "Capacity 85 %/SFP 1,5: 3700 m3/h @ 250 Pa; commercial properties"),
    # Salda UAB. Representative product AmberAir Compact S-R-3000.
    "EPD-IES-0030270": EpdFact(3000, "byggnad", "Airflow (m3/hr) 3000, representative product S-R-3000-H; no application stated, placed by flow"),
    "EPD-IES-0030272": EpdFact(3000, "byggnad", "Airflow (m3/hr) 3000, representative product S-R-3000-V; no application stated, placed by flow"),
    # Flexit AS, Nordic. No flow in the EPD ("please visit our webpage").
    "NEPD-7226-6608": EpdFact(None, "lägenhet", "Nordic CL2: no flow stated; kitchen-hood connection (dwelling unit)"),
    "NEPD-7225-6609": EpdFact(None, "lägenhet", "Nordic CL3: no flow stated; apartments, detached houses, villas"),
    "NEPD-7224-6610": EpdFact(None, "lägenhet", "Nordic CL4: no flow stated; apartments, detached houses, villas"),
    "NEPD-11163-9191": EpdFact(None, "lägenhet", "Nordic L6 SW: no flow stated; apartments, houses, villas"),
    "NEPD-6159-5426-EN": EpdFact(None, "lägenhet", "Nordic S2: no flow stated; apartments, houses, villas"),
    "NEPD-6160-5425-EN": EpdFact(None, "lägenhet", "Nordic S3: no flow stated; apartments, houses, villas"),
    "NEPD-6161-5424-EN": EpdFact(None, "lägenhet", "Nordic S4: no flow stated; apartments, houses, villas"),
    "NEPD-9539-9190": EpdFact(None, "lägenhet", "Nordic S7 SW: no flow stated; apartments, houses, villas"),
    # Swegon Group AB, GOLD/SILVER C RX (HENRIC-3386). PDF: "Declared unit 1
    # finished product"; the ILCD gives a Mass equal to the unit's weight.
    # Flow: the B6 scenario's "Design airflow rate" (100 %) in m3/s x 3600, at
    # the annual average SFP 1.6 kW/(m3/s). PDFs read from Swegon's site and
    # its Baltic distributor (ecowise.lv); Environdec's library no longer
    # serves them, its data hub does.
    "EPD-IES-0013087": EpdFact(1476, "byggnad", "GOLD RX 005: Design airflow rate 0.41 m3/s (max 0.65), SFP 1.6; 266 kg; comfort ventilation, placed by flow", 266),
    "EPD-IES-0013088": EpdFact(2340, "byggnad", "GOLD RX 008: Design airflow rate 0.65 m3/s (max 1), SFP 1.6; 344 kg; comfort ventilation, placed by flow", 344),
    "EPD-IES-0005063": EpdFact(None, "byggnad", "GOLD RX 012, S-P-05063 (2022): no flow and no use scenario; 488 kg; a size of the series between 007/008 and 014/020", 488),
    "EPD-IES-0013089": EpdFact(5040, "byggnad", "GOLD RX 020: Design airflow rate 1.40 m3/s (max 2.1), SFP 1.6; 679 kg; comfort ventilation, placed by flow", 679),
    "EPD-IES-0013090": EpdFact(6840, "byggnad", "GOLD RX 030: Design airflow rate 1.9 m3/s (max 3.2), SFP 1.6; 861 kg; comfort ventilation, placed by flow", 861),
    "EPD-IES-0013091": EpdFact(10800, "byggnad", "GOLD RX 040: Design airflow rate 3 m3/s (max 5), SFP 1.6; 1254 kg; comfort ventilation, placed by flow", 1254),
    "EPD-IES-0013092": EpdFact(14040, "byggnad", "GOLD RX 060: Design airflow rate 3.9 m3/s (max 6.5), SFP 1.6; 1534 kg; comfort ventilation, placed by flow", 1534),
    "EPD-IES-0013343": EpdFact(19440, "byggnad", "GOLD RX 080: Design airflow rate 5.4 m3/s (max 9.5), SFP 1.6; 2482 kg; comfort ventilation, placed by flow", 2482),
    "EPD-IES-0013344": EpdFact(31680, "byggnad", "GOLD RX 120: Design airflow rate 8.80 m3/s (max 14.0), SFP 1.6; 3920 kg; comfort ventilation, placed by flow", 3920),
    # S&P Sistemas de Ventilación (Soler & Palau). Declared per unit in the
    # dataset itself (Number of pieces).
    "EPD-IES-0013020": EpdFact(None, "lägenhet", "SABIK 350: no flow stated ('its reference flow rate'); Range of domestic MVHR units"),
    "EPD-IES-0025462": EpdFact(700, None, "PURECLASS 800 CL: 'constant representative operating point of 700 m3/h'; non-ducted room unit for schools, offices, hotels"),
    # Acetec AB. Weighted average of the EvoAir A series, 36-3 960 m3/h.
    "EPD-IES-0033292": EpdFact(None, None, "weighted average over a series from 36 to 3 960 m3/h, residential and larger buildings"),
    # Zehnder. Declared per kg of unit; the flow (max 374 m3/h) is the
    # reference product's, not the declared unit's.
    "EPD-IES-0032015": EpdFact(None, None, "declared per kg; Residential ventilation, maximum airflow 374 m3/h"),
}


def reg_stem(reg_no: str) -> str:
    """Registration number without its version suffix: 'EPD-IES-0025633:002'
    -> 'EPD-IES-0025633'."""
    return (reg_no or "").split(":", 1)[0].strip()


def facts_for(reg_no: str) -> EpdFact | None:
    return EPD_FACTS.get(reg_stem(reg_no))


# ---------------------------------------------------------------------------
# Component side
# ---------------------------------------------------------------------------

# A number with Swedish or English grouping: "3000", "3 000" (also with a
# no-break or thin space), "10,000", "10.000", "0,8", "2.5". Never the tail of
# a word or of a longer number: "LB01 300" is 300, not 1 300.
_SPACES = r"[    ]"
_NUM = (r"(?<![\w.,])"
        r"(\d{1,3}(?:" + _SPACES + r"[0-9]{3})+|\d{1,3}(?:[.,][0-9]{3})+(?![0-9])|\d+(?:[.,]\d+)?)")
_FLOW = re.compile(
    _NUM + r"\s*(m\s*(?:³|3|\^3)\s*/\s*(?:hr|h|timme|tim)|m\s*(?:³|3|\^3)\s*/\s*(?:sek|s)"
    r"|l\s*/\s*(?:sek|s))"
    r"(?![a-zåäö])",
    re.IGNORECASE,
)
# A flow per person, per m2 or per room is a design rate from BBR or a brief,
# not the unit's flow: "7 l/s per person", "0,35 l/s,m²", "à 35 l/s",
# "per person: 15 l/s". Per unit, system or building is the unit's: "3 000
# m³/h per aggregat". A comma only makes a rate before an area ("3 000 m³/h,
# rum 2.14" names the plant room).
_RATE_WORDS = (r"(?:m2|m²|m\^2|kvm|kvadratmeter|pers|person|elev|barn|lgh|lägenhet"
               r"|rum|klassrum|plats|sittplats|brukare|bädd)(?![a-zåäö])")
_UNITS_OWN = r"(?!\w*aggregat|system|byggnad|hus|st\b|styck|enhet|anläggning)"
_RATE_AFTER = re.compile(
    r"\s*(?:(?:per\b|/)\s*" + _RATE_WORDS
    + r"|,\s*(?:m2|m²|m\^2|kvm)(?![a-zåäö0-9])"
    + r"|per\s+" + _UNITS_OWN + ")",
    re.IGNORECASE,
)
# "à" names a rate unless it follows the unit itself ("24 st FTX-aggregat à
# 350 m³/h" is each unit's flow). A bare "a" only after a count ("25 rum a 35
# l/s"): alone it is a letter ("Hus A 3 000 m³/h").
_NOT_THE_UNIT = r"(?<![\w-])(?![\w-]*aggregat\b)[\w-]+"
_RATE_BEFORE = re.compile(
    r"(?:(?:\d+\s+(?:st\.?\s+)?)?" + _NOT_THE_UNIT + r"\s+à"
    r"|\d+\s+(?:st\.?\s+)?" + _NOT_THE_UNIT + r"\s+a"
    r"|\bper\s+" + _UNITS_OWN + r"\w+\s*:?)\s*$",
    re.IGNORECASE,
)
# "1 000-2 000 m³/h", "3000/2800 m³/h", "2 x 1 000 m³/h", "2 000 eller 3 000
# m³/h": the unit belongs to more than the number it follows, and which one is
# meant would be a guess. The number before must stand alone: in "LB01 - 3 000
# m³/h" the dash only separates the designation.
_SEVERAL_BEFORE = re.compile(
    # After a designation only a number of three digits or more: "TA 2 - 3 000"
    # is TA 2 at 3 000, "TF 3000/2800" a pair.
    r"(?<!\w)(?:" + "".join(f"(?<!{d}{sep})" for d in ("LB", "TA", "TF", "LA") for sep in " -")
    + r"\d+|\d{3,})(?:" + _SPACES + r"\d{3}|[.,]\d+)*\s*"
    r"(?:-|–|—|/|x|×|\*|eller|och|till|resp\.?|respektive|or|and|to)\s*$",
    re.IGNORECASE,
)
# "TA 2 500 m³/h" is TA 2 at 500 or TA at 2 500, "TA1-2 500" TA1 at 2 500 or
# TA1-2 at 500.
_DESIGNATION_BEFORE = re.compile(r"\b(?:LB|TA|TF|LA)(?:[ -]?|[ -]?\d{1,2}-)$")
# The span a unit's nominal flow can have, from a flat's unit to the largest
# built-up units. Outside it the reading is wrong, not the unit.
_MIN_FLOW, _MAX_FLOW = 50.0, 100_000.0


def _to_float(raw: str, per_second: bool) -> float | None:
    s = re.sub(_SPACES, "", raw)
    # Per hour, a comma or point followed by exactly three digits and nothing
    # more is a thousands separator ("10,000 m³/h"): a unit's flow never has
    # three decimals. Per second the same form is a decimal ("0,833 m³/s",
    # "1.250 l/s"), and so is anything that starts with a zero.
    if (not per_second and not s.startswith("0")
            and re.fullmatch(r"\d{1,3}([.,]\d{3})+", s)):
        s = re.sub(r"[.,]", "", s)
    elif s.count(",") + s.count(".") > 1:
        return None
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _fmt(value: float) -> str:
    if value >= 10:
        return f"{value:,.0f}".replace(",", " ")
    return f"{value:g}".replace(".", ",")


# "tilluft 3 000 m³/h" or "3 000 m³/h tilluft", the label next to its own flow.
# Not "tilluftsaggregat": that names a unit.
_LABEL = r"(tilluft|frånluft)(?![\w-]*aggregat)\w*"
_LABEL_BEFORE = re.compile(_LABEL + r"\s*:?\s*$", re.IGNORECASE)
_LABEL_AFTER = re.compile(r"\s*\(?\s*" + _LABEL, re.IGNORECASE)
_LABELLED_NUMBER = re.compile(_LABEL + r"\s*:?\s*\d", re.IGNORECASE)


def _scan(text: str) -> tuple[list[float], str]:
    """The airflows the text states for the unit, in m3/h, and why a flow it
    mentions cannot be used ('' when nothing stands in the way)."""
    found, doubt = _scan_labelled(text)
    return [v for v, _ in found], doubt


def _scan_labelled(text: str) -> tuple[list[tuple[float, str]], str]:
    """_scan, with each flow's supply/exhaust label ('' when it has none)."""
    text = text or ""
    flows: list[tuple[float, str]] = []
    labelled_before = 0
    doubt = ""
    for m in _FLOW.finditer(text):
        before = text[:m.start()]
        if _RATE_AFTER.match(text, m.end()) or _RATE_BEFORE.search(before):
            continue
        unit = re.sub(r"sek$", "s", m.group(2).lower().replace(" ", ""))
        digits = re.sub(_SPACES, "", m.group(1))
        if unit.startswith("l") and re.fullmatch(r"[1-9]\d{0,2}[.,]\d{3}", digits):
            # "1,250 l/s": 1,25 or 1 250. Nobody states l/s to three decimals,
            # but a point or comma for thousands is not Swedish either.
            doubt = doubt or f"anger {m.group(1)} l/s, som kan vara både tusental och decimaler"
            continue
        value = _to_float(m.group(1), per_second=unit.endswith("/s"))
        if value is None or value <= 0:
            continue
        if unit.startswith("l"):
            value *= 3.6          # l/s -> m3/h
        elif unit.endswith("/s"):
            value *= 3600         # m3/s -> m3/h
        value = round(value, 1)
        if _SEVERAL_BEFORE.search(before):
            doubt = doubt or "anger ett intervall eller flera luftflöden"
            continue
        if re.search(_SPACES, m.group(1)) and _DESIGNATION_BEFORE.search(before.rstrip()):
            doubt = doubt or "skiljer inte systembeteckningen från luftflödet"
            continue
        if not _MIN_FLOW <= value <= _MAX_FLOW:
            doubt = doubt or (f"anger {_fmt(value)} m³/h, vilket inte är ett rimligt "
                              f"luftflöde för ett aggregat")
            continue
        label = _LABEL_BEFORE.search(before) or _LABEL_AFTER.match(text, m.end())
        labelled_before += bool(_LABEL_BEFORE.search(before))
        flows.append((value, label.group(1).lower() if label else ""))
    if len(_LABELLED_NUMBER.findall(text)) > labelled_before:
        # "tilluft 3 000 / frånluft 2 800 m³/h": the unit follows the last only.
        doubt = doubt or "anger till- och frånluft, men enheten bara för ett av flödena"
    return flows, doubt


def _distinct(values: list[float]) -> list[float]:
    """The same flow written twice, in two units, is one flow:
    "3 000 m³/h (833 l/s)" reads 3 000 and 2 998.8."""
    out: list[float] = []
    for v in values:
        if not any(abs(v - d) <= 0.01 * max(v, d) for d in out):
            out.append(v)
    return out


def airflows_in(text: str) -> list[float]:
    """Every usable airflow the text states for the unit, in m3/h, in order."""
    return _scan(text)[0]


# A renovation's description often names the unit being replaced, and its
# flow: "ersätter ett aggregat på 2 000 m³/h med ett på 3 000". Which flow is
# the new unit's is then a guess, so a description that names an old unit
# gives none. The name is the new unit's.
_OLD_UNIT = re.compile(
    r"\b(?:befintlig\w*|tidigare|nuvarande|gaml[ae]|gammalt|ersätt\w*|byts|bytas|rivs"
    r"|rivas|demonter\w*)\b",
    re.IGNORECASE,
)
_TOTAL = re.compile(r"\b(?:totalt|sammanlagt|tillsammans|summa|total)\b", re.IGNORECASE)


def component_airflow(name: str, usage_context: str = "",
                      quantity: float = 1) -> tuple[float | None, str]:
    """The component's airflow in m3/h, and where it was read, or (None, why).

    The name first: it is where the chat and the intake write the flow, and it
    names one unit. The usage_context only when the name says nothing about a
    flow, only when it states exactly one, only when it does not also name the
    unit being replaced, and only for a single unit: for two, "12 000 m³/h" may
    be each unit's or both together (so may "totalt 12 000 m³/h" in the name).
    Supply and exhaust named as a pair give the larger.
    """
    n = f"{quantity:g}".replace(".", ",")
    for text, where in ((name, "namnet"), (usage_context, "beskrivningen")):
        found, doubt = _scan_labelled(text)
        flows = _distinct([v for v, _ in found])
        # An FTX unit moves both: "3 000 m³/h tilluft, 2 800 m³/h frånluft" is
        # one unit sized for the larger. Only with each label on its own flow,
        # and for one unit.
        if (len(found) == 2 and not doubt and quantity == 1
                and {lab for _, lab in found} == {"tilluft", "frånluft"}):
            flows = [max(flows)]
        if doubt and not flows:
            return None, f"{where} {doubt}"
        if len(flows) > 1 or (doubt and flows):
            return None, f"{where} anger flera luftflöden"
        if flows:
            if where == "beskrivningen" and _OLD_UNIT.search(text):
                return None, ("beskrivningen nämner också aggregatet som byts ut, så "
                              "luftflödet kan vara det gamla aggregatets")
            if quantity != 1 and (where == "beskrivningen" or _TOTAL.search(text)):
                return None, (f"{where} anger ett luftflöde, men inte om det gäller "
                              f"vart och ett av de {n} aggregaten eller alla tillsammans")
            return flows[0], where
    return None, "inget luftflöde angivet"


# Words that name the class the way the EPDs do. lägenhet: a unit per dwelling
# ("apartments, houses, villas"). byggnad: a unit serving a building, named as
# such. A building type alone ("skola", "kontor") is NOT used: the lägenhet EPDs
# also claim "small commercial buildings", so a small office can have either.
#
# The name is read broadly, the usage_context strictly. Intake writes the
# building into the usage_context ("flerbostadshus med 24 lägenheter"), and a
# bare "lägenheter" there describes the house, not the unit, which may be one
# central unit for all 24. There only a phrase that puts the unit IN a dwelling
# counts.
_LAGENHET_NAME = re.compile(
    r"lägenhet|\blgh\b|\bvilla|småhus|radhus|parhus|enbostadshus|bostadsaggregat",
    re.IGNORECASE,
)
_LAGENHET_CONTEXT = re.compile(
    r"(?:per|varje|i varje|en i varje|ett i varje) (?:lägenhet|lgh)\b|lägenhetsaggregat"
    r"|bostadsaggregat|\bvilla\b|småhus|radhus|parhus|enbostadshus",
    re.IGNORECASE,
)
# Same split for byggnad. In the name "Centralt FTX-aggregat" is the unit; in
# the usage_context "skolan ligger centralt" is the address, so there "central"
# only counts next to the word aggregat. "Hela huset" is in neither: a villa's
# unit serves the whole house too.
_BYGGNAD_NAME = re.compile(
    r"byggnadsaggregat|centralaggregat|\bcentral(?:t|a)?\b|huvudaggregat"
    r"|hela (?:byggnaden|skolan|förskolan|fastigheten)",
    re.IGNORECASE,
)
_BYGGNAD_CONTEXT = re.compile(
    r"byggnadsaggregat|centralaggregat|huvudaggregat"
    r"|\bcentral(?:t|a)?\s+(?:[\w-]*aggregat|ftx)"
    r"|hela (?:byggnaden|skolan|förskolan|fastigheten)",
    re.IGNORECASE,
)
# A system designation from the ventilation drawings (LB01, TA 2, TF-03).
# Case-sensitive on purpose: "ta 2 aggregat" is the verb. In the usage_context
# only LB: there "TA 12" is as likely a technical guideline as a system.
_DESIGNATION = re.compile(r"\b(?:LB|TA|TF|LA)[ -]?\d{1,2}\b")
_DESIGNATION_CONTEXT = re.compile(r"\bLB[ -]?\d{1,2}\b")
# "24 lägenheter" in a name counts the building's flats, it does not put the
# unit in one: "FTX-aggregat för flerbostadshus, 24 lägenheter".
_COUNTED_DWELLINGS = re.compile(r"\d[\d\s]*(?:st\.?\s+)?\w*(?:lägenheter|lgh)\b(?!-)",
                                re.IGNORECASE)
# "10 l/s per lägenhet" is a design rate, not a unit in each flat.
_RATE_PER_DWELLING = re.compile(
    r"(?:l|m\s*(?:³|3))\s*/\s*(?:sek|s|h)\s*(?:per|/)\s*(?:lägenhet|lgh)\b", re.IGNORECASE)


def component_class(name: str, usage_context: str = "") -> tuple[str | None, str]:
    """("lägenhet" | "byggnad", where it was read) or (None, why), from the
    component's own words.

    Both classes named ("centralt aggregat för 24 lägenheter") is not decided
    here: which one is meant needs the flow, and the row asks for it."""
    name = name or ""
    ctx = _RATE_PER_DWELLING.sub(" ", usage_context or "")
    lag_name = bool(_LAGENHET_NAME.search(_COUNTED_DWELLINGS.sub(" ", name)))
    lag = lag_name or bool(_LAGENHET_CONTEXT.search(ctx))
    byg_name = bool(_BYGGNAD_NAME.search(name) or _DESIGNATION.search(name))
    byg = byg_name or bool(_BYGGNAD_CONTEXT.search(ctx) or _DESIGNATION_CONTEXT.search(ctx))
    if lag and byg:
        return None, "både lägenhet och hela byggnaden nämns"
    if lag:
        return "lägenhet", "namnet" if lag_name else "beskrivningen"
    if byg:
        return "byggnad", "namnet" if byg_name else "beskrivningen"
    return None, "det framgår inte om det är ett lägenhets- eller byggnadsaggregat"
