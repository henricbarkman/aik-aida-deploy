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
    use scenario and states no flow. Since 2026-10-02 the GOLD rows come from
    Swegon's EPD Hub declaration HUB-6058 instead (see PRODUCT_FACTS): it
    states the design flow of size 012 (0.95 m3/s), and the other sizes keep
    the 2024 design flows read here.
  - S&P's PURECLASS 800 CL states a "constant representative operating point
    of 700 m3/h" for its school scenario, the same kind of figure as Swegon's
    design flow. NASHIRA S (residential, "airflow rates of up to 150 m3/h")
    states only a maximum, so it counts in its class and not per flow
    (HENRIC-3369). SABIK states no flow ("its reference flow rate", no number).
  - Systemair's 20 SAVE units and Vallox's 16 (HENRIC-3369) state no flow in
    the dataset, nor in the one document of each kind read in full (the SAVE
    VTR 300/B PDF; Environdec's page for Vallox 096 MV).
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
  - S&P SABIK: lägenhet, "Range of domestic MVHR units". NASHIRA S:
    lägenhet, "Double-flow VMC for homes".
  - Systemair SAVE: lägenhet, "residential air handling unit" in every one.
  - Vallox MV and TSK Multi MV: lägenhet, the living-comfort text ("the
    structures of your house"; 096 MV "suited for small and medium-sized
    apartments"). MyVallox CFi says "homes and other buildings" and is left
    out of both, like Acetec.
  - S&P PURECLASS 800 CL is neither: a non-ducted unit for one room, "in
    schools, offices, hotels". A building unit's alternatives must not be a
    classroom's, and a flat's must not be a school's, so it counts only where
    its flow does.
  - Acetec is left out of both: the declared unit is the series average, and
    the series is sold for both.

Until HENRIC-3369 all lägenhet units but SABIK were Flexit (8 of 9), and the
dominance rule withheld the class. With Systemair's SAVE, Vallox and NASHIRA it
has four makers, Systemair the largest at about half, and is published. None of
its EPDs states a flow, so its value is the per-piece upper-half median
(_class_values_from_flow keeps it when the class states too few flows).
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
    # Swegon Group AB, GOLD RX: one EPD Hub declaration (HUB-6058, 2026) with
    # a GWP per size; facts per size in PRODUCT_FACTS. It replaced nine
    # Environdec EPDs (S-P-05063, 13087-13092, 13343, 13344, HENRIC-3386) that
    # the library stopped serving (HENRIC-3405). Their piece_mass_kg went with
    # them: the hand rows are per piece already.
    "HUB-6058": EpdFact(None, None, "GOLD RX, one EPD with a GWP per size; facts per size in PRODUCT_FACTS"),
    # S&P Sistemas de Ventilación (Soler & Palau). Declared per unit in the
    # dataset itself (Number of pieces).
    "EPD-IES-0013020": EpdFact(None, "lägenhet", "SABIK 350: no flow stated ('its reference flow rate'); Range of domestic MVHR units"),
    "EPD-IES-0025462": EpdFact(700, None, "PURECLASS 800 CL: 'constant representative operating point of 700 m3/h'; non-ducted room unit for schools, offices, hotels"),
    # Acetec AB. Weighted average of the EvoAir A series, 36-3 960 m3/h.
    "EPD-IES-0033292": EpdFact(None, None, "weighted average over a series from 36 to 3 960 m3/h, residential and larger buildings"),
    # Zehnder. Declared per kg of unit; the flow (max 374 m3/h) is the
    # reference product's, not the declared unit's.
    "EPD-IES-0032015": EpdFact(None, None, "declared per kg; Residential ventilation, maximum airflow 374 m3/h"),
    # UAB Systemair, SAVE (HENRIC-3369). EPD Norge, generated with the NPCR 030
    # EPD generator. The dataset and the PDF (VTR 300/B read in full) call each a
    # "residential air handling unit" and state no airflow.
    "NEPD-7165-6570": EpdFact(None, "lägenhet", "SAVE VSC 100: residential air handling unit; no flow stated"),
    "NEPD-7301-6569": EpdFact(None, "lägenhet", "SAVE VSC 200: residential air handling unit; no flow stated"),
    "NEPD-7302-6571": EpdFact(None, "lägenhet", "SAVE VSC 300: residential air handling unit; no flow stated"),
    "NEPD-8063-7703": EpdFact(None, "lägenhet", "SAVE VSR 150/B: residential air handling unit; no flow stated"),
    "NEPD-7303-6572": EpdFact(None, "lägenhet", "SAVE VSR 200/B: residential air handling unit; no flow stated"),
    "NEPD-7296-6564": EpdFact(None, "lägenhet", "SAVE VSR 300: residential air handling unit; no flow stated"),
    "NEPD-7300-6568": EpdFact(None, "lägenhet", "SAVE VSR 400: residential air handling unit; no flow stated"),
    "NEPD-7295-6563": EpdFact(None, "lägenhet", "SAVE VSR 500: residential air handling unit; no flow stated"),
    "NEPD-7294-6561": EpdFact(None, "lägenhet", "SAVE VSR 700: residential air handling unit; no flow stated"),
    "NEPD-7166-6562": EpdFact(None, "lägenhet", "SAVE VTC 200-1: residential air handling unit; no flow stated"),
    "NEPD-7297-6565": EpdFact(None, "lägenhet", "SAVE VTC 300: residential air handling unit; no flow stated"),
    "NEPD-7298-6566": EpdFact(None, "lägenhet", "SAVE VTC 500: residential air handling unit; no flow stated"),
    "NEPD-6360-5625-EN": EpdFact(None, "lägenhet", "SAVE VTR 100/B: residential air handling unit; no flow stated"),
    "NEPD-6359-5626-EN": EpdFact(None, "lägenhet", "SAVE VTR 150/B: residential air handling unit; no flow stated"),
    "NEPD-6365-5620-EN": EpdFact(None, "lägenhet", "SAVE VTR 150/K: residential air handling unit; no flow stated"),
    "NEPD-6367-5618-EN": EpdFact(None, "lägenhet", "SAVE VTR 250/B: residential air handling unit; no flow stated"),
    "NEPD-6366-5619-EN": EpdFact(None, "lägenhet", "SAVE VTR 275/B: residential air handling unit; no flow stated"),
    "NEPD-6364-5621-EN": EpdFact(None, "lägenhet", "SAVE VTR 300/B: residential air handling unit; no flow stated"),
    "NEPD-6363-5622-EN": EpdFact(None, "lägenhet", "SAVE VTR 500: residential air handling unit; no flow stated"),
    "NEPD-6362-5623-EN": EpdFact(None, "lägenhet", "SAVE VTR 700: residential air handling unit; no flow stated"),
    # Vallox Oy (not part of Zehnder Group: its 2024 list of companies has only
    # Enervent in Finland). MV and TSK Multi MV: the dataset's own application
    # text is the living-comfort one ("the structures of your house"), and
    # Environdec's page for 096 MV says "suited for small and medium-sized
    # apartments". No flow in the dataset.
    "EPD-IES-0010356": EpdFact(None, "lägenhet", "Vallox 096 MV: comfort of living, 'your house'; no flow stated"),
    "EPD-IES-0010357": EpdFact(None, "lägenhet", "Vallox 110 MV: comfort of living, 'your house'; no flow stated"),
    "EPD-IES-0011397": EpdFact(None, "lägenhet", "Vallox 125 MV: comfort of living, 'your house'; no flow stated"),
    "EPD-IES-0010358": EpdFact(None, "lägenhet", "Vallox 145 MV: comfort of living, 'your house'; no flow stated"),
    "EPD-IES-0010359": EpdFact(None, "lägenhet", "Vallox 245 MV: comfort of living, 'your house'; no flow stated"),
    "EPD-IES-0009030": EpdFact(None, "lägenhet", "Vallox 51 MV: comfort of living, 'your house'; no flow stated"),
    "EPD-IES-0010355": EpdFact(None, "lägenhet", "Vallox 99 MV: comfort of living, 'your house'; no flow stated"),
    "EPD-IES-0010360": EpdFact(None, "lägenhet", "Vallox TSK Multi 50 MV: comfort of living, 'your house'; no flow stated"),
    "EPD-IES-0010361": EpdFact(None, "lägenhet", "Vallox TSK Multi 80 MV: comfort of living, 'your house'; no flow stated"),
    # MyVallox CFi: "ventilation for homes and other buildings", so both, and
    # left out of both classes like Acetec. No flow stated.
    "EPD-IES-0025180": EpdFact(None, None, "MyVallox 119 CFi: homes and other buildings; no flow stated"),
    "EPD-IES-0029384": EpdFact(None, None, "MyVallox 125 CFi: homes and other buildings; no flow stated"),
    "EPD-IES-0025182": EpdFact(None, None, "MyVallox 149 CFi: homes and other buildings; no flow stated"),
    "EPD-IES-0029385": EpdFact(None, None, "MyVallox 245 CFi: homes and other buildings; no flow stated"),
    "EPD-IES-0029386": EpdFact(None, None, "MyVallox 245 CFi VKL: homes and other buildings; no flow stated"),
    "EPD-IES-0029335": EpdFact(None, None, "MyVallox 51 CFi: homes and other buildings; no flow stated"),
    "EPD-IES-0029383": EpdFact(None, None, "MyVallox 99 CFi: homes and other buildings; no flow stated"),
    # S&P NASHIRA S: "Double-flow VMC for homes"; states only a maximum.
    "EPD-IES-0017092": EpdFact(None, "lägenhet", "NASHIRA S: double-flow VMC for homes; only a maximum flow (up to 150 m3/h) stated"),
    # Swegon Group AB, CASA (HENRIC-3398). One EPD, S-P-05388:003, with ten
    # units each declared per piece; the class is per unit (PRODUCT_FACTS).
    # Entered by hand: Environdec's data hub has no dataset for it.
    "EPD-IES-0005388": EpdFact(None, None, "CASA, ten units in one EPD; class per unit in PRODUCT_FACTS"),
}

# Facts per product, for an EPD that declares several units with different
# facts. Keyed by (registration stem, catalog name); facts_for reads these
# before EPD_FACTS.
#
# Swegon CASA, S-P-05388:003 (read 2026-10-02, HENRIC-3398). Flow: the EPD's
# only figure is the B6 scenario's "Average air handling capacity" (25 to 320
# l/s), an operating average and not a design or maximum flow, the figure
# GOLD's weighted mean was turned down for above. None of them enters the flow
# typvärde. Class: the EPD says only "buildings". Swegon's CASA catalogue 2026
# lists all ten under "Residential ventilation units" and gives nine an
# Ecodesign energy class, which only a residential ventilation unit (RVU)
# carries: under Regulation (EU) 1253/2014 art. 2(2) a unit up to 250 m3/h, or
# up to 1 000 m3/h that the manufacturer declares exclusively residential.
# That declaration is the manufacturer's own statement of use, so the nine are
# lägenhet. R15V is "NRVU" in the same catalogue (max 1 710 m3/h) while Swegon
# describes the R7-R15 sizes for "large residences, operating plants, and
# meeting spaces alike", so it is left out of both classes, like Acetec.
_CASA = "EPD-IES-0005388"
_CASA_RVU = "Swegon CASA catalogue 2026: residential ventilation unit with an Ecodesign energy class (RVU, EU 1253/2014 art. 2(2)); EPD flow is a use-phase average"
PRODUCT_FACTS: dict[tuple[str, str], EpdFact] = {
    (_CASA, "Swegon CASA W3xs air handling unit"): EpdFact(None, "lägenhet", "W3xs: average 30 l/s; catalogue 36-288 m3/h, class A. " + _CASA_RVU),
    (_CASA, "Swegon CASA W5 air handling unit"): EpdFact(None, "lägenhet", "W5: average 60 l/s; catalogue 108-468 m3/h, class A+. " + _CASA_RVU),
    (_CASA, "Swegon CASA R2 air handling unit"): EpdFact(None, "lägenhet", "R2: average 25 l/s; catalogue 65-216 m3/h. " + _CASA_RVU),
    (_CASA, "Swegon CASA R3 air handling unit"): EpdFact(None, "lägenhet", "R3: average 30 l/s; catalogue 90-295 m3/h. " + _CASA_RVU),
    (_CASA, "Swegon CASA R5 air handling unit"): EpdFact(None, "lägenhet", "R5: average 60 l/s; catalogue 108-421 m3/h. " + _CASA_RVU),
    (_CASA, "Swegon CASA R5H air handling unit"): EpdFact(None, "lägenhet", "R5-H (horizontal): average 60 l/s; catalogue 108-439 m3/h. " + _CASA_RVU),
    (_CASA, "Swegon CASA R7H air handling unit"): EpdFact(None, "lägenhet", "R7-H (horizontal), use-phase column 'R7H': average 120 l/s; catalogue 216-749 m3/h. " + _CASA_RVU),
    (_CASA, "Swegon CASA R7V air handling unit"): EpdFact(None, "lägenhet", "R7 (standing, R07V), use-phase column 'R7': average 100 l/s; catalogue 216-677 m3/h. " + _CASA_RVU),
    (_CASA, "Swegon CASA R9V air handling unit"): EpdFact(None, "lägenhet", "R9 (R09V): average 170 l/s; catalogue 270-871 m3/h. " + _CASA_RVU),
    (_CASA, "Swegon CASA R15V air handling unit"): EpdFact(None, None, "R15 (R15V): average 320 l/s; catalogue 360-1 710 m3/h, marked NRVU; for large residences and meeting spaces alike, so neither class"),
}

# Swegon GOLD RX, HUB-6058 (EPD Hub, published 2026-04-17, read 2026-10-02,
# HENRIC-3405). The declared unit is one GOLD RX 012; appendix 1 gives the
# GWP-fossil A1-A3 of every size, each modelled individually. The catalog keeps
# nine of the eighteen sizes, the larger of each pair Swegon's nine earlier
# Environdec EPDs declared (004/005 ... 100/120), for two reasons: the pairs
# differ by 0-7 %, and the flows below exist for those sizes only, so the
# other nine could not be sized and would only have doubled one maker's share
# of the building class.
#
# Flow. HUB-6058 states a design flow for the declared size alone: the B6
# scenario's "Design airflow rate" 0.95 m3/s for RX 012 (the functional unit
# says "up to 3420 m3/hr"). The other eight carry the design airflow rate of
# the same size in Swegon's 2024 Environdec EPDs (the same B6 scenario, SFP
# 1.6), which were withdrawn as LCA results, not as product ratings; 012's
# 0.95 sits between 008's 0.65 and 020's 1.40, so the two declarations size
# the series the same way. Without them eight GOLD rows would drop out of
# every sized alternatives list and of the per-airflow typvärde. Class: the
# smallest design flow, 1 476 m3/h, is above ProNordic L110R's 1 000, so
# byggnad, as before.
_GOLD = "HUB-6058"
_GOLD_2024 = "the design airflow rate of this size in Swegon's 2024 Environdec EPD"
PRODUCT_FACTS.update({
    (_GOLD, "Swegon GOLD RX 005 air handling unit"): EpdFact(1476, "byggnad", f"GOLD RX 005: Design airflow rate 0.41 m3/s (S-P-13087), {_GOLD_2024}; comfort ventilation, placed by flow"),
    (_GOLD, "Swegon GOLD RX 008 air handling unit"): EpdFact(2340, "byggnad", f"GOLD RX 008: Design airflow rate 0.65 m3/s (S-P-13088), {_GOLD_2024}; comfort ventilation, placed by flow"),
    (_GOLD, "Swegon GOLD RX 012 air handling unit"): EpdFact(3420, "byggnad", "GOLD RX 012: Design airflow rate 0.95 m3/s, HUB-6058 appendix 2 (B6, 100 %), 'up to 3420 m3/hr' in its functional unit; comfort ventilation, placed by flow"),
    (_GOLD, "Swegon GOLD RX 020 air handling unit"): EpdFact(5040, "byggnad", f"GOLD RX 020: Design airflow rate 1.40 m3/s (S-P-13089), {_GOLD_2024}; comfort ventilation, placed by flow"),
    (_GOLD, "Swegon GOLD RX 030 air handling unit"): EpdFact(6840, "byggnad", f"GOLD RX 030: Design airflow rate 1.9 m3/s (S-P-13090), {_GOLD_2024}; comfort ventilation, placed by flow"),
    (_GOLD, "Swegon GOLD RX 040 air handling unit"): EpdFact(10800, "byggnad", f"GOLD RX 040: Design airflow rate 3 m3/s (S-P-13091), {_GOLD_2024}; comfort ventilation, placed by flow"),
    (_GOLD, "Swegon GOLD RX 060 air handling unit"): EpdFact(14040, "byggnad", f"GOLD RX 060: Design airflow rate 3.9 m3/s (S-P-13092), {_GOLD_2024}; comfort ventilation, placed by flow"),
    (_GOLD, "Swegon GOLD RX 080 air handling unit"): EpdFact(19440, "byggnad", f"GOLD RX 080: Design airflow rate 5.4 m3/s (S-P-13343), {_GOLD_2024}; comfort ventilation, placed by flow"),
    (_GOLD, "Swegon GOLD RX 120 air handling unit"): EpdFact(31680, "byggnad", f"GOLD RX 120: Design airflow rate 8.80 m3/s (S-P-13344), {_GOLD_2024}; comfort ventilation, placed by flow"),
})


def reg_stem(reg_no: str) -> str:
    """Registration number without its version suffix: 'EPD-IES-0025633:002'
    -> 'EPD-IES-0025633'."""
    return (reg_no or "").split(":", 1)[0].strip()


def facts_for(reg_no: str, name: str = "") -> EpdFact | None:
    stem = reg_stem(reg_no)
    product = PRODUCT_FACTS.get((stem, (name or "").strip()))
    if product is not None:
        return product
    return EPD_FACTS.get(stem)


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
