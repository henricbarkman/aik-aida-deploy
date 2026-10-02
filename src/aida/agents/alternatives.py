"""Alternatives agent: finds climate-optimized and reuse alternatives per component.

Uses pre-categorized Environdec EPD data to give the LLM real product-specific
GWP values. The LLM acts as expert, selecting and reasoning about the best
alternatives from the EPD data it receives.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

logger = logging.getLogger(__name__)

from aida.agents.aggregate import _number, article_price
from aida.api_client import (
    DEFAULT_MODEL,
    EFFORT_HIGH,
    EFFORT_MEDIUM,
    REASONING_MAX_TOKENS,
    call_model,
    extract_text,
    get_client,
)
from aida.data.climate_data import (
    names_built_in_storage,
    normalize_component_name,
    resolve_category,
)
from aida.data.nordic_supply import availability_label, nordic_supplier
from aida.errors import UserFacingError
from aida.llm_json import extract_json_object, extract_json_value
from aida.models import (
    Alternative,
    AlternativesResult,
    Baseline,
    ComponentAlternatives,
    NeedsAnalysis,
    Project,
)
from aida.name_match import match_key, tokens

EPD_ALTERNATIVES_PATH = Path(__file__).parent.parent / "data" / "epd_alternatives.json"

# The catalog stores every validated EPD per category (so baseline tier 2 can
# compute an honest upper-half median over the full GWP distribution). For the
# alternatives prompt we only want the best candidates to suggest, so we slice
# the N lowest-GWP per category — per component, after narrowing to its unit.
# This keeps the per-component prompt bounded.
_MAX_ALTERNATIVES_PER_CATEGORY = 80

# How many rows in a queue must come from a Nordic supplier, when the catalog
# can supply that many. Five, because the model picks two to four alternatives
# from the queue: fewer than five and a category could still produce a set with
# nothing orderable in it, while a much larger number would start displacing
# rows the model would actually have chosen.
_NORDIC_QUOTA = 5

# Heterogeneous categories whose candidates are capped and filtered PER
# subcategory (a toilet, a tap and a basin live in "sanitet" but are not
# interchangeable alternatives). Mirrors epd_baseline_medians.
_SUBCATEGORIZED_CATEGORIES = {"sanitet", "belysning", "vitvaror", "fast_inredning",
                              "los_inredning"}

# Smallest share of a component's need a Palats listing must cover to be shown
# as a reuse alternative. A single door against a need of 70 (coverage 0.014)
# was rendered as a full-quantity row, so the table said "70 återbrukade
# dörrar" about a listing that could supply one. The figures are still
# computed for the full need (Henric 2026-08-15: stock turns over before
# procurement), which is exactly why a find that covers almost none of it is
# misleading rather than merely optimistic. Starting point 0.2; below it the
# listing is dropped from the table and reported in the log and, when nothing
# else is shown, in an info row that carries the ratio. Unknown quantities
# (Palats reports 0 when unstated, or the project counts in m2 and the listing
# in articles) are never hidden: no data is not the same as no coverage.
MIN_REUSE_COVERAGE = 0.2

# Material routing is a correctness step (the toalett/kakel mis-routing lived
# here), so it runs on the default model, not the cheap one the chat's intent
# classifier uses.
_ROUTER_MODEL = DEFAULT_MODEL

SYSTEM_PROMPT = """Du är Aidas alternativanalys-agent — en byggnadsexpert som hittar klimatsmartare alternativ till konventionella byggmaterial.

UPPDRAG:
Hjälpa förvaltare och byggledare att hitta renoveringslösningar som kraftigt minskar klimatpåverkan utan att ge avkall på praktiska behov. Varje procentenhet reduktion räknas.

Du får:
1. En komponent med baslinjevärde (Boverket Typical, konventionellt standardmaterial)
2. En lista med FAKTISKA EPD:er (Environmental Product Declarations) från Environdec-databasen, med verifierade GWP-värden. Rader märkta [EPD].
3. Ibland också en lista med ÅTERBRUKSANNONSER från Palats, Karlstads kommuns interna marknadsplats för begagnat byggmaterial. Rader märkta [Palats återbruk]. De är redan filtrerade på produkttyp och lagersaldo innan du får dem.

Din uppgift:
1. Rangordna EPD:er och Palats-annonser TILLSAMMANS i en lista. Återbruk och klimatoptimerat nyinköp är likvärdiga alternativ som ska vägas mot varandra på matchning, klimat, pris och praktiska behov.
2. Välj de 2-4 mest relevanta EPD-alternativen, och ta med VARJE Palats-annons i listan om den inte är uppenbart fel produkt för komponenten.
3. Beräkna total CO2e baserat på EPD-värdet × antal enheter. För Palats-annonser: använd de CO2e- och prissiffror som står på raden, räkna inte om dem.
4. Resonera om varför varje alternativ är bättre eller sämre — beskriv BÅDE klimatvinsten och hur det uppfyller praktiska behov. Det gäller återbruk lika mycket som nyinköp: säg vad annonsen är, hur den passar komponenten, vad täckningen och priset betyder i praktiken och vad förvaltaren behöver kontrollera (skick, mått, antal) innan den kan räknas in.

PRINCIPER FÖR ALTERNATIV:
- Ett nytt alternativ (EPD) ska ha lägre CO2e än baslinjen. Systemet tar bort EPD-alternativ på eller över baslinjen och säger själv till användaren när katalogen bara har sådana, så föreslå dem inte: de tar en plats som ett riktigt alternativ kunde haft. Bland alternativen under baslinjen gäller att användaren optimerar TOTALEN över hela projektet, inte per komponent: ett alternativ med mindre besparing men bättre funktion, pris eller leveranskedja kan vara rätt val, så visa spridningen med tydlig jämförelse i reasoning. Rangordna med lägst CO2e först så användaren ser besparingen.
- Uttryckta behov är oförhandlingsbara — inget alternativ som inte uppfyller dem.
- Resonera om hur alternativen möter behov: både uttryckta och antagna (ljudmiljö, inomhusklimat, underhåll, estetik, arbetsmiljö vid installation).
- Presentera spridning i pris — det är användarens beslut att väga ekonomi mot klimat.
- Var innovativ — föreslå kombinationer som löser flera behov samtidigt, till exempel återbruk för en del av behovet och nyinköp för resten när annonsen inte täcker allt.
- Förklara installationsaspekter som påverkar totalkostnaden (enklare montering kan kompensera dyrare material).

TEKNISKA REGLER:
- VÄLJ BARA alternativ från listorna du får. Fabricera INGA egna alternativ, varken EPD:er eller återbruksannonser.
- Om ingen rad i någon lista passar komponenten, returnera en tom array [].
- Använd GWP-värdena från EPD-listan — de är GWP-fossil A1-A3 (samma metod som Boverket-baslinjen), verifierade och direkt jämförbara.
- Om ett omräknat värde visas (efter →), använd det omräknade värdet för beräkningar.
- Ange EPD-registreringsnummer i source-fältet för EPD-alternativ.
- Använd fältet "Tillgänglighet", inte "Geo", för att bedöma om en förvaltare kan köpa produkten. Geo är deklarationens giltighetsområde, inte var varan finns: Ahlsell AB deklarerar GLO.
- Är klimatskillnaden liten mellan två alternativ, välj det med "nordisk leverantör". Är skillnaden stor, välj det bästa ändå och nämn i reasoning att leverantören är utländsk.
- Minst ett av dina EPD-alternativ ska ha "nordisk leverantör" om listan innehåller något sådant.
- Välj EPD:er i projektets enhet. Systemet räknar co2e_kg från EPD:ns eget värde efteråt och jämför bara samma enhet (undantag anges i uppgiften), så en EPD i en annan enhet visas inte och en egen omräkning används inte.
- co2e_kg MÅSTE vara > 0 — alla byggmaterial har klimatpåverkan, även återbruk (transport och renovering). Returnera aldrig 0.
- Föreslå KOMPLETTA system, inte enskilda komponenter.
- alternative_type är "climate_optimized" för EPD-rader och "reuse" för Palats-rader. Aldrig "reuse" på en EPD, aldrig "climate_optimized" på en annons.
- source för en Palats-rad ska vara exakt "[Palats] palats.app/listing/<id>" med id från raden, så annonsen går att slå upp.

PRISER:
- Alla priser avser installerat pris (material + arbete) i SEK exklusive moms.
- Sätt cost_sek till 0 om du inte vet — priser hämtas automatiskt via webbsökning efteråt.
- Palats-priser är annonspriser för begagnat, inte installerat pris. Säg det i reasoning när det påverkar jämförelsen.

Svara med giltig JSON-array:
[
  {
    "name": "Produktnamn (Tillverkare)",
    "co2e_kg": <total CO2e i kg>,
    "cost_sek": <uppskattad kostnad i SEK, 0 om okänt>,
    "source": "[EPD] Environdec <registreringsnummer>",
    "reasoning": "Varför detta alternativ är bättre (klimat + praktiska behov)",
    "alternative_type": "climate_optimized"
  },
  {
    "name": "Annonsens titel (Palats återbruk, Plats)",
    "co2e_kg": <CO2e från raden>,
    "cost_sek": <pris från raden>,
    "source": "[Palats] palats.app/listing/<id>",
    "reasoning": "Vad annonsen är, hur den passar, vad täckning och pris betyder, vad som ska kontrolleras",
    "alternative_type": "reuse"
  }
]"""


def _load_epd_alternatives() -> dict[str, list[dict]]:
    """Load pre-categorized EPD alternatives, grouped by Aida category.

    Returns every positive-GWP row per category. Capping to the best-N happens
    in _select_epd_candidates instead, because it has to run AFTER the
    candidates are narrowed to the component's declared unit — a cap applied
    here spends its budget on rows the component can never be compared to.
    """
    if not EPD_ALTERNATIVES_PATH.exists():
        return {}
    try:
        with open(EPD_ALTERNATIVES_PATH) as f:
            data = json.load(f)
        result: dict[str, list[dict]] = {}
        for epd in data:
            cat = epd.get("category", "")
            if cat and epd.get("gwp_a1a3", 0) > 0:
                result.setdefault(cat, []).append(epd)
        return result
    except (json.JSONDecodeError, OSError):
        return {}


# Units that describe an area-measured building element.
_AREA_UNITS = {"m2", "m²", "kvm"}

# Units that can share one queue. Two rows may be ranked against each other only
# if their units fall in the same class, because ranking is what decides which
# alternatives a förvaltare is shown and a number in the wrong unit is not a
# better product, it is a different question answered.
#
# Mass and count share a class deliberately: a washbasin mixer declared per
# styck and one declared per kilo are both offered for the same fixture, and
# splitting them starved the sanitet bucket once already.
#
# Length is its own class, and that is the 2026-09-06 addition. Only the area
# branch used to narrow at all, so a kg-measured component drew on the whole
# bucket: 18 of `el`'s 28 rows are cables declared per linear metre, and a
# per-metre figure for a thin signal cable is numerically tiny next to a per-kilo
# one, so those 18 swept the front. The queue's top five read as savings of 30x
# to 80x against a 2.48 kg CO2e/kg typvärde, and every one of them was a phantom
# in the same way as the parquet example below.
#
# The baseline side already knew this. `epd_baseline_medians` keeps a separate
# el/lm typvärde at 24.19 next to el/kg at 2.48; only the alternatives side was
# putting the two in one list.
_UNIT_CLASSES: tuple[frozenset[str], ...] = (
    frozenset(_AREA_UNITS),
    frozenset({"lm", "m", "meter", "löpmeter"}),
    frozenset({"kg", "st", "styck", "pcs"}),
)


def _unit_class(unit: str) -> frozenset[str] | None:
    """The comparability class a unit belongs to, or None if it has no peers."""
    lowered = unit.strip().lower()
    for klass in _UNIT_CLASSES:
        if lowered in klass:
            return klass
    return None


def _epd_comparable(epd: dict) -> tuple[float, str]:
    """The (gwp, unit) pair an EPD can actually be compared against a component.

    Uses the derived functional unit for m3-declared rows, and for kg-declared
    rows whose conversion is marked `fu_basis == "areal_density"`.

    Those are the two bridges that describe the product rather than guess at the
    category. The m3 one is geometric: a facade panel IS 22 mm. The åtgång one
    is an application rate, which is what paint and render are actually
    specified by, and it is restricted by an allow-list to product families
    where such a rate exists (see AREAL_DENSITY_KG_M2). A general kg -> m2
    bridge remains refused: it rests on a category-wide density, and folding
    that in moved yttervägg/m2 from 39.4 to 207.9 the one time it was tried.
    The marker is what separates the third case from the second, since all
    three arrive here as a number and a unit.

    Both bridges are honoured on the baseline side too. A candidate ranked in m2
    against a baseline that ignored the same rows would be compared with the
    wrong reference, which is how the queue and the report end up disagreeing.

    Returning both together matters: for a converted row the comparable figure
    is the functional-unit one, and ranking on the raw declared value instead
    puts the most expensive rows at the front of the queue. A kg figure is
    numerically small for any product, so "TIMBABUILD EWS epoxy wood repair"
    sorted as 5.52 while actually costing 115.92 kg CO2e/m2 — seven times the
    floor baseline it was being offered as an improvement on.
    """
    unit = str(epd.get("unit", "")).lower()
    gwp = epd.get("gwp_a1a3", 0)
    # A third bridge since HENRIC-3363, and the same kind as the m3 one: a
    # steel row per kg restated per metre of the component's own profile,
    # whose weight per metre is a standard table's (steel_profiles). Only on
    # copies made for one component; no catalog row carries it.
    # And a fourth (HENRIC-3362): a precast concrete row per kg restated per
    # m², m³ or metre of the element by a weight per unit whose source the
    # copy names (betongstomme; _element_mass_copy).
    # And a fifth (HENRIC-3366): a rug's or a curtain's per-m² row restated
    # per piece by the size the component's name states (_textile_rows).
    # And a sixth (HENRIC-3371): a levelling compound's or a membrane's kg row
    # restated per m² by its own datasheet's application rate (_coverage_rows).
    if unit == "m3" or epd.get("fu_basis") in ("areal_density", "profile_mass",
                                               "element_mass", "piece_area",
                                               "coverage"):
        fu_gwp = epd.get("gwp_per_functional_unit")
        fu_unit = epd.get("functional_unit")
        if fu_unit and isinstance(fu_gwp, (int, float)):
            return float(fu_gwp), str(fu_unit).lower()
    return float(gwp if isinstance(gwp, (int, float)) else 0), unit


def _select_epd_candidates(
    epds: list[dict], project_unit: str, category: str,
) -> list[dict]:
    """Narrow a category's EPDs to those this component can be compared against,
    then keep the best-N by GWP.

    Unit has to be settled before GWP is, for two reasons.

    It decides whether a saving is real. A kg-declared EPD offered to an
    area-measured component produces a phantom: 0.57 kg CO2e/kg parquet reads as
    a 97% cut against a 17.4 kg CO2e/m2 baseline. The prompt used to paper over
    this by asking the model for "en rimlig omräkning" — a density guess that
    epd_baseline_medians (see its m3 comment) explicitly refuses to make for the
    baseline. The alternatives side should not be making it either.

    It also decides what the cap can see. Ranking by GWP across mixed units puts
    the kg rows first, because a per-kg figure is numerically smaller than a
    per-m2 one for the same product. The golv bucket spent 40 of its 80 slots on
    kg-declared epoxy pipe entries and cut 56 real m2 floors to do it.

    Only area-measured components are narrowed. st/kg components (fixtures,
    appliances) legitimately draw on both units, and narrowing that side starved
    the sanitet bucket once already.
    """
    klass = _unit_class(project_unit)
    if klass is None:
        matching = epds
    else:
        matching = [e for e in epds if _epd_comparable(e)[1] in klass]
        if not matching:
            # Better a unit-mismatched suggestion than none at all, but say so.
            logger.warning(
                "No EPDs in category %s share a unit class with %r; falling "
                "back to the full bucket (%d rows, mixed units)",
                category, project_unit, len(epds),
            )
            matching = epds

    def rank(e: dict) -> float:
        return _epd_comparable(e)[0]

    if category in _SUBCATEGORIZED_CATEGORIES:
        # Heterogeneous categories are capped PER SUBCATEGORY — otherwise
        # low-GWP taps fill the category-wide cap and starve a WC-stol
        # component of toilets (it then gets offered a tap or a toilet seat).
        by_sub: dict[str, list[dict]] = {}
        for e in matching:
            by_sub.setdefault(e.get("subcategory", ""), []).append(e)
        flat: list[dict] = []
        for sub_epds in by_sub.values():
            sub_epds.sort(key=rank)
            flat.extend(sub_epds[:_MAX_ALTERNATIVES_PER_CATEGORY])
        return _apply_nordic_quota(flat, matching, rank, category)
    selected = sorted(matching, key=rank)[:_MAX_ALTERNATIVES_PER_CATEGORY]
    return _apply_nordic_quota(selected, matching, rank, category)


def _split_subtype_rows(epds: list[dict], proj_comp, category: str) -> list[dict]:
    """Rows of a category with a split subtype (ventilation/aggregat) narrowed
    to the side the component is on.

    Unlike the heterogeneous categories, where the model picks from a balanced
    set, the two sides here are not alternatives to each other in either
    direction: a 50 kg duct offered against an air handling unit reads as a
    97 % saving, and a 5 000 kg unit against a duct is noise. A component in a
    unit the subtype is not counted in (a whole system per m2) sees the
    category side, the same rule the baseline lookup uses.
    """
    from aida.data.epd_baseline_medians import _SPLIT_SUBCATEGORIES
    from aida.data.palats_client import component_subcategory

    splits = _SPLIT_SUBCATEGORIES.get(category)
    if not splits:
        return epds
    sub = component_subcategory(proj_comp.name, category)
    if sub not in splits or proj_comp.unit not in splits[sub]:
        sub = ""
    return [e for e in epds
            if (e.get("subcategory", "") if e.get("subcategory", "") in splits else "") == sub]


def _aggregat_size_rows(rows: list[dict], proj_comp, category: str) -> tuple[list[dict], str]:
    """Aggregat rows narrowed to units that can do this component's job, or
    the reason there are none.

    The baseline is sized (per airflow, or per size class), and a row per piece
    is only an alternative to it if it is the same size. Unnarrowed, a school's
    3 000 m3/h unit was offered apartment units of 270-564 kg as an 85-93 %
    saving against its 3 720 kg baseline (found in review, 2026-09-30).

    The size is the baseline's own reading (aggregat_typvärde), so the two
    steps cannot disagree about one component. A flow keeps the rows whose own
    nominal flow is at least as large: a smaller unit cannot ventilate the same
    rooms, and a larger one is a real if heavier option. No margin is added
    below the flow, since the EPDs give no ground for one. Rows without a
    stated flow cannot be checked and are left out, apartment units included.
    Without a flow, a size class keeps the rows of that class, also when the
    class's typvärde is withheld. With neither, the rows are left as they were;
    the baseline is then an estimate as well.
    """
    from aida.claims import format_value
    from aida.data.epd_baseline_medians import aggregat_typvärde
    from aida.data.palats_client import component_subcategory

    if category != "ventilation" or proj_comp.unit != "st" or not rows:
        return rows, ""
    if component_subcategory(proj_comp.name, category) != "aggregat":
        return rows, ""
    size = aggregat_typvärde(proj_comp.name, proj_comp.usage_context, proj_comp.quantity)
    flow, klass = size.get("airflow_m3h"), size.get("klass")
    label = {"lägenhet": "lägenhetsaggregat", "byggnad": "byggnadsaggregat"}
    if flow is None:
        if not klass:
            return rows, ""
        sized = [e for e in rows if e.get("aggregat_class") == klass]
        if sized:
            return sized, ""
        return [], f"Katalogen har inga EPD:er för {label[klass]}."
    sized = [e for e in rows if (e.get("airflow_m3h") or 0) >= flow]
    if sized:
        return sized, ""
    largest = max((e.get("airflow_m3h") or 0 for e in rows), default=0)
    if not largest:
        return [], "Ingen av katalogens EPD:er för ventilationsaggregat anger luftflöde."
    return [], (
        f"Katalogens största aggregat med angivet luftflöde klarar {format_value(largest)} m³/h, "
        f"och komponenten behöver {format_value(flow)} m³/h. Ett mindre aggregat är inget "
        f"alternativ, eftersom det inte ventilerar samma lokaler."
    )


# Words that name a stud or framing profile in an EPD. The innervägg bucket is
# almost all plasterboard per m2, so without narrowing a "Reglar" component was
# offered gypsum boards; a board is not an alternative to the frame it is
# screwed to. Still read here because older builds filed steel profiles under
# innervägg, and a stud should see them wherever they sit.
# Matched from the start of a word: bare "stud" is inside "Gyproc® Studio", an
# acoustic plasterboard.
_STUD_ROW_RE = re.compile(r"\b(studs?\b|profil|regel|reglar|c-section|framing)")

# The stomme families a stud is compared across. Timber and steel both, in
# either direction: swapping a steel stud for a timber one is exactly the kind
# of saving the tool exists to show (HENRIC-3290).
_STUD_FAMILIES = ("virke", "stålregel")

# Families whose m3 rows can be restated per löpmeter or m2 of a member, from
# the member's own section or thickness. Solid material only; see
# epd_baseline_medians._GEOMETRY_BRIDGE_SUBCATEGORIES for why steel is not.
_SOLID_FAMILIES = {"virke", "limträ", "konstruktionsskiva"}

# Families declared per kg whose rows can be restated per metre of a member,
# from the weight per metre of its named standard profile (HENRIC-3363).
_STEEL_FAMILIES = {"konstruktionsstål", "stålregel"}

# How each family is named in a reason, in Swedish.
_FAMILY_LABELS = {
    "virke": "sågat virke", "limträ": "limträ, KL-trä och fanerträ",
    "konstruktionsskiva": "konstruktionsskivor", "konstruktionsstål":
    "konstruktionsstål", "stålregel": "stålreglar", "betong": "betongstomme",
}

_LENGTH_UNITS = {"lm", "m", "meter", "löpmeter"}
_COUNT_UNITS = {"st", "styck", "pcs"}


def _is_frame_component(proj_comp, category: str) -> bool:
    """A stud, batten or structural board routed to the part it stands in."""
    from aida.data.climate_data import _FRAME_HOST_CATEGORIES, names_frame_member

    return category in _FRAME_HOST_CATEGORIES and names_frame_member(proj_comp.name)


def _bridge_member_rows(rows: list[dict], proj_comp) -> list[dict]:
    """Copies of solid-family m3 rows restated per unit of THIS member.

    "Reglar 45x95" in lm is 0.004275 m3 per metre, so a sawmill's 25.6 kg
    CO2e/m3 is 0.109 per metre of that stud. Same geometric bridge as the
    facade one in build_epd_alternatives (a panel IS 22 mm), with the
    dimension read from the component instead of the category: a stud has no
    typical section the way a facade panel has a typical thickness. Copies, so
    the catalog rows loaded once per analysis are never written to.
    """
    from aida.data.unit_conversion import member_volume_per_unit

    geometry = member_volume_per_unit(proj_comp.name, proj_comp.unit)
    if not geometry:
        return []
    factor, label = geometry
    unit = proj_comp.unit.strip().lower()
    out = []
    for e in rows:
        gwp = e.get("gwp_a1a3")
        if (e.get("unit") != "m3" or e.get("subcategory") not in _SOLID_FAMILIES
                or not isinstance(gwp, (int, float))):
            continue
        bridged = dict(e)
        bridged["gwp_per_functional_unit"] = round(gwp * factor, 4)
        bridged["functional_unit"] = unit
        bridged["fu_basis"] = "dimension"
        bridged["fu_note"] = label
        out.append(bridged)
    return out


# The shape a konstruktionsstål row declares: open sections (I, H, U, angles,
# merchant bars) or hollow sections. A hollow-section EPD is no alternative to
# an HEA beam, nor a beam EPD to a VKR column: swapping one for the other is a
# structural redesign, not a choice of supplier. Read from the name first.
_HOLLOW_ROW_RE = re.compile(r"hollow|\btubes?\b|\bpipes?\b")
_OPEN_ROW_RE = re.compile(r"\bbeams?\b|i-section|h-beam|\bangles?\b|\bchannels?\b|merchant bar")
# Rows whose name states no shape, placed by what the EPD itself says the
# product is (technology description and applicability, data.environdec.com,
# read 2026-10-01). (owner, product name or "" for any, shape):
#   kardemir: "hot-rolled ... IPE, NPI, NPU, HEA, HEB, angles"
#   kocaer: "structural steel profiles" rolled from steel billet
#   jindal ("Average Structural Steel Product" only): hot rolled from billets
#     and blooms, applicability "the channels are ideal for frames ..."
#   melewar: "pipe forming or roll forming ... structural hollow sections"
# Every other shape-less row is left out of a profile's comparison: heavy plate
# (SIMAXX, SIQUAL 0577), fabricated members of any section (GOLDBECK, Chenxin,
# Grædstrup), a light-gauge framing system (CINTAC Metalcon) and a scrap-route
# "Structural Steel" with no product description (review 2026-10-01: the
# first version let every one of them meet both shapes).
_ROW_SHAPE_BY_OWNER = (
    ("kardemir", "", "open"),
    ("kocaer", "", "open"),
    ("jindal", "average structural steel product", "open"),
    ("melewar", "", "hollow"),
)


def _row_shapes(row: dict) -> set[str]:
    """{"open"}, {"hollow"}, both, or none for a konstruktionsstål row."""
    name = str(row.get("name") or "").lower()
    shapes = set()
    if _HOLLOW_ROW_RE.search(name):
        shapes.add("hollow")
    if _OPEN_ROW_RE.search(name):
        shapes.add("open")
    if not shapes:
        owner = str(row.get("owner") or "").lower().replace("̇", "")
        shapes = {shape for who, product, shape in _ROW_SHAPE_BY_OWNER
                  if who in owner and product in name}
    return shapes


def _fits_profile_shape(row: dict, profile) -> bool:
    """True when a konstruktionsstål row declares the profile's own shape."""
    if profile.family != "konstruktionsstål":
        return True
    shape = "hollow" if profile.designation.startswith(("VKR", "KKR")) else "open"
    return shape in _row_shapes(row)


def _bridge_profile_rows(rows: list[dict], proj_comp) -> tuple[list[dict], str]:
    """Copies of the steel family's kg rows restated per metre of THIS profile.

    The steel twin of _bridge_member_rows (HENRIC-3363). "Stålbalk HEA 200" in
    lm is 42,3 kg per metre (EN 10365), so a mill's 0,74 kg CO2e/kg is 31,3 per
    metre of that beam. The weight comes from steel_profiles' standard tables
    and the designation from the name; no designation, no copies, and the
    second value is the Swedish reason (or "" when the name names no profile).
    Rows of the profile's own family only, so a stud's 0,59 kg/m never meets
    a beam mill's figure, and of its own shape (_fits_profile_shape).
    """
    from aida.data.steel_profiles import profile_mass

    if proj_comp.unit.strip().lower() not in _LENGTH_UNITS:
        return [], ""
    profile, why = profile_mass(proj_comp.name)
    if not profile:
        return [], why
    unit = proj_comp.unit.strip().lower()
    out = []
    for e in rows:
        gwp = e.get("gwp_a1a3")
        # The innervägg profile rows a stud also sees carry no stomme family;
        # all of them are steel studs (see _stomme_rows).
        family = e.get("subcategory") or "stålregel"
        if (e.get("unit") != "kg" or family != profile.family
                or not isinstance(gwp, (int, float)) or not _fits_profile_shape(e, profile)):
            continue
        bridged = dict(e)
        bridged["gwp_per_functional_unit"] = round(gwp * profile.kg_per_m, 4)
        bridged["functional_unit"] = unit
        bridged["fu_basis"] = "profile_mass"
        bridged["fu_note"] = profile.label
        bridged["kg_per_m"] = profile.kg_per_m
        bridged["profile"] = profile.designation
        bridged["profile_source"] = profile.source
        out.append(bridged)
    return out, ""


def _stomme_rows(epd_data: dict[str, list[dict]], proj_comp,
                 category: str) -> tuple[list[dict], str]:
    """Rows a frame member can be compared against, or the reason there are none.

    Replaces `_stud_rows` (HENRIC-3290, second round). Returns (rows, reason);
    `reason` is "" whenever rows are returned, and for any category but stomme
    the function is a no-op.

    Within stomme a component only ever meets its own family: a stud meets
    timber and steel studs, a glulam beam glulam, a board boards. A median-free
    queue of beams, boards and sections sorted by GWP would put a per-kg steel
    figure next to a per-metre stud and call the smaller number better.

    Units are settled here rather than left to _select_epd_candidates, because
    its fallback ("better a unit-mismatched suggestion than none") is exactly
    wrong for a frame: the answer to a stud in st is a question, not a list.
    - st: nothing. A piece of timber has no length, so nothing is comparable;
      the reason asks for a section and a number of löpmeter.
    - lm / m2: native rows in that unit, plus the solid families' m3 rows
      bridged by the member's own section or thickness. Native m2 rows are left
      out: every one declares a board or element at a thickness its name does
      not state, so it cannot be set against "OSB-skiva 12 mm". In lm also the
      steel families' kg rows, bridged by the weight per metre of a standard
      profile the name states ("HEA 200", "C-regel 70"; HENRIC-3363).
    - kg / m3: native rows in that unit.
    """
    from aida.data.climate_data import names_stud
    from aida.data.palats_client import component_subcategory

    if category != "stomme":
        return epd_data.get(category, []), ""

    stomme = epd_data.get("stomme", [])
    if names_stud(proj_comp.name):
        pool = [e for e in stomme if e.get("subcategory") in _STUD_FAMILIES]
        pool += [e for e in epd_data.get("innervägg", [])
                 if _STUD_ROW_RE.search(e.get("name", "").lower())]
        seen: set = set()
        deduped = []
        for e in pool:
            key = e.get("uuid") or _row_key(e)
            if key not in seen:
                seen.add(key)
                deduped.append(e)
        pool, family = deduped, "reglar"
        # Across materials only per metre of stud. A kilo of timber and a kilo
        # of steel do different amounts of work, so in kg or m3 a stud meets
        # its own material: "Stålreglar" steel, anything else timber.
        # The innervägg profile rows carry no stomme family; all are steel.
        if proj_comp.unit.strip().lower() in ("kg", "m3"):
            own = component_subcategory(proj_comp.name, "stomme")
            own = own if own == "stålregel" else "virke"
            pool = [e for e in pool if (e.get("subcategory") or "stålregel") == own]
            family = own
    else:
        family = component_subcategory(proj_comp.name, "stomme")
        if not family:
            return [], (
                "Namnet säger inte vilket stommaterial det gäller, och alternativ "
                "jämförs bara inom samma material. Ange material och dimension, "
                "till exempel \"Limträbalk 90x315\", \"Stålbalk HEA 200\" eller "
                "\"OSB-skiva 12 mm\"."
            )
        pool = [e for e in stomme if e.get("subcategory") == family]
    label = _FAMILY_LABELS.get(family, family)
    if not pool:
        return [], f"Katalogen har inga EPD:er för {label}, så ingen jämförelse görs."
    if family == "betong":
        return _concrete_rows(pool, proj_comp)

    unit = proj_comp.unit.strip().lower()
    if unit in _COUNT_UNITS and family == "konstruktionsstål":
        return [], (
            f"Komponenten är angiven i styck, och en balk eller pelare i styck "
            f"säger inget om längd eller profil. EPD:erna för {label} anges per kg. "
            f"Ange profilen (till exempel HEA 200 eller VKR 100x100x5) och antal "
            f"löpmeter (antal × längd) så kan de jämföras."
        )
    if unit in _COUNT_UNITS:
        return [], (
            f"Komponenten är angiven i styck, och en regel eller balk i styck "
            f"säger inget om längd eller dimension. EPD:erna för {label} anges "
            f"per m³, löpmeter eller kg. Ange dimensionen (till exempel 45x95) "
            f"och antal löpmeter (antal × längd) så kan de jämföras."
        )
    if unit == "m3":
        native = [e for e in pool if _epd_comparable(e)[1] == "m3"]
    elif unit in _AREA_UNITS:
        native = []
    else:
        klass = _unit_class(unit)
        native = [e for e in pool if klass is not None and _epd_comparable(e)[1] in klass]
    profile_rows, profile_why = _bridge_profile_rows(pool, proj_comp)
    rows = native + _bridge_member_rows(pool, proj_comp) + profile_rows
    if rows:
        return rows, ""

    units = sorted({str(e.get("unit", "")) for e in pool})
    declared = ", ".join(units)
    steel = any(e.get("subcategory") in _STEEL_FAMILIES for e in pool)
    if unit in _LENGTH_UNITS and steel:
        # A steel member in löpmeter whose profile could not be read. Said
        # with what would make it comparable, the way a stud without a
        # section is: the designation, from which the standard table gives
        # the weight per metre (HENRIC-3363), or the quantity in kg.
        why = profile_why or (
            "Vikten per meter beror på profilen, och namnet anger ingen "
            "standardprofil.")
        return [], (
            f"Katalogen har {len(pool)} EPD:er för {label}, men de anges per "
            f"{declared} och komponenten i {proj_comp.unit}. {why} Ange profilen "
            f"i namnet, till exempel \"HEA 200\", \"IPE 200\", \"VKR 100x100x5\" "
            f"eller \"Stålregel 70\", så räknas vikten per meter ur standardtabellen, "
            f"eller ange mängden i kg."
        )
    if unit in _LENGTH_UNITS and any(e.get("subcategory") in _SOLID_FAMILIES for e in pool):
        return [], (
            f"EPD:erna för {label} anges per {declared}. För att räkna om till "
            f"löpmeter behövs tvärsnittet i namnet, till exempel \"45x95\" eller "
            f"\"90x315\". Ange det så jämförs de."
        )
    if unit in _AREA_UNITS and any(e.get("subcategory") in _SOLID_FAMILIES for e in pool):
        return [], (
            f"EPD:erna för {label} anges per {declared}. För att räkna om till m² "
            f"behövs tjockleken i namnet, till exempel \"OSB-skiva 12 mm\" eller "
            f"\"KL-trä 200 mm\". Ange den så jämförs de."
        )
    return [], (
        f"Katalogen har {len(pool)} EPD:er för {label}, men de anges per "
        f"{declared} och komponenten i {proj_comp.unit}. Att räkna om kräver en "
        f"vikt per meter eller m² som beror på profilen, så ingen jämförelse görs. "
        f"Ange mängden i kg för att jämföra, eller läs baslinjen som den står."
    )


def _units_text(rows: list[dict]) -> str:
    """"7 per kg och 1 per m3", for a reason that says what the catalog has."""
    counts: dict[str, int] = {}
    for e in rows:
        unit = str(e.get("unit", ""))
        counts[unit] = counts.get(unit, 0) + 1
    parts = [f"{n} per {u}" for u, n in sorted(counts.items())]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " och " + parts[-1]


def _element_mass_copy(e: dict, kg_per_unit: float, unit: str, note: str,
                       source: str, own: bool = False) -> dict:
    """A precast row per kg restated per unit of the element (HENRIC-3362).

    Same kind of bridge as _bridge_profile_rows: the GWP is the EPD's, the mass
    per unit is named with where it comes from. `own` marks a mass the EPD
    states for its own product rather than one read from the component."""
    bridged = dict(e)
    bridged["gwp_per_functional_unit"] = round(e["gwp_a1a3"] * kg_per_unit, 4)
    bridged["functional_unit"] = unit
    bridged["fu_basis"] = "element_mass"
    bridged["fu_note"] = note
    bridged["fu_note_own"] = own
    bridged["kg_per_unit"] = round(kg_per_unit, 2)
    bridged["mass_source"] = source
    return bridged


def _volume_copy(e: dict, factor: float, unit: str, note: str) -> dict:
    """A solid element's m3 row restated per m² or metre by the component's
    own thickness or section, like _bridge_member_rows does for timber."""
    bridged = dict(e)
    bridged["gwp_per_functional_unit"] = round(e["gwp_a1a3"] * factor, 4)
    bridged["functional_unit"] = unit
    bridged["fu_basis"] = "dimension"
    bridged["fu_note"] = note
    return bridged


def _concrete_rows(pool: list[dict], proj_comp) -> tuple[list[dict], str]:
    """Rows a precast concrete component can be compared against, or the reason
    there are none (HENRIC-3362). `pool` is the catalog's betong family.

    Per element kind (betongstomme.element_kind): a hollow-core slab meets
    hollow-core slabs, a solid slab solid slabs, a beam or column beams and
    columns. The EPDs are mostly per tonne, so the unit decides the rest:

    - kg: the kind's kg rows as declared.
    - m3: the kind's m3 rows; for solid slabs and beams also the kg rows at
      2 500 kg/m³ (solid concrete, Svensk Betong). Not for hollow-core, whose
      weight per m³ depends on its voids.
    - m2, solid slab: the thickness in the name times the m3 rows, and times
      2 500 kg/m³ for the kg rows.
    - m2, hollow-core: rows whose own EPD states a thickness within 5 mm of the
      component's, per m² as declared or by the weight per m² the EPD states;
      and, when the name states the slab's weight ("290 kg/m2"), every kg row
      by that weight. Never a weight derived from the thickness: Svensk Betong
      gives 255-330 kg/m² for a 200 mm slab, depending on the maker's profile.
    - lm, beam: the section in the name, the same way as m2 for a solid slab.
    - anything else: a reason that says what to state.
    """
    from aida.data import betongstomme as bs

    name = proj_comp.name
    unit = proj_comp.unit.strip().lower()
    if bs.is_reinforcement(name):
        return [], ("Armering är stål och inget betongelement, och katalogen har "
                    "inga EPD:er för armeringsstål, så ingen jämförelse görs.")
    if bs.is_lightweight(name):
        return [], ("Katalogens betong-EPD:er gäller element av vanlig betong. "
                    "Lättklinker är lättballastbetong, som väger mindre per m³, så "
                    "ingen jämförelse görs.")
    if bs.is_cast_in_place(name):
        return [], ("Katalogen har EPD:er för prefabricerade betongelement "
                    "(håldäck, massiva bjälklag, balkar och pelare), inte för "
                    "platsgjuten betong, så ett platsgjutet bjälklag jämförs inte "
                    "med dem.")
    kind = bs.element_kind(name)
    if not kind:
        return [], ("Namnet säger inte vilket sorts betongelement det är, och "
                    "håldäck, massiva bjälklag och balkar jämförs bara med sin "
                    "egen sort. Skriv till exempel \"Håldäck 200 mm\", "
                    "\"Massivbjälklag 200 mm\" eller \"Betongbalk 300x500\".")
    label = bs.KIND_LABELS[kind]
    rows = [e for e in pool if bs.element_kind(e.get("name", "")) == kind]
    if not rows:
        return [], f"Katalogen har inga EPD:er för {label}, så ingen jämförelse görs."
    have = f"Katalogen har {len(rows)} EPD:er för {label}, {_units_text(rows)}."
    kg_rows = [e for e in rows if e.get("unit") == "kg"
               and isinstance(e.get("gwp_a1a3"), (int, float))]
    m3_rows = [e for e in rows if e.get("unit") == "m3"
               and isinstance(e.get("gwp_a1a3"), (int, float))]
    solid = kind in (bs.MASSIV, bs.BALK)
    density_note = f"{bs.SOLID_DENSITY_KG_M3:g} kg/m³ massiv betong"

    if unit == "kg":
        return kg_rows, "" if kg_rows else f"{have} Ingen av dem anges per kg."
    if unit == "m3":
        out = list(m3_rows)
        if solid:
            out += [_element_mass_copy(e, bs.SOLID_DENSITY_KG_M3, "m3", density_note,
                                       bs.SOLID_DENSITY_SOURCE) for e in kg_rows]
        if out:
            return out, ""
        return [], (f"{have} Ett håldäcks vikt per m³ beror på hålen, så EPD:erna "
                    f"per kg räknas inte om till m³. Ange mängden i kg.")

    if unit in _AREA_UNITS:
        if kind == bs.BALK:
            return [], (f"{have} Balkar och pelare räknas i löpmeter med tvärsnittet "
                        f"i namnet (till exempel \"Betongbalk 300x500\"), i m³ "
                        f"eller i kg, inte i m².")
        thickness = bs.slab_thickness_mm(name)
        if kind == bs.MASSIV:
            if not thickness:
                return [], (f"{have} För att räkna om till m² behövs bjälklagets "
                            f"tjocklek i namnet, till exempel \"Massivbjälklag "
                            f"200 mm\". Ange den så jämförs de.")
            t_mm, how = thickness
            t_m = t_mm / 1000
            out = [_volume_copy(e, t_m, unit, how) for e in m3_rows]
            out += [_element_mass_copy(
                e, t_m * bs.SOLID_DENSITY_KG_M3, unit,
                f"{how} × {density_note}", bs.SOLID_DENSITY_SOURCE) for e in kg_rows]
            return out, "" if out else f"{have} Ingen av dem går att räkna om till m²."
        # Hollow-core.
        # A row whose EPD states its own thickness and weight is compared at
        # its own weight: the alternative is that maker's slab. Others only
        # by the weight the name states, as if they weighed the same.
        out = []
        stated = bs.stated_kg_per_m2(name)
        t_mm = thickness[0] if thickness else None
        for e in rows:
            own_t = e.get("element_thickness_mm")
            same = (t_mm is not None and isinstance(own_t, (int, float))
                    and abs(own_t - t_mm) <= bs.THICKNESS_TOLERANCE_MM)
            if e.get("unit") in _AREA_UNITS:
                if same:
                    out.append(e)
            elif e.get("unit") != "kg" or e not in kg_rows:
                continue
            elif same and isinstance(e.get("element_kg_per_m2"), (int, float)):
                out.append(_element_mass_copy(
                    e, e["element_kg_per_m2"], unit,
                    f"EPD:ns egen vikt {e['element_kg_per_m2']:g} kg/m² vid "
                    f"{own_t:g} mm", e.get("element_facts_source", ""), own=True))
            elif stated:
                out.append(_element_mass_copy(
                    e, stated, unit, f"vikt {stated:g} kg/m² enligt namnet",
                    "komponentens namn"))
        if out:
            return out, ""
        if not thickness:
            return [], (f"{have} Ett håldäck per m² går bara att jämföra med håldäck "
                        f"av samma tjocklek. Ange tjockleken i namnet, till exempel "
                        f"\"Håldäck 200 mm\" eller \"HD/F 120/20\", och gärna vikten "
                        f"per m² från leverantören (\"Håldäck 200 mm, XXX kg/m2\").")
        return [], (f"{have} Ingen av dem gäller ett håldäck på {t_mm:g} mm per m². "
                    f"EPD:erna per kg går inte att räkna om utan håldäckets vikt per "
                    f"m², och den följer inte av tjockleken: {bs.hdf_weight_text(t_mm)}. "
                    f"Ange leverantörens vikt per m² efter tjockleken i namnet, i "
                    f"formen \"Håldäck {t_mm:g} mm, XXX kg/m2\", eller mängden i kg, "
                    f"så jämförs de.")

    if unit in _LENGTH_UNITS:
        if kind != bs.BALK:
            return [], (f"{have} Ett bjälklag räknas i m² med tjockleken i namnet "
                        f"(till exempel \"Håldäck 200 mm\"), inte i löpmeter.")
        section = bs.beam_section_mm(name)
        if not section:
            return [], (f"{have} För att räkna om till löpmeter behövs tvärsnittet i "
                        f"namnet, till exempel \"Betongbalk 300x500\". Ange det så "
                        f"jämförs de.")
        w, h = section
        area = (w / 1000) * (h / 1000)
        how = f"tvärsnitt {w:g}×{h:g} mm"
        out = [_volume_copy(e, area, unit, how) for e in m3_rows]
        out += [_element_mass_copy(e, area * bs.SOLID_DENSITY_KG_M3, unit,
                                   f"{how} × {density_note}", bs.SOLID_DENSITY_SOURCE)
                for e in kg_rows]
        return out, "" if out else f"{have} Ingen av dem går att räkna om till löpmeter."

    what = ("tvärsnittet och antal löpmeter (till exempel \"Betongbalk 300x500\")"
            if kind == bs.BALK else
            "tjockleken och antal m² (till exempel \"Håldäck 200 mm\")")
    return [], (f"{have} Komponenten är angiven i {proj_comp.unit}, som inte säger hur "
                f"mycket betong det är. Ange {what}, eller mängden i kg.")


# Category keys that are not words, as a reader should see them.
_CATEGORY_TEXT = {"los_inredning": "lös inredning", "fast_inredning": "fast inredning",
                  "undertak": "undertak och akustik"}

# How each loose-furniture subcategory is named in a reason, in Swedish.
_FURNITURE_LABELS = {
    "stol": "stolar och pallar", "kontorsstol": "kontorsstolar",
    "bord": "bord och skrivbord",
    "förvaring": "förvaring (skåp, hyllor, garderober)",
    "soffa": "soffor och fåtöljer", "akustik": "fristående skärmar",
    "matta": "mattor", "gardin": "gardiner",
}


def _built_in_storage_rows(epd_data: dict[str, list[dict]],
                           proj_comp) -> tuple[list[dict], str]:
    """Rows built-in storage can be compared against, or the reason there are
    none (HENRIC-3394).

    "Platsbyggd garderob" is fast inredning since HENRIC-3367, and until
    2026-10-02 it then met the whole category: kitchen fronts, worktops,
    sinks and bathroom cabinets. A built-in wardrobe, shelf or cabinet is
    joinery, a carcass with doors, so its only comparable rows are the
    catalog's cabinet carcasses (fast_inredning/köksskåp).

    Units: kg meets kg, which compares the material of the carcass. A carcass
    per piece is a kitchen cabinet of a size the EPD does not set against a
    wardrobe's, and nothing in the catalog is declared per metre or per m² of
    built-in storage, so st, lm and m² get the reason instead of a list.
    """
    pool = [e for e in epd_data.get("fast_inredning", [])
            if e.get("subcategory") == "köksskåp"]
    unit = proj_comp.unit.strip().lower()
    if unit == "kg":
        rows = [e for e in pool if _epd_comparable(e)[1] == "kg"]
        if rows:
            return rows, ""
    per_kg = sum(1 for e in pool if _epd_comparable(e)[1] == "kg")
    if not per_kg:
        return [], (
            "Katalogen har ingen miljödeklaration för platsbyggd förvaring "
            "(garderober, skåp och hyllor byggda på plats) och inga för "
            "skåpstommar per kg, så ingen jämförelse med nyköp görs. "
            "Bänkskivor, diskbänkar och köksluckor jämförs inte.")
    return [], (
        "Katalogen har ingen miljödeklaration för platsbyggd förvaring "
        "(garderober, skåp och hyllor byggda på plats). Det som liknar mest är "
        f"{per_kg} deklaration{'er' if per_kg != 1 else ''} för skåpstommar till "
        "kök per kg. Ett köksskåp per styck är en annan storlek än en garderob, "
        "så de jämförs bara när mängden anges i kg. Bänkskivor, diskbänkar och "
        "köksluckor jämförs inte."
    )


def _furniture_rows(epd_data: dict[str, list[dict]], proj_comp) -> tuple[list[dict], str]:
    """Rows a piece of loose furniture can be compared against, or the reason
    there are none (HENRIC-3290 del 2).

    Strict per subcategory, the way stomme is per family: thirty elevstolar are
    not replaced by a bookcase, and a queue sorted by GWP across chairs, sofas
    and lockers would put a 4 kg stool first for every one of them.

    Units are settled here, not in _select_epd_candidates, for the same reason
    as in _stomme_rows: its unit classes put kg and st together, and its
    fallback hands over the whole bucket when nothing matches. A chair in kg
    says nothing about how many chairs it is, so it meets no per-piece row and
    gets a question instead. The catalog holds no kg rows for this category
    (build_epd_alternatives.CATEGORY_DECLARED_UNITS), so st meets st and m2
    meets m2.
    """
    from aida.data.climate_data import furniture_subcategory

    sub = furniture_subcategory(proj_comp.name)
    if not sub:
        return [], (
            "Namnet säger inte vilken sorts möbel det gäller, och alternativ "
            "jämförs bara inom samma sort. Ange till exempel \"Elevstol\", "
            "\"Skrivbord\", \"Förvaringsskåp\", \"Soffa\" eller \"Golvskärm\"."
        )
    label = _FURNITURE_LABELS.get(sub, sub)
    pool = [e for e in epd_data.get("los_inredning", []) if e.get("subcategory") == sub]
    if not pool:
        return [], f"Katalogen har inga EPD:er för {label}, så ingen jämförelse med nyköp görs."
    if sub in ("matta", "gardin"):
        return _textile_rows(pool, proj_comp, label)

    unit = proj_comp.unit.strip().lower()
    if unit in _COUNT_UNITS:
        wanted = {"st"}
    elif unit in _AREA_UNITS:
        wanted = {"m2"}
    else:
        wanted = set()
    rows = [e for e in pool if _epd_comparable(e)[1] in wanted]
    if rows:
        return rows, ""
    declared = ", ".join(sorted({str(e.get("unit", "")) for e in pool}))
    return [], (
        f"EPD:erna för {label} anges per {declared}, och komponenten i "
        f"{proj_comp.unit}. Ange antalet i styck så jämförs de."
    )


def _textile_rows(pool: list[dict], proj_comp, label: str) -> tuple[list[dict], str]:
    """Rows a rug or a curtain can be compared against, or the reason there are
    none (HENRIC-3366). `pool` is the kind's own rows, all declared per m².

    - m2: the rows as declared.
    - st: copies restated per piece by the size the name states ("Matta 2x3 m"
      is 6 m² a piece, aida.data.textil.piece_area_m2). No size, no copies: a
      rug in st says nothing about how much rug it is, and the reason asks for
      the size or the quantity in m².
    """
    from aida.data.textil import piece_area_m2

    per_m2 = [e for e in pool if _epd_comparable(e)[1] == "m2"]
    unit = proj_comp.unit.strip().lower()
    example = ("\"Matta 2x3 m\"" if label == "mattor" else "\"Gardin 140x250 cm\"")
    if unit in _AREA_UNITS:
        if per_m2:
            return per_m2, ""
        return [], f"Katalogen har inga EPD:er för {label} per m²."
    if unit in _COUNT_UNITS:
        # `why` is the area's label when there is an area, the reason when not.
        area, why = piece_area_m2(proj_comp.name)
        if area:
            out = []
            for e in per_m2:
                gwp, _ = _epd_comparable(e)
                bridged = dict(e)
                bridged["gwp_per_functional_unit"] = round(gwp * area, 4)
                bridged["functional_unit"] = unit
                bridged["fu_basis"] = "piece_area"
                bridged["fu_note"] = why
                bridged["area_m2_per_unit"] = area
                bridged["per_m2"] = gwp
                out.append(bridged)
            return out, ""
        return [], (
            f"EPD:erna för {label} anges per m², och komponenten i styck. "
            + (why + " " if why else "Namnet anger ingen storlek. ")
            + f"Ange storleken i namnet med enhet, till exempel {example}, "
            f"eller mängden i m²."
        )
    return [], (
        f"EPD:erna för {label} anges per m², och komponenten i {proj_comp.unit}. "
        f"Ange mängden i m², eller antal i styck med storleken i namnet, till "
        f"exempel {example}."
    )


# How each ceiling kind is named in a reason, in Swedish.
_CEILING_LABELS = {
    "akustikplatta": "undertaksplattor", "väggabsorbent": "väggabsorbenter",
    "baffel": "bafflar och akustiköar", "bärverk": "bärverk (T-profiler)",
    "metallundertak": "metallundertak (plåtkassetter, ribbor och bärverk som ett system)",
    "gipstak": "gipstak",
}

# A gypsum ceiling is plasterboard screwed to the joists or a furring, the same
# boards innervägg holds per m2 (HENRIC-3370). Its rows are read from there by
# name, since innervägg also holds glazed partitions, insulation sold as
# "Drywall", prefab wall panels and MDF.
_PLASTERBOARD_RE = re.compile(
    r"plasterboard|gypsum|\bgips|gyproc|gyprock|\byeso\b|rigidur|\bba13\b|"
    r"chapa de drywall")
# What a board is fixed or finished with, named after the board.
_NOT_BOARD_RE = re.compile(r"adhesive|paste|putt|framing|profile")


def _plasterboard_rows(epd_data: dict[str, list[dict]]) -> list[dict]:
    return [e for e in epd_data.get("innervägg", [])
            if not e.get("subcategory")
            and _PLASTERBOARD_RE.search(e.get("name", "").lower())
            and not _NOT_BOARD_RE.search(e.get("name", "").lower())]


# An innervägg component that names its boards (HENRIC-3399): "Gipsskivor
# 12,5 mm", "Innervägg gips 2x13". A glazed partition is read before this.
_NAMES_PLASTERBOARD_RE = re.compile(r"\bgips|gyproc|plasterboard|gipsvägg|gipsskiv")


def _board_class_rows(pool: list[dict], proj_comp) -> tuple[list[dict], str]:
    """Plasterboard rows of the component's thickness class, each carrying a
    note that says which thickness was compared, or the reason there are none.

    A thinner board is a different product, with another strength and fire
    class, so a 9.5 mm board against a 12.5 mm baseline shows a saving that is
    partly the thickness (HENRIC-3399). A board whose thickness is known from
    neither its name nor its EPD is left out: unknown is not standard.
    """
    from aida.data.gipsskiva import (
        STANDARD_TEXT,
        board_thickness,
        component_class,
        format_mm,
        thickness_class,
    )

    want, stated = component_class(proj_comp.name)
    rows: list[dict] = []
    unknown = 0
    for e in pool:
        mm, src = board_thickness(e)
        if mm is None:
            unknown += 1
            continue
        if thickness_class(mm) != want:
            continue
        b = dict(e)
        b["board_mm"] = mm
        if stated:
            b["board_note"] = (f"Skivan är {format_mm(mm)} mm ({src}), samma "
                               f"tjocklek som komponentens {want} mm.")
        else:
            b["board_note"] = (f"Skivan är {format_mm(mm)} mm ({src}). Komponenten "
                               f"anger ingen tjocklek, så den jämförs med "
                               f"standardskivor på {STANDARD_TEXT}.")
        rows.append(b)
    if rows:
        return rows, ""
    left_out = (f" {unknown} skivor anger ingen tjocklek och jämförs inte." if unknown else "")
    return [], (f"Katalogen har inga gipsskivor på {want} mm att jämföra med, och "
                f"skivor av en annan tjocklek är en annan produkt.{left_out}")


def _ceiling_rows(epd_data: dict[str, list[dict]], proj_comp) -> tuple[list[dict], str]:
    """Rows a ceiling, an acoustic absorber or a grid can be compared against,
    or the reason there are none (HENRIC-3290 del 3).

    Strict per kind, like _furniture_rows: a ceiling tile is not replaced by a
    wall absorber, and the grid is a steel profile. Units are matched exactly
    and not through _select_epd_candidates' unit classes, which put kg and st
    together and fall back to the whole bucket when nothing matches: a grid
    per kg against "T24 tvärprofil 40 st" is exactly the comparison that must
    not be made, and a ceiling in m2 must only meet rows per m2.
    """
    from aida.data.climate_data import ceiling_subcategory

    sub = ceiling_subcategory(proj_comp.name)
    if not sub:
        return [], (
            "Namnet säger inte vilken sorts undertak eller akustikprodukt det "
            "gäller, och alternativ jämförs bara inom samma sort. Ange till "
            "exempel \"Undertaksplattor\", \"Väggabsorbenter\", \"Bafflar\", "
            "\"Bärverk T24\", \"Gipstak\" eller \"Putsat innertak\"."
        )
    if sub == "putstak":
        return [], (
            "Ett putsat innertak jämförs inte med nyköp. Katalogen har puts "
            "bara för fasader, per kg, och ingen EPD för invändig puts på tak."
        )
    label = _CEILING_LABELS.get(sub, sub)
    if sub == "gipstak":
        rows, why = _board_class_rows(_plasterboard_rows(epd_data), proj_comp)
        if why:
            return [], why
        pool = rows
    else:
        pool = [e for e in epd_data.get("undertak", []) if e.get("subcategory") == sub]
    if not pool:
        return [], f"Katalogen har inga EPD:er för {label}, så ingen jämförelse med nyköp görs."

    rows = _exact_unit_rows(pool, proj_comp.unit)
    if rows:
        return rows, ""
    return [], _unit_mismatch_reason(label, pool, proj_comp.unit)


def _exact_unit_rows(pool: list[dict], comp_unit: str) -> list[dict]:
    """Rows of `pool` declared in the component's own unit, spellings folded
    (st/styck, m2/m², lm/m) but kg and st kept apart."""
    unit = comp_unit.strip().lower()
    if unit in _COUNT_UNITS:
        wanted = {"st"}
    elif unit in _AREA_UNITS:
        wanted = {"m2"}
    elif unit in _LENGTH_UNITS:
        wanted = {"lm", "m"}
    else:
        wanted = {unit}
    return [e for e in pool if _epd_comparable(e)[1] in wanted]


def _unit_mismatch_reason(label: str, pool: list[dict], comp_unit: str) -> str:
    """Why no row meets a component in `comp_unit`, naming the unit most of
    `pool` is declared in as the one to give (not the alphabetically first:
    the grid is 35 rows per lm and one per kg)."""
    counts: dict[str, int] = {}
    for e in pool:
        u = _epd_comparable(e)[1]
        counts[u] = counts.get(u, 0) + 1
    declared = ", ".join(sorted(counts))
    common = max(sorted(counts), key=lambda u: counts[u])
    return (
        f"EPD:erna för {label} anges per {declared}, och komponenten i "
        f"{comp_unit}. Ange mängden i {common} så jämförs de."
    )


def _names_del3_subtype(proj_comp) -> bool:
    """True when the component, in the category it resolves to itself, names
    glasparti or avjämning. Not aggregat: that split predates del 3 and the
    router's hand on it is left as it was."""
    from aida.data.epd_baseline_medians import UNLIKE_THEIR_CATEGORY
    from aida.data.palats_client import component_subcategory

    own = resolve_category(proj_comp.name, proj_comp.category)
    return (own, component_subcategory(proj_comp.name, own)) in UNLIKE_THEIR_CATEGORY


# Split subtypes named in a reason, in Swedish.
_SPLIT_LABELS = {"avjämning": "avjämningsmassa", "glasparti": "glaspartier",
                 "aggregat": "ventilationsaggregat"}


def _split_unit_reason(rows: list[dict], proj_comp, category: str) -> tuple[list[dict], str]:
    """Rows of a split subtype declared in this component's own unit, or the
    reason there are none (HENRIC-3290 del 3).

    _select_epd_candidates falls back to the whole bucket when no row shares
    the component's unit, which on the category side is a deliberate choice.
    On a split subtype the bucket is the subtype, and every row in it is in
    the wrong unit: an "Avjämningsmassa" in m2 would be offered compounds per
    kg as a 97 % saving. The answer there is a question, not a list.

    The unit is matched exactly, as for ceilings, and not by unit class. The
    class puts kg and st together, so "Avjämningsmassa 20 st" (bags) met all
    60 compounds per kg, and a unit outside the subtype's own ("Glasparti
    12 lm") was let through to that same fallback. Found in review.
    """
    from aida.data.epd_baseline_medians import _SPLIT_SUBCATEGORIES
    from aida.data.palats_client import component_subcategory

    splits = _SPLIT_SUBCATEGORIES.get(category)
    if not splits or not rows:
        return rows, ""
    sub = component_subcategory(proj_comp.name, category)
    if sub not in splits:
        return rows, ""
    matching = _exact_unit_rows(rows, proj_comp.unit)
    if matching:
        return matching, ""
    return [], _unit_mismatch_reason(_SPLIT_LABELS.get(sub, sub), rows, proj_comp.unit)


def _coverage_rows(rows: list[dict], proj_comp, category: str) -> tuple[list[dict], str]:
    """A levelling compound or waterproofing in m²: the kg rows restated per
    m² by each product's own datasheet rate, or the reason there are none
    (HENRIC-3371). Any other component passes through unchanged.

    Same kind of bridge as _bridge_profile_rows: the GWP is the EPD's, the kg
    per m² is the maker's application rate (data/coverage.py, with the URL
    and the sentence it came from) and, for levelling, the thickness the
    component's name states. A row whose datasheet is not in the table is
    left out, not given a borrowed rate. The catalog's native m² rows stay.
    """
    from aida.data import coverage

    unit = (proj_comp.unit or "").strip().lower()
    if unit not in _AREA_UNITS or not rows:
        return rows, ""
    levelling = category == "golv" and _component_subcategory(proj_comp, category) == "avjämning"
    if not levelling and category != "tätskikt":
        return rows, ""
    native = [e for e in rows if _epd_comparable(e)[1] in _AREA_UNITS]
    kg_rows = [e for e in rows if str(e.get("unit", "")).lower() == "kg"
               and isinstance(e.get("gwp_a1a3"), (int, float))]
    out = []
    if levelling:
        mm = coverage.thickness_mm(proj_comp.name)
        if mm is None:
            if native:
                return native, ""
            return [], (
                "EPD:erna för avjämningsmassa anges per kg, och åtgången per m² "
                "beror på skiktets tjocklek. Ange tjockleken i namnet, till "
                "exempel \"Avjämning 10 mm\", så räknas varje produkt om med "
                "åtgången i sitt tekniska datablad. Ange mängden i kg om tjockleken "
                "inte är känd."
            )
        for e in kg_rows:
            rate = coverage.levelling(e.get("reg_no"))
            if not rate:
                continue
            kg = rate.kg_per_m2_mm * mm
            out.append(_coverage_copy(
                e, kg, unit,
                f"åtgång {rate.kg_per_m2_mm:g} kg/m² per mm × {mm:g} mm = {kg:g} kg/m²",
                rate))
    else:
        side = coverage.surface(proj_comp.name)
        for e in kg_rows:
            rate = coverage.membrane(e.get("reg_no"))
            if not rate:
                continue
            if side == "vägg":
                kg, where = rate.wall_kg_per_m2, "vägg"
            else:
                kg, where = rate.floor_kg_per_m2, "golv"
            if rate.floor_kg_per_m2 == rate.wall_kg_per_m2:
                note = f"åtgång {kg:g} kg/m²"
            elif side is None:
                note = (f"åtgång {kg:g} kg/m² på golv; namnet säger inte golv eller "
                        f"vägg, och väggens åtgång är {rate.wall_kg_per_m2:g}")
            else:
                note = f"åtgång {kg:g} kg/m² på {where}"
            out.append(_coverage_copy(e, kg, unit, note, rate))
    if native or out:
        return native + out, ""
    label = "avjämningsmassa" if levelling else "tätskikt"
    return [], (
        f"EPD:erna för {label} anges per kg, och ingen av dem har ett tekniskt "
        f"datablad med åtgång per m² inläst. Ange mängden i kg för att jämföra."
    )


def _coverage_copy(e: dict, kg_per_m2: float, unit: str, note: str, rate) -> dict:
    """One kg row restated per m² by a datasheet rate (_coverage_rows)."""
    bridged = dict(e)
    bridged["gwp_per_functional_unit"] = round(e["gwp_a1a3"] * kg_per_m2, 4)
    bridged["functional_unit"] = unit
    bridged["fu_basis"] = "coverage"
    bridged["fu_note"] = note
    bridged["fu_note_own"] = True
    bridged["kg_per_unit"] = round(kg_per_m2, 3)
    bridged["coverage_url"] = rate.url
    bridged["coverage_quote"] = rate.quote
    return bridged


def _row_key(epd: dict) -> tuple:
    """Identity for a catalog row. uuid where present, name+category otherwise."""
    return (epd.get("uuid") or "", epd.get("category", ""), epd.get("name", ""))


def _apply_nordic_quota(
    selected: list[dict],
    pool: list[dict],
    rank,
    category: str,
    quota: int = _NORDIC_QUOTA,
) -> list[dict]:
    """Guarantee `quota` Nordic-supplier rows in the queue, taking from the tail.

    A quota, not a filter, and the distinction is the whole design. Förvaltare
    want to buy from Nordic wholesalers, so a queue with no Nordic row in it is
    useless to them in practice. But the lowest-GWP product in a category is
    often not Nordic, and hiding it would make the tool worse at the thing it
    exists for. So: keep the best rows by GWP, and if fewer than `quota` of them
    come from a Nordic supplier, promote the best Nordic rows from the pool and
    drop an equal number from the BACK of the selection.

    Three properties this preserves, each of which a filter would break:

    - The count never falls. Displacement is one-for-one, so "flera alternativ
      visas" survives the guarantee.
    - Nothing near the front is displaced. The rows that go are the worst in a
      selection that is already capped at 80 and from which the model picks two
      to four; they were never going to be offered.
    - A category with no Nordic rows at all is left alone rather than padded.
      That is a coverage gap, and the honest response is to report it, not to
      manufacture a Nordic-looking queue.

    Promotion cannot displace another Nordic row, so calling this twice is a
    no-op the second time.
    """
    if quota <= 0 or not selected:
        return selected

    chosen = {_row_key(e) for e in selected}
    have = sum(1 for e in selected if nordic_supplier(e))
    if have >= quota:
        return selected

    available = sorted(
        (e for e in pool
         if nordic_supplier(e) and _row_key(e) not in chosen),
        key=rank,
    )
    wanted = min(quota - have, len(available))
    if wanted <= 0:
        # Either the pool has no more Nordic rows, or every one is already in.
        # Both mean the catalog cannot meet the quota here; say which category,
        # because that list is the input to the sourcing work.
        if have < quota:
            logger.info(
                "Nordisk kvot ej uppfylld för %s: %d av %d möjliga i kön, "
                "katalogen har inga fler", category, have, quota,
            )
        return selected

    # Drop from the back, worst GWP first, and never a Nordic row — displacing
    # one to make room for another would leave the count unchanged and the
    # queue no more available.
    keep = sorted(selected, key=rank)
    droppable = [i for i in range(len(keep) - 1, -1, -1)
                 if not nordic_supplier(keep[i])]
    for i in droppable[:wanted]:
        keep[i] = None  # type: ignore[call-overload]
    result = [e for e in keep if e is not None]
    result.extend(available[:wanted])
    logger.info(
        "Nordisk kvot för %s: befordrade %d nordiska rader (%d -> %d av %d)",
        category, wanted, have, have + wanted, quota,
    )
    return sorted(result, key=rank)


# Both of these moved to aida.name_match. The price matcher used a bare
# .lower() instead and therefore discarded prices the web search had genuinely
# found (2026-08-20: three of five). One implementation, one set of rules, so
# the two cannot drift apart again.
_match_key = match_key
_tokens = tokens


def _name_candidates(name: str, epds: list[dict]) -> list[dict]:
    """The catalog rows an LLM-written name may refer to, best match only.

    The model paraphrases and truncates product names, so exact equality is
    useless. Containment either way first, longest match wins, and every row
    tied for it is returned; then a token overlap for the cases containment
    cannot reach, such as "Fibre cement cladding HardiePanel® / Hardie®
    Architectural Panel" for a catalog entry called "S-P-10857 Fibre cement
    cladding: HardiePanel®, Hardie® Architectural Panel" — same product,
    reordered and re-punctuated. [] when nothing matches or the token overlap
    has no clear winner.

    Returning the whole tie lets each caller decide whether the tied rows
    disagree about the thing it needs: the GWP label for match_epd_by_name,
    the figure itself for _catalog_row_for.
    """
    if not name:
        return []
    needle = _match_key(name)
    if not needle:
        return []
    best_len = 0
    tied: list[dict] = []
    exact: list[dict] = []
    for epd in epds:
        epd_name = _match_key(epd.get("name") or "")
        if not epd_name:
            continue
        if epd_name == needle:
            # All of them: 96 names occur more than once in the catalog, some
            # with different figures, so the first exact hit is not the answer.
            exact.append(epd)
            continue
        if epd_name in needle or needle in epd_name:
            overlap = min(len(epd_name), len(needle))
            if overlap > best_len:
                best_len, tied = overlap, [epd]
            elif overlap == best_len:
                tied.append(epd)
    if exact:
        return exact
    if tied:
        return tied

    # Token overlap, for names the model reordered or re-punctuated past what
    # containment can follow. Deliberately strict: at least three quarters of
    # the catalog entry's distinctive words must appear, and the winner has to
    # be clearly ahead of the runner-up. A wrong match here would put a
    # GWP-GHG label on a product that does not deserve one, which is worse than
    # leaving a fossil figure unlabelled.
    needle_tokens = _tokens(needle)
    if len(needle_tokens) < 2:
        return []
    scored: list[tuple[float, dict]] = []
    for epd in epds:
        epd_tokens = _tokens(_match_key(epd.get("name") or ""))
        if len(epd_tokens) < 2:
            continue
        score = len(epd_tokens & needle_tokens) / len(epd_tokens)
        if score >= 0.75:
            scored.append((score, epd))
    if not scored:
        return []
    scored.sort(key=lambda pair: pair[0], reverse=True)
    if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.15:
        return []  # ambiguous, better to say nothing
    return [scored[0][1]]


def match_epd_by_name(name: str, epds: list[dict]) -> dict | None:
    """Find which candidate EPD an LLM-written alternative name refers to.

    Used to carry facts the model cannot be trusted to relay (which GWP
    indicator a figure rests on) from the catalog onto the alternative.
    """
    tied = _name_candidates(name, epds)
    if not tied:
        return None
    # A tie is only a problem when the tied entries disagree about the thing
    # we are carrying across. Two equally-matching fossil products give the
    # same answer either way; one fossil and one GHG do not, and guessing
    # there would put a label on a product that may not deserve it.
    if len({e.get("gwp_basis", "") for e in tied}) > 1:
        return None
    return tied[0]


def _model_number(value) -> float | None:
    """A finite number from a field the model wrote, or None.

    The model writes JSON by hand, and "ca 400", "1 200", null and true all
    arrive in number fields. Before 2026-09-30 they were stored as-is, and a
    string reached `alt.co2e_kg <= 0` in _validate_alternatives as a
    TypeError that took the whole component out of the analysis.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def _reg_tokens(reg_no: str) -> set[str]:
    """The forms a registration number may be cited in, normalised.

    A third of the catalog stores two ids in one field, "EPD-IES-0006778:001
    (S-P-06778)", and the model cites either one, with or without the
    ":001" product suffix. Each is a form; short ones are dropped as too
    ambiguous to count as a citation.
    """
    out: set[str] = set()
    for tok in re.findall(r"[\w][\w\-:./]*", _match_key(reg_no)):
        tok = tok.rstrip(".:/")
        out.add(tok)
        if ":" in tok:
            out.add(tok.split(":", 1)[0])
    return {t for t in out if len(t) >= 5}


def _cites_reg_no(source_key: str, epd: dict) -> bool:
    """True when a normalised source string names one of this row's
    registration numbers as a whole token. Bounded on both sides, because
    "S-P-0111" is a substring of "S-P-01111"."""
    if not source_key:
        return False
    return any(
        re.search(rf"(?<![\w-]){re.escape(tok)}(?![\w-])", source_key)
        for tok in _reg_tokens(str(epd.get("reg_no") or ""))
    )


def _name_accounts_for(needle_key: str, epd: dict) -> bool:
    """Whether a name match on this row explains the whole model-written name.

    A catalog name found inside a longer model name is only evidence when
    what is left over is the manufacturer (the prompt asks for "Produktnamn
    (Tillverkare)") or punctuation. "Fire rated Doors Acme" contains the
    catalog's "Doors", and without this check it took that row's figure and
    an [EPD] tag, which promoted an invented product to a declared one. The
    other shapes are left as they were: an exact name, the model shortening
    the catalog name, or the strict token-overlap winner.
    """
    name_key = _match_key(epd.get("name") or "")
    if not name_key or name_key == needle_key or name_key not in needle_key:
        return True
    return _leftover_words(needle_key, epd) <= _words(_match_key(epd.get("owner") or ""))


def _words(text: str) -> set[str]:
    """Words of three letters or more, punctuation stripped first ("(Bolon AB)"
    is {"bolon"}: _tokens would keep "ab)" as "ab")."""
    return {w for w in re.findall(r"\w+", text) if len(w) >= 3}


def _leftover_words(needle_key: str, epd: dict) -> set[str]:
    """The words of a model-written name that the catalog row's name does not
    cover: normally the manufacturer the prompt asks for in parentheses."""
    name_key = _match_key(epd.get("name") or "")
    rest = needle_key.replace(name_key, " ", 1) if name_key else needle_key
    return _words(rest)


def _catalog_row_for(item: dict, epds: list[dict]) -> dict | None:
    """The one catalog row a model-written EPD alternative refers to, or None.

    Two pieces of evidence: the name (the same matcher the GWP label uses)
    and the registration number the prompt asks the model to cite in
    `source`. When both are there they must agree, and the rows both point
    at are the answer; that also settles a multi-product EPD, where one
    number covers rows with different figures. When they disagree the row is
    refused: a short catalog name inside a longer model name ("LED
    DOWNLIGHT" inside "Planar LED Downlight") is a name match to the wrong
    product, and a number the model misquoted is a number match to one.
    Without a citation the name has to account for itself
    (_name_accounts_for). The manufacturer breaks a remaining tie. Rows that
    agree on the comparable figure and the GWP basis are one answer however
    many there are; rows that disagree are not, and guessing between them
    would put one product's number on another's name.
    """
    name = str(item.get("name") or "")
    needle = _match_key(name)
    by_name = _name_candidates(name, epds)
    source_key = _match_key(str(item.get("source") or ""))
    cited = [e for e in epds if _cites_reg_no(source_key, e)]
    if by_name and cited:
        cands = [e for e in by_name if any(e is c for c in cited)]
    elif cited:
        cands = _name_candidates(name, cited) or cited
    else:
        cands = [e for e in by_name if _name_accounts_for(needle, e)]

    def distinct(rows):
        return {(_epd_comparable(e), e.get("gwp_basis", "")) for e in rows}

    if len(distinct(cands)) > 1:
        # "Floor Tile (Graniser CERAMICS)": the catalog has three Floor Tiles.
        # Only words beyond the product name count, or an owner whose name is
        # also in the product's ("Kingspan Kooltherm") would always win.
        by_owner = [e for e in cands
                    if _words(_match_key(e.get("owner") or "")) & _leftover_words(needle, e)]
        cands = by_owner if len(distinct(by_owner)) == 1 else []
    return cands[0] if cands else None


def _catalog_co2e(matched: dict, proj_comp, category: str) -> tuple[float | None, str]:
    """(co2e_kg, note) for an EPD alternative, computed from its catalog row.

    The model used to supply co2e_kg itself, and the per-component prompt
    left unit mismatches to "en rimlig omräkning": a per-kg toilet EPD was
    multiplied by a count of toilets (0.9 kg/kg × 30 = 27 kg against a
    1 545 kg baseline, shown as a 98 % saving). The figure is now the
    catalog's, times the component quantity, and only in the component's own
    unit: the declared unit first (an m3 component meets an m3 row's own
    figure, not its per-m2 restatement), then the functional unit
    _epd_comparable derives. One bridge is allowed, the one the baseline side
    already uses for count-denominated products: a per-kg row for a component
    counted in st, through the typical item mass for its (category,
    subcategory), and only when the row is that kind of product or of no
    stated kind; a bathtub's kg figure times a toilet's mass is neither.
    Anything else is refused rather than guessed.

    On success `note` is "" or a Swedish sentence for the reasoning; on
    refusal co2e is None and `note` says why, in Swedish, for the row that
    tells the förvaltare nothing could be compared.
    """
    from aida.data.unit_conversion import typical_item_mass
    from aida.followup import normalize_unit, units_comparable

    quantity = _model_number(getattr(proj_comp, "quantity", None))
    if not quantity or quantity <= 0:
        return None, "komponenten saknar en mängd att räkna på"
    comp_unit = proj_comp.unit or ""
    declared = _model_number(matched.get("gwp_a1a3"))
    if declared and declared > 0 and units_comparable(str(matched.get("unit") or ""), comp_unit):
        # A plasterboard row says which thickness it was compared at.
        return round(declared * quantity, 1), str(matched.get("board_note") or "")
    gwp, epd_unit = _epd_comparable(matched)
    if gwp <= 0:
        return None, "EPD:n saknar ett användbart klimatvärde"
    if units_comparable(epd_unit, comp_unit):
        co2e = round(gwp * quantity, 1)
        if matched.get("fu_basis") == "profile_mass":
            # The weight per metre is the one figure here that is not the
            # EPD's, so the row says which it is and where it comes from.
            return co2e, (
                f"CO2e räknat från EPD:ns deklarerade värde: "
                f"{matched.get('gwp_a1a3'):g} kg CO2e/kg × "
                f"{matched.get('kg_per_m'):g} kg/m ({matched.get('profile')}, "
                f"{matched.get('profile_source')}) × {quantity:g} {comp_unit} = "
                f"{co2e:g} kg."
            )
        if matched.get("fu_basis") == "piece_area":
            # The area per piece is the component's own size, not the EPD's.
            return co2e, (
                f"CO2e räknat från EPD:ns deklarerade värde: "
                f"{matched.get('per_m2'):g} kg CO2e/m² × "
                f"{matched.get('area_m2_per_unit'):g} m²/{comp_unit} "
                f"(komponentens mått, {matched.get('fu_note')}) × "
                f"{quantity:g} {comp_unit} = {co2e:g} kg."
            )
        if matched.get("fu_basis") == "coverage":
            # The kg per m² is the maker's datasheet's, so the row names it
            # and links the sheet (HENRIC-3371).
            return co2e, (
                f"CO2e räknat från EPD:ns deklarerade värde: "
                f"{matched.get('gwp_a1a3'):g} kg CO2e/kg × "
                f"{matched.get('kg_per_unit'):g} kg/m² ({matched.get('fu_note')}, "
                f"enligt tekniskt datablad: {matched.get('coverage_url')}) × "
                f"{quantity:g} {comp_unit} = {co2e:g} kg."
            )
        if matched.get("fu_basis") == "element_mass":
            # Same reason: the mass per unit is not the EPD's figure (or is
            # the EPD's own, which the source then says).
            return co2e, (
                f"CO2e räknat från EPD:ns deklarerade värde: "
                f"{matched.get('gwp_a1a3'):g} kg CO2e/kg × "
                f"{matched.get('kg_per_unit'):g} kg/{comp_unit} "
                f"({matched.get('fu_note')}; {matched.get('mass_source')}) × "
                f"{quantity:g} {comp_unit} = {co2e:g} kg."
            )
        return co2e, ""
    if normalize_unit(epd_unit) == "kg" and normalize_unit(comp_unit) == "st":
        sub = _component_subcategory(proj_comp, category)
        mass = typical_item_mass(category, sub) if category else None
        if not mass:
            return None, (
                "EPD:n anges per kg och komponenten räknas i styck, och ingen "
                "typisk vikt per styck är känd för den här sortens produkt"
            )
        row_sub = matched.get("subcategory") or ""
        if row_sub and row_sub != sub:
            return None, (
                f"EPD:n anges per kg och gäller en annan produkttyp ({row_sub}) än "
                f"komponenten ({sub}), så den typiska vikten per styck går inte att använda"
            )
        co2e = round(gwp * mass * quantity, 1)
        return co2e, (
            f"CO2e räknat från EPD:ns deklarerade värde: {gwp:g} kg CO2e/kg × "
            f"{mass:g} kg/st (antagen typisk vikt, approximation) × "
            f"{quantity:g} st = {co2e:g} kg."
        )
    return None, f"EPD:n anges per {epd_unit or 'okänd enhet'} och komponenten i {comp_unit}"


def _component_subcategory(proj_comp, category: str | None) -> str:
    """The subcategory the component's name gives within `category`, or ""."""
    from aida.data.palats_client import component_subcategory

    return component_subcategory(proj_comp.name, category) if category else ""


def _item_mass(proj_comp, category: str | None) -> float | None:
    """Typical kg per piece for a component counted in st, or None.

    Same lookup as the baseline's kg->st bridge (baseline._apply_epd_median_
    fallback, alternatives._effective_baseline_co2e): the routed category and
    the subcategory the component's name gives.
    """
    from aida.data.unit_conversion import typical_item_mass
    from aida.followup import normalize_unit

    if normalize_unit(getattr(proj_comp, "unit", "") or "") != "st" or not category:
        return None
    return typical_item_mass(category, _component_subcategory(proj_comp, category))


def _mass_bridge_clause(proj_comp, category: str | None) -> str:
    """The one exception to "skip an EPD in another unit", told to the model
    with the mass the code will use, so its reasoning ("−45 % CO2e") is
    computed the same way as the figure the row ends up carrying."""
    mass = _item_mass(proj_comp, category)
    if not mass:
        return ""
    return (
        f"\nUndantag: en EPD deklarerad per kg för samma sorts produkt räknas om med "
        f"en antagen typisk vikt på {mass:g} kg/st, alltså EPD-värdet × {mass:g} × "
        f"{proj_comp.quantity} st. Använd den omräkningen när du jämför med baslinjen."
    )


# Registry names as a reader knows them, for a source string the code builds.
_REGISTRY_LABELS = {"environdec": "Environdec", "environdec_manual": "Environdec",
                    "epd_norge": "EPD-Norge", "epd_hub": "EPD Hub", "ibu": "IBU",
                    "ul_environment": "UL Environment"}


class RankingFailed(RuntimeError):
    """The ranking call for one component failed: the API call raised, or the
    answer was not readable JSON. Not the same as an empty answer, which is
    the model saying nothing in the lists fits. `timed_out` is kept apart
    because its advice differs: fewer components, not just another try."""

    def __init__(self, component_name: str, *, timed_out: bool = False) -> None:
        super().__init__(component_name)
        self.timed_out = timed_out


def _format_epd_list(epds: list[dict]) -> str:
    """Format EPD list for inclusion in prompt."""
    lines = []
    for epd in epds:
        reg = epd.get("reg_no", "")
        reg_str = f" ({reg})" if reg else ""

        # GWP-fossil A1-A3 normally, matching Boverket's standard so
        # alternatives are comparable to the baseline. GWP-total (which includes
        # biogenic carbon credit and can be negative for bio-based products) is
        # intentionally never shown, to avoid mixing bases in the same list.
        # The one exception is an EPD whose own components did not add up, where
        # the build falls back to GWP-GHG; that is named here rather than
        # blended in, so the model does not present it as like for like.
        basis_label = "GWP-GHG" if epd.get("gwp_basis") == "ghg" else "GWP-fossil"
        gwp_str = f"{basis_label} A1-A3: {epd['gwp_a1a3']} kg CO2e/{epd['unit']}"
        fu_gwp = epd.get("gwp_per_functional_unit")
        fu_unit = epd.get("functional_unit")
        if fu_gwp is not None and fu_unit:
            gwp_str += f" \u2192 {fu_gwp} kg CO2e/{fu_unit}"
            # A frame member's bridge rests on the component's own dimension,
            # so the model can say which section the per-metre figure is for.
            if epd.get("fu_note") and epd.get("fu_note_own"):
                gwp_str += f" ({epd['fu_note']})"
            elif epd.get("fu_note"):
                gwp_str += f" (komponentens {epd['fu_note']})"
        elif isinstance(epd.get("element_thickness_mm"), (int, float)):
            # A hollow-core row per m² is per m² at its own thickness.
            gwp_str += f" (tjocklek {epd['element_thickness_mm']:g} mm enligt EPD:n)"
        elif isinstance(epd.get("board_mm"), (int, float)):
            # A plasterboard row was picked for its thickness (HENRIC-3399).
            gwp_str += f" (skiva {epd['board_mm']:g} mm)"

        source = epd.get("source_registry", "environdec")
        source_tag = f" [{source}]" if source != "environdec" else ""

        # Availability is shown separately from Geo, and both are shown, because
        # they answer different questions and the model was previously told to
        # use Geo for a question Geo cannot answer. Geo is the declaration's
        # validity region; the label is about the supplier. Ahlsell AB declares
        # GLO and is the most orderable row in the catalog.
        # Insulation only. An m2-declared insulation figure is per square metre
        # at that product's own thickness, and the thickness is nowhere in the
        # data, so two rows in this list can describe the same material at very
        # different depths. Where the declaration states a thermal resistance it
        # is shown, because that is the one thing that makes two rows genuinely
        # comparable -- and showing it on some rows and not others is the point:
        # the model can then see which comparisons it is entitled to make.
        r_str = ""
        r_value = epd.get("r_value")
        if r_value:
            r_str = f" | deklarerad vid R={r_value} m2K/W"

        lines.append(
            f"- {epd['name']} | {epd.get('owner', '?')} | "
            f"{gwp_str}{r_str} | "
            f"Tillgänglighet: {availability_label(epd)} | "
            f"Geo: {epd.get('geo', '?')}{reg_str}{source_tag}"
        )
    return "\n".join(lines)


# Warning appended to the EPD list for insulation, where the declared unit hides
# a variable the förvaltare is actually choosing.
ISOLERING_R_CAVEAT = (
    "\nOBS om isolering: ett m2-värde gäller produktens EGEN tjocklek, som "
    "sällan står i deklarationen. Rader märkta \"deklarerad vid R=...\" är "
    "räknade på en angiven värmemotstånd och kan jämföras med varandra. Rader "
    "utan sådan märkning kan INTE jämföras rakt av, varken med varandra eller "
    "med de märkta: en tunn skiva ser billigare ut enbart för att den är tunn. "
    "Rangordna dem inte som om skillnaden vore klimatprestanda, och säg i "
    "reasoning när en jämförelse vilar på olika tjocklekar."
)


# Keywords that indicate a component part rather than a complete system.
# Used to filter out alternatives that aren't apples-to-apples with a full baseline.
_COMPONENT_ONLY_KEYWORDS = [
    "membran",
    "ångspärr",
    "ångbroms",
    "underlagsduk",
    "underlagstak",
    "diffusionsspärr",
    "tätskikt",
    "fuktspärr",
    "vindskydd",
    "vapor barrier",
    "vapour barrier",
    "membrane",
    "underlayment",
    "underlag",
]


# Components that are themselves one layer of a build-up, HENRIC-3290 del 3.
# Wet-room waterproofing is a membrane, and a levelling compound is sold as an
# "underlayment"; for them the words above name the product, not a part of it.
_LAYER_CATEGORIES = {"tätskikt"}
_LAYER_SUBTYPES = {("golv", "avjämning")}


def _is_component_only(name: str) -> bool:
    """Check if an alternative name suggests it's just a component part, not a complete system.

    E.g. a vapor barrier membrane is not a complete roofing alternative.
    """
    name_lower = name.lower()
    return any(kw in name_lower for kw in _COMPONENT_ONLY_KEYWORDS)


def _validate_alternatives(
    alternatives: list[Alternative],
    baseline_co2e: float,
    component_name: str,
    quantity: float = 0,
    category: str | None = None,
    unit: str = "",
    dropped: list[str] | None = None,
) -> list[Alternative]:
    """Filter out alternatives with data quality issues.

    ``dropped``, when given, collects a Swedish clause per filtered row that
    is not about the baseline (a part, not a complete system; no figure), so
    a component left with nothing can say what happened to its candidates.
    The worse-than-baseline filter is explained from the pool itself
    (_filtered_pool_reason), which also covers rows the model left out.

    Removes:
    - Alternatives with co2e_kg <= 0 (unrealistic for building materials)
    - Component-only products when the baseline is a complete system
    - climate_optimized alternatives that don't beat the baseline (mislabeled)
    Flags:
    - Alternatives with cost_sek == 0 get "Pris ej tillgängligt"
    - LLM-estimated prices get "Approximerat pris"
    - Out-of-range prices get "Oväntat pris — verifiera"
    - climate_optimized alternatives with an implausibly low CO2e (broken EPD)
    """
    from aida.data.price_validation import validate_total_price

    # Use the routed category when provided so a kakel-routed wall validates
    # prices against kakel ranges, not the name-derived innervägg ranges.
    if category is None:
        category = normalize_component_name(component_name)
    # A component that IS a layer (del 3): its alternatives are membranes and
    # underlayments by definition, and the complete-system filter below would
    # drop every one of them.
    from aida.data.palats_client import component_subcategory
    is_layer = (category in _LAYER_CATEGORIES
                or (category, component_subcategory(component_name, category or ""))
                in _LAYER_SUBTYPES)
    valid = []
    for alt in alternatives:
        # A) Filter zero/negative CO2 — all building materials have emissions
        if alt.co2e_kg is None or alt.co2e_kg <= 0:
            logger.info(
                "Filtered alternative '%s' for %s: co2e_kg=%s (unrealistic)",
                alt.name, component_name, alt.co2e_kg,
            )
            if dropped is not None and alt.alternative_type != "info":
                dropped.append(f"{alt.name} saknar ett klimatvärde över noll")
            continue

        # Reuse rows are built from a Palats listing with a deterministic CO2e
        # and the listing's own asking price, and the coverage gate in
        # _palats_candidates has already judged them. Until 2026-09-14 they
        # were appended AFTER this validator ran and so never passed through
        # it; now that the model ranks them in the same call as the EPDs they
        # arrive here, and the checks below (complete-system name filter,
        # market-price plausibility) are about new products and would only
        # add wrong notes to a second-hand listing.
        if alt.alternative_type == "reuse":
            valid.append(alt)
            continue

        # B) Filter component-only products (membranes, vapor barriers etc.)
        if not is_layer and _is_component_only(alt.name):
            logger.info(
                "Filtered alternative '%s' for %s: component part, not complete system",
                alt.name, component_name,
            )
            if dropped is not None:
                dropped.append(f"{alt.name} är en del av en byggdel (membran eller "
                               f"underlag), inte en hel byggdel")
            continue

        # B2) Drop climate_optimized options that don't actually beat the
        #     baseline — a "climate-optimized" choice with co2e >= baseline is
        #     mislabeled and produces absurd kr/sparat-kg in the ranking. Reuse
        #     and info entries are exempt (different comparison / no number).
        #     This is the rule metod.md and both prompts state (HENRIC-3396,
        #     scripts/test_alternativ_under_baslinjen.py); until 2026-10-02 the
        #     text said the opposite. A pool with only worse rows is explained
        #     to the user by _filtered_pool_reason.
        if (alt.alternative_type == "climate_optimized"
                and baseline_co2e > 0
                and alt.co2e_kg >= baseline_co2e):
            logger.info(
                "Filtered alternative '%s' for %s: co2e %.1f >= baseline %.1f "
                "(not an improvement)",
                alt.name, component_name, alt.co2e_kg, baseline_co2e,
            )
            continue

        # B3) Flag implausibly-low climate_optimized values — a new product at
        #     <3%% of the baseline is almost certainly a broken EPD (partial
        #     module / wrong unit) that slipped past the catalog floor. Keep it
        #     for transparency but mark it for verification.
        if (alt.alternative_type == "climate_optimized"
                and baseline_co2e > 0
                and alt.co2e_kg < baseline_co2e * 0.03):
            if "verifiera" not in alt.reasoning.lower():
                alt.reasoning = alt.reasoning.rstrip(". ") + (
                    ". Ovanligt lågt CO2e — verifiera mot EPD:n "
                    "(kan vara partiell modul eller felaktig enhet)."
                )

        # C) Price validation — flag zero prices (filtered after enrichment)
        if alt.cost_sek is None or alt.cost_sek <= 0:
            alt.cost_sek = 0
            if "pris ej tillgängligt" not in alt.reasoning.lower():
                alt.reasoning = alt.reasoning.rstrip(". ") + ". Pris ej tillgängligt."
        elif quantity > 0:
            is_estimate = "[uppskattning]" in alt.source.lower()
            _cost, note = validate_total_price(
                alt.cost_sek, quantity, category, is_estimate=is_estimate, unit=unit,
            )
            if note and note.lower() not in alt.reasoning.lower():
                alt.reasoning = alt.reasoning.rstrip(". ") + f". {note}."

        valid.append(alt)

    return valid


def _b1_keep_alternative(alt) -> bool:
    """DoD B1: should this alternative survive the post-enrichment zero-price cut?

    Keep an alternative if it is actionable on at least one axis:
    - baseline/info/reuse entries are never price-cut (reuse is legitimately free)
    - it has a real price (cost_sek > 0) → buyable
    - it is EPD-backed ([EPD] source) → carries a verified CO2 number, the core
      value of a climate tool, even when its obscure product name can't be priced

    Only unpriced AND non-EPD alternatives (pure LLM guesses, unactionable on both
    price and climate verifiability) are dropped.
    """
    if alt.alternative_type in ("baseline", "info", "reuse"):
        return True
    if alt.cost_sek is not None and alt.cost_sek > 0:
        return True
    return "[epd]" in (alt.source or "").lower()


def _effective_baseline_co2e(
    proj_comp, bl_comp, routed_category: str | None, has_directive: bool = False,
) -> float:
    """Baseline reference that follows the routed material category.

    Retroactive-directive fix (Fas 2): when a directive (or usage_context)
    reroutes a component to a different material than its stored baseline was
    computed for — e.g. a "Väggytskikt" whose baseline is innervägg, rerouted to
    kakel by a "ge kakel"-directive added AFTER the baseline was set — the stored
    baseline is for the WRONG material. Comparing kakel alternatives against an
    innervägg baseline makes RC5 (climate_optimized must beat the baseline) drop
    every kakel option ("noll alternativ"), and the report would show an
    innervägg baseline next to kakel choices.

    So when the routed category diverges from the baseline's original category,
    recompute the conventional baseline on the routed category from its EPD
    typvärde. This keeps baseline + alternatives on the SAME material (the #420
    invariant) for the retroactive case, without a separate baseline rerun.

    A genuine Boverket-material baseline (Tier 1) is more accurate than a
    category typvärde, so it is NOT downgraded on a mere name/usage_context
    disagreement between the router and resolve_category. It IS rerouted when the
    user gave an explicit directive (``has_directive``) — there the user is
    actively changing the material, so the old material match no longer applies.

    Returns the stored ``bl_comp.co2e_kg`` unchanged when nothing diverged, the
    baseline is a protected Boverket hit, or the routed category has no usable
    typvärde (can't honestly recompute). Only the CO2e reference moves; cost
    stays as-is (no routed-category price source).
    """
    from aida.data.epd_baseline_medians import (
        aggregat_typvärde,
        get_baseline_typvärde,
        member_typvärde,
    )
    from aida.data.palats_client import component_subcategory
    from aida.data.unit_conversion import typical_item_mass

    orig_category = resolve_category(proj_comp.name, proj_comp.category)
    if not routed_category or routed_category == orig_category:
        return bl_comp.co2e_kg

    # Tier 1 Boverket baselines: don't silently downgrade to a typvärde unless an
    # explicit directive overrides the material (boverket_product is set only for
    # genuine Boverket hits — cleared for typvärde/uppskattning in baseline.py).
    if getattr(bl_comp, "boverket_product", "") and not has_directive:
        logger.debug(
            "Component %r has Boverket baseline and no directive; keeping it "
            "(router said %s, baseline category %s).",
            proj_comp.name, routed_category, orig_category,
        )
        return bl_comp.co2e_kg

    subcat = component_subcategory(proj_comp.name, routed_category)
    if routed_category == "ventilation" and subcat == "aggregat" and proj_comp.unit == "st":
        # Per airflow or size class, mirroring _apply_aggregat_typvärde in
        # baseline.py; None (keep the stored baseline) when neither is known.
        tv = aggregat_typvärde(proj_comp.name, proj_comp.usage_context,
                               proj_comp.quantity)["payload"]
    else:
        tv = get_baseline_typvärde(routed_category, proj_comp.unit, subcat)
    # kg->st bridge, mirroring _apply_epd_median_fallback in baseline.py: a
    # count-denominated component whose routed category only has a kg typvärde.
    if not tv and proj_comp.unit == "st":
        kg_tv = get_baseline_typvärde(routed_category, "kg", subcat)
        mass = typical_item_mass(routed_category, subcat)
        if kg_tv and mass:
            tv = {"baseline_co2e_per_unit": kg_tv["baseline_co2e_per_unit"] * mass}
    # m3 -> lm/m2 for a frame member, mirroring the same bridge in baseline.py.
    if not tv:
        tv = member_typvärde(routed_category, proj_comp.name, proj_comp.unit, subcat)
    if not tv:
        logger.warning(
            "Component %r rerouted %s->%s but routed category has no typvärde; "
            "keeping stored baseline (RC5 may still mismatch).",
            proj_comp.name, orig_category, routed_category,
        )
        return bl_comp.co2e_kg

    rerouted = round(tv["baseline_co2e_per_unit"] * proj_comp.quantity, 1)
    logger.info(
        "Component %r rerouted %s->%s: baseline %s -> %s kg (follows directive)",
        proj_comp.name, orig_category, routed_category, bl_comp.co2e_kg, rerouted,
    )
    return rerouted


def _reuse_coverage(available: int | None, need: float, units_match: bool) -> float | None:
    """Share of the component need a listing's stock covers, or None when the
    ratio cannot be formed: mismatched units (articles against m2), no need
    quantity, or a listing quantity Palats did not state (it reports 0).
    """
    if not units_match:
        return None
    if not need or need <= 0:
        return None
    if not available or available <= 0:
        return None
    return available / need


def _coverage_label(available: int, need: float, coverage: float) -> str:
    """'3 av 30 (10 %)', or 'hela behovet' once the stock covers it."""
    if coverage >= 1:
        return f"{available} av {int(need)} (hela behovet)"
    return f"{available} av {int(need)} ({coverage * 100:.0f} %)"


# Cap on Palats listings shown per component. Five, as before the unified
# prompt: enough to show a choice, few enough that the section stays a handful
# of lines next to an EPD list of up to 80.
_MAX_PALATS_PER_COMPONENT = 5

# Reasoning used ONLY when the ranking call itself failed and the reuse rows
# are appended without the model having seen them. Factual and short: the
# detail string (price, coverage, location, link) is appended after it, so
# the row still carries every number the table needs. Not the pre-2026-09-14
# climate paragraph, which Johanna read as boilerplate on every reuse row.
_FALLBACK_REUSE_REASONING = (
    "Återbruksannons på Palats som matchar komponenten. Analysen kunde inte "
    "rangordna den mot nyinköpsalternativen den här gången, så siffrorna "
    "nedan är annonsens egna utan bedömning av skick eller passform."
)


def _palats_candidates(
    component_name: str,
    quantity: float,
    project_unit: str,
    palats_listings: list[dict],
    category: str | None = None,
) -> tuple[list[tuple], Alternative | None]:
    """The Palats listings a component may be offered, and an info row if none.

    Shared by the unified ranking prompt (the normal path since 2026-09-14)
    and the deterministic fallback. Three filters, in order:

    1. Category match, on the ROUTED category when the caller has one, so the
       reuse list and the EPD list for a component describe the same
       material.
    2. Strict subcategory: when the component name names a subcategory
       ("Toalettstol" -> toalett), listings from other subcategories are
       dropped, not shown. A washbasin is not an alternative to a toilet.
    3. Coverage gate (MIN_REUSE_COVERAGE), before the cap of five so a listing
       that covers the need is not crowded out by five that cover a sliver.

    Returns ``(shown, info)``: ``shown`` is a list of ``(listing, coverage)``
    pairs (coverage None when the ratio cannot be formed), capped; ``info`` is
    an info-type Alternative explaining why nothing is shown, or None. Every
    listing dropped on the way is logged with the reason.
    """
    from aida.data.palats_client import (
        PalatsListing,
        component_subcategory,
        search_listings_for_component,
    )

    category = category or normalize_component_name(component_name)
    matched = search_listings_for_component(
        component_name, palats_listings, category=category or None,
    )
    target_subcat = component_subcategory(component_name, category) if category else ""

    # Strict subcategory filter: when the user asked for a specific subcategory
    # (e.g. "Toalettstol" → subcat "toalett"), drop listings from other
    # subcategories (handfat, dusch, etc.) entirely. They're wrong product
    # type for this component — showing them as "alternatives" is misleading,
    # not graceful degradation. Only fall back to broader category matches
    # when no subcategory keyword was inferable from the component name.
    if target_subcat:
        subcat_matches = [m for m in matched if m.subcategory == target_subcat]
        if not subcat_matches and matched:
            # Tell the user nothing of the type they asked for is listed, and
            # what the category does hold, so they can judge whether to look
            # manually. Since #347 the other subcategories are filtered out
            # rather than shown, so the wording has to say that too.
            other_subcats = sorted({m.subcategory for m in matched if m.subcategory})
            other_label = ", ".join(other_subcats) if other_subcats else "annan typ"
            # The component name is a noun of unknown gender, so any phrasing
            # that needs an article is wrong half the time: "Inget toalettstol"
            # was right only for the ett-words and wrong for every en-word
            # (dörr, belysning, toalettstol itself). Leading with the user's own
            # term verbatim removes the agreement problem entirely.
            #
            # Number agreement is the same kind of tell: "Palats har 1
            # produkter" is what a reader notices first, and it was there
            # because the count was interpolated into a fixed plural.
            one = len(matched) == 1
            count_label = "1 produkt" if one else f"{len(matched)} produkter"
            mismatch = "men den matchar inte" if one else "men ingen av dem matchar"
            hidden = "så den visas inte" if one else "så de visas inte"
            logger.info(
                "Palats: %d listings in %s but none in subcategory %r for %r; "
                "info row instead of reuse (%s)",
                len(matched), category, target_subcat, component_name, other_label,
            )
            return [], Alternative(
                name=f"{component_name}: inget på Palats just nu",
                co2e_kg=0,
                cost_sek=0,
                source="[Palats] palats.app",
                reasoning=(
                    f"Palats har {count_label} i kategorin "
                    f"{_CATEGORY_TEXT.get(category, category)} just nu "
                    f"({other_label}), {mismatch} {component_name.lower()}, "
                    f"{hidden} som alternativ här. Kolla tillbaka när nya "
                    "annonser publicerats, eller sök bredare manuellt på palats.app."
                ),
                alternative_type="info",
            )
        matched = subcat_matches

    if not matched:
        return [], None

    units_match = project_unit.lower() in ("st", "styck", "stk")

    # Coverage gate, applied before the cap so a listing that covers the need
    # is not crowded out by five that cover a sliver of it. Each hidden find
    # is logged with its ratio; the ratio also reaches the table for every
    # shown find, so the threshold can be audited from the output alone.
    shown: list[tuple[PalatsListing, float | None]] = []
    hidden_finds: list[tuple[PalatsListing, float]] = []
    for listing in matched:
        coverage = _reuse_coverage(listing.quantity, quantity, units_match)
        if coverage is not None and coverage < MIN_REUSE_COVERAGE:
            hidden_finds.append((listing, coverage))
        else:
            shown.append((listing, coverage))
    for listing, coverage in hidden_finds:
        logger.info(
            "Palats-fynd dolt för %r: %s täcker %s, under tröskeln %.0f %%",
            component_name, listing.title,
            _coverage_label(listing.quantity, quantity, coverage),
            MIN_REUSE_COVERAGE * 100,
        )
    if not shown:
        # Everything the category holds covers too little of the need. Say
        # so with the numbers, in the same info form as the subcategory miss
        # above, rather than showing nothing and looking like an empty search.
        ratios = "; ".join(
            f"{listing.title}: {_coverage_label(listing.quantity, quantity, coverage)}"
            for listing, coverage in hidden_finds
        )
        one = len(hidden_finds) == 1
        count_label = "1 annons" if one else f"{len(hidden_finds)} annonser"
        return [], Alternative(
            name=f"{component_name}: för lite på Palats just nu",
            co2e_kg=0,
            cost_sek=0,
            source="[Palats] palats.app",
            reasoning=(
                f"Palats har {count_label} som matchar {component_name.lower()}, "
                f"men lagret täcker under {MIN_REUSE_COVERAGE * 100:.0f} % av "
                f"behovet på {int(quantity)} {project_unit} ({ratios}), så "
                "återbruk visas inte som alternativ här. Kolla tillbaka när nya "
                "annonser publicerats, eller sök bredare manuellt på palats.app."
            ),
            alternative_type="info",
        )

    # Dedupe on title: Palats carries the same washbasin under a dozen
    # identical titles, and the model cannot tell twelve rows called
    # "Porslinstvättställ" apart, nor can the table. Keep the first of each.
    seen_titles: set[str] = set()
    unique: list[tuple[PalatsListing, float | None]] = []
    for listing, coverage in shown:
        key = listing.title.lower()
        if key in seen_titles:
            continue
        seen_titles.add(key)
        unique.append((listing, coverage))
    if len(unique) < len(shown):
        logger.info(
            "Palats: %d duplicate-title listings collapsed for %r",
            len(shown) - len(unique), component_name,
        )
    if len(unique) > _MAX_PALATS_PER_COMPONENT:
        logger.info(
            "Palats: %d matching listings for %r, showing the first %d",
            len(unique), component_name, _MAX_PALATS_PER_COMPONENT,
        )
    unique = unique[:_MAX_PALATS_PER_COMPONENT]

    # A reused frame member's climate figure is its transport, derived from
    # the weight the name gives (stomme_reuse, HENRIC-3364). Without that
    # weight there is no figure, and a default would be a number with no
    # basis: 2 kg CO2e per metre of stud is more than a new stud. The
    # listings are named in an info row instead, with the reason.
    if category == "stomme":
        from aida.data.stomme_reuse import family_label, reuse_figure

        figure = reuse_figure(component_name, project_unit)
        if figure.per_unit is None:
            label = family_label(target_subcat)
            one = len(unique) == 1
            listed = "; ".join(
                f"{listing.title}" + (f" ({listing.url})" if listing.url else "")
                for listing, _ in unique)
            logger.info("Palats: %d stomme listings for %r without a climate "
                        "figure: %s", len(unique), component_name, figure.note)
            return [], Alternative(
                name=f"{component_name}: återbruk på Palats utan klimatsiffra",
                co2e_kg=0,
                cost_sek=0,
                source="[Palats] palats.app",
                reasoning=(
                    f"Palats har {'1 annons' if one else f'{len(unique)} annonser'} "
                    f"med {label} som matchar {component_name.lower()}: {listed}. "
                    f"Klimatvärdet för återbruket är transporten, och den räknas ur "
                    f"vikten, som inte går att få fram här: "
                    f"{figure.note[:1].lower()}{figure.note[1:]}. "
                    f"{'Annonsen visas' if one else 'Annonserna visas'} därför utan "
                    "klimatsiffra och kan inte väljas som alternativ i tabellen."
                ),
                alternative_type="info",
            )
    return unique, None


def _reuse_figures(
    listing, coverage: float | None, quantity: float, project_unit: str, category: str,
    component_name: str = "",
) -> tuple[float | None, float, str, bool]:
    """(total_co2e, total_cost, detail, cost_is_per_article) for a listing.

    The numbers the table needs, computed once and used both in the prompt
    (so the model reasons about the same figures the row will carry) and in
    the Alternative built from the model's answer.

    A frame member (stomme) has no set figure: it is derived from the
    component's name (stomme_reuse), and total_co2e is None when the name does
    not give the weight. _palats_candidates turns such a component's listings
    into an info row before they get here.

    Pricing logic:
    - If project counts in "st" (fönster, dörr), Palats price * quantity
      gives a directly comparable total.
    - If project counts in "m2" (golv, vägg), we can't calculate total
      (unknown coverage per article). total_cost is then 0, unpriced, and the
      caller keeps listing.price as the row's per-article price. Until
      2026-09-30 the per-article price was returned as the total, and a 45 m2
      floor was summed as costing 725 kr.
    """
    from aida.data.palats_client import (
        _DEFAULT_REUSE_CO2E,
        LOGIN_REQUIRED_NOTE,
        REUSE_CO2E_PER_UNIT,
        listing_requires_login,
    )

    stomme_note = ""
    if category == "stomme":
        from aida.data.stomme_reuse import reuse_figure

        figure = reuse_figure(component_name, project_unit)
        co2e_per_unit = figure.per_unit
        stomme_note = figure.note
    else:
        co2e_per_unit = REUSE_CO2E_PER_UNIT.get(category, _DEFAULT_REUSE_CO2E)
    units_match = project_unit.lower() in ("st", "styck", "stk")
    total_co2e = co2e_per_unit * quantity if co2e_per_unit is not None else None

    if units_match and listing.price > 0:
        # Units match (both "st") — total is directly comparable.
        #
        # The total covers the FULL component quantity even when fewer are
        # in stock, and so does total_co2e above. That is deliberate: Aida
        # plans early, and stock turns over long before procurement
        # (Henric, 2026-08-15). What was wrong was saying nothing about it.
        # Live check 2026-08-14: 30 windows needed, best listing had 3, and
        # the row read "9 600 kr" with no hint that 27 were assumed.
        total_cost = listing.price * quantity
        price_note = f"Pris: {listing.price:.0f} SEK/st × {int(quantity)} = {int(total_cost)} SEK"
        if coverage is not None and coverage < 1:
            price_note += (
                f" | OBS: täckning {_coverage_label(listing.quantity, quantity, coverage)}"
                " finns i lager just nu."
                " Pris och klimatnytta räknas på hela behovet, alltså som om"
                " resten går att få tag på begagnat. Kontrollera tillgången"
                " innan siffran används i ett beslutsunderlag."
            )
        elif coverage is not None:
            price_note += f" | Täckning: {_coverage_label(listing.quantity, quantity, coverage)}"
        else:
            price_note += " | Täckning: okänd (antal ej angivet i annonsen)"
        cost_is_estimate = False
    elif listing.price > 0:
        # Units don't match: the price is for one article, and how much of the
        # need an article covers is unknown, so there is no total to give.
        total_cost = 0
        price_note = (
            f"Pris: {listing.price:.0f} SEK/st ({listing.quantity} tillgängliga)"
            " — yta per artikel okänd | Täckning: okänd (antal per "
            f"{project_unit} saknas)"
        )
        cost_is_estimate = True
    else:
        total_cost = 0
        price_note = f"{listing.quantity} tillgängliga"
        if coverage is not None:
            price_note += f" | Täckning: {_coverage_label(listing.quantity, quantity, coverage)}"
        else:
            price_note += " | Täckning: okänd"
        cost_is_estimate = False

    # A category without its own reuse figure gets the default, and the row
    # says so: the number is a placeholder, not something derived for this
    # kind of product, and a reader comparing it with a new product's EPD
    # should know which of the two is the soft one.
    default_note = stomme_note
    if category not in REUSE_CO2E_PER_UNIT and category != "stomme":
        default_note = (
            f"Klimatvärdet för återbruket är en schablon ({_DEFAULT_REUSE_CO2E:g} "
            f"kg CO2e per {project_unit or 'enhet'}): kategorin saknar eget "
            f"underlag för transport och upprustning"
        )

    location_note = f"Plats: {listing.location}" if listing.location else ""
    # A seller without a public shop (Sola's furniture, HENRIC-3365) can only
    # be linked to Palats' internal page, which shows a login form to anyone
    # without an account. The row says so rather than pass it off as a link.
    url_note = ""
    if listing.url:
        url_note = (f"Se annons ({LOGIN_REQUIRED_NOTE}): {listing.url}"
                    if listing_requires_login(listing.url)
                    else f"Se annons: {listing.url}")
    detail = " | ".join(p for p in [price_note, default_note, location_note, url_note] if p)
    total = round(total_co2e, 1) if total_co2e is not None else None
    return total, round(total_cost), detail, cost_is_estimate


def _reuse_alternative(
    listing, coverage: float | None, quantity: float, project_unit: str,
    category: str, reasoning: str, component_name: str = "",
) -> Alternative:
    """Build the table row for a Palats listing.

    ``reasoning`` is the model's own text from the unified ranking (or the
    fallback sentence when the call failed). The factual detail string is
    appended after it either way: the price arithmetic, the coverage ratio
    and the link are facts about the listing, not something the model is
    asked to reproduce.
    """
    total_co2e, total_cost, detail, cost_is_estimate = _reuse_figures(
        listing, coverage, quantity, project_unit, category, component_name,
    )
    text = reasoning.strip()
    if detail:
        text = f"{text} {detail}" if text else detail
    if cost_is_estimate:
        text += " OBS: Priset avser en artikel, inte totalbehovet."
    if listing.description:
        desc_preview = listing.description[:150]
        if len(listing.description) > 150:
            desc_preview += "..."
        text += f" Beskrivning: {desc_preview}"

    # The listing number is part of the name. Sola alone has 22 listings titled
    # "Porslinstvättställ" and 18 "Träfönster" at one location, and the table,
    # selection intent and multi-pick all identify a row by name: without the
    # number two such listings were one row to them and could not be combined.
    # The id is stable across reruns, so the name is too. Mark with * when the
    # price is per-article, not total: the table's footnote and canCombine read
    # the marker. The totals read article_price_sek and cost_sek instead.
    where = f"Palats återbruk, {listing.location}" if listing.location else "Palats återbruk"
    display_name = f"{listing.title} ({where}, annons {listing.id})" if listing.id else f"{listing.title} ({where})"
    if cost_is_estimate:
        display_name += " *"

    return Alternative(
        name=display_name,
        co2e_kg=total_co2e,
        cost_sek=total_cost,
        source=f"[Palats] palats.app/listing/{listing.id}",
        reasoning=text,
        alternative_type="reuse",
        available_quantity=listing.quantity,
        price_basis="listing" if listing.price > 0 else "",
        url=listing.url or "",
        article_price_sek=round(listing.price) if cost_is_estimate else 0,
    )


def _format_palats_list(
    candidates: list[tuple], quantity: float, project_unit: str, category: str,
    component_name: str = "",
) -> str:
    """The [Palats återbruk] rows of the unified prompt.

    One line per listing with the same figures the table will carry (CO2e for
    the full need, price arithmetic, coverage, location, subcategory, id), so
    the model ranks the row it will actually be shown next to the EPDs.
    """
    lines = []
    for listing, coverage in candidates:
        total_co2e, _total_cost, detail, _per_article = _reuse_figures(
            listing, coverage, quantity, project_unit, category, component_name,
        )
        sub = f" | Subkategori: {listing.subcategory}" if listing.subcategory else ""
        lines.append(
            f"- [Palats återbruk] {listing.title} | id: {listing.id} | "
            f"Uppskattad CO2e: {total_co2e} kg totalt för {quantity} {project_unit} "
            f"(transport och renovering, ingen nytillverkning) | {detail}{sub}"
        )
    return "\n".join(lines)


_LISTING_ID_RE = re.compile(r"listing/([A-Za-z0-9_-]+)")


def _match_palats_candidate(item: dict, candidates: list[tuple]):
    """Which candidate a model-written reuse row refers to, or None.

    By listing id from the source field first (the prompt asks for it
    verbatim), then by title containment, the same tolerance
    match_epd_by_name gives EPD names.
    """
    source = str(item.get("source", "") or "")
    m = _LISTING_ID_RE.search(source)
    if m:
        for listing, coverage in candidates:
            if str(listing.id) == m.group(1):
                return listing, coverage
    name = _match_key(str(item.get("name", "") or ""))
    if name:
        for listing, coverage in candidates:
            title = _match_key(listing.title)
            if title and (title in name or name in title):
                return listing, coverage
    return None


def _add_palats_reuse(
    alternatives: list[Alternative],
    component_name: str,
    quantity: float,
    project_unit: str,
    palats_listings: list[dict],
    category: str | None = None,
) -> None:
    """Deterministic fallback: append matching Palats listings (in-place).

    Until 2026-09-14 this was THE reuse path, run after the EPD ranking call
    and with a fixed paragraph as reasoning. Now the ranking prompt sees the
    same candidates (see _find_alternatives_with_epds) and writes the
    reasoning itself; this function remains for the case where that call
    failed outright, so a Palats find is never lost to an API error on the
    EPD side. Same candidates, same figures, factual fallback sentence.
    """
    category = category or normalize_component_name(component_name)
    shown, info = _palats_candidates(
        component_name, quantity, project_unit, palats_listings, category,
    )
    if info is not None:
        alternatives.append(info)
        return
    existing_names = {a.name.lower() for a in alternatives}
    for listing, coverage in shown:
        if listing.title.lower() in existing_names:
            continue
        alternatives.append(_reuse_alternative(
            listing, coverage, quantity, project_unit, category,
            _FALLBACK_REUSE_REASONING, component_name,
        ))
        existing_names.add(listing.title.lower())

def _route_components(
    project: Project,
    baseline_components: list,
    available_categories: set[str],
    user_feedback: str | None = None,
) -> dict[str, str]:
    """{component_id: category}. See _route_components_with_directives."""
    return _route_components_with_directives(
        project, baseline_components, available_categories, user_feedback,
    )[0]


# Key in the router's JSON answer listing the components whose material the
# user's directive changes. Leading underscore so it can never collide with a
# component id (c1, c2, ...).
_DIRECTED_KEY = "_direktiv"


def _route_components_with_directives(
    project: Project,
    baseline_components: list,
    available_categories: set[str],
    user_feedback: str | None = None,
) -> tuple[dict[str, str], set[str]]:
    """LLM-route each component to the EPD category its alternatives fit best.

    Why: retrieval is otherwise locked to normalize_component_name(name), which
    maps a "Väggytskikt" to innervägg even when it's a tiled wet-room wall, and
    a user directive ("ge kakel-alternativ") can't pull kakel EPDs. The router
    reads the component name + usage_context + directive and picks the apt
    catalog category. Falls back to normalize_component_name on ANY failure —
    it never blocks the pipeline.

    Returns ``(routing, directed)``. ``directed`` holds the ids of the
    components whose material the directive itself changes, as the router read
    it. Only those may have a Boverket baseline replaced (see
    _effective_baseline_co2e). Until 2026-09-26 that permission came from "is
    there any feedback at all", so "byt golvet till kakel", or even "bara
    svenska tillverkare", released every component's Boverket baseline, and a
    wet-room wall the router moved to kakel on its usage_context alone lost
    its correct baseline to a kakel typvärde without anyone asking for it
    (Fable audit 2026-07-19, P2 #3). When the router does not say, nothing is
    directed: a kept Boverket baseline is a visible mismatch at worst, a
    silently replaced one is a wrong number.
    """
    fallback: dict[str, str] = {}
    items: list[dict] = []
    for bl in baseline_components:
        proj_comp = next(
            (c for c in project.components if c.id == bl.component_id), None)
        name = proj_comp.name if proj_comp else bl.component_name
        declared = proj_comp.category if proj_comp else ""
        fallback[bl.component_id] = resolve_category(name, declared)
        if proj_comp:
            items.append({
                "id": bl.component_id,
                "name": proj_comp.name,
                "unit": proj_comp.unit,
                "usage_context": (proj_comp.usage_context or "")[:300],
            })

    if not items or not available_categories:
        return fallback, set()

    directive = (user_feedback or "").strip()[:500]
    # Skip the LLM call when there's nothing to disambiguate: with no directive
    # and no usage_context, name-based routing is already the right answer.
    if not directive and not any(it["usage_context"] for it in items):
        return fallback, set()

    cats = sorted(available_categories)
    directive_clause = ""
    answer_shape = '{"c1": "kategori", "c2": "kategori", ...}'
    if directive:
        directive_clause = (
            f'\nANVÄNDARENS ÖNSKEMÅL (kan gälla en eller flera komponenter): '
            f'"{directive}". Om önskemålet anger ett material (t.ex. kakel), '
            f'välj den kategorin för den komponent önskemålet rör, även om '
            f'komponentnamnet antyder ett annat material.\n'
            f'Lägg också till nyckeln "{_DIRECTED_KEY}": en lista med id för de '
            f'komponenter vars MATERIAL önskemålet uttryckligen byter. Bara '
            f'dem önskemålet handlar om, inte komponenter du flyttar till en '
            f'annan kategori av andra skäl (namn, användningskontext). Tom lista '
            f'om önskemålet inte byter material för någon komponent (t.ex. '
            f'"bara svenska tillverkare", "tänk bredare").'
        )
        answer_shape = ('{"c1": "kategori", "c2": "kategori", ..., '
                        f'"{_DIRECTED_KEY}": ["c1"]}}')
    prompt = (
        "Du mappar byggkomponenter till rätt materialkategori i en EPD-databas, "
        "så att klimatalternativ hämtas från rätt sorts material.\n\n"
        f"Tillgängliga kategorier: {', '.join(cats)}.\n\n"
        "Regler:\n"
        "- Utgå från komponentens NAMN och ANVÄNDNINGSKONTEXT (vad det faktiskt "
        "är för material/yta).\n"
        "- Kaklad våtrumsvägg eller -golv → kakel (inte innervägg/golv).\n"
        "- Målad yta / ommålning → farg. Rör/stambyte → vvs. Elkabel → el. "
        "Radiator/värmeelement → radiator.\n"
        "- Välj EXAKT en kategori per komponent, ur listan ovan."
        f"{directive_clause}\n\n"
        f"Komponenter:\n{json.dumps(items, ensure_ascii=False)}\n\n"
        f"Svara ENBART med JSON: {answer_shape}"
    )
    try:
        client = get_client()
        resp = call_model(
            client,
            model=_ROUTER_MODEL,
            max_tokens=REASONING_MAX_TOKENS,
            effort=EFFORT_HIGH,  # correctness step — material routing (toalett/kakel)
            messages=[{"role": "user", "content": prompt}],
        )
        text = extract_text(resp)
        data = extract_json_object(text, what="kategori-routingen")
    except Exception:
        logger.warning("Component routing failed; using name-based fallback",
                       exc_info=True)
        return fallback, set()

    if not isinstance(data, dict):
        logger.warning("Router returned %s, not a dict; name-based fallback",
                       type(data).__name__)
        return fallback, set()

    routed: dict[str, str] = {}
    for cid, default_cat in fallback.items():
        chosen = data.get(cid)
        if isinstance(chosen, str) and chosen in available_categories:
            if chosen != default_cat:
                logger.info("Routed %s: %s -> %s (was name-based)",
                            cid, default_cat, chosen)
            routed[cid] = chosen
        else:
            routed[cid] = default_cat

    directed: set[str] = set()
    if directive:
        raw = data.get(_DIRECTED_KEY)
        if isinstance(raw, list):
            directed = {str(x) for x in raw if str(x) in fallback}
        else:
            logger.warning(
                "Router gave no %s list for directive %r; treating it as "
                "changing no component's material (Boverket baselines kept)",
                _DIRECTED_KEY, directive[:80],
            )
        if directed:
            logger.info("Directive changes material for: %s",
                        ", ".join(sorted(directed)))
    return routed, directed


def find_alternatives(
    project: Project,
    baseline: Baseline,
    user_feedback: str | None = None,
) -> AlternativesResult:
    """Find climate-optimized alternatives for each component.

    Strategy:
    1. Load pre-categorized EPD data from Environdec
    2. Fetch available reuse listings from Palats marketplace
    3. For each component, gather the relevant EPDs AND the matching Palats
       listings (routed category, strict subcategory, coverage gate, cap 5)
    4. One LLM call ranks both sources together and writes the reasoning for
       every row, reuse included; the CO2e on every row is computed from the
       catalog row or the listing, not taken from the model
    5. If any component fails (the ranking call, or our own processing), the
       whole step raises a UserFacingError naming it. A partial result would
       be read as the project total downstream.
    """
    from aida.data import palats_client
    from aida.data.palats_client import fetch_listings

    epd_data = _load_epd_alternatives()

    # Route each component to the apt EPD category (name + usage_context +
    # directive), so a tiled "Väggytskikt" or a "ge kakel"-directive pulls kakel
    # EPDs instead of being locked to normalize_component_name's guess.
    routing, directed = _route_components_with_directives(
        project, baseline.components, set(epd_data.keys()), user_feedback,
    )

    # Fetch Palats listings once for the entire analysis
    palats_listings = fetch_listings()
    palats_status = palats_client.last_fetch_status
    has_palats = len(palats_listings) > 0
    if has_palats:
        logger.info("Palats: %d listings available for reuse matching", len(palats_listings))
    elif palats_status != "ok":
        logger.warning("Palats unavailable (status: %s)", palats_status)

    def _process_component(bl_comp):
        proj_comp = next(
            (c for c in project.components if c.id == bl_comp.component_id),
            None,
        )
        if not proj_comp:
            return None

        comp_key = routing.get(bl_comp.component_id) or resolve_category(
            proj_comp.name, proj_comp.category)
        # A stud or board the router filed under the wall, roof or floor it
        # stands in is still a frame member: its comparable rows, its reuse
        # listings and its baseline all live in stomme (HENRIC-3290).
        if _is_frame_component(proj_comp, comp_key):
            comp_key = "stomme"
        # Same for a piece of furniture the router filed elsewhere (a
        # "Förvaringsskåp" under fast_inredning, a "Golvskärm" under golv): its
        # rows and reuse are strict per kind and only exist here. A category
        # the component itself declares still wins, as in resolve_category.
        elif resolve_category(proj_comp.name, proj_comp.category) in (
                "los_inredning", "undertak", "tätskikt"):
            # Ceilings, absorbers and wet-room waterproofing the same way
            # (del 3): "Undertak akustikplattor" filed under tak met 39 roofs,
            # and "Tätskikt våtrum" under kakel met the tiles laid on it.
            comp_key = resolve_category(proj_comp.name, proj_comp.category)
        elif _names_del3_subtype(proj_comp):
            # And a glazed partition or a levelling compound, whose rows only
            # exist as their category's split subtype. In the production smoke
            # the router filed "Glasparti mot korridor" under fönster, and it
            # was offered used wooden windows against a window baseline.
            comp_key = resolve_category(proj_comp.name, proj_comp.category)
        if names_built_in_storage(proj_comp.name):
            # Built-in storage is joinery wherever the router filed it
            # (HENRIC-3367), and meets only cabinet carcasses (HENRIC-3394).
            comp_key = "fast_inredning"
            category_rows, no_alt_reason = _built_in_storage_rows(epd_data, proj_comp)
        elif comp_key == "los_inredning":
            category_rows, no_alt_reason = _furniture_rows(epd_data, proj_comp)
        elif comp_key == "undertak":
            category_rows, no_alt_reason = _ceiling_rows(epd_data, proj_comp)
        elif (comp_key == "innervägg"
              and _NAMES_PLASTERBOARD_RE.search(proj_comp.name.lower())
              and not _names_del3_subtype(proj_comp)):
            # Plasterboard in a wall meets boards of its own thickness
            # (HENRIC-3399), not the prefab panels and MDF in the bucket.
            category_rows, no_alt_reason = _board_class_rows(
                _plasterboard_rows(epd_data), proj_comp)
        else:
            category_rows, no_alt_reason = _stomme_rows(epd_data, proj_comp, comp_key)
        category_rows = _split_subtype_rows(category_rows, proj_comp, comp_key)
        if category_rows and not no_alt_reason:
            category_rows, no_alt_reason = _coverage_rows(
                category_rows, proj_comp, comp_key)
        if category_rows and not no_alt_reason:
            category_rows, no_alt_reason = _split_unit_reason(
                category_rows, proj_comp, comp_key)
        if category_rows and not no_alt_reason:
            category_rows, no_alt_reason = _aggregat_size_rows(
                category_rows, proj_comp, comp_key)
        epds_for_category = _select_epd_candidates(
            category_rows, proj_comp.unit, comp_key,
        ) if category_rows else []
        # Baseline reference must follow the routed material (retroactive
        # directive): a kakel-routed wall is validated against the kakel
        # baseline, not the stored innervägg one — else RC5 drops every kakel
        # alternative and the report shows the wrong-material baseline.
        # Per component: only a directive that names THIS component's material
        # may replace its Boverket baseline (see _route_components_with_directives).
        eff_baseline_co2e = _effective_baseline_co2e(
            proj_comp, bl_comp, comp_key,
            has_directive=bl_comp.component_id in directed,
        )

        # Note: candidates are NOT narrowed to the component's subcategory.
        # Narrowing to e.g. "toalett" only starved sanitet components of the
        # generic "Ceramic Sanitaryware" EPDs (which carry subcategory "") that
        # are the actual usable alternatives — and the "" bucket also holds
        # non-fixture noise, so admitting it wholesale is wrong too. Instead the
        # loader keeps best-N PER subcategory (so real toilets aren't cut by the
        # category cap), and the LLM picks the apt ones from the balanced set.
        # Precise per-component subcategory retrieval is Fas 2 (semantic).

        # Palats candidates are gathered BEFORE the ranking call so the model
        # sees reuse and new purchase side by side and ranks both. The
        # deterministic post-injection this replaced (2026-09-14) produced
        # Johanna's four April symptoms: Palats-only lists, unranked mixing,
        # and a fixed paragraph as "reasoning" on every reuse row. The routed
        # category goes along so the reuse search looks in the same material
        # bucket as the EPD search.
        palats_candidates: list[tuple] = []
        palats_info: Alternative | None = None
        if has_palats:
            palats_candidates, palats_info = _palats_candidates(
                proj_comp.name, proj_comp.quantity, proj_comp.unit,
                palats_listings, comp_key,
            )

        refused: list[str] = []
        alternatives = _find_alternatives_with_epds(
            proj_comp, bl_comp, epds_for_category, user_feedback,
            needs_analysis=project.needs_analysis,
            effective_baseline_co2e=eff_baseline_co2e,
            palats_candidates=palats_candidates,
            category=comp_key,
            refused=refused,
        )

        # Validate data quality: filter zero CO2, component-only parts, flag prices
        dropped: list[str] = []
        alternatives = _validate_alternatives(
            alternatives, eff_baseline_co2e, proj_comp.name, proj_comp.quantity,
            category=comp_key, unit=proj_comp.unit, dropped=dropped,
        )

        # Palats had listings in the category but none of the asked-for type,
        # or none covering enough of the need: say so, in the same row form
        # as before, instead of leaving the reuse side silent.
        if palats_info is not None:
            alternatives.append(palats_info)

        # Show Palats status: connection error vs no matches for this category
        palats_reuse_count = sum(
            1 for a in alternatives
            if a.alternative_type == "reuse" and "[Palats]" in a.source
        )
        if palats_reuse_count == 0:
            if palats_status in ("no_credentials", "auth_failed"):
                alternatives.append(Alternative(
                    name="Palats ej tillgänglig (autentisering)",
                    co2e_kg=0,
                    cost_sek=0,
                    source="[Palats] palats.app",
                    reasoning=(
                        "Kunde inte ansluta till Palats — autentisering misslyckades. "
                        "Återbruksprodukter kan inte sökas. Kontakta systemadministratör."
                    ),
                    alternative_type="info",
                ))
            elif palats_status == "api_error":
                alternatives.append(Alternative(
                    name="Palats ej tillgänglig (anslutningsfel)",
                    co2e_kg=0,
                    cost_sek=0,
                    source="[Palats] palats.app",
                    reasoning=(
                        "Kunde inte hämta data från Palats — anslutningsfel eller "
                        "timeout. Återbruksprodukter kan inte sökas just nu. "
                        "Försök igen senare."
                    ),
                    alternative_type="info",
                ))
            elif has_palats:
                alternatives.append(Alternative(
                    name="Inget tillgängligt i Palats",
                    co2e_kg=0,
                    cost_sek=0,
                    source="[Palats] palats.app",
                    reasoning=(
                        "Inga matchande återbruksprodukter hittades i Palats "
                        "(Karlstads kommuns interna marknadsplats) för denna kategori "
                        "just nu. Utbudet ändras löpande — kolla igen senare."
                    ),
                    alternative_type="info",
                ))

        selectable = [a for a in alternatives if a.alternative_type != "info"]
        if not selectable:
            alternatives.append(Alternative(
                name=f"Inga alternativ hittades för {proj_comp.name}",
                co2e_kg=eff_baseline_co2e,
                cost_sek=bl_comp.cost_sek,
                source="N/A",
                # A pool that had rows says what became of them (HENRIC-3362
                # follow-up): on stage "Håldäck 200 mm" had one comparable
                # slab, above the baseline, and read "Inga alternativ
                # identifierade" as if the catalog were empty.
                reasoning=(no_alt_reason
                           or _filtered_pool_reason(
                               epds_for_category, proj_comp, comp_key,
                               eff_baseline_co2e, dropped, refused)
                           or _refused_reason(refused)
                           or "Inga alternativ identifierade."),
                alternative_type="baseline",
            ))

        return ComponentAlternatives(
            component_id=bl_comp.component_id,
            component_name=bl_comp.component_name,
            baseline_co2e_kg=eff_baseline_co2e,
            baseline_cost_sek=bl_comp.cost_sek,
            alternatives=alternatives,
        )

    # Run per-component LLM calls in parallel (I/O-bound).
    # max(1, ...) guards against an empty baseline — ThreadPoolExecutor(0) raises.
    max_workers = max(1, min(len(baseline.components), 5))
    results_map = {}
    failures: dict[str, Exception] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_comp = {
            executor.submit(_process_component, bl): bl
            for bl in baseline.components
        }
        for future in as_completed(future_to_comp):
            bl = future_to_comp[future]
            try:
                r = future.result()
                if r:
                    results_map[bl.component_id] = r
            except Exception as exc:
                logger.warning("Component failed: %s", bl.component_name, exc_info=True)
                failures[bl.component_id] = exc

    # A component that failed used to be dropped here with a log line, and
    # nothing downstream noticed: the report gate iterates the components this
    # returns, so "every component chosen" held for the survivors and the
    # report presented their sum as the project total. The baseline refuses
    # the same thing in _complete_baseline. Raised before pricing and
    # commentary, so a run that is going to be discarded does not pay for them.
    if failures:
        raise _alternatives_failed(baseline.components, failures)

    # Preserve original component order
    component_results = [
        results_map[bl.component_id]
        for bl in baseline.components
        if bl.component_id in results_map
    ]

    # Batch price enrichment for alternatives missing prices
    _enrich_alternative_prices(component_results, project, routing)

    # DoD B1: drop alternatives still at cost_sek=0 after enrichment — but only
    # those unactionable on BOTH axes (no price AND no verified EPD). An
    # alternative earns its place by being actionable on at least one:
    #   - has a real price → buyable
    #   - is EPD-backed ([EPD] source) → carries a verified CO2 number, which is
    #     the whole point of a climate tool. These obscure EPD products (foreign
    #     ceramic manufacturers etc.) often can't be web-priced, but deleting
    #     them lost real kakel/facade climate options entirely (Johanna's
    #     toalettblock: every kakel alternative vanished here even after the
    #     baseline reroute). Keep them flagged "Pris ej tillgängligt" (the
    #     validator already adds that note) instead of hiding the climate signal.
    # "reuse" stays exempt: Palats reuse listings are legitimately free.
    for comp in component_results:
        before = len(comp.alternatives)
        comp.alternatives = [a for a in comp.alternatives if _b1_keep_alternative(a)]
        removed = before - len(comp.alternatives)
        if removed:
            logger.info("B1 filter: removed %d unpriced non-EPD alternatives from %s", removed, comp.component_name)

    result = AlternativesResult(components=component_results)
    result.commentary = _generate_commentary(project, baseline, result)
    return result


def _alternatives_failed(baseline_components: list, failures: dict) -> UserFacingError:
    """The error for a run in which one or more components failed.

    Names every failed component, in project order, and says why nothing is
    shown. A failed ranking call (API error, unreadable answer) is 502 and
    worth retrying as is, a timed-out one 504; anything else is a fault on
    our side, 500.
    """
    names = [bl.component_name for bl in baseline_components
             if bl.component_id in failures]
    upstream = all(isinstance(e, RankingFailed) for e in failures.values())
    timed_out = upstream and all(e.timed_out for e in failures.values())
    if timed_out:
        # Same advice as the route's own 504 branch, which a timeout inside
        # the ranking call never reached (it was swallowed here before).
        why, status = "Modellen svarade inte i tid.", 504
        retry = "Försök igen, eller minska antalet komponenter."
    elif upstream:
        why, status = "Modellen svarade inte eller gav ett svar som inte gick att läsa.", 502
        retry = "Försök igen, det brukar gå igenom vid nästa försök."
    else:
        why, status = "Ett fel uppstod när förslagen sammanställdes.", 500
        retry = "Försök igen. Står felet kvar, hör av dig så tittar vi i loggarna."
    return UserFacingError(
        f"Alternativen kunde inte tas fram för alla komponenter (saknas: "
        f"{', '.join(names)}). {why} Inga alternativ visas, eftersom en "
        f"jämförelse där komponenter saknas ser ut som hela projektet. {retry}",
        status_code=status,
    )


def _kg_text(value: float) -> str:
    """4340.0 -> '4 340', 12.34 -> '12,3': a kg figure the way the table reads."""
    if abs(value) >= 100:
        return f"{round(value):,}".replace(",", " ")
    return f"{round(value, 1):g}".replace(".", ",")


def _filtered_pool_reason(
    epds: list[dict], proj_comp, category: str | None, baseline_co2e: float | None,
    dropped: list[str] | None = None, refused: list[str] | None = None,
) -> str:
    """Why a component whose candidate pool was NOT empty ends with nothing.

    "" when the pool was empty: that case has its own messages (no_alt_reason
    and the default sentence). Otherwise every pool row is figured the way the
    row would have been (_catalog_co2e for this component), and the sentence
    says how many there were, which were above the baseline, by how much, and
    what else removed the rest. Never a bare "Inga alternativ identifierade".
    """
    if not epds:
        return ""
    figured: list[tuple[float, str]] = []
    incomparable: list[str] = []
    for e in epds:
        co2e, note = _catalog_co2e(e, proj_comp, category or "")
        if co2e is None:
            incomparable.append(note)
            continue
        name = str(e.get("name") or "okänd produkt")
        owner = str(e.get("owner") or "").strip()
        figured.append((co2e, f"{name} ({owner})" if owner and owner.lower()
                        not in name.lower() else name))
    if not figured and refused:
        # Nothing in the pool fits the component's unit and the model's rows
        # were refused for exactly that: _refused_reason says it, with advice.
        return ""
    figured.sort()
    base = baseline_co2e if isinstance(baseline_co2e, (int, float)) else None
    over = [f for f in figured if base is not None and base > 0 and f[0] >= base]
    under = [f for f in figured if f not in over]
    n = len(figured)
    parts: list[str] = []

    if figured and not under:
        lowest, name = over[0]
        if n == 1:
            parts.append(
                f"Katalogen har 1 jämförbar EPD för komponenten, {name}, men den "
                f"ligger över baslinjen ({_kg_text(lowest)} mot {_kg_text(base)} kg "
                f"CO2e), så inget bättre alternativ finns i katalogen.")
        else:
            parts.append(
                f"Katalogen har {n} jämförbara EPD:er för komponenten, men alla "
                f"ligger över baslinjen (lägst {name}, {_kg_text(lowest)} mot "
                f"{_kg_text(base)} kg CO2e), så inget bättre alternativ finns i "
                f"katalogen.")
    elif under:
        lowest, name = under[0]
        count = "1 jämförbar EPD" if n == 1 else f"{n} jämförbara EPD:er"
        below = ("den ligger" if n == 1 else
                 f"{len(under)} av dem ligger" if len(under) < n else "alla ligger")
        against = (f" ({name}, {_kg_text(lowest)} mot {_kg_text(base)} kg CO2e)"
                   if base else f" ({name}, {_kg_text(lowest)} kg CO2e)")
        why = "; ".join(dict.fromkeys(dropped or []))
        if why:
            tail = f"men den togs bort: {why}." if len(under) == 1 else f"men de togs bort: {why}."
        elif refused:
            tail = ("men förslagen som kom tillbaka gick inte att koppla till "
                    "katalogen (" + "; ".join(dict.fromkeys(refused)) + ").")
        else:
            tail = ("men analysen tog inte med den den här gången. Kör alternativen "
                    "igen för att få den jämförd." if len(under) == 1 else
                    "men analysen tog inte med någon av dem den här gången. Kör "
                    "alternativen igen för att få dem jämförda.")
        parts.append(f"Katalogen har {count} för komponenten, och {below} under "
                     f"baslinjen{against}, {tail}")
    if incomparable:
        k = len(incomparable)
        reasons = "; ".join(dict.fromkeys(incomparable))
        lead = ("Ytterligare " if figured else "Katalogen har ")
        noun = "1 EPD" if k == 1 else f"{k} EPD:er"
        parts.append(f"{lead}{noun} för kategorin går inte att jämföra med "
                     f"komponenten ({reasons}).")
        if not figured and any("typisk vikt" in r for r in incomparable):
            parts.append("Ange vilken sorts produkt det är (till exempel toalettstol "
                         "eller handfat), eller mängden i kg, så kan de jämföras.")
    return " ".join(parts)


def _refused_reason(refused: list[str]) -> str:
    """Why a component ended with no comparable alternative although the model
    proposed some, for the "Inga alternativ hittades" row. "" when nothing was
    refused, so the caller falls back to its own sentence."""
    if not refused:
        return ""
    reasons = "; ".join(dict.fromkeys(refused))
    text = f"Inget av förslagen från EPD-listan gick att jämföra med baslinjen ({reasons})."
    if any("typisk vikt" in r for r in refused):
        text += (
            " Ange vilken sorts produkt det är (till exempel toalettstol eller "
            "handfat), eller mängden i kg, så kan de jämföras."
        )
    return text


def _enrich_alternative_prices(
    components: list[ComponentAlternatives],
    project: Project,
    routing: dict[str, str] | None = None,
) -> None:
    """Price the alternatives that came back at cost_sek == 0.

    Two passes, both single batch calls. Pass 1 is a web search for a real
    market price. Pass 2 asks the model to estimate the ones the search could
    not resolve, because an empty cost column is useless to someone choosing
    between two materials while a labelled estimate is not (Henric,
    2026-08-20). Which pass produced a number is carried on price_basis so the
    table and the report can say so.

    Prices come back per unit (SEK/m², SEK/st) and are multiplied by the
    project's component quantity here — storing the per-unit value as the total
    would understate cost by the quantity factor (725 SEK vs 45 × 725 for 45 m²
    of flooring). validate_total_price then runs as a safety net for both that
    regression and out-of-range prices.

    Never per-product lookups: sequential calls are what caused the 5+ minute
    timeouts this path was restructured to avoid. Pass 2 is skipped when the
    request no longer has time for it.
    """
    import time

    from aida.api_client import remaining_budget
    from aida.data.price_validation import check_total_price
    from aida.data.pricing_provider import (
        BASIS_ADJUSTED,
        BASIS_LLM_ESTIMATE,
        BASIS_WEB_SEARCH,
        estimate_prices_batch,
        lookup_prices_batch,
        price_unit_matches,
    )

    started_at = time.monotonic()
    quantity_by_cid = {c.id: c.quantity for c in project.components}
    unit_by_cid = {c.id: c.unit for c in project.components}
    routing = routing or {}
    category_by_cid = {
        c.component_id: (routing.get(c.component_id)
                         or normalize_component_name(c.component_name))
        for c in components
    }

    products_needing_prices: list[tuple[str, str]] = []
    alt_index: list[tuple[int, int]] = []  # (comp_idx, alt_idx) for mapping back

    for ci, comp in enumerate(components):
        for ai, alt in enumerate(comp.alternatives):
            # "reuse" excluded: a Palats reuse listing is a specific second-hand
            # item, so a generic market price for the material would not be its
            # price. Those keep whatever the listing said.
            if alt.cost_sek <= 0 and alt.alternative_type not in ("baseline", "info", "reuse"):
                # The component's unit goes into the question ("enhet: m2"),
                # so the answer comes back in the unit the quantity is counted
                # in, and is checked against it in apply() below.
                products_needing_prices.append(
                    (alt.name, unit_by_cid.get(comp.component_id, "") or ""))
                alt_index.append((ci, ai))

    if not products_needing_prices:
        return

    def apply(prices: dict, wanted: list[tuple[str, str]],
              index: list[tuple[int, int]], basis: str) -> list[int]:
        """Write prices onto their alternatives. Returns positions left unpriced."""
        unresolved: list[int] = []
        is_estimate = basis == BASIS_LLM_ESTIMATE
        for pos, ((ci, ai), (name, _unit)) in enumerate(zip(index, wanted)):
            price_result = prices.get(name.lower())
            if not price_result:
                unresolved.append(pos)
                continue
            price_per_unit, unit, source = price_result
            comp = components[ci]
            alt = comp.alternatives[ai]
            quantity = quantity_by_cid.get(comp.component_id, 0) or 0
            # A per-unit price is only multiplied by a quantity counted in the
            # same unit. SEK/st times 45 m2 is not a cost, it is a number, and
            # before 2026-09-26 it went into the total unchecked. Left unpriced
            # instead, so the estimate pass gets a go and, failing that, the row
            # says "Pris saknas" rather than a figure off by the pack size.
            comp_unit = unit_by_cid.get(comp.component_id, "") or ""
            if comp_unit and not price_unit_matches(unit, comp_unit):
                logger.warning(
                    "Price for '%s' is per %r but the component is counted in %r; "
                    "discarded (%s)", name, unit, comp_unit, basis,
                )
                unresolved.append(pos)
                continue
            alt.cost_sek = round(price_per_unit * quantity) if quantity > 0 else round(price_per_unit)
            # A typical installed price for this KIND of material, or the
            # model's own guess at one. Neither is this product's asking price,
            # and the table renders all three in the same column, so which one
            # it is has to travel with the number.
            alt.price_basis = basis
            alt.reasoning = alt.reasoning.replace(". Pris ej tillgängligt.", "")
            alt.reasoning = alt.reasoning.replace("Pris ej tillgängligt.", "")

            # An outlier is replaced by the category's typical price. The row
            # then says so, and no longer names the source of a figure it does
            # not show (review L2; the baseline does the same with
            # ADJUSTED_PRICE_SOURCE).
            category = category_by_cid.get(comp.component_id, "")
            note = ""
            if quantity > 0 and category:
                checked = check_total_price(
                    alt.cost_sek, quantity, category, is_estimate=is_estimate,
                    unit=comp_unit,
                )
                alt.cost_sek = checked.cost
                note = checked.note
                if checked.replaced:
                    alt.price_basis = BASIS_ADJUSTED
                    source = ""
            if source and source.lower() not in alt.reasoning.lower():
                alt.reasoning = alt.reasoning.rstrip(". ") + f". Prisunderlag: {source}."
            if note and note.lower() not in alt.reasoning.lower():
                alt.reasoning = alt.reasoning.rstrip(". ") + f". {note}."

            logger.info(
                "Priced '%s' via %s: %d SEK/%s x %g = %d SEK total",
                alt.name, basis, round(price_per_unit), unit, quantity, alt.cost_sek,
            )
        return unresolved

    # Pass 1: web search for a real market price.
    unresolved = apply(
        lookup_prices_batch(products_needing_prices),
        products_needing_prices, alt_index, BASIS_WEB_SEARCH,
    )

    # Pass 2: the model's own estimate for whatever the search left. This is the
    # fallback that already existed for a single product (lookup_price falls
    # back to _estimate_price_without_search) but was unreachable in a batch, so
    # any analysis with two or more unpriced alternatives never got it. That is
    # why "Pris saknas" appeared as often as it did.
    if unresolved:
        budget = remaining_budget(started_at)
        if budget < 20:
            logger.warning(
                "%d alternatives unpriced and only %.0fs budget left; skipping estimate pass",
                len(unresolved), budget,
            )
        else:
            still_wanted = [products_needing_prices[p] for p in unresolved]
            still_index = [alt_index[p] for p in unresolved]
            unresolved = [
                still_index[p] for p in apply(
                    estimate_prices_batch(still_wanted, timeout=budget),
                    still_wanted, still_index, BASIS_LLM_ESTIMATE,
                )
            ]

    if unresolved:
        # Genuinely unpriceable. Not zero kronor — see compute_aggregate.
        logger.info("%d alternatives unpriced after web search and estimate", len(unresolved))


def _find_alternatives_with_epds(
    proj_comp,
    bl_comp,
    epds: list[dict],
    user_feedback: str | None = None,
    needs_analysis: NeedsAnalysis | None = None,
    effective_baseline_co2e: float | None = None,
    palats_candidates: list[tuple] | None = None,
    category: str | None = None,
    refused: list[str] | None = None,
) -> list[Alternative]:
    """Use LLM to rank EPD alternatives and Palats reuse listings together.

    ``effective_baseline_co2e`` is the baseline reference after any
    routed-category reroute (see _effective_baseline_co2e). When a component was
    rerouted (e.g. innervägg→kakel by a directive), the LLM MUST reason about
    savings against the rerouted baseline — otherwise its "−45% CO2e" reasoning
    is computed against the wrong (stale) material and the report text
    contradicts the actual savings math.

    ``palats_candidates`` are ``(listing, coverage)`` pairs from
    _palats_candidates. They go into the same prompt as the EPDs, tagged
    [Palats återbruk], and the model ranks both sources in one list and
    writes the reasoning for the reuse rows too. The figures on a reuse row
    (CO2e from REUSE_CO2E_PER_UNIT, the listing's own price) are computed
    here, not taken from the model.

    The figures on an EPD row are ours too (_catalog_co2e): the model picks
    and argues, the catalog row times the quantity is the number. A row that
    maps to no catalog row, or only to one in another unit, is dropped and
    logged, and the reason is appended to ``refused`` when the caller passes
    a list, so the component can say why nothing was comparable.

    If the call itself fails (API error, unreadable answer) this raises
    RankingFailed. It used to return [] or the Palats rows alone, which the
    table then showed as "Inga alternativ hittades", i.e. as the model having
    looked and found no better product.
    """
    client = get_client()
    palats_candidates = palats_candidates or []
    category = category or normalize_component_name(proj_comp.name)

    if not epds and not palats_candidates:
        # Nothing to rank. The old prompt still made the call here and asked
        # the model to return [], which cost a request per empty category.
        logger.info("No EPDs and no Palats candidates for %s; skipping ranking call",
                    proj_comp.name)
        return []

    # Baseline the LLM reasons against: the rerouted value when it differs from
    # the stored one, else the stored baseline.
    rerouted = (
        effective_baseline_co2e is not None
        and effective_baseline_co2e != bl_comp.co2e_kg
    )
    baseline_for_prompt = effective_baseline_co2e if rerouted else bl_comp.co2e_kg

    baseline_source = (getattr(bl_comp, "source", "") or "").lower()
    if rerouted:
        # A rerouted baseline is the routed category's EPD typvärde, regardless
        # of what the original (wrong-material) baseline source was.
        baseline_label = "EPD-typvärde (kategori-aggregat)"
    elif "uppskattning" in baseline_source:
        baseline_label = "uppskattning"
    elif "epd-typvärde" in baseline_source or "epd-medel" in baseline_source or "epd-median" in baseline_source:
        baseline_label = "EPD-typvärde (kategori-aggregat)"
    else:
        baseline_label = "Boverket Typical"
    prompt = f"""Komponent: {proj_comp.name}
Antal: {proj_comp.quantity} {proj_comp.unit}
Baslinje CO2e: {baseline_for_prompt} kg ({baseline_label})
Baslinje kostnad: {_prompt_cost(bl_comp.cost_sek)}

Föreslå alternativ ur listorna nedan: 2-4 EPD-alternativ, plus varje Palats-annons som passar komponenten. Rangordna dem TILLSAMMANS i en lista. Välj bara EPD-alternativ med lägre CO2e än baslinjen: en ny produkt som inte sänker klimatpåverkan visas inte för användaren, och systemet förklarar självt när katalogen bara har sådana. Rangordna med lägst CO2e först. I reasoning: ange explicit hur alternativet jämför mot baslinjen (t.ex. "−45% CO2e, och nordisk leverantör").
"""

    # Project-level needs (user-approved) — overarching framing for the whole
    # selection. Per-component usage_context below is the targeted derivation
    # of these for this specific component.
    inferred = getattr(needs_analysis, "inferred", "") if needs_analysis else ""
    if inferred:
        prompt += f"""
PROJEKTETS BEHOV (godkänt av användaren):
{inferred}

→ Detta är överordnad kontext. Föreslå alternativ från EPD-listan som vanligt och flagga i reasoning hur varje alternativ möter eller utmanar dessa behov. Utelämna ENDAST om alternativet är uppenbart fel byggdel för funktionen (t.ex. utomhusgolv för innerentré). I tveksamma fall: föreslå med tydlig caveat i reasoning — användaren har sista ordet, inte agenten.
"""

    usage_context = getattr(proj_comp, "usage_context", "")
    if usage_context:
        prompt += f"""
ANVÄNDNINGSKONTEXT FÖR DENNA KOMPONENT (funktionella krav från intake):
{usage_context}

→ Använd kontexten för att resonera om hur varje alternativ möter kraven. Föreslå alternativ från EPD-listan även om någon egenskap är osäker — flagga osäkerheten i reasoning (t.ex. "lägre CO2e men kräver halksäkringsbehandling"). Utelämna bara om alternativet är uppenbart fel byggdel.
"""

    if epds:
        r_caveat = ISOLERING_R_CAVEAT if any(
            e.get("category") == "isolering" for e in epds
        ) else ""
        prompt += f"""
TILLGÄNGLIGA EPD:er FÖR DENNA KATEGORI ({len(epds)} st):
{_format_epd_list(epds)}{r_caveat}

Välj de 2-4 bästa alternativen från listan ovan. Beräkna total CO2e baserat på EPD-värdet × {proj_comp.quantity} {proj_comp.unit}.
Om EPD-enheten inte matchar projektenheten: hoppa över den EPD:n. Räkna aldrig om mellan enheter med en antagen densitet eller tjocklek — en sådan omräkning ser ut som en besparing men är en gissning.{_mass_bridge_clause(proj_comp, category)}
Se till att minst ett alternativ har "nordisk leverantör", om listan innehåller något sådant. Finns inget, säg det i reasoning i stället för att låtsas.
"""
    else:
        prompt += """
Inga EPD:er tillgängliga för denna kategori. Föreslå inga nyinköp; rangordna bara återbruksannonserna nedan.
"""

    if palats_candidates:
        prompt += f"""
ÅTERBRUKSANNONSER PÅ PALATS FÖR DENNA KATEGORI ({len(palats_candidates)} st, redan filtrerade på produkttyp och lagersaldo):
{_format_palats_list(palats_candidates, proj_comp.quantity, proj_comp.unit, category, proj_comp.name)}

Ta med varje annons ovan i din rankade lista, med alternative_type "reuse" och source "[Palats] palats.app/listing/<id>". Använd CO2e- och prissiffrorna från raden. Skriv ett eget resonemang per annons: vad den är, hur den passar komponenten och behoven, vad täckningen betyder i praktiken, och vad som ska kontrolleras innan den räknas in. Utelämna en annons bara om den är uppenbart fel produkt för komponenten, och säg då varför i reasoning på ett av de andra alternativen.
"""
    elif epds:
        prompt += """
Inga återbruksannonser på Palats matchar denna komponent just nu, så listan består bara av nyinköp.
"""

    if user_feedback:
        prompt += f"\nAnvändarens önskemål: {user_feedback}\n"

    prompt += "\nSvara med JSON-array."

    # Only the call and the parse are "the model failed". A fault in the row
    # handling below is our own bug and must not be reported as a busy API.
    try:
        response = call_model(
            client,
            model=DEFAULT_MODEL,
            max_tokens=REASONING_MAX_TOKENS,
            effort=EFFORT_HIGH,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = extract_text(response)
        data = extract_json_value(text, what="alternativsökningen")
    except Exception as exc:
        logger.warning("Ranking call failed for %s", proj_comp.name, exc_info=True)
        raise RankingFailed(
            proj_comp.name,
            timed_out=isinstance(exc, (anthropic.APITimeoutError, TimeoutError)),
        ) from exc

    if isinstance(data, dict):
        data = data.get("alternatives", [data])
    if not isinstance(data, list):
        data = [data]

    results = []
    used_candidates: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            logger.info("Skipping non-object item in alternatives: %r", item)
            continue
        source = str(item.get("source", "") or "")
        is_reuse = (
            item.get("alternative_type") == "reuse"
            or "[palats]" in source.lower()
        )
        if is_reuse:
            # A reuse row is only ever a Palats candidate the prompt showed.
            # Its figures are ours (REUSE_CO2E_PER_UNIT and the listing
            # price), its reasoning is the model's. Anything that does not
            # map back to a candidate is fabricated and dropped, logged.
            hit = _match_palats_candidate(item, palats_candidates)
            if hit is None:
                logger.warning(
                    "Dropped reuse row %r for %s: matches no Palats candidate "
                    "(source=%r)", item.get("name"), proj_comp.name, source,
                )
                continue
            listing, coverage = hit
            if str(listing.id) in used_candidates:
                logger.info("Duplicate reuse row for listing %s; keeping the first",
                            listing.id)
                continue
            used_candidates.add(str(listing.id))
            results.append(_reuse_alternative(
                listing, coverage, proj_comp.quantity, proj_comp.unit,
                category, str(item.get("reasoning", "") or ""), proj_comp.name,
            ))
            continue

        # An EPD row, the same rule as a reuse row: it must map back to a
        # row the prompt showed, and its figure is the catalog's, not the
        # model's multiplication. What the model contributes is the choice
        # and the reasoning.
        alt_name = str(item.get("name") or "").strip()
        matched = _catalog_row_for(item, epds)
        if matched is None:
            logger.warning(
                "Dropped EPD row %r for %s: matches no single catalog candidate "
                "(source=%r)", alt_name, proj_comp.name, source,
            )
            if refused is not None:
                refused.append("ett förslag motsvarade ingen rad i EPD-listan")
            continue
        co2e, note = _catalog_co2e(matched, proj_comp, category)
        if co2e is None:
            logger.warning(
                "Dropped EPD row %r for %s: %s is declared per %s, component "
                "counted in %s (%s)", alt_name, proj_comp.name, matched.get("name"),
                _epd_comparable(matched)[1], proj_comp.unit, note,
            )
            if refused is not None:
                refused.append(note)
            continue
        model_co2e = _model_number(item.get("co2e_kg"))
        if model_co2e is None or abs(model_co2e - co2e) > 0.05 * co2e:
            logger.info(
                "CO2e for %r (%s) set from the catalog: %s kg, the model wrote %r",
                alt_name, proj_comp.name, co2e, item.get("co2e_kg"),
            )
            if not note:
                # The model's reasoning ("−98 % CO2e") was argued from its own
                # figure; the arithmetic that replaced it goes next to it.
                gwp, unit = _epd_comparable(matched)
                note = (
                    f"CO2e räknat från EPD:ns deklarerade värde: {gwp:g} kg CO2e/"
                    f"{unit} × {_model_number(proj_comp.quantity):g} "
                    f"{proj_comp.unit} = {co2e:g} kg."
                )
        # A price the model could not write as a number is an unknown price,
        # which the enrichment pass fills in; never a reason to lose the row.
        cost = _model_number(item.get("cost_sek"))
        if cost is None or cost < 0:
            if item.get("cost_sek") not in (None, 0, "", "0"):
                logger.info("Unreadable cost_sek %r for %r; treated as unknown",
                            item.get("cost_sek"), alt_name)
            cost = 0

        # Every row that gets here is a catalog row, whatever the model put in
        # `source`, so it is tagged as one (the B1 filter keeps unpriced EPD
        # rows and drops unpriced estimates).
        if not source.lower().startswith("[epd]"):
            registry = matched.get("source_registry") or "environdec"
            source = " ".join(p for p in (
                "[EPD]", _REGISTRY_LABELS.get(registry, registry),
                str(matched.get("reg_no") or "")) if p)

        # Which GWP indicator this rests on is a fact about the catalog, not
        # something to hope the model repeats.
        gwp_basis = matched.get("gwp_basis", "") or ""
        if gwp_basis == "ghg":
            # Rides in `source` as well as its own field: source is what the
            # report's component table prints, so the basis reaches the
            # document without a separate plumbing path.
            source = f"{source} (GWP-GHG)"

        reasoning = str(item.get("reasoning") or "").strip()
        if note:
            reasoning = f"{reasoning.rstrip('. ')}. {note}" if reasoning else note
        results.append(Alternative(
            name=alt_name or str(matched.get("name") or "Okänt alternativ"),
            co2e_kg=co2e,
            cost_sek=cost,
            source=source,
            reasoning=reasoning,
            alternative_type="climate_optimized",
            gwp_basis=gwp_basis,
        ))

    # Candidates the model left out. The prompt asks for every one of
    # them, so an omission is either a judgement it was told to voice on
    # another row or a lapse. Either way the listing exists on Palats and
    # Johanna's April complaint was exactly a toilet that did not appear,
    # so the row is appended with the factual fallback text and the
    # omission is logged with the listing named. Data on the floor must
    # say so.
    for listing, coverage in palats_candidates:
        if str(listing.id) in used_candidates:
            continue
        logger.warning(
            "Ranking omitted Palats listing %s (%r) for %s; appending with "
            "fallback reasoning", listing.id, listing.title, proj_comp.name,
        )
        results.append(_reuse_alternative(
            listing, coverage, proj_comp.quantity, proj_comp.unit,
            category, _FALLBACK_REUSE_REASONING, proj_comp.name,
        ))

    return results


COMMENTARY_PROMPT = """Du är Aida — en byggnadsexpert som hjälper förvaltare och byggledare att hitta renoveringslösningar med kraftigt minskad klimatpåverkan.

Du har just tagit fram alternativ för ett ombyggnadsprojekt. Skriv en kort kommentar om förslagen. Kommentaren ska:
- Lyfta de mest intressanta alternativen och varför de sticker ut
- Nämna om det finns återbruksmöjligheter och vad det innebär
- Peka på eventuella avvägningar (t.ex. lägre CO2 men högre installerat pris, eller enklare montering som sänker totalkostnaden)
- Resonera kort om hur alternativen uppfyller praktiska behov (ljudmiljö, underhåll, inomhusklimat etc)
- Ge ett helhetsintryck av besparingspotentialen

Format:
- Dela upp texten så den är lätt att skumma: korta stycken, punktlistor eller en kombination.
- Max 5-6 meningar/punkter totalt. Korta formuleringar.
- Konkret och direkt, med materialnamn och siffror.
- Skriv som en kunnig byggnadsexpert som pratar med en projektledare.
- Skriv på svenska."""


def _prompt_cost(value) -> str:
    """A cost for a model prompt. The model reads "0 SEK" as free and reasons
    about cost from it, so a missing price must say it is missing."""
    n = _number(value)
    return f"{n:.0f} SEK" if n is not None and n > 0 else "pris saknas"


def _generate_commentary(
    project: Project,
    baseline: Baseline,
    result: AlternativesResult,
) -> str:
    """Generate a natural language commentary about the alternatives found."""
    client = get_client()

    # Map component_id -> usage_context so commentary can reason about
    # whether alternatives actually meet the functional requirements.
    usage_by_id = {
        c.id: getattr(c, "usage_context", "") for c in project.components
    }

    summary_lines = []
    for comp in result.components:
        bl_co2 = comp.baseline_co2e_kg
        bl_cost = comp.baseline_cost_sek
        summary_lines.append(f"\n{comp.component_name} (baslinje: {bl_co2:.0f} kg CO2e, {_prompt_cost(bl_cost)}):")
        usage = usage_by_id.get(comp.component_id, "")
        if usage:
            summary_lines.append(f"  Användning: {usage}")
        for alt in comp.alternatives:
            # Convention (matches the per-component prompt): minus = reduction.
            # pct = (alt - baseline)/baseline, so a 45% saving renders as "-45%".
            pct = ((alt.co2e_kg - bl_co2) / bl_co2 * 100) if bl_co2 > 0 else 0
            per_article = article_price(alt.to_dict())
            cost_str = (f"annonspris {per_article:.0f} kr/st, täckning okänd" if per_article
                        else _prompt_cost(alt.cost_sek))
            summary_lines.append(
                f"  - {alt.name} ({alt.alternative_type}): {alt.co2e_kg:.0f} kg CO2e, "
                f"{cost_str} ({pct:+.0f}% CO2e) | {alt.source}"
            )

    needs_block = ""
    inferred = getattr(project.needs_analysis, "inferred", "") if project.needs_analysis else ""
    if inferred:
        needs_block = f"""

Projektets behov (användargodkänt):
{inferred}
"""

    # One line per row. Joined with "" until 2026-10-01, which ran every
    # alternative onto its component's heading line.
    alternatives_text = "\n".join(summary_lines)
    prompt = f"""Projekt: {project.building_type}, {project.area_bta} m2{needs_block}

Alternativ som hittats:
{alternatives_text}

Skriv din kommentar."""

    try:
        response = call_model(
            client,
            model=DEFAULT_MODEL,
            max_tokens=REASONING_MAX_TOKENS,
            effort=EFFORT_MEDIUM,
            system=COMMENTARY_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return extract_text(response).strip()
    except Exception:
        return ""


def main():
    """CLI entry point for alternatives."""
    if len(sys.argv) < 5 or sys.argv[1] != "--project" or sys.argv[3] != "--baseline":
        print("Usage: python -m aida.agents.alternatives --project <project.json> --baseline <baseline.json>", file=sys.stderr)
        sys.exit(1)

    project_path = sys.argv[2]
    baseline_path = sys.argv[4]

    print("Steg 1/2: Läser projekt och baslinje...", file=sys.stderr)
    project = Project.from_json_file(project_path)
    baseline = Baseline.from_json_file(baseline_path)

    print(f"Steg 2/2: Söker alternativ för {len(project.components)} komponenter...", file=sys.stderr)
    result = find_alternatives(project, baseline)
    print(result.to_json())


if __name__ == "__main__":
    main()
