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

logger = logging.getLogger(__name__)

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
    normalize_component_name,
    resolve_category,
)
from aida.data.nordic_supply import availability_label, nordic_supplier
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
_SUBCATEGORIZED_CATEGORIES = {"sanitet", "belysning", "vitvaror", "fast_inredning"}

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

# Cheap, fast model for the retrieval router (same as the orchestrator's
# intent classifier). Routing is a short structured-classification task.
# Material routing is a correctness step (the toalett/kakel mis-routing lived
# here) — runs on the default Opus 4.8 model now, not the old Haiku.
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
- Användaren optimerar TOTALEN över hela projektet, inte per komponent. En komponent kan välja ett dyrare eller högre CO2e-alternativ om totalen blir bättre tack vare stora vinster på andra komponenter. Filtrera därför INTE bort alternativ enbart för att deras CO2e råkar vara högre än baslinjen — visa relevanta valmöjligheter med tydlig +/- jämförelse i reasoning. Rangordna gärna med lägst CO2e först så användaren ser besparingen, men inkludera även likvärdiga eller marginellt högre alternativ när de är funktionellt relevanta.
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
- Om EPD-värdet är i en annan enhet (kg) än projektets enhet (m2, st), gör en rimlig omräkning och notera det.
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
    if unit == "m3" or epd.get("fu_basis") == "areal_density":
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


def match_epd_by_name(name: str, epds: list[dict]) -> dict | None:
    """Find which candidate EPD an LLM-written alternative name refers to.

    The model paraphrases and truncates product names, so exact equality is
    useless. Containment either way first, longest match wins; then a token
    overlap for the cases containment cannot reach, such as
    "Fibre cement cladding HardiePanel® / Hardie® Architectural Panel" for a
    catalog entry called "S-P-10857 Fibre cement cladding: HardiePanel®,
    Hardie® Architectural Panel" — same product, reordered and re-punctuated.

    Used to carry facts the model cannot be trusted to relay (which GWP
    indicator a figure rests on) from the catalog onto the alternative.
    """
    if not name:
        return None
    needle = _match_key(name)
    if not needle:
        return None
    best = None
    best_len = 0
    tied: list[dict] = []
    for epd in epds:
        epd_name = _match_key(epd.get("name") or "")
        if not epd_name:
            continue
        if epd_name == needle:
            return epd
        if epd_name in needle or needle in epd_name:
            overlap = min(len(epd_name), len(needle))
            if overlap > best_len:
                best, best_len, tied = epd, overlap, [epd]
            elif overlap == best_len:
                tied.append(epd)
    if best is not None:
        # A tie is only a problem when the tied entries disagree about the
        # thing we are carrying across. Two equally-matching fossil products
        # give the same answer either way; one fossil and one GHG do not, and
        # guessing there would put a label on a product that may not deserve it.
        if len({e.get("gwp_basis", "") for e in tied}) > 1:
            return None
        return best

    # Token overlap, for names the model reordered or re-punctuated past what
    # containment can follow. Deliberately strict: at least three quarters of
    # the catalog entry's distinctive words must appear, and the winner has to
    # be clearly ahead of the runner-up. A wrong match here would put a
    # GWP-GHG label on a product that does not deserve one, which is worse than
    # leaving a fossil figure unlabelled.
    needle_tokens = _tokens(needle)
    if len(needle_tokens) < 2:
        return None
    scored: list[tuple[float, dict]] = []
    for epd in epds:
        epd_tokens = _tokens(_match_key(epd.get("name") or ""))
        if len(epd_tokens) < 2:
            continue
        score = len(epd_tokens & needle_tokens) / len(epd_tokens)
        if score >= 0.75:
            scored.append((score, epd))
    if not scored:
        return None
    scored.sort(key=lambda pair: pair[0], reverse=True)
    if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.15:
        return None  # ambiguous, better to say nothing
    return scored[0][1]


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
) -> list[Alternative]:
    """Filter out alternatives with data quality issues.

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
    valid = []
    for alt in alternatives:
        # A) Filter zero/negative CO2 — all building materials have emissions
        if alt.co2e_kg is None or alt.co2e_kg <= 0:
            logger.info(
                "Filtered alternative '%s' for %s: co2e_kg=%s (unrealistic)",
                alt.name, component_name, alt.co2e_kg,
            )
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
        if _is_component_only(alt.name):
            logger.info(
                "Filtered alternative '%s' for %s: component part, not complete system",
                alt.name, component_name,
            )
            continue

        # B2) Drop climate_optimized options that don't actually beat the
        #     baseline — a "climate-optimized" choice with co2e >= baseline is
        #     mislabeled and produces absurd kr/sparat-kg in the ranking. Reuse
        #     and info entries are exempt (different comparison / no number).
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
                alt.cost_sek, quantity, category, is_estimate=is_estimate,
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
    from aida.data.epd_baseline_medians import get_baseline_typvärde
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
    tv = get_baseline_typvärde(routed_category, proj_comp.unit, subcat)
    # kg->st bridge, mirroring _apply_epd_median_fallback in baseline.py: a
    # count-denominated component whose routed category only has a kg typvärde.
    if not tv and proj_comp.unit == "st":
        kg_tv = get_baseline_typvärde(routed_category, "kg", subcat)
        mass = typical_item_mass(routed_category, subcat)
        if kg_tv and mass:
            tv = {"baseline_co2e_per_unit": kg_tv["baseline_co2e_per_unit"] * mass}
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
                    f"Palats har {count_label} i kategorin {category} just nu "
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
    return unique[:_MAX_PALATS_PER_COMPONENT], None


def _reuse_figures(
    listing, coverage: float | None, quantity: float, project_unit: str, category: str,
) -> tuple[float, float, str, bool]:
    """(total_co2e, total_cost, detail, cost_is_per_article) for a listing.

    The numbers the table needs, computed once and used both in the prompt
    (so the model reasons about the same figures the row will carry) and in
    the Alternative built from the model's answer.

    Pricing logic:
    - If project counts in "st" (fönster, dörr), Palats price * quantity
      gives a directly comparable total.
    - If project counts in "m2" (golv, vägg), we can't calculate total
      (unknown coverage per article). Show per-article price instead.
    """
    from aida.data.palats_client import _DEFAULT_REUSE_CO2E, REUSE_CO2E_PER_UNIT

    co2e_per_unit = REUSE_CO2E_PER_UNIT.get(category, _DEFAULT_REUSE_CO2E)
    units_match = project_unit.lower() in ("st", "styck", "stk")
    total_co2e = co2e_per_unit * quantity

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
        # Units don't match — show per-article price only
        total_cost = listing.price
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

    location_note = f"Plats: {listing.location}" if listing.location else ""
    url_note = f"Se annons: {listing.url}" if listing.url else ""
    detail = " | ".join(p for p in [price_note, location_note, url_note] if p)
    return round(total_co2e, 1), round(total_cost), detail, cost_is_estimate


def _reuse_alternative(
    listing, coverage: float | None, quantity: float, project_unit: str,
    category: str, reasoning: str,
) -> Alternative:
    """Build the table row for a Palats listing.

    ``reasoning`` is the model's own text from the unified ranking (or the
    fallback sentence when the call failed). The factual detail string is
    appended after it either way: the price arithmetic, the coverage ratio
    and the link are facts about the listing, not something the model is
    asked to reproduce.
    """
    total_co2e, total_cost, detail, cost_is_estimate = _reuse_figures(
        listing, coverage, quantity, project_unit, category,
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
    # The id is stable across reruns, so the name is too. Mark with * when cost
    # is per-article, not total.
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
    )


def _format_palats_list(
    candidates: list[tuple], quantity: float, project_unit: str, category: str,
) -> str:
    """The [Palats återbruk] rows of the unified prompt.

    One line per listing with the same figures the table will carry (CO2e for
    the full need, price arithmetic, coverage, location, subcategory, id), so
    the model ranks the row it will actually be shown next to the EPDs.
    """
    lines = []
    for listing, coverage in candidates:
        total_co2e, _total_cost, detail, _per_article = _reuse_figures(
            listing, coverage, quantity, project_unit, category,
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
            _FALLBACK_REUSE_REASONING,
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
       every row, reuse included
    5. If that call fails, the Palats rows are appended deterministically so
       an EPD-side error never hides a reuse find
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
        epds_for_category = _select_epd_candidates(
            epd_data.get(comp_key, []), proj_comp.unit, comp_key,
        )
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

        alternatives = _find_alternatives_with_epds(
            proj_comp, bl_comp, epds_for_category, user_feedback,
            needs_analysis=project.needs_analysis,
            effective_baseline_co2e=eff_baseline_co2e,
            palats_candidates=palats_candidates,
            category=comp_key,
        )

        # Validate data quality: filter zero CO2, component-only parts, flag prices
        alternatives = _validate_alternatives(
            alternatives, eff_baseline_co2e, proj_comp.name, proj_comp.quantity,
            category=comp_key,
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
                reasoning="Inga alternativ identifierade.",
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
            except Exception:
                logger.warning("Component failed: %s", bl.component_name, exc_info=True)

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
    from aida.data.price_validation import validate_total_price
    from aida.data.pricing_provider import (
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
            if source and source.lower() not in alt.reasoning.lower():
                alt.reasoning = alt.reasoning.rstrip(". ") + f". Prisunderlag: {source}."

            category = category_by_cid.get(comp.component_id, "")
            if quantity > 0 and category:
                validated_cost, note = validate_total_price(
                    alt.cost_sek, quantity, category, is_estimate=is_estimate,
                )
                if validated_cost != alt.cost_sek:
                    alt.cost_sek = validated_cost
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
    here, not taken from the model. If the call fails, the candidates are
    still appended through the deterministic fallback so an EPD-side API
    error never hides a reuse find.
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
Baslinje kostnad: {bl_comp.cost_sek} SEK

Föreslå alternativ ur listorna nedan: 2-4 EPD-alternativ, plus varje Palats-annons som passar komponenten. Rangordna dem TILLSAMMANS i en lista. Inkludera hela spannet av CO2e-värden — användaren optimerar totalen över hela projektet, inte per komponent, så ett alternativ som ligger något över baslinjen kan vara värt att visa om det möter behoven bättre. Rangordna med lägst CO2e först. I reasoning: ange explicit hur alternativet jämför mot baslinjen (t.ex. "−45% CO2e" eller "+12% CO2e — men kortare leveranskedja och tystare drift").
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
Om EPD-enheten inte matchar projektenheten: hoppa över den EPD:n. Räkna aldrig om mellan enheter med en antagen densitet eller tjocklek — en sådan omräkning ser ut som en besparing men är en gissning.
Se till att minst ett alternativ har "nordisk leverantör", om listan innehåller något sådant. Finns inget, säg det i reasoning i stället för att låtsas.
"""
    else:
        prompt += """
Inga EPD:er tillgängliga för denna kategori. Föreslå inga nyinköp; rangordna bara återbruksannonserna nedan.
"""

    if palats_candidates:
        prompt += f"""
ÅTERBRUKSANNONSER PÅ PALATS FÖR DENNA KATEGORI ({len(palats_candidates)} st, redan filtrerade på produkttyp och lagersaldo):
{_format_palats_list(palats_candidates, proj_comp.quantity, proj_comp.unit, category)}

Ta med varje annons ovan i din rankade lista, med alternative_type "reuse" och source "[Palats] palats.app/listing/<id>". Använd CO2e- och prissiffrorna från raden. Skriv ett eget resonemang per annons: vad den är, hur den passar komponenten och behoven, vad täckningen betyder i praktiken, och vad som ska kontrolleras innan den räknas in. Utelämna en annons bara om den är uppenbart fel produkt för komponenten, och säg då varför i reasoning på ett av de andra alternativen.
"""
    elif epds:
        prompt += """
Inga återbruksannonser på Palats matchar denna komponent just nu, så listan består bara av nyinköp.
"""

    if user_feedback:
        prompt += f"\nAnvändarens önskemål: {user_feedback}\n"

    prompt += "\nSvara med JSON-array."

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
                    category, str(item.get("reasoning", "") or ""),
                ))
                continue

            # Tag source based on whether it references an EPD
            if not source.startswith("["):
                if "epd" in source.lower() or "environdec" in source.lower():
                    source = f"[EPD] {source}"
                else:
                    source = f"[Uppskattning] {source}"

            # Which GWP indicator this rests on is a fact about the catalog, not
            # something to hope the model repeats, so match it back by name.
            alt_name = item.get("name", "Okänt alternativ")
            matched = match_epd_by_name(alt_name, epds)
            gwp_basis = (matched or {}).get("gwp_basis", "") if matched else ""
            if gwp_basis == "ghg":
                # Rides in `source` as well as its own field: source is what the
                # report's component table prints, so the basis reaches the
                # document without a separate plumbing path.
                source = f"{source} (GWP-GHG)"

            results.append(Alternative(
                name=alt_name,
                co2e_kg=item.get("co2e_kg", baseline_for_prompt),
                cost_sek=item.get("cost_sek", 0),
                source=source,
                reasoning=item.get("reasoning", ""),
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
                category, _FALLBACK_REUSE_REASONING,
            ))

        return results
    except Exception:
        logger.warning("Failed to parse alternatives for %s", proj_comp.name, exc_info=True)
        if palats_candidates:
            # The EPD side failed; the reuse side is deterministic and must not
            # go down with it. Same rows, fallback reasoning, logged.
            logger.warning(
                "Ranking call failed for %s; appending %d Palats listings via fallback",
                proj_comp.name, len(palats_candidates),
            )
            return [
                _reuse_alternative(
                    listing, coverage, proj_comp.quantity, proj_comp.unit,
                    category, _FALLBACK_REUSE_REASONING,
                )
                for listing, coverage in palats_candidates
            ]
        return []


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
        summary_lines.append(f"\n{comp.component_name} (baslinje: {bl_co2:.0f} kg CO2e, {bl_cost:.0f} SEK):")
        usage = usage_by_id.get(comp.component_id, "")
        if usage:
            summary_lines.append(f"  Användning: {usage}")
        for alt in comp.alternatives:
            # Convention (matches the per-component prompt): minus = reduction.
            # pct = (alt - baseline)/baseline, so a 45% saving renders as "-45%".
            pct = ((alt.co2e_kg - bl_co2) / bl_co2 * 100) if bl_co2 > 0 else 0
            cost_str = f"{alt.cost_sek:.0f} SEK" if alt.cost_sek > 0 else "Pris ej tillgängligt"
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

    prompt = f"""Projekt: {project.building_type}, {project.area_bta} m2{needs_block}

Alternativ som hittats:
{''.join(summary_lines)}

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
