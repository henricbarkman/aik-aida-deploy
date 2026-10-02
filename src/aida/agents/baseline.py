"""Baseline agent: calculates baseline (conventional standard materials) per component."""

from __future__ import annotations

import json
import logging
import sys
import time

from aida.api_client import (
    DEFAULT_MODEL,
    EFFORT_HIGH,
    REASONING_MAX_TOKENS,
    call_model,
    extract_text,
    get_client,
)
from aida.data.climate_data import (
    REASONING,
    resolve_category,
)
from aida.data.climate_provider import ClimateProvider, resolve_boverket_product
from aida.errors import UserFacingError
from aida.llm_json import ModelOutputError, extract_json_value
from aida.models import Baseline, BaselineResult, Project

logger = logging.getLogger(__name__)


# Source of a CO2e figure that is a category range's midpoint rather than
# anything looked up. It contains "Uppskattning", so every renderer badges it
# as an estimate.
CLAMPED_CO2E_SOURCE = "Uppskattning (intervall)"

# cost_source of a price that was an extreme outlier and is now the category
# range midpoint times the quantity: no web search found it.
ADJUSTED_PRICE_SOURCE = "Schablonpris (justerat)"


def _validate_baseline(results: list[BaselineResult], components: list) -> list[BaselineResult]:
    """Validate prices and CO2 values on baseline results.

    Extreme outliers get clamped to reasonable ranges, and a clamped row is
    relabelled as the range estimate it has become. Mild outliers get flagged.
    Every row leaves with co2e_kg == co2e_per_unit x quantity, which is the
    multiplication the baseline view prints under the total.
    """
    from aida.data.epd_baseline_medians import UNLIKE_THEIR_CATEGORY
    from aida.data.price_validation import check_co2e, check_total_price

    comp_map = {c.id: c for c in components}
    for r in results:
        comp = comp_map.get(r.component_id)
        quantity = comp.quantity if comp else 0
        unit = comp.unit if comp else ""
        category = resolve_category(r.component_name, comp.category if comp else "")
        is_estimate = "uppskattning" in (r.cost_source or "").lower()

        # Validate price
        if r.cost_sek <= 0:
            r.cost_sek = 0
            if "pris ej tillgängligt" not in r.description.lower():
                r.description = r.description.rstrip(". ") + ". Pris ej tillgängligt."
        else:
            price = check_total_price(
                r.cost_sek, quantity, category, is_estimate=is_estimate, unit=unit,
            )
            if price.cost != r.cost_sek:
                r.cost_sek = price.cost
            if price.replaced:
                # The total is now the range midpoint, not the price the search
                # found, so "Webbsökning (AI)" would attribute a number to a
                # search that never produced it (handover review L2).
                r.cost_source = ADJUSTED_PRICE_SOURCE
            if price.note and price.note.lower() not in r.description.lower():
                r.description = r.description.rstrip(". ") + f". {price.note}."

        # Validate CO2, except an EPD typvärde. The ranges are hand-written and
        # the typvärde is the catalog's own median, so checking one against the
        # other is circular: published typvärden for betongvägg, tak and
        # yttervägg sat outside their ranges and every such row was stamped
        # "Oväntat", and the broken hiss/st typvärde (13.4 kg per elevator) was
        # clamped to 16 000 under a label that still read "13,4 kg/st x 1 st"
        # (handover review C5). A mis-tagged catalog row is caught where it can
        # be fixed instead: test_baseline_clamp checks every published
        # typvärde against these ranges.
        #
        # A bridged typvärde keeps the check. kg -> st through an assumed item
        # mass (a 45 kg bathtub) is a catalog median times a hand-written
        # number, and that number is what the range can catch; no test sees
        # it, since it is computed per component. Except the subtypes their
        # category's range says nothing about (glasparti, avjämning).
        basis = r.basis or {}
        exempt = basis.get("kind") == "epd_typvärde" and (
            not basis.get("bridge")
            or (category, basis.get("subcategory")) in UNLIKE_THEIR_CATEGORY)
        if not exempt and quantity > 0 and r.co2e_kg > 0:
            co2 = check_co2e(r.co2e_kg / quantity, quantity, category, unit)
            if co2.clamped:
                _relabel_clamped_co2e(r, co2, quantity, unit, category)
            elif co2.note and co2.note.lower() not in r.description.lower():
                r.description = r.description.rstrip(". ") + f". {co2.note}."

        _reconcile_per_unit(r, quantity, unit)

    return results


def _relabel_clamped_co2e(r: BaselineResult, co2, quantity: float, unit: str,
                          category: str) -> None:
    """Rewrite a clamped row so every field describes the number it now shows.

    Until 2026-09-30 only co2e_kg changed. The per-unit figure, source, basis
    and description kept describing the discarded value, and the view printed
    16 000 kg over "13,4 kg CO2e/st x 1 st", sourced as an EPD typvärde.
    """
    lo, hi, range_unit = co2.bounds
    unit = unit or range_unit
    was = r.co2e_kg / quantity
    was_source = r.source or "okänd källa"
    direction = "högt" if was > hi else "lågt"

    r.co2e_kg = co2.total
    r.co2e_per_unit = co2.per_unit
    r.unit = unit
    r.quantity = quantity
    r.source = CLAMPED_CO2E_SOURCE
    # The Boverket product line would otherwise sit under a number that is
    # no longer that product's.
    r.boverket_product = ""
    r.basis = {
        "kind": "intervall",
        "label": f"Uppskattning ur rimlighetsintervall för {category}",
        "min": lo,
        "max": hi,
        "discarded_per_unit": round(was, 3),
        "discarded_source": was_source,
        "reason": (f"Beräknat värde {was:.4g} kg CO2e/{unit} ({was_source}) var "
                   f"orimligt {direction} och används inte"),
    }
    # assumed_material stays on the row (it is still what the agent assumed is
    # built), but the description does not repeat it: the number is the
    # category's midpoint, not that material's.
    r.description = (
        f"Uppskattning ur kategorins rimlighetsintervall: mittpunkten av "
        f"{lo:g} till {hi:g} kg CO2e/{unit} för {category} ({co2.per_unit:g} kg "
        f"CO2e/{unit}) × {quantity:g} {unit}. Det beräknade värdet, {was:.4g} kg "
        f"CO2e/{unit} ({was_source}), var orimligt {direction} och används inte."
    )


def _reconcile_per_unit(r: BaselineResult, quantity: float, unit: str) -> None:
    """Hold co2e_kg == co2e_per_unit x quantity, in the component's unit.

    The view prints that multiplication under the total so it can be checked
    by hand, and a multiplication that does not add up is worse than none. The
    total wins: it is what every sum and every alternative is measured
    against, and the per-unit figure is the one that can be in the wrong unit
    (a Boverket kg value next to a count in m2). Rounding is allowed for: the
    total carries one decimal, the per-unit figure two to four.
    """
    if quantity <= 0:
        return
    tolerance = 0.05 + 0.0005 * quantity + 1e-9
    if abs(r.co2e_per_unit * quantity - r.co2e_kg) > tolerance:
        logger.warning(
            "Baseline %s: %s kg CO2e/%s x %g %s is not the total %s; per-unit "
            "figure rederived from the total",
            r.component_id, r.co2e_per_unit, r.unit, quantity, unit, r.co2e_kg,
        )
        r.co2e_per_unit = round(r.co2e_kg / quantity, 4)
    r.quantity = quantity
    if unit:
        r.unit = unit


MATCH_SYSTEM_PROMPT = """Du är Aidas baslinjeberäknare — en byggnadsexpert som beräknar baslinjen för klimatpåverkan.

Baslinjen representerar standardfallet enligt NollCO2-metoden: vad det kostar klimatmässigt om projektet använder konventionella material utan särskild klimathänsyn.

KLIMATMETOD (gäller hela analysen):
- GWP-fossil, livscykelskedena A1-A3 (cradle-to-gate), enligt Boverkets klimatdatabas.
- Inkludera ALDRIG biogenic carbon credit i baslinjen. Värden måste vara konsistenta med Boverket Typical A1-A3.

Du får:
1. En lista med projektets komponenter (id, namn, antal, enhet, kategori, samt "Funktion
   och krav" när det finns — vad komponenten ska klara av, vilka som använder den och i
   vilken miljö)
2. Boverkets kompletta produktlista med CO2e-värden (GWP-fossil, Typical A1-A3)

UPPGIFT — följ dessa steg för VARJE komponent:

STEG 1 — BESTÄM STANDARDMATERIAL:
Fundera på vad det konventionella/typiska materialet är för denna komponent, givet
byggnadstypen OCH komponentens funktion och krav (fältet "Funktion och krav").
Exempel: golv i skolentré med saltslask → homogen vinylmatta (PVC). Innervägg → gipsskiva
på stålreglar.

Skriv ut valet i fältet `assumed_material` som en kort produktbenämning, inte en mening:
"Homogen vinylmatta (PVC)", "Linoleum 2,5 mm", "Gipsskiva 13 mm på stålreglar",
"Keramisk klinker". Det är den som visas för användaren som svar på frågan "vilket
material har ni räknat på?", så den ska gå att känna igen och ifrågasätta.

VIKTIGT — baslinjen är INTE användarens val:
Baslinjen ska svara på "vad hade man normalt byggt här?", inte "vad tänker användaren
köpa?" och inte "vad sitter där idag?". Väljer du material efter vad projektet planerar
blir jämförelsen cirkulär och besparingen noll per definition. Nämner beskrivningen ett
önskat material, bortse från det och välj det byggnadstypiska. (Samma princip som NollCO2,
där baslinjen räknas fram ur byggnadsparametrar innan projektet har projekterat något.)

Undantaget är innertakets sort, som är en annan byggdel och inte ett materialval:
undertaksplattor i bärverk, ett gipstak och ett putsat innertak jämförs var för sig, så
baslinjen följer sorten i namnet. Ett gipstak ("Gipstak", "Nytt innertak av gips") är
gipsskivor i ett lager i taket: matcha "Gipsskiva, standardskiva". Ett putsat innertak är
puts på tak: matcha "Putsbruk C (CS II)" och skriv skikttjockleken du räknar med i
beskrivningen. Byt aldrig ett gipstak eller putsat innertak mot ett akustikundertak.

Välj det konventionella valet utan särskild klimathänsyn — inte det bästa tillgängliga,
och inte det sämsta tänkbara.

STEG 2 — MATCHA MOT BOVERKET ENDAST VID SAMMA MATERIAL:
Boverkets databas är organiserad efter materialtyp, inte byggnadsfunktion. Välj en Boverket-
produkt ENBART när den faktiskt ÄR komponentens standardmaterial:
- Gipsskiva på innervägg matchar "Gipsskiva, standardskiva" (gips är gips). OK.
- Betongvägg matchar en betongprodukt (betong är betong). OK.
- Stålreglar matchar "Lättreglar av stål, primär" (stål är stål). OK.
- Träreglar och konstruktionsvirke matchar "Sågat virke, u 16 %, barrträ" eller
  "Hyvlat virke, u 16 %, barrträ", OSB-skiva matchar "OSB" (trä är trä). OK.
  Systemet räknar själv om ett kg-värde för stommens trä och skivor, för skivan i ett
  gipstak och för en träpanel på fasad ("Hyvlat virke", panelen utan läkt och målning,
  22 mm om namnet inte anger tjocklek), till löpmeter eller m² ur tvärsnittet eller
  tjockleken i komponentens namn, så räkna inte om de raderna själv. Andra skivor, t.ex. gips på en innervägg, räknar du om
  själv enligt STEG 3.
  Detsamma gäller stålbalkar, stålpelare och stålreglar i löpmeter med en
  standardprofil i namnet ("HEA 200", "VKR 100x100x5", "Stålregel 70"): vikten per
  meter tas ur standardtabeller. Konstruktionsstål matchar "Konstruktionsstål, alla
  sorter, 80 % primär råvara".
  Och prefabricerade betongelement: håldäck matchar "Hålbjälklag, HD/F", massiva
  bjälklag "Massivplattor, RD, RD/F", betongbalkar och pelare "Balkar B". Systemet
  räknar om dem till m² eller löpmeter ur tjockleken, tvärsnittet eller vikten i namnet.
- Mineralull som isolering matchar en mineralullsprodukt. OK.

Låna ALDRIG en produkt av annan typ bara för att den delar basmaterial. Det ger en
vilseledande baslinje. Exempel på vad som är FÖRBJUDET:
- Vinylgolv mot "Takduk, PVC": golvbeläggning och takduk är olika produkter även om båda
  är PVC. Sätt boverket_product=null och source="Uppskattning" istället.
- En stol eller ett bord mot "Sågat virke" eller en stålprodukt: en möbel är en färdig
  produkt, inte sitt stommaterial.
Boverket saknar bl.a. golvbeläggning, sanitetsporslin, vitvaror, belysning och möbler
(lös inredning) som egna produkter. Leta inte efter en ersättare för dem i Boverket.

STEG 3 — JUSTERA FÖR MATERIALEGENSKAPER:
När du valt en Boverket-produkt av rätt material men dimensionerna skiljer (tjocklek,
densitet, vikt per m²), justera co2e_per_unit proportionellt och beskriv resonemanget i
description-fältet.

STEG 4 — UPPSKATTNING NÄR BOVERKET SAKNAR MATERIALET:
Om komponentens standardmaterial inte finns som egen produkt i Boverket, sätt
boverket_product till null och source="Uppskattning". Systemet ersätter då uppskattningen
med ett EPD-typvärde där sådant finns, och använder ditt `assumed_material` för att välja
rätt undertyp (t.ex. vinyl snarare än golv generellt) — så ju mer precis produktbenämning
du skriver, desto träffsäkrare blir baslinjen. Uppskattningen ska alltid avse
GWP-fossil A1-A3 (cradle-to-gate, exkl. biogenic carbon credit) så värdet är jämförbart med
övriga komponenter.

PRISER:
Sätt cost_sek till 0 — priser hämtas separat via webbsökning.

Svara med ENBART giltig JSON (ingen markdown, inga kommentarer):
[
  {
    "component_id": "string (exakt id från komponentlistan)",
    "component_name": "string",
    "assumed_material": "string (kort produktbenämning på det antagna standardmaterialet, se STEG 1)",
    "boverket_product": "string (exakt produktnamn från Boverket-listan, eller null)",
    "co2e_per_unit": number,
    "unit": "string (enhet från Boverket-produkten, konverterad till komponentens enhet vid behov)",
    "co2e_kg": number (co2e_per_unit x quantity),
    "cost_sek": 0,
    "method": "NollCO2",
    "description": "Beskriv: 1) antaget standardmaterial, 2) vald Boverket-produkt ELLER varför ingen passar (uppskattning), 3) eventuell justering och varför",
    "source": "Boverkets klimatdatabas" eller "Uppskattning"
  }
]"""


# The clock the request budget is read from. A module attribute so a test can
# move time forward instead of sleeping.
_clock = time.monotonic

# Request budget for the pricing passes. The estimate batch is one call
# without tools (17 s for four products, measured 2026-08-20), so the web
# call's timeout keeps _ESTIMATE_RESERVE back, and the estimate pass starts
# with as little as _MIN_ESTIMATE_BUDGET. The floor sits well under the
# reserve on purpose: a timed-out call returns a moment after its timeout, and
# with the two equal the estimate pass could never follow a search that timed
# out. The web pass itself needs _MIN_WEB_PRICING_BUDGET to be worth starting,
# which leaves its search 45 s.
_ESTIMATE_RESERVE = 45.0
_MIN_ESTIMATE_BUDGET = 20.0
_MIN_WEB_PRICING_BUDGET = 90.0

WEB_PRICE_SOURCE = "Webbsökning (AI)"
ESTIMATED_PRICE_SOURCE = "Uppskattning (AI)"


def calculate_baseline(project: Project, *, started_at: float | None = None) -> Baseline:
    """Calculate NollCO2 baseline for each component.

    Matches components against the full Boverket product list (about 660 rows)
    in parallel LLM calls of BASELINE_CHUNK_SIZE components each. The LLM picks
    the best Boverket product per component, or estimates when none fits.

    ``started_at`` is the request's start on the time.monotonic clock, for the
    pricing budget; omitted, the budget counts from this call.
    """
    started_at = _clock() if started_at is None else started_at
    provider = ClimateProvider()
    provider.ensure_synced()

    # Phase 1: LLM-based matching against full Boverket product list, with
    # exactly one row per component (re-asked once for any the model skipped).
    boverket_products = provider._cache.get_all_boverket()
    results = _complete_baseline(project, boverket_products)
    _apply_member_geometry(results, project, boverket_products)

    # Phase 1b: For components where the LLM fell back to "Uppskattning"
    # (no Boverket material proxy fit), substitute an EPD-median where the
    # component's category has reliable aggregated data.
    _apply_epd_median_fallback(results, project)

    # Phase 2 and 3: prices, within what is left of the request.
    _price_baseline(results, project, provider, started_at=started_at)

    results = _validate_baseline(results, project.components)
    return Baseline(components=results)


def _price_baseline(results: list[BaselineResult], project: Project,
                    provider: ClimateProvider, *, started_at: float) -> None:
    """Price every row, in the unit its component is counted in (in-place).

    Two passes, each one call for all the rows it covers: a web search batch,
    then the model's own estimate for whatever the search left. Until
    2026-09-30 the second pass was a loop of lookup_price per product, a web
    search plus an estimate each, run one after the other with no clock: a
    batch answer that came back as prose (zero rows parsed) turned a
    31-component project into up to 62 sequential calls, Vercel killed the
    function at 300 s, and the matching already paid for was lost (handover
    review L3). The alternatives' pricing was restructured the same way in
    August.

    A pass the request no longer has time for is skipped, and its rows keep
    cost 0, which _validate_baseline marks "Pris ej tillgängligt". A failing
    lookup or cache write does not fail the baseline: a price is worth less
    than the baseline it belongs to.
    """
    from aida.api_client import remaining_budget
    from aida.data.pricing_provider import (
        estimate_prices_batch,
        lookup_prices_batch,
        price_unit_matches,
    )

    comp_map = {c.id: c for c in project.components}
    # Asked in the unit the component is counted in, and a price that comes
    # back in another unit is neither used nor cached: SEK/st times 45 m2 is
    # not a cost (Fable audit 2026-07-19, P2 #9). The row then keeps the
    # baseline agent's own estimate, which is labelled as one.
    unit_by_name = {
        r.component_name.lower(): (comp_map[r.component_id].unit
                                   if r.component_id in comp_map else "")
        for r in results
    }
    def cached(name: str) -> bool:
        try:
            return _is_price_cached(provider, name)
        except Exception as e:  # noqa: BLE001 - a cache read is not worth the baseline
            logger.warning("Price cache read failed for '%s': %s", name, e)
            return False

    wanted = [
        (r.component_name, unit_by_name.get(r.component_name.lower(), ""))
        for r in results
        if not cached(r.component_name)
    ]
    if not wanted:
        return

    # lowercase name -> (price per unit, cost_source)
    prices: dict[str, tuple[float, str]] = {}

    def take(found: dict, default_source: str) -> None:
        for key, (price, price_unit, src) in (found or {}).items():
            if key in prices:
                continue
            want = unit_by_name.get(key, "")
            if want and not price_unit_matches(price_unit, want):
                logger.warning(
                    "Baseline price for '%s' is per %r but the component is "
                    "counted in %r; discarded", key, price_unit, want,
                )
                continue
            # A single-product "batch" is lookup_price, which falls back to an
            # estimate by itself; that one must not be labelled a search. Read
            # off the prefix pricing_provider gives every estimate, not a
            # substring: a search's source label carries its citation URL,
            # and a slug like "prisuppskattning" is still a search.
            source = (ESTIMATED_PRICE_SOURCE if (src or "").startswith("LLM-uppskattning")
                      else default_source)
            prices[key] = (price, source)
            if source == WEB_PRICE_SOURCE:
                try:
                    provider._cache.update_cost(key, price)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Price cache write failed for '%s': %s", key, e)

    def left() -> float:
        return remaining_budget(started_at, now=_clock())

    budget = left()
    if budget >= _MIN_WEB_PRICING_BUDGET:
        try:
            take(lookup_prices_batch(wanted, timeout=budget - _ESTIMATE_RESERVE),
                 WEB_PRICE_SOURCE)
        except Exception as e:  # noqa: BLE001 - pricing must not fail the baseline
            logger.warning("Baseline web pricing failed: %s", e)
    else:
        logger.warning("Baseline web pricing skipped: %.0fs left of the request", budget)

    missing = [(name, unit) for name, unit in wanted if name.lower() not in prices]
    if missing:
        budget = left()
        if budget >= _MIN_ESTIMATE_BUDGET:
            try:
                take(estimate_prices_batch(missing, timeout=budget),
                     ESTIMATED_PRICE_SOURCE)
            except Exception as e:  # noqa: BLE001
                logger.warning("Baseline price estimate failed: %s", e)
        else:
            logger.warning(
                "%d baseline rows left unpriced: %.0fs left of the request",
                len(missing), budget,
            )

    for r in results:
        hit = prices.get(r.component_name.lower())
        if hit:
            comp = comp_map.get(r.component_id)
            quantity = comp.quantity if comp else 1
            r.cost_sek = round(hit[0] * quantity)
            r.cost_source = hit[1]


# Boverket categories whose kg value is a solid material, so that kg per m3 is a
# density and a member's section or thickness gives its volume.
_GEOMETRY_BOVERKET_CATEGORIES = {"Trävaror", "Byggskivor"}


# Boverket's steel products a beam, a column or a stud is made of. Not plate,
# fasteners or reinforcement: a weight per metre of a profile says nothing about
# those, and the model has no business matching them to a member anyway.
_STEEL_PROFILE_BOVERKET_PREFIXES = ("konstruktionsstål", "lättreglar av stål",
                                    "galvaniserade stålprodukter")


def _apply_profile_mass(r: BaselineResult, comp, product, extra: dict) -> bool:
    """Redo a steel Boverket baseline per löpmeter from the profile's standard
    weight per metre (in-place). True when the product is a steel profile
    product, whether or not the name allowed the conversion.

    The steel half of _apply_member_geometry (HENRIC-3363). Boverket declares
    "Konstruktionsstål" per kg and a beam is counted in löpmeter; an HEA 200
    weighs 42,3 kg per metre by EN 10365 (steel_profiles), so the figure per
    metre is arithmetic, done here and written out in the row. A name without
    a designation keeps the model's figure, labelled as resting on an assumed
    weight per metre.
    """
    from aida.data.steel_profiles import profile_mass

    if extra.get("category") != "Stål och andra metaller":
        return False
    if not (product.name or "").strip().lower().startswith(_STEEL_PROFILE_BOVERKET_PREFIXES):
        return False
    if comp.unit != "lm":
        return True
    profile, _ = profile_mass(comp.name)
    if not profile:
        note = (" Namnet anger ingen standardprofil, så omräkningen från kg bygger "
                "på en antagen vikt per meter.")
        if note.strip() not in (r.description or ""):
            r.description = (r.description or "").rstrip() + note
        return True
    per_unit = round(product.co2e_per_unit * profile.kg_per_m, 4)
    r.co2e_per_unit = per_unit
    r.unit = comp.unit
    r.quantity = comp.quantity
    r.co2e_kg = round(per_unit * comp.quantity, 1)
    r.description = (r.description or "").rstrip() + (
        f" Omräknat ur komponentens profil: {product.co2e_per_unit} kg CO2e/kg "
        f"(Boverket) × {profile.kg_per_m:g} kg/m ({profile.designation}, "
        f"{profile.source}) = {per_unit} kg CO2e/lm."
    )
    return True


# Boverket's precast element records, by the element kind they are
# (betongstomme.element_kind). A record of another kind is left alone: its
# weight per m² or metre is not the component's.
_CONCRETE_BOVERKET_PREFIXES = {"håldäck": "hålbjälklag", "massiv": "massivplattor",
                               "balk": "balkar"}


def _apply_concrete_element(r: BaselineResult, comp, product, extra: dict) -> bool:
    """Redo a precast concrete Boverket baseline per m² or löpmeter (in-place).
    True when the product is a precast element record of the component's kind.

    Boverket declares hollow-core, solid slabs and beams per kg, and a slab is
    counted in m². Until 2026-10-02 the model did that conversion with a
    weight of its own choosing (270 kg/m² for a 200 mm hollow-core slab on
    stage), while the alternatives use only weights a name or an EPD states
    (betongstomme). The same rule now holds for the baseline (HENRIC-3395):

    - A hollow-core slab uses the weight per m² its name states, else
      Boverket's own conversion for the record (kg/m³ times the height, which
      is what the record's kg value is declared against), and the row gives
      Svensk Betong's span for the height, since makers' voids differ.
    - A solid slab or a beam uses 2 500 kg/m³ times the thickness or section,
      as the alternatives do (Svensk Betong's element table).
    Without the thickness or section the model's figure stays, labelled.
    """
    from aida.data import betongstomme as bs

    if extra.get("category") != "Betong":
        return False
    kind = bs.element_kind(comp.name)
    prefix = _CONCRETE_BOVERKET_PREFIXES.get(kind)
    if not prefix or not (product.name or "").strip().lower().startswith(prefix):
        return False
    kg_value = product.co2e_per_unit
    aside = ""
    if kind == bs.BALK:
        section = bs.beam_section_mm(comp.name) if comp.unit == "lm" else None
        if not section:
            if comp.unit == "lm":
                note = (" Namnet anger inget tvärsnitt, så omräkningen från kg bygger "
                        "på ett antaget tvärsnitt.")
                if note.strip() not in (r.description or ""):
                    r.description = (r.description or "").rstrip() + note
            return True
        w, h = section
        weight = w / 1000 * h / 1000 * bs.SOLID_DENSITY_KG_M3
        how = (f"tvärsnitt {w:g}×{h:g} mm × 2 500 kg/m³ ({bs.SOLID_DENSITY_SOURCE}) "
               f"= {weight:g} kg/lm")
    elif comp.unit != "m2":
        return True
    else:
        thickness = bs.slab_thickness_mm(comp.name)
        stated = bs.stated_kg_per_m2(comp.name) if kind == bs.HÅLDÄCK else None
        if stated:
            weight = stated
            how = f"vikten {stated:g} kg/m² som står i namnet"
        elif not thickness:
            note = (" Namnet anger ingen tjocklek, så omräkningen från kg bygger på "
                    "en antagen vikt per m².")
            if kind == bs.HÅLDÄCK:
                note += f" {bs.hdf_weight_text(200)}."
            if note.strip() not in (r.description or ""):
                r.description = (r.description or "").rstrip() + note
            return True
        elif kind == bs.HÅLDÄCK:
            density = extra.get("density_kg_m3")
            if not density:
                return True
            weight = round(density * thickness[0] / 1000, 1)
            how = (f"{weight:g} kg/m², Boverkets omräkning för {product.name.strip()} "
                   f"({density:g} kg/m³ × {thickness[0]:g} mm)")
            aside = (f" {bs.hdf_weight_text(thickness[0])}, så leverantörens vikt i "
                     f"namnet (\"Håldäck {thickness[0]:g} mm, XXX kg/m2\") ger en "
                     f"säkrare siffra.")
        else:
            weight = thickness[0] / 1000 * bs.SOLID_DENSITY_KG_M3
            how = (f"{thickness[1]} × 2 500 kg/m³ ({bs.SOLID_DENSITY_SOURCE}) "
                   f"= {weight:g} kg/m²")
    per_unit = round(kg_value * weight, 4)
    r.co2e_per_unit = per_unit
    r.unit = comp.unit
    r.quantity = comp.quantity
    r.co2e_kg = round(per_unit * comp.quantity, 1)
    r.description = (r.description or "").rstrip() + (
        f" Omräknat ur elementets vikt: {kg_value} kg CO2e/kg (Boverket) × {how}. "
        f"Det ger {per_unit} kg CO2e/{comp.unit}.{aside}"
    )
    return True


def _apply_ceiling_board(r: BaselineResult, comp, product, extra: dict) -> None:
    """Redo a gypsum ceiling's Boverket baseline per m² (in-place).

    Boverket declares "Gipsskiva, standardskiva" per kg with a density (710
    kg/m³), and a gipstak is counted in m². The prompt tells the model the
    system converts boards from the thickness in the name, which until
    2026-10-02 held for the frame only: "Gipstak 15 mm" kept 0,227 kg CO2e
    per kg as its value per m², about a tenth of the board, and every
    alternative came out above it. The thickness is the name's own board
    thickness (gipsskiva), else a standard board of 12,5 mm, said in the row;
    the layers are the name's ("2x13"), else one, as the prompt describes a
    gipstak.
    """
    from aida.data.climate_data import ceiling_kind
    from aida.data.gipsskiva import format_mm, layers_from_name, thickness_from_name

    if comp.unit != "m2" or ceiling_kind(comp.name) != "gipstak":
        return
    if not (product.name or "").strip().lower().startswith(("gipsskiva", "fibergipsskiva")):
        return
    density = extra.get("density_kg_m3")
    if extra.get("category") != "Byggskivor" or not density:
        return
    mm = thickness_from_name(comp.name)
    assumed = mm is None
    if assumed:
        mm = 12.5
    layers = layers_from_name(comp.name)
    per_unit = round(product.co2e_per_unit * density * mm / 1000 * layers, 4)
    r.co2e_per_unit = per_unit
    r.unit = comp.unit
    r.quantity = comp.quantity
    r.co2e_kg = round(per_unit * comp.quantity, 1)
    board = f"{layers} × {format_mm(mm)} mm" if layers > 1 else f"{format_mm(mm)} mm"
    why = (" Namnet anger ingen skivtjocklek, så en standardskiva på 12,5 mm antas."
           if assumed else "")
    r.description = (r.description or "").rstrip() + (
        f" Omräknat ur skivan, {board}: {product.co2e_per_unit} kg CO2e/kg × "
        f"{density:g} kg/m³ (Boverket) × {layers * mm / 1000:g} m = {per_unit} kg CO2e/m².{why}"
    )


# Panel thickness when a wood facade's name gives none: the catalog's own
# fasadskikt bridge (unit_conversion.CONVERSIONS["fasadskikt"], 22 mm, the
# usual 22x120 to 22x170 panel), so the baseline and the alternatives it is
# measured against rest on the same panel.
_FACADE_PANEL_DEFAULT_MM = 22.0


def _apply_facade_panel(r: BaselineResult, comp, product, extra: dict) -> None:
    """Redo a wood facade cladding's Boverket baseline per m² (in-place).

    HENRIC-3410. "Träpanel fasad", 150 m² in a school, got 1,61, 1,75, 5,83
    and 5,83 kg CO2e/m² in four runs on 2026-10-02: the model matched
    "Hyvlat virke" per kg and converted to m² in its head, each time with its
    own panel volume, with or without battens and paint. The conversion is the
    frame's: Boverket's figure per kg × Boverket's density × the panel's
    thickness, from the name ("22 mm", "22x145") or 22 mm. The panel only:
    paint is its own component (farg), and the cladding EPDs it is compared
    with declare the board without battens.
    """
    from aida.data.climate_data import component_facade_kind
    from aida.data.unit_conversion import cross_section_mm, thickness_mm

    if comp.unit != "m2" or extra.get("category") != "Trävaror":
        return
    if component_facade_kind(comp.name) not in ("trä", ""):
        return
    density = extra.get("density_kg_m3")
    if not density:
        return
    mm = thickness_mm(comp.name)
    if mm is None:
        section = cross_section_mm(comp.name)
        mm = min(section) if section else None
    assumed = mm is None
    if assumed:
        mm = _FACADE_PANEL_DEFAULT_MM
    per_unit = round(product.co2e_per_unit * density * mm / 1000, 4)
    r.co2e_per_unit = per_unit
    r.unit = comp.unit
    r.quantity = comp.quantity
    r.co2e_kg = round(per_unit * comp.quantity, 1)
    why = (f" Namnet anger ingen paneltjocklek, så en vanlig fasadpanel på "
           f"{mm:g} mm antas." if assumed else "")
    r.description = (r.description or "").rstrip() + (
        f" Omräknat ur panelen, {mm:g} mm: {product.co2e_per_unit} kg CO2e/kg × "
        f"{density:g} kg/m³ (Boverket) × {mm / 1000:g} m = {per_unit} kg CO2e/m². "
        f"Bara panelen: läkt och målning ingår inte.{why}"
    )


def _apply_member_geometry(results: list[BaselineResult], project: Project,
                           boverket_products) -> None:
    """Redo a timber or board Boverket baseline per löpmeter or m2 (in-place).

    Boverket declares "Sågat virke" per kg; a stud is counted in löpmeter.
    Until 2026-09-28 the model did that conversion in its head (STEG 3 asks it
    to), guessing a section and a density for every stud. Both are known: the
    density is in Boverket's own record and the section is in the name
    ("Reglar 45x95"), so the arithmetic is done here instead and written out in
    the row. Frame members (stomme), and the boards of a gypsum ceiling
    (_apply_ceiling_board, HENRIC-3399).

    A name without a dimension keeps the model's figure, labelled as resting on
    an assumed section, since the chat is meant to ask for it before this runs.
    """
    from aida.data.unit_conversion import member_volume_per_unit

    by_name = {(p.name or "").strip().lower(): p for p in boverket_products}
    comp_map = {c.id: c for c in project.components}
    for r in results:
        comp = comp_map.get(r.component_id)
        if not r.boverket_product or not comp or comp.unit not in ("lm", "m2"):
            continue
        category = resolve_category(comp.name, comp.category)
        if category not in ("stomme", "undertak", "fasadskikt"):
            continue
        product = by_name.get(r.boverket_product.strip().lower())
        if not product or (product.unit or "").lower() != "kg":
            continue
        try:
            extra = json.loads(product.extra_json or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        if category == "undertak":
            _apply_ceiling_board(r, comp, product, extra)
            continue
        if category == "fasadskikt":
            _apply_facade_panel(r, comp, product, extra)
            continue
        if _apply_profile_mass(r, comp, product, extra):
            continue
        if _apply_concrete_element(r, comp, product, extra):
            continue
        density = extra.get("density_kg_m3")
        if extra.get("category") not in _GEOMETRY_BOVERKET_CATEGORIES or not density:
            continue
        geometry = member_volume_per_unit(comp.name, comp.unit)
        if not geometry:
            what = "tvärsnitt" if comp.unit == "lm" else "tjocklek"
            note = (f" Namnet anger inget {what}, så omräkningen från kg bygger "
                    f"på ett antaget {what}.")
            if note.strip() not in (r.description or ""):
                r.description = (r.description or "").rstrip() + note
            continue
        factor, label = geometry
        per_unit = round(product.co2e_per_unit * density * factor, 4)
        r.co2e_per_unit = per_unit
        r.unit = comp.unit
        r.quantity = comp.quantity
        r.co2e_kg = round(per_unit * comp.quantity, 1)
        r.description = (r.description or "").rstrip() + (
            f" Omräknat ur komponentens {label}: {product.co2e_per_unit} kg CO2e/kg "
            f"× {density:g} kg/m³ (Boverket) × {factor:.6f} m³/{comp.unit} = "
            f"{per_unit} kg CO2e/{comp.unit}."
        )


def _geo_note(data: dict, n: int, where: str = "") -> str:
    """Which population a typvärde rests on, and the true reason when it is the
    global one. Until 2026-10-01 every global key said "only N European EPDs,
    under the floor", also when N was 197 and the European bucket had been set
    aside because one group held most of it (undertak/akustikplatta/m2)."""
    scope = data.get("geo_scope")
    if scope == "europa":
        return (f" Urvalet är europeiska EPD:er (SE/Norden/EU), {n} av "
                f"{data.get('sample_size_global', n)}{where}.")
    if scope != "global":
        return ""
    n_eu = data.get("sample_size_europe", 0)
    reason = data.get("scope_reason", "thin")
    if reason == "dominated":
        return (f" De {n_eu} europeiska EPD:erna kommer till största delen från en och "
                f"samma koncern, så hela det globala urvalet används.")
    if reason == "scope_free":
        return (" Hela det globala urvalet används, så att golvtyperna och golvet "
                "som helhet vilar på samma underlag.")
    return (f" Bara {n_eu} europeiska EPD:er, under golvet "
            f"{data.get('min_samples_europe', '')}, så hela det globala urvalet används.")


def _apply_aggregat_typvärde(r: BaselineResult, comp) -> None:
    """An air handling unit counted per piece (in-place): per airflow when the
    flow is known, per size class when only the class is, and otherwise the
    estimate, with the reason and what would give a number. aggregat.py has
    the decision and how every fact behind it was read.

    ventilation/aggregat/st has no typvärde of its own: its EPDs span 270 to
    16 700 kg/st by unit size. So this replaces the generic path, which could
    only say that.
    """
    from aida.claims import format_value
    from aida.data.epd_baseline_medians import aggregat_typvärde

    res = aggregat_typvärde(comp.name, comp.usage_context, comp.quantity)
    data = res["payload"]
    if data is None:
        reason = res["reason"]
        if res.get("flow_available"):
            advice = ("Ange aggregatets luftflöde (m³/h) i chatten eller i namnet, "
                      "så räknas baslinjen per luftflöde ur katalogens EPD:er.")
        else:
            advice = ""
        note = (f" Ventilationsaggregat räknas per luftflöde eller per storleksklass, "
                f"men här går ingetdera: {reason[:1].lower()}{reason[1:]}. "
                f"Ventilationskategorins typvärde gäller kanaler och don och används "
                f"inte. Siffran är därför en uppskattning.")
        if advice:
            note += f" {advice}"
        if note.strip() not in (r.description or ""):
            r.description = (r.description or "").rstrip() + note
        r.basis = {
            "kind": "saknar_typvärde",
            "label": "Inget EPD-typvärde för aggregatet",
            "subcategory": "aggregat",
            "reason": f"{reason}. {advice}".strip(),
        }
        return

    per_piece = data["baseline_co2e_per_unit"]
    n = data["sample_size"]
    material_note = (f" Antaget standardmaterial: {r.assumed_material}."
                     if r.assumed_material else "")
    # Same population statement as the generic path below.
    geo_note = _geo_note(data, n)
    from aida.data.koncern import concentration_note
    geo_note += concentration_note(data.get("concentration"))
    if res["method"] == "luftflöde":
        flow = res["airflow_m3h"]
        per_m3h = res["per_m3h"]
        n_all = data.get("sample_size_per_piece", n)
        label = (f"EPD-typvärde per luftflöde, {format_value(per_m3h)} kg CO2e per m³/h "
                 f"× {format_value(flow)} m³/h")
        left_out = (f" De {n_all - n} aggregat-EPD:er som inte anger något nominellt "
                    f"luftflöde ingår inte." if n_all > n else "")
        span = (f" EPD:ernas flöden spänner {format_value(data['airflow_min'])} till "
                f"{format_value(data['airflow_max'])} m³/h")
        if res["extrapolated"]:
            span += (f", och {format_value(flow)} m³/h ligger utanför, så värdet är "
                     f"en extrapolering.")
        else:
            span += "."
        text = (
            f"Baslinje från EPD-typvärde per luftflöde: median av övre halvan av "
            f"{n} aggregat-EPD:er (Environdec, EPD Norge), räknat per m³/h av "
            f"nominellt luftflöde ({format_value(per_m3h)} kg CO2e per m³/h), × "
            f"aggregatets {format_value(flow)} m³/h (ur {res['airflow_source']}) = "
            f"{format_value(per_piece)} kg CO2e/st × {format_value(comp.quantity)} st."
            f"{left_out}{span}{geo_note}{material_note} Per luftflöde skiljer "
            f"EPD:erna en faktor {format_value(data['max'] / data['min'])}, per styck "
            f"mycket mer eftersom storleken varierar, därför räknas det per "
            f"luftflöde. Boverket saknar ventilationsaggregat."
        )
        extra = {"airflow_m3h": flow, "per_m3h": per_m3h,
                 "extrapolated": res["extrapolated"]}
    else:
        klass = res["klass"]
        flow_median = data.get("airflow_median")
        where = res.get("class_why") or "komponenten"
        label = f"EPD-typvärde, {klass}saggregat"
        if flow_median:
            label += f" ({format_value(flow_median)} m³/h)"
            how = (f"ett aggregat av klassens mellanstorlek, median av de "
                   f"{n} EPD:ernas nominella luftflöden ({format_value(flow_median)} m³/h), "
                   f"× {format_value(data['per_m3h'])} kg CO2e per m³/h")
        else:
            how = f"median av övre halvan av klassens {n} EPD:er"
        flow_note = res.get("flow_note") or ""
        if flow_note:
            flow_note = f" Luftflödet används inte: {flow_note}."
        text = (
            f"Baslinje från EPD-typvärde för {klass}saggregat (klassen ur {where}): "
            f"{how} = {format_value(per_piece)} kg CO2e/st × "
            f"{format_value(comp.quantity)} st. Klassens EPD:er spänner "
            f"{format_value(data['min'])} till {format_value(data['max'])} kg CO2e/st "
            f"beroende på storlek.{flow_note}{geo_note}{material_note} Ange aggregatets "
            f"luftflöde (m³/h) "
            f"i chatten eller i namnet för ett säkrare värde. Boverket saknar "
            f"ventilationsaggregat."
        )
        extra = {"klass": klass, "airflow_median": flow_median}

    r.co2e_kg = round(per_piece * comp.quantity, 1)
    r.source = "Environdec EPD-typvärde"
    r.boverket_product = ""
    r.co2e_per_unit = per_piece
    r.unit = comp.unit
    r.quantity = comp.quantity
    r.basis = {
        "kind": "epd_typvärde",
        "label": label,
        "method": res["method"],
        "level": "subtype",
        "subcategory": data.get("subcategory", "aggregat"),
        "requested_subtype": "",
        "sample_size": n,
        "full_median": data.get("full_median"),
        "min": data.get("min"),
        "max": data.get("max"),
        "geo_scope": data.get("geo_scope", ""),
        "sample_size_global": data.get("sample_size_global", n),
        "sample_size_europe": data.get("sample_size_europe", 0),
        "concentration": data.get("concentration"),
        **extra,
    }
    r.description = text


def _apply_epd_median_fallback(results: list[BaselineResult], project: Project) -> None:
    """Substitute LLM-uppskattning with EPD-typvärde where available (in-place).

    Three-tier baseline strategy:
    1. Boverket material proxy (handled in _match_components_to_boverket)
    2. EPD-typvärde per category — median of upper-half EPDs by GWP, matching
       NollCO2's "Typical" framing for conventional standard materials
    3. LLM uppskattning (kept when neither tier 1 nor 2 fits)

    Only kicks in when the LLM returned source="Uppskattning". Boverket
    matches are left alone — they're more precise.

    Why upper-half (not full) median: EPD databases skew toward climate-
    conscious producers (selection bias — voluntary disclosure). Full-median
    underestimates "standardval utan klimathänsyn", which is the NollCO2
    reference point. Upper-half median better approximates the conventional
    default a user would pick if they weren't actively climate-optimizing.
    """
    from aida.data.epd_baseline_medians import (
        _SPLIT_SUBCATEGORIES,
        SUBTYPE_LABELS,
        get_baseline_typvärde,
        lookup_withheld_reason,
        member_typvärde,
        split_subcategory_miss,
        subtype_from_material,
        subtype_named,
        textile_typvärde,
        textile_typvärde_unit,
        withheld_reason,
    )
    from aida.data.koncern import concentration_note
    from aida.data.palats_client import component_subcategory
    from aida.data.unit_conversion import typical_item_mass

    comp_map = {c.id: c for c in project.components}
    for r in results:
        if "uppskattning" not in (r.source or "").lower():
            continue
        if r.boverket_product:
            continue  # genuine Boverket material hit, don't touch
        comp = comp_map.get(r.component_id)
        if not comp:
            continue
        category = resolve_category(comp.name, comp.category)
        if not category:
            continue
        # Two ways to reach a subcategory, and they answer different questions.
        #
        # component_subcategory reads the component NAME, which is how the
        # heterogeneous categories (sanitet, belysning, vitvaror) tell a toilet
        # from a tap. That works there because the name IS the product type.
        #
        # For a subtype-preferred category the name is useless: "Golv i
        # tambur/toalett" says nothing about vinyl or linoleum. What identifies
        # the material is the standard material the agent just named from the
        # building type and the component's function, which is the NollCO2
        # question ("byggt på ett idag byggnadstypiskt sätt"). So prefer that,
        # and keep the name-derived one as the fallback.
        subcategory = component_subcategory(comp.name, category)
        if category == "ventilation" and subcategory == "aggregat" and comp.unit == "st":
            _apply_aggregat_typvärde(r, comp)
            continue
        material_subtype = subtype_from_material(category, r.assumed_material)
        # Except when the name already names a split subtype: an
        # "Avjämningsmassa" whose assumed material reads "flytspackel under
        # vinylmatta" is still a levelling compound, not a vinyl floor
        # (HENRIC-3290 del 3).
        if material_subtype and subcategory not in _SPLIT_SUBCATEGORIES.get(category, {}):
            subcategory = material_subtype
        if (category, subcategory, comp.unit) == ("stomme", "konstruktionsstål", "kg"):
            # A named profile in kg meets the EPDs of its own form, as it does
            # in löpmeter (_profile_typvärde; HENRIC-3390).
            from aida.data.steel_profiles import form_subcategory, profile_mass
            profile, _ = profile_mass(comp.name)
            if profile and profile.form and get_baseline_typvärde(
                    category, "kg", form_subcategory(subcategory, profile.form)):
                subcategory = form_subcategory(subcategory, profile.form)
        typvärde_data = get_baseline_typvärde(category, comp.unit, subcategory)

        # kg->st bridge: count-denominated components (a toilet, a radiator) are
        # entered in st, but their EPDs are declared per kg. Convert the kg
        # typvärde to per-st via a typical item mass so these get a baseline
        # instead of falling through to LLM-uppskattning.
        mass_note = ""
        if not typvärde_data and comp.unit == "st":
            kg_data = get_baseline_typvärde(category, "kg", subcategory)
            mass = typical_item_mass(category, subcategory)
            if kg_data and mass:
                typvärde_data = {
                    "baseline_co2e_per_unit": round(kg_data["baseline_co2e_per_unit"] * mass, 2),
                    "sample_size": kg_data["sample_size"],
                    "full_median": round(kg_data["full_median"] * mass, 2),
                    "min": round(kg_data["min"] * mass, 2),
                    "max": round(kg_data["max"] * mass, 2),
                    # Carried through from the kg lookup. The label downstream
                    # reads the RETURNED subcategory rather than the requested
                    # one (a thin subtype can fall back to the category), so a
                    # bridge dict without these would silently relabel a
                    # sanitet/handfat baseline as plain "sanitet".
                    "subcategory": kg_data.get("subcategory", ""),
                    "level": kg_data.get("level", ""),
                    # Read by _validate_baseline, which range-checks a value
                    # resting on an assumed mass (a published one it does not).
                    "bridge": "mass",
                    "concentration": kg_data.get("concentration"),
                }
                mass_note = (
                    f" Omräknat kg→st via antagen typisk vikt {mass} kg/st "
                    f"(approximation)."
                )

        # m3 -> lm/m2 for a frame member, from its own section or thickness
        # (HENRIC-3290). Not an approximation like the mass bridge above: the
        # dimension is the one the component's name states.
        if not typvärde_data and comp.unit in ("lm", "m2"):
            bridged = member_typvärde(category, comp.name, comp.unit, subcategory)
            if bridged:
                typvärde_data = bridged
                # A steel profile is bridged from kg by its standard weight per
                # metre (HENRIC-3363), timber and boards from m3.
                if "per_kg" in bridged:
                    mass_note = (
                        f" Omräknat från {bridged['per_kg']} kg CO2e/kg via "
                        f"komponentens {bridged['geometry']}."
                    )
                else:
                    mass_note = (
                        f" Omräknat från {bridged['per_m3']} kg CO2e/m³ via "
                        f"komponentens {bridged['geometry']}."
                    )

        # m2 -> st for a rug or a curtain, from the size its name states
        # (HENRIC-3366). Same kind of bridge: the size is the component's own.
        if not typvärde_data and comp.unit in ("st", "styck"):
            bridged = textile_typvärde(category, comp.name, comp.unit, subcategory)
            if bridged:
                typvärde_data = bridged
                mass_note = (
                    f" Omräknat från {bridged['per_m2']} kg CO2e/m² via "
                    f"komponentens mått, {bridged['geometry']}."
                )

        if not typvärde_data:
            # A rug's or a curtain's value is kept per m², so that is the key
            # whose reason applies, whatever unit the component is in.
            reason_unit = textile_typvärde_unit(category, comp.unit, subcategory)
            if split_subcategory_miss(category, comp.unit, subcategory):
                # The category has a typvärde, just not for this kind of
                # product, and the estimate below is not that number. Said in
                # the row, because a reader who knows ventilation/st exists
                # would otherwise assume it was used.
                why = (withheld_reason(category, subcategory, comp.unit)
                       or f"katalogen har för få EPD:er för {subcategory} per "
                          f"{comp.unit} (minst 5 krävs)")
                note = (
                    f" Inget EPD-typvärde för {category}/{subcategory}: {why}. "
                    f"{category.capitalize()}-kategorins typvärde gäller andra "
                    f"produkter och används inte. Siffran är därför en uppskattning."
                )
                if note.strip() not in (r.description or ""):
                    r.description = (r.description or "").rstrip() + note
                r.basis = {
                    "kind": "saknar_typvärde",
                    "label": f"Inget EPD-typvärde för {category}/{subcategory}",
                    "subcategory": subcategory,
                    "reason": why[0].upper() + why[1:],
                }
            elif lookup_withheld_reason(category, reason_unit, subcategory):
                # Enough EPDs, deliberately not published (one company
                # group's range, los_inredning/förvaring; a mis-read unit,
                # hiss/st). Same reasoning as above: a reader who can see the
                # catalog rows would otherwise assume the estimate is their
                # median. Looked up along get_baseline_typvärde's own
                # fallback, so a vinyl floor in st is told that golv/st is
                # withheld rather than nothing.
                why = lookup_withheld_reason(category, reason_unit, subcategory)
                key = f"{category}/{subcategory}" if subcategory else category
                note = (f" Inget EPD-typvärde för {key}: {why}. "
                        f"Siffran är därför en uppskattning.")
                if note.strip() not in (r.description or ""):
                    r.description = (r.description or "").rstrip() + note
                r.basis = {
                    "kind": "saknar_typvärde",
                    "label": f"Inget EPD-typvärde för {key}",
                    "subcategory": subcategory,
                    "reason": why[0].upper() + why[1:],
                }
            continue  # no usable EPD-typvärde for this (category[, subcat], unit)

        baseline_per_unit = typvärde_data["baseline_co2e_per_unit"]
        n = typvärde_data["sample_size"]
        full_med = typvärde_data["full_median"]
        new_co2e = round(baseline_per_unit * comp.quantity, 1)
        # The level the lookup actually landed on, which is not always the one
        # asked for: a subtype too thin to publish falls back to the category
        # aggregate. Labelling that as the subtype would be the exact claim this
        # whole change exists to stop making.
        used_sub = typvärde_data.get("subcategory", "")
        level = typvärde_data.get("level", "subtype" if used_sub else "category")
        cat_label = f"{category}/{used_sub}" if used_sub else category

        if level == "category" and material_subtype:
            scope_note = (
                f" Katalogen har för få EPD:er för {material_subtype} för att "
                f"ge ett eget typvärde, så siffran är hela {category}-kategorin "
                f"och spänner över flera materialtyper."
            )
        elif level == "subtype":
            scope_note = f" Avser {used_sub}, inte {category} generellt."
        else:
            scope_note = ""

        # A name that gives another floor than the baseline's (HENRIC-3406):
        # "Textilmatta" against vinyl. The number stays the typical floor's,
        # by method; the row says so, since unexplained it reads as a mistake.
        named = subtype_named(category, comp.name)
        named_note = ""
        # Not for a levelling compound under a named floor ("Avjämning under
        # vinylmatta"): that baseline is the compound, and rightly so.
        if (named and named != used_sub
                and subcategory not in _SPLIT_SUBCATEGORIES.get(category, {})
                and (not used_sub or used_sub in SUBTYPE_LABELS)):
            used_label = (SUBTYPE_LABELS.get(used_sub, used_sub) if used_sub
                          else f"{category} i allmänhet")
            named_note = (
                f" Namnet anger {SUBTYPE_LABELS.get(named, named)}, men baslinjen är "
                f"{used_label}: den beskriver vad som är typiskt för byggnaden och "
                f"användningen, inte det material projektet har valt, eftersom "
                f"jämförelsen annars blir cirkulär."
            )

        # Which population the number rests on, stated with both counts. A
        # "global" key is not a weaker typvärde, it is a typvärde with a known
        # limitation the reader can weigh; hiding it behind a bare n would
        # imply a Swedish context the data does not have.
        geo_scope = typvärde_data.get("geo_scope", "")
        n_global = typvärde_data.get("sample_size_global", n)
        n_europe = typvärde_data.get("sample_size_europe", 0)
        geo_note = _geo_note(typvärde_data, n, " i kategorin")

        material_note = (
            f" Antaget standardmaterial: {r.assumed_material}."
            if r.assumed_material else ""
        )
        # Who is behind the number, on every row (HENRIC-3368): the largest
        # company groups and their counts, and a plain warning when the key
        # rests on two groups' ranges.
        conc_note = concentration_note(typvärde_data.get("concentration"))

        r.co2e_kg = new_co2e
        r.source = "Environdec EPD-typvärde"
        r.boverket_product = ""  # signal: not a Boverket match
        r.co2e_per_unit = baseline_per_unit
        r.unit = comp.unit
        r.quantity = comp.quantity
        r.basis = {
            "kind": "epd_typvärde",
            "label": f"EPD-typvärde, {cat_label}",
            "level": level,
            "subcategory": used_sub,
            "requested_subtype": material_subtype,
            "sample_size": n,
            "full_median": full_med,
            "min": typvärde_data.get("min"),
            "max": typvärde_data.get("max"),
            "geo_scope": geo_scope,
            "sample_size_global": n_global,
            "sample_size_europe": n_europe,
            "bridge": typvärde_data.get("bridge", ""),
            "concentration": typvärde_data.get("concentration"),
        }
        # assumed_material is deliberately NOT cleared. Before 2026-09-01 this
        # assignment replaced the whole description, and the standard material
        # the agent had just reasoned its way to disappeared with it — which is
        # why "vilket golv har den räknat på?" had no answer.
        r.description = (
            f"Baslinje från EPD-typvärde: median av övre halvan av "
            f"{n} EPD:er (Environdec, EPD Norge) i kategorin {cat_label} "
            f"({baseline_per_unit} kg CO2e/{comp.unit}) × {comp.quantity} {comp.unit}."
            f"{material_note}{scope_note}{named_note}{geo_note}{conc_note}{mass_note} "
            f"Övre halvan används för att approximera 'standardval utan "
            f"klimathänsyn' — full median ({full_med}) hade underskattat "
            f"konventionellt val pga selection bias i EPD-databasen. "
            f"Boverket saknar denna materialtyp."
        )


def _complete_baseline(project: Project, boverket_products, match=None) -> list[BaselineResult]:
    """Exactly one baseline row per project component, in project order.

    The matching call returns a JSON list, and nothing checked that the list
    covered the project. A component the model left out vanished from the
    baseline, the alternatives, the choices and the report, and the total was
    still presented as the whole project; a component it returned twice was
    counted twice (Fable audit 2026-07-19, P2 #5). Duplicates and rows for
    unknown ids are settled in _match_components_to_boverket. Here, the
    components still missing are asked for once more on their own. If the
    model skips them again the step fails with the names, because a baseline
    that silently covers part of the project is worse than none.

    ``match`` is the matching function, injectable for tests.
    """
    match = match or _match_components_to_boverket
    results = match(project, boverket_products)
    have = {r.component_id for r in results}
    missing = [c for c in project.components if c.id not in have]
    if missing:
        logger.warning(
            "Baseline match left out %d of %d components (%s); asking again for those",
            len(missing), len(project.components), ", ".join(c.id for c in missing),
        )
        sub = Project(
            building_type=project.building_type,
            area_bta=project.area_bta,
            components=missing,
            name=project.name,
            description=project.description,
            needs_analysis=project.needs_analysis,
        )
        missing_ids = {c.id for c in missing}
        results += [r for r in match(sub, boverket_products)
                    if r.component_id in missing_ids and r.component_id not in have]
        have = {r.component_id for r in results}
        still = [c for c in project.components if c.id not in have]
        if still:
            names = ", ".join(c.name for c in still)
            raise UserFacingError(
                f"Baslinjen kunde inte räknas för alla komponenter (saknas: {names})."
                " Försök igen. Står felet kvar, dela upp projektet i färre komponenter.",
                status_code=502,
            )
    order = {c.id: i for i, c in enumerate(project.components)}
    return sorted(results, key=lambda r: order.get(r.component_id, len(order)))


def _is_price_cached(provider: ClimateProvider, product_name: str) -> bool:
    """Check if a product already has a cached enriched price."""
    cached = provider._cache.get(product_name.lower().strip())
    return bool(cached and cached.price_enriched and cached.cost_per_unit > 0)


def _format_boverket_list(products) -> str:
    """Format Boverket products as compact text for LLM context."""
    lines = []
    for p in products:
        lines.append(f"- {p.name} | {p.co2e_per_unit} kg CO2e/{p.unit}")
    return "\n".join(lines)


# Components per matching call. One call for the whole project stopped fitting:
# Patrik's Nobelgymnasiet project (31 components, 2026-09-19) spent all 16k
# output tokens (8.4k thinking, the rest JSON) and was cut off mid-list, twice,
# and the half-JSON surfaced as a generic error. Re-run 2026-09-26: 153 s,
# stop_reason=max_tokens. Chunks run in parallel, so a large project takes
# about as long as a small one instead of timing out. A chunk that is still
# cut off is split in half and asked again.
BASELINE_CHUNK_SIZE = 8
BASELINE_MAX_PARALLEL = 4


class _Truncated(Exception):
    """The matching answer hit max_tokens; the JSON is incomplete."""


def _sub_project(project: Project, components: list) -> Project:
    return Project(
        building_type=project.building_type,
        area_bta=project.area_bta,
        components=list(components),
        name=project.name,
        description=project.description,
        needs_analysis=project.needs_analysis,
    )


def _match_components_to_boverket(project: Project, boverket_products) -> list[BaselineResult]:
    """Match every component to a Boverket product, in chunks of
    BASELINE_CHUNK_SIZE run in parallel. Row order follows the project."""
    comps = list(project.components)
    if len(comps) <= BASELINE_CHUNK_SIZE:
        return _match_or_split(project, boverket_products)

    n_chunks = -(-len(comps) // BASELINE_CHUNK_SIZE)
    size = -(-len(comps) // n_chunks)  # balanced: 31 -> 8, 8, 8, 7
    chunks = [comps[i:i + size] for i in range(0, len(comps), size)]
    logger.info("Baseline matching %d components in %d parallel chunks",
                len(comps), len(chunks))

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(BASELINE_MAX_PARALLEL, len(chunks))) as ex:
        parts = list(ex.map(
            lambda cs: _match_or_split(_sub_project(project, cs), boverket_products),
            chunks,
        ))
    return [r for part in parts for r in part]


def _match_or_split(project: Project, boverket_products) -> list[BaselineResult]:
    """One matching call; if the answer is cut off, halve and ask again."""
    try:
        return _match_chunk(project, boverket_products)
    except _Truncated:
        comps = list(project.components)
        if len(comps) <= 1:
            raise UserFacingError(
                f"Baslinjen för \"{comps[0].name if comps else '?'}\" blev för lång "
                "för modellens svar. Försök igen. Står felet kvar, korta ner "
                "komponentens beskrivning.",
                status_code=502,
            ) from None
        half = len(comps) // 2
        logger.warning("Baseline match cut off at %d components; splitting in two",
                       len(comps))
        return (_match_or_split(_sub_project(project, comps[:half]), boverket_products)
                + _match_or_split(_sub_project(project, comps[half:]), boverket_products))


def _match_chunk(project: Project, boverket_products) -> list[BaselineResult]:
    """Single LLM call: match the given components to Boverket products."""
    client = get_client()

    # usage_context is what makes the standard material choosable. Intake
    # already writes the functional requirements per component ("entré med
    # blötsnö och saltslask, kräver halksäker och våtmoppbar yta"), and until
    # 2026-09-01 this call threw all of it away and passed only name, quantity
    # and unit. STEG 1 asked the model which material is typical "för denna
    # komponent i denna byggnadstyp" while withholding everything about what the
    # component actually has to do.
    #
    # Truncated because the baseline is a single call covering every component
    # and already sits near max_tokens on large projects. Intake puts the
    # functional requirements first, so a head slice keeps the deciding part.
    def _fmt(c) -> str:
        line = f"- {c.id}: {c.name}, {c.quantity} {c.unit}"
        if c.category:
            line += f" [kategori: {c.category}]"
        ctx = (c.usage_context or "").strip()
        if ctx:
            if len(ctx) > 400:
                ctx = ctx[:400].rsplit(" ", 1)[0] + "…"
            line += f"\n  Funktion och krav: {ctx}"
        return line

    comp_list = "\n".join(_fmt(c) for c in project.components)
    boverket_list = _format_boverket_list(boverket_products)

    logger.info("Baseline LLM matching: %d components against %d Boverket products",
                len(project.components), len(boverket_products))

    response = call_model(
        client,
        model=DEFAULT_MODEL,
        max_tokens=REASONING_MAX_TOKENS,
        effort=EFFORT_HIGH,  # correctness step — bump to "max" if matching regresses
        system=MATCH_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"""Projekt: {project.building_type}, {project.area_bta} m² BTA

KOMPONENTER:
{comp_list}

BOVERKETS PRODUKTLISTA (Typical A1-A3):
{boverket_list}

Matcha varje komponent ovan mot bästa Boverket-produkt. Använd EXAKT de component_id som anges (t.ex. c1, c2, c3)."""
        }],
    )

    text = extract_text(response)

    # Check before parsing: a cut-off list can still contain a parseable
    # fragment (the 2026-09-26 re-run parsed one inner object as "dict, not a
    # list"), and that must never be read as the answer.
    if getattr(response, "stop_reason", None) == "max_tokens":
        logger.warning(
            "Baseline match cut off at max_tokens (%d components, %d chars)",
            len(project.components), len(text),
        )
        raise _Truncated()

    try:
        data = extract_json_value(text, what="baslinje-matchningen")
    except ModelOutputError as e:
        # Opus 4.8 adaptive thinking shares the 16k max_tokens budget. If a very
        # large project pushes thinking + output past the cap, stop_reason is
        # "max_tokens" and the JSON is truncated. Surface it clearly instead of a
        # cryptic decode error in the request handler.
        logger.error(
            "Baseline match returned unparseable JSON (stop_reason=%s, %d chars): %s",
            getattr(response, "stop_reason", "?"), len(text), e,
        )
        raise UserFacingError(
            "Baslinje-matchningen gav ett ofullständigt svar."
            " Försök igen eller dela upp projektet i färre komponenter."
        ) from e
    if isinstance(data, dict) and "components" in data:
        data = data["components"]

    # Build lookups to force correct IDs
    id_by_name = {c.name.lower(): c.id for c in project.components}
    id_by_index = {i: c.id for i, c in enumerate(project.components)}
    comp_map = {c.id: c for c in project.components}

    if not isinstance(data, list):
        raise ModelOutputError(
            f"baslinje-matchningen gav {type(data).__name__}, inte en lista", text)

    known_ids = {c.id for c in project.components}
    # (how sure the id is, position, row). Lower rank is surer: the model's own
    # id, then its component name, then the row's position in the list.
    ranked: list[tuple[int, int, BaselineResult]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            logger.warning("Baseline match row %d is %s, not an object; skipped",
                           i, type(item).__name__)
            continue
        llm_id = str(item.get("component_id", "") or "")
        llm_name = str(item.get("component_name", "") or "")

        if llm_id in known_ids:
            comp_id, rank = llm_id, 0
        elif llm_name.lower() in id_by_name:
            comp_id, rank = id_by_name[llm_name.lower()], 1
        elif i in id_by_index:
            comp_id, rank = id_by_index[i], 2
        else:
            # A row for a component the project does not have. Kept until
            # 2026-09-26, and then summed into the totals as quantity 1 of
            # something nobody asked about.
            logger.warning(
                "Baseline match row %d has unknown id %r (%r); dropped",
                i, llm_id, llm_name,
            )
            continue

        comp = comp_map.get(comp_id)
        quantity = comp.quantity if comp else 1
        co2e_per_unit = item.get("co2e_per_unit", 0)
        co2e_kg = item.get("co2e_kg", co2e_per_unit * quantity)

        boverket_match = str(item.get("boverket_product") or "").strip()
        if boverket_match:
            product = resolve_boverket_product(boverket_match, boverket_products)
            if product is None:
                # A name that is not on the list is the model's own product and
                # figure; labelling it "Boverkets klimatdatabas" credits Boverket
                # with a number it never published. As an estimate the row goes
                # on to the EPD typvärde fallback like any other.
                logger.warning(
                    "Baseline match: %r for %s is not on Boverket's list; "
                    "treated as an estimate", boverket_match, comp_id,
                )
                boverket_match = ""
            else:
                # The list's own spelling, so _apply_member_geometry's lookup by
                # name finds the product the model retyped.
                boverket_match = product.name or product.product_name
        source = "Boverkets klimatdatabas" if boverket_match else "Uppskattning"
        cost_source = "Uppskattning (AI)" if not boverket_match else ""

        unit = item.get("unit", comp.unit if comp else "st")
        description = item.get("description", "")
        if boverket_match and not description:
            description = f"Baslinje (NollCO2): {boverket_match}, {co2e_per_unit} kg CO2e/{unit} x {quantity} {comp.unit if comp else 'st'}. {REASONING['conventional']}"
        elif not description:
            description = f"LLM-uppskattning (ej i Boverkets databas). {REASONING['conventional']}"

        ranked.append((rank, i, BaselineResult(
            component_id=comp_id,
            component_name=item.get("component_name", ""),
            co2e_kg=round(co2e_kg, 1),
            cost_sek=round(item.get("cost_sek", 0)),
            method="NollCO2",
            description=description,
            source=source,
            cost_source=cost_source,
            boverket_product=boverket_match or "",
            assumed_material=(item.get("assumed_material") or "").strip(),
            co2e_per_unit=round(float(co2e_per_unit or 0), 3),
            unit=unit,
            quantity=quantity,
            # Only Boverket hits get their basis here. An "Uppskattning" is
            # about to be replaced by an EPD typvärde in
            # _apply_epd_median_fallback, which sets its own basis; writing one
            # now would leave a stale label behind whenever that substitution
            # does not fire.
            basis={
                "kind": "boverket",
                "label": f"Boverkets klimatdatabas: {boverket_match}",
            } if boverket_match else {},
        )))

    # One row per component. A second row for the same component would be
    # counted twice in every total, so keep the one whose id the model gave
    # itself (surest), else the first, and say that the rest were dropped.
    best: dict[str, tuple[int, int, BaselineResult]] = {}
    for entry in ranked:
        cid = entry[2].component_id
        kept = best.get(cid)
        if kept is None or entry[:2] < kept[:2]:
            if kept is not None:
                logger.warning("Baseline match: duplicate row for %s (row %d); "
                               "keeping row %d", cid, kept[1], entry[1])
            best[cid] = entry
        else:
            logger.warning("Baseline match: duplicate row for %s (row %d); "
                           "keeping row %d", cid, entry[1], kept[1])
    return [entry[2] for entry in sorted(best.values(), key=lambda e: e[1])]


def main():
    """CLI entry point for baseline."""
    if len(sys.argv) < 3 or sys.argv[1] != "--project":
        print("Usage: python -m aida.agents.baseline --project <project.json>", file=sys.stderr)
        sys.exit(1)

    project_path = sys.argv[2]
    print("Steg 1/2: Läser projektbeskrivning...", file=sys.stderr)

    try:
        project = Project.from_json_file(project_path)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Fel: Kunde inte läsa projektfilen: {e}", file=sys.stderr)
        sys.exit(1)

    if not project.components:
        print("Fel: Projektet har inga komponenter.", file=sys.stderr)
        sys.exit(1)

    print(f"Steg 2/2: Beräknar baslinje (NollCO2) för {len(project.components)} komponenter...", file=sys.stderr)
    baseline = calculate_baseline(project)
    print(baseline.to_json())


if __name__ == "__main__":
    main()
