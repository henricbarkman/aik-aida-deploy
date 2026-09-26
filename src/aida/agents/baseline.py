"""Baseline agent: calculates baseline (conventional standard materials) per component."""

from __future__ import annotations

import json
import logging
import sys

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
    normalize_component_name,
    resolve_category,
)
from aida.data.climate_provider import ClimateProvider
from aida.errors import UserFacingError
from aida.llm_json import ModelOutputError, extract_json_value
from aida.models import Baseline, BaselineResult, Project

logger = logging.getLogger(__name__)


def _validate_baseline(results: list[BaselineResult], components: list) -> list[BaselineResult]:
    """Validate prices and CO2 values on baseline results.

    Extreme outliers get clamped to reasonable ranges. Mild outliers get flagged.
    """
    from aida.data.price_validation import validate_co2e, validate_total_price

    comp_map = {c.id: c for c in components}
    for r in results:
        comp = comp_map.get(r.component_id)
        quantity = comp.quantity if comp else 0
        category = resolve_category(r.component_name, comp.category if comp else "")
        is_estimate = "uppskattning" in (r.cost_source or "").lower()

        # Validate price
        if r.cost_sek <= 0:
            r.cost_sek = 0
            if "pris ej tillgängligt" not in r.description.lower():
                r.description = r.description.rstrip(". ") + ". Pris ej tillgängligt."
        else:
            validated_cost, price_note = validate_total_price(
                r.cost_sek, quantity, category, is_estimate=is_estimate,
            )
            if validated_cost != r.cost_sek:
                r.cost_sek = validated_cost
            if price_note and price_note.lower() not in r.description.lower():
                r.description = r.description.rstrip(". ") + f". {price_note}."

        # Validate CO2
        if quantity > 0 and r.co2e_kg > 0:
            co2e_per_unit = r.co2e_kg / quantity
            validated_co2, co2_note = validate_co2e(co2e_per_unit, quantity, category)
            if validated_co2 != r.co2e_kg:
                r.co2e_kg = validated_co2
            if co2_note and co2_note.lower() not in r.description.lower():
                r.description = r.description.rstrip(". ") + f". {co2_note}."

    return results


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

Välj det konventionella valet utan särskild klimathänsyn — inte det bästa tillgängliga,
och inte det sämsta tänkbara.

STEG 2 — MATCHA MOT BOVERKET ENDAST VID SAMMA MATERIAL:
Boverkets databas är organiserad efter materialtyp, inte byggnadsfunktion. Välj en Boverket-
produkt ENBART när den faktiskt ÄR komponentens standardmaterial:
- Gipsskiva på innervägg matchar "Gipsskiva, standardskiva" (gips är gips). OK.
- Betongvägg matchar en betongprodukt (betong är betong). OK.
- Stålreglar matchar "Lättreglar av stål, primär" (stål är stål). OK.
- Mineralull som isolering matchar en mineralullsprodukt. OK.

Låna ALDRIG en produkt av annan typ bara för att den delar basmaterial. Det ger en
vilseledande baslinje. Exempel på vad som är FÖRBJUDET:
- Vinylgolv mot "Takduk, PVC": golvbeläggning och takduk är olika produkter även om båda
  är PVC. Sätt boverket_product=null och source="Uppskattning" istället.
Boverket saknar bl.a. golvbeläggning, sanitetsporslin, vitvaror och belysning som egna
produkter. Leta inte efter en ersättare för dem i Boverket.

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


def calculate_baseline(project: Project) -> Baseline:
    """Calculate NollCO2 baseline for each component.

    Uses a single LLM call with the full Boverket product list (~229 products,
    ~2200 tokens) for semantic matching. The LLM picks the best Boverket
    product per component, or estimates when no match exists.
    """
    provider = ClimateProvider()
    provider.ensure_synced()

    # Phase 1: LLM-based matching against full Boverket product list, with
    # exactly one row per component (re-asked once for any the model skipped).
    boverket_products = provider._cache.get_all_boverket()
    results = _complete_baseline(project, boverket_products)

    # Phase 1b: For components where the LLM fell back to "Uppskattning"
    # (no Boverket material proxy fit), substitute an EPD-median where the
    # component's category has reliable aggregated data.
    _apply_epd_median_fallback(results, project)

    # Phase 2: Batch price enrichment
    from aida.data.pricing_provider import (
        lookup_price,
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
    products_needing_prices = [
        (r.component_name, unit_by_name.get(r.component_name.lower(), ""))
        for r in results
        if not _is_price_cached(provider, r.component_name)
    ]

    def _usable(product_key: str, price_unit: str) -> bool:
        want = unit_by_name.get(product_key, "")
        if not want or price_unit_matches(price_unit, want):
            return True
        logger.warning(
            "Baseline price for '%s' is per %r but the component is counted in "
            "%r; discarded", product_key, price_unit, want,
        )
        return False

    batch_prices: dict[str, tuple[float, str, str]] = {}
    if products_needing_prices:
        batch_prices = {
            key: val
            for key, val in lookup_prices_batch(products_needing_prices).items()
            if _usable(key, val[1])
        }
        for product_key, (price, _unit, _source) in batch_prices.items():
            provider._cache.update_cost(product_key, price)

        for name, unit in products_needing_prices:
            if name.lower() not in batch_prices:
                result = lookup_price(name, unit)
                if result and _usable(name.lower(), result[1]):
                    price, u, src = result
                    batch_prices[name.lower()] = (price, u, src)
                    provider._cache.update_cost(name.lower(), price)

    # Phase 3: Apply prices to results
    for r in results:
        batch_result = batch_prices.get(r.component_name.lower())
        if batch_result:
            comp = comp_map.get(r.component_id)
            quantity = comp.quantity if comp else 1
            r.cost_sek = round(batch_result[0] * quantity)
            r.cost_source = "Webbsökning (AI)"

    results = _validate_baseline(results, project.components)
    return Baseline(components=results)


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
        get_baseline_typvärde,
        subtype_from_material,
    )
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
        material_subtype = subtype_from_material(category, r.assumed_material)
        if material_subtype:
            subcategory = material_subtype
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
                }
                mass_note = (
                    f" Omräknat kg→st via antagen typisk vikt {mass} kg/st "
                    f"(approximation)."
                )

        if not typvärde_data:
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

        # Which population the number rests on, stated with both counts. A
        # "global" key is not a weaker typvärde, it is a typvärde with a known
        # limitation the reader can weigh; hiding it behind a bare n would
        # imply a Swedish context the data does not have.
        geo_scope = typvärde_data.get("geo_scope", "")
        n_global = typvärde_data.get("sample_size_global", n)
        n_europe = typvärde_data.get("sample_size_europe", 0)
        if geo_scope == "europa":
            geo_note = (
                f" Urvalet är europeiska EPD:er (SE/Norden/EU), {n} av "
                f"{n_global} i kategorin."
            )
        elif geo_scope == "global":
            geo_note = (
                f" Bara {n_europe} europeiska EPD:er, under golvet "
                f"{typvärde_data.get('min_samples_europe', '')}, så hela det globala "
                f"urvalet används."
            )
        else:
            geo_note = ""

        material_note = (
            f" Antaget standardmaterial: {r.assumed_material}."
            if r.assumed_material else ""
        )

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
        }
        # assumed_material is deliberately NOT cleared. Before 2026-09-01 this
        # assignment replaced the whole description, and the standard material
        # the agent had just reasoned its way to disappeared with it — which is
        # why "vilket golv har den räknat på?" had no answer.
        r.description = (
            f"Baslinje från EPD-typvärde: median av övre halvan av "
            f"{n} EPD:er (Environdec, EPD Norge) i kategorin {cat_label} "
            f"({baseline_per_unit} kg CO2e/{comp.unit}) × {comp.quantity} {comp.unit}."
            f"{material_note}{scope_note}{geo_note}{mass_note} "
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

        boverket_match = item.get("boverket_product")
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
