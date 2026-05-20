"""Baseline agent: calculates baseline (conventional standard materials) per component."""

from __future__ import annotations

import json
import logging
import sys

from aida.api_client import (
    DEFAULT_MODEL,
    THINKING_DEEP,
    extract_text,
    get_client,
    thinking_config,
)
from aida.data.climate_data import REASONING, normalize_component_name
from aida.data.climate_provider import ClimateProvider
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
        category = normalize_component_name(r.component_name)
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


MATCH_SYSTEM_PROMPT = """Du är AIda:s baslinjeberäknare — en byggnadsexpert som beräknar baslinjen för klimatpåverkan.

Baslinjen representerar standardfallet enligt NollCO2-metoden: vad det kostar klimatmässigt om projektet använder konventionella material utan särskild klimathänsyn.

KLIMATMETOD (gäller hela analysen):
- GWP-fossil, livscykelskedena A1-A3 (cradle-to-gate), enligt Boverkets klimatdatabas.
- Inkludera ALDRIG biogenic carbon credit i baslinjen. Värden måste vara konsistenta med Boverket Typical A1-A3.

Du får:
1. En lista med projektets komponenter (med id, namn, antal, enhet)
2. Boverkets kompletta produktlista med CO2e-värden (GWP-fossil, Typical A1-A3)

UPPGIFT — följ dessa steg för VARJE komponent:

STEG 1 — BESTÄM STANDARDMATERIAL:
Fundera på vad det konventionella/typiska materialet är för denna komponent i denna byggnadstyp.
Exempel: golv i skola → homogen vinylmatta (PVC). Innervägg → gipsskiva på stålreglar.

STEG 2 — HITTA BOVERKET-PROXY:
Boverkets databas är organiserad efter materialtyp, inte byggnadsfunktion. Den saknar t.ex.
kategorier för "golvbeläggning" och "sanitetsprodukter". Matcha därför efter MATERIALSAMMAN-
SÄTTNING, inte funktion:
- Vinylgolv (PVC) → "Takduk, PVC" är samma basmaterial
- Gipsskiva på innervägg → "Gipsskiva, standardskiva"
- Stålreglar → "Lättreglar av stål, primär"

STEG 3 — JUSTERA FÖR MATERIALEGENSKAPER:
När du använder en proxy, tänk på skillnader i vikt/tjocklek/densitet mellan proxyn och
det verkliga materialet. Beskriv resonemanget i description-fältet.
Exempel: Takduk PVC är ~1.2 mm, homogent vinylgolv är ~2.0 mm och tätare.
Justera co2e_per_unit proportionellt baserat på kg/m² eller tjocklek.

STEG 4 — UPPSKATTA ENBART SOM SISTA UTVÄG:
Bara om INGEN Boverket-produkt har liknande materialsammansättning (t.ex. sanitets-
porslin, elektronik). Sätt då boverket_product till null. Uppskattningen ska
alltid avse GWP-fossil A1-A3 (cradle-to-gate, exkl. biogenic carbon credit) så
värdet är jämförbart med övriga komponenter.

PRISER:
Sätt cost_sek till 0 — priser hämtas separat via webbsökning.

Svara med ENBART giltig JSON (ingen markdown, inga kommentarer):
[
  {
    "component_id": "string (exakt id från komponentlistan)",
    "component_name": "string",
    "boverket_product": "string (exakt produktnamn från Boverket-listan, eller null)",
    "co2e_per_unit": number,
    "unit": "string (enhet från Boverket-produkten, konverterad till komponentens enhet vid behov)",
    "co2e_kg": number (co2e_per_unit x quantity),
    "cost_sek": 0,
    "method": "NollCO2",
    "description": "Beskriv: 1) antaget standardmaterial, 2) vald Boverket-proxy, 3) eventuell justering och varför",
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

    # Phase 1: LLM-based matching against full Boverket product list
    boverket_products = provider._cache.get_all_boverket()
    results = _match_components_to_boverket(project, boverket_products)

    # Phase 2: Batch price enrichment
    from aida.data.pricing_provider import lookup_price, lookup_prices_batch

    products_needing_prices = [
        (r.component_name, "")
        for r in results
        if not _is_price_cached(provider, r.component_name)
    ]

    batch_prices: dict[str, tuple[float, str, str]] = {}
    if products_needing_prices:
        batch_prices = lookup_prices_batch(products_needing_prices)
        for product_key, (price, _unit, _source) in batch_prices.items():
            provider._cache.update_cost(product_key, price)

        for name, unit in products_needing_prices:
            if name.lower() not in batch_prices:
                result = lookup_price(name, unit)
                if result:
                    price, u, src = result
                    batch_prices[name.lower()] = (price, u, src)
                    provider._cache.update_cost(name.lower(), price)

    # Phase 3: Apply prices to results
    comp_map = {c.id: c for c in project.components}
    for r in results:
        batch_result = batch_prices.get(r.component_name.lower())
        if batch_result:
            comp = comp_map.get(r.component_id)
            quantity = comp.quantity if comp else 1
            r.cost_sek = round(batch_result[0] * quantity)
            r.cost_source = "Webbsökning (AI)"

    results = _validate_baseline(results, project.components)
    return Baseline(components=results)


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


def _match_components_to_boverket(project: Project, boverket_products) -> list[BaselineResult]:
    """Single LLM call: match all components to Boverket products."""
    client = get_client()

    comp_list = "\n".join(
        f"- {c.id}: {c.name}, {c.quantity} {c.unit}"
        for c in project.components
    )
    boverket_list = _format_boverket_list(boverket_products)

    logger.info("Baseline LLM matching: %d components against %d Boverket products",
                len(project.components), len(boverket_products))

    response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=3000 + THINKING_DEEP,
        thinking=thinking_config(THINKING_DEEP),
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
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    data = json.loads(text.strip())
    if isinstance(data, dict) and "components" in data:
        data = data["components"]

    # Build lookups to force correct IDs
    id_by_name = {c.name.lower(): c.id for c in project.components}
    id_by_index = {i: c.id for i, c in enumerate(project.components)}
    comp_map = {c.id: c for c in project.components}

    results = []
    for i, item in enumerate(data):
        llm_id = item.get("component_id", "")
        llm_name = item.get("component_name", "")
        known_ids = {c.id for c in project.components}

        if llm_id in known_ids:
            comp_id = llm_id
        elif llm_name.lower() in id_by_name:
            comp_id = id_by_name[llm_name.lower()]
        elif i in id_by_index:
            comp_id = id_by_index[i]
        else:
            comp_id = llm_id

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

        results.append(BaselineResult(
            component_id=comp_id,
            component_name=item.get("component_name", ""),
            co2e_kg=round(co2e_kg, 1),
            cost_sek=round(item.get("cost_sek", 0)),
            method="NollCO2",
            description=description,
            source=source,
            cost_source=cost_source,
        ))

    return results


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
