"""Report agent: generates exportable summary report."""

from __future__ import annotations

import json
import sys
from datetime import date

from aida import overrides as overrides_mod
from aida.agents.aggregate import compute_aggregate
from aida.api_client import (
    DEFAULT_MODEL,
    EFFORT_HIGH,
    REASONING_MAX_TOKENS,
    call_model,
    extract_text,
    get_client,
)
from aida.models import Project, Selections


def cell(text) -> str:
    """A markdown table cell that cannot break its own row.

    Every appendix here is deterministic precisely so a marking cannot go
    missing, and an unescaped "|" would undo that: it splits the row into more
    cells than the table has columns, and the docx exporter drops a row it
    cannot fit. The line saying a figure is not Aida's would vanish from the one
    artifact that leaves the tool, with no error anywhere.

    Component names come from the project and override notes are typed by hand,
    so a pipe is not exotic ("Ramavtal 2024 | pos 14"). GFM and `marked` both
    render the escaped form as a plain pipe.
    """
    return str(text).replace("|", "\\|").replace("\n", " ")


REPORT_SYSTEM_PROMPT = """Du är Aidas rapportgenerator — en byggnadsexpert som skapar strukturerade beslutsunderlag för ombyggnadsprojekt.

Aidas uppdrag är att hjälpa förvaltare och byggledare att hitta renoveringslösningar som kraftigt minskar klimatpåverkan utan att ge avkall på praktiska behov.

Varje rapport ska följa denna rubrikstruktur exakt:

# Klimatanalys: [Projekttyp]

## Sammanfattning
Kort projektbeskrivning, total klimatbesparing (kg CO2e och %), kostnadsjämförelse. Max 4-5 meningar.

## Projektförutsättningar
Byggnadstyp, area, antal komponenter. Kort och sakligt.

## Baslinjeberäkning
Vad baslinjen representerar (konventionella material, NollCO2-metod). Totalt baslinjevärde.

## Valda alternativ
Komponenttabellen (använd den som ges i prompten). Kort kommentar per komponent om varför alternativet valdes, klimatvinst och eventuella praktiska fördelar.

## Klimatbesparing
Sammanställning av total besparing. Jämförelse mot baslinjen i absoluta tal och procent.

## Kostnadsbedömning
Totalkostnad jämfört med baslinjen, men bara om prompten anger en totalkostnad. Saknar någon komponent pris redovisas i stället delsumman för de prissatta komponenterna, och vilka som saknar pris. Notera att priser avser uppskattat installerat pris (material + arbete) exkl. moms. Kommentera om det finns Palats-alternativ med styckpris som inte är direkt jämförbara.

## Osäkerheter och begränsningar
Datakällor som använts (Boverket, Environdec, webbsökning). Vad som är verifierat vs uppskattat. Att detta är beslutsstöd, inte certifierade beräkningar.

## Rekommendation
Kort rekommendation baserat på analysen.

REGLER:
- Formell svenska, lämplig för tjänsteskrivelser
- Alla siffror med enheter (kg CO2e, SEK)
- Inga anglicismer, inga AI-typiska fraser, inga em-dashes
- Markdown-format"""


def generate_report_markdown(
    project: Project, selections: Selections, overrides: dict | None = None,
) -> str:
    """Generate a markdown report from project and selections.

    Reuse figures cover the full component quantity even when Palats holds
    fewer units. Henric settled that on 2026-08-15: Aida plans early and stock
    turns over long before anything is procured, so capping to today's stock
    would be its own distortion. The report is what leaves the tool, though, so
    the assumption is appended deterministically rather than left to whether
    the model happens to mention it.
    """
    aggregate = compute_aggregate(project, selections)

    # Build context for LLM
    component_table = ""
    for comp in aggregate.components:
        saving_pct = (comp["co2e_besparing_kg"] / comp["baslinje_co2e_kg"] * 100) if comp["baslinje_co2e_kg"] > 0 else 0
        # A missing price is not zero kronor. Printing "0" in this column let
        # the model write about a cost saving that came out of a data gap.
        cost_cell = "Pris saknas" if comp.get("pris_saknas") else f"{comp['kostnad_sek']:,.0f}"
        if comp.get("pris_saknas") and comp.get("artikelpris_sek"):
            # Named, so the model can do what the system prompt asks about
            # Palats per-article prices, but never as the component's cost.
            cost_cell += f" (annonspris {comp['artikelpris_sek']:,.0f} per artikel, täckning okänd)"
        component_table += (
            f"| {comp['name']} | {comp['valt_alternativ']} | "
            f"{comp['co2e_kg']:.0f} | {comp['baslinje_co2e_kg']:.0f} | "
            f"{comp['co2e_besparing_kg']:.0f} ({saving_pct:.0f}%) | "
            f"{cost_cell} | {comp['källa']} |\n"
        )

    stock_caveats = build_stock_caveats(aggregate.components)
    gwp_caveats = build_gwp_basis_caveats(aggregate.components)
    no_choice = build_missing_selection_caveat(project, aggregate)
    price_gap = build_missing_price_caveat(aggregate)
    estimated_prices = build_estimated_price_caveats(aggregate.components)
    override_rows = overrides_mod.listing(project.to_dict(), overrides)

    saving_pct_total = (
        aggregate.co2e_savings_kg / aggregate.baseline_total_co2e_kg * 100
    ) if aggregate.baseline_total_co2e_kg > 0 else 0

    client = get_client()

    response = call_model(
        client,
        model=DEFAULT_MODEL,
        max_tokens=REASONING_MAX_TOKENS,
        effort=EFFORT_HIGH,
        system=REPORT_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"""Generera en rapport för detta ombyggnadsprojekt:

Projekttyp: {project.building_type}
Area: {project.area_bta} m² BTA
Datum: {date.today().isoformat()}

Sammanställning:
- Total klimatpåverkan (valt): {aggregate.total_co2e_kg:.0f} kg CO2e
- Baslinje (konventionellt): {aggregate.baseline_total_co2e_kg:.0f} kg CO2e
- Klimatbesparing: {aggregate.co2e_savings_kg:.0f} kg CO2e ({saving_pct_total:.0f}%)
{_coverage_prompt_line(no_choice, len(aggregate.components))}{_cost_prompt_lines(aggregate)}
Komponenttabell:
| Komponent | Valt alternativ | CO2e (kg) | Baslinje (kg) | Besparing | Kostnad (SEK) | Källa |
|-----------|----------------|-----------|---------------|-----------|---------------|-------|
{component_table}
{_missing_selection_prompt_block(no_choice)}{_caveat_prompt_block(stock_caveats)}{_price_gap_prompt_block(price_gap)}{_estimated_price_prompt_block(estimated_prices)}{_override_prompt_block(override_rows)}
Skriv en komplett rapport i markdown. Inkludera disclaimer om att detta är uppskattningar för beslutsstöd."""
        }],
    )

    markdown = extract_text(response)
    if no_choice:
        markdown = markdown.rstrip() + "\n\n" + render_missing_selection_caveat(
            no_choice, len(aggregate.components))
    if stock_caveats:
        markdown = markdown.rstrip() + "\n\n" + render_stock_caveats(stock_caveats)
    if gwp_caveats:
        markdown = markdown.rstrip() + "\n\n" + render_gwp_basis_caveats(gwp_caveats)
    if price_gap:
        markdown = markdown.rstrip() + "\n\n" + render_missing_price_caveat(price_gap)
    if estimated_prices:
        markdown = markdown.rstrip() + "\n\n" + render_estimated_price_caveats(estimated_prices)
    if override_rows:
        markdown = markdown.rstrip() + "\n\n" + render_override_caveats(override_rows)
    return markdown


def build_stock_caveats(components: list[dict]) -> list[dict]:
    """Selected reuse alternatives whose figures assume more units than Palats
    currently lists.

    Returns one entry per affected component: name, chosen alternative, units
    in stock, units needed.
    """
    caveats = []
    for comp in components:
        available = comp.get("tillgangligt_antal")
        needed = comp.get("behov_antal")
        if available is None or needed is None:
            continue
        try:
            available = int(available)
            needed = float(needed)
        except (TypeError, ValueError):
            continue
        if needed <= 0 or available >= needed:
            continue
        caveats.append({
            "komponent": comp.get("name", ""),
            "alternativ": comp.get("valt_alternativ", ""),
            "tillgangligt": available,
            "behov": needed,
        })
    return caveats


def build_missing_selection_caveat(project: Project, aggregate) -> list[str]:
    """Names of the project components the totals do not cover.

    A component whose alternatives step failed never reaches the selections,
    and the browser only asks for a choice per component that HAS alternatives.
    Until 2026-09-30 the report then summed the rest and called it "Total
    klimatpåverkan" with nothing to say a component was missing.
    """
    names = {c.id: c.name for c in project.components}
    return [names.get(cid) or cid for cid in aggregate.missing_selection_ids]


def _coverage_prompt_line(no_choice: list[str], counted: int) -> str:
    if not no_choice:
        return ""
    return (
        f"- Summorna ovan gäller {counted} av {counted + len(no_choice)} "
        f"komponenter. Utan valt alternativ: {', '.join(no_choice)}.\n"
    )


def _missing_selection_prompt_block(no_choice: list[str]) -> str:
    if not no_choice:
        return ""
    return (
        "\nKomponenter utan valt alternativ (viktigt, nämn i sammanfattningen "
        "och under osäkerheter):\n"
        + "\n".join(f"- {name}" for name in no_choice)
        + "\nDe ingår varken i klimat- eller kostnadssummorna. Kalla inte "
        "summorna projektets totala klimatpåverkan.\n"
    )


def render_missing_selection_caveat(no_choice: list[str], counted: int) -> str:
    """The appendix that always lands, regardless of what the model wrote."""
    total = counted + len(no_choice)
    rows = "\n".join(f"| {cell(name)} |" for name in no_choice)
    return (
        "## Komponenter utan valt alternativ\n\n"
        f"{len(no_choice)} av {total} komponenter i projektet saknar ett valt "
        "alternativ med klimatvärde och ingår inte i rapportens summor. "
        "Klimatpåverkan, besparing och kostnad gäller alltså "
        f"{counted} av {total} komponenter, inte hela projektet.\n\n"
        "| Komponent utan val |\n|---|\n"
        f"{rows}\n"
    )


def build_missing_price_caveat(aggregate) -> dict | None:
    """Facts about the part of the basket that has no price.

    Returns None when every selected alternative and every baseline figure is
    priced. Otherwise which side lacks a price, per component, plus the two
    totals that ARE comparable, computed over the components priced on both
    sides only. A baseline without a price is the same gap as an alternative
    without one: until 2026-09-30 its zero entered the comparison and a floor
    read as a 40 000 kr overrun against a baseline nobody had priced.
    """
    unpriced = list(aggregate.unpriced_components)
    baseline_unpriced = list(aggregate.baseline_unpriced_components)
    if not unpriced and not baseline_unpriced:
        return None
    rows = []
    comparable = 0
    for c in aggregate.components:
        alt_gap, bl_gap = bool(c.get("pris_saknas")), bool(c.get("baslinje_pris_saknas"))
        if alt_gap and bl_gap:
            rows.append({"komponent": c.get("name", ""), "saknas": "Valt alternativ och baslinje"})
        elif alt_gap:
            rows.append({"komponent": c.get("name", ""), "saknas": "Valt alternativ"})
        elif bl_gap:
            rows.append({"komponent": c.get("name", ""), "saknas": "Baslinje"})
        else:
            comparable += 1
    return {
        "utan_pris": unpriced,
        "antal_utan_pris": len(unpriced),
        "baslinje_utan_pris": baseline_unpriced,
        "rader": rows,
        "antal_prissatta": comparable,
        "antal_totalt": len(aggregate.components),
        "jamforbar_kostnad": aggregate.comparable_cost_sek,
        "jamforbar_baslinje": aggregate.comparable_baseline_cost_sek,
    }


def _comparable_scope(n: int) -> str:
    """Singular and plural: "de 1 komponenter" reads as generated text."""
    if n == 1:
        return "den komponent som har pris både i valt alternativ och i baslinjen"
    return f"de {n} komponenter som har pris både i valt alternativ och i baslinjen"


def _cost_prompt_lines(aggregate) -> str:
    """The cost block of the prompt.

    When part of the basket is unpriced, a bare "Total kostnad" against a full
    baseline is not a comparison, it is a subtraction with a hole in it. In that
    case the model gets the priced subset instead, and is told not to present a
    total or a percentage.
    """
    gap = build_missing_price_caveat(aggregate)
    if not gap:
        return (
            f"- Total kostnad (valt): {aggregate.total_cost_sek:,.0f} SEK\n"
            f"- Baslinje kostnad: {aggregate.baseline_total_cost_sek:,.0f} SEK\n"
            f"- Kostnadsskillnad: {aggregate.cost_difference_sek:+,.0f} SEK\n"
        )
    lines = []
    if gap["utan_pris"]:
        lines.append(
            f"- Kostnad går INTE att totalsummera: {gap['antal_utan_pris']} av "
            f"{gap['antal_totalt']} komponenter saknar pris "
            f"({', '.join(gap['utan_pris'])}).\n"
        )
    if gap["baslinje_utan_pris"]:
        lines.append(
            f"- Kostnad går INTE att jämföra mot baslinjen för "
            f"{len(gap['baslinje_utan_pris'])} av {gap['antal_totalt']} "
            f"komponenter, eftersom baslinjen saknar pris "
            f"({', '.join(gap['baslinje_utan_pris'])}).\n"
        )
    if gap["antal_prissatta"]:
        diff = gap["jamforbar_kostnad"] - gap["jamforbar_baslinje"]
        lines.append(
            f"- För {_comparable_scope(gap['antal_prissatta'])}: "
            f"{gap['jamforbar_kostnad']:,.0f} SEK mot baslinjens "
            f"{gap['jamforbar_baslinje']:,.0f} SEK, alltså {diff:+,.0f} SEK.\n"
        )
    else:
        lines.append(
            "- Ingen komponent har pris både i valt alternativ och i baslinjen, "
            "så ingen kostnadsjämförelse går att göra.\n"
        )
    lines.append(
        "- Ange ALDRIG en totalkostnad eller en procentuell kostnadsförändring "
        "för hela projektet. Redovisa bara delsumman ovan och säg vilka "
        "komponenter som saknar pris.\n"
    )
    return "".join(lines)


def _price_gap_prompt_block(gap: dict | None) -> str:
    if not gap:
        return ""
    return (
        "\nKomponenter utan pris (viktigt, ta upp under kostnadsbedömning och "
        "osäkerheter):\n"
        + "\n".join(f"- {r['komponent']} (saknas: {r['saknas'].lower()})" for r in gap["rader"])
        + "\nEtt saknat pris betyder att ingen prisuppgift hittades, inte att "
        "posten är gratis. Skriv detta rakt ut.\n"
    )


def render_missing_price_caveat(gap: dict) -> str:
    """The appendix that always lands, regardless of what the model wrote."""
    rows = "\n".join(f"| {cell(r['komponent'])} | {r['saknas']} |" for r in gap["rader"])
    said = []
    if gap["utan_pris"]:
        said.append(f"{gap['antal_utan_pris']} av {gap['antal_totalt']} valda "
                    "alternativ saknar prisuppgift.")
    if gap["baslinje_utan_pris"]:
        said.append(f"Baslinjen saknar prisuppgift för {len(gap['baslinje_utan_pris'])} "
                    f"av {gap['antal_totalt']} komponenter.")
    if gap["antal_prissatta"]:
        diff = gap["jamforbar_kostnad"] - gap["jamforbar_baslinje"]
        comparison = (
            "och den kostnadsjämförelse som redovisas gäller bara "
            f"{_comparable_scope(gap['antal_prissatta'])}: "
            f"{gap['jamforbar_kostnad']:,.0f} SEK mot baslinjens "
            f"{gap['jamforbar_baslinje']:,.0f} SEK, alltså {diff:+,.0f} SEK."
        )
    else:
        comparison = ("och ingen komponent har pris både i valt alternativ och "
                      "i baslinjen, så någon kostnadsjämförelse går inte att göra.")
    return (
        "## Komponenter utan prisuppgift\n\n"
        + " ".join(said)
        + " Det betyder att ingen prisuppgift gick att hitta, inte att posten "
        "är kostnadsfri. Någon totalkostnad för projektet går därför inte att "
        f"ange, {comparison}\n\n"
        "| Komponent | Pris saknas för |\n|---|---|\n"
        f"{rows}\n"
    )


def build_estimated_price_caveats(components: list[dict]) -> list[dict]:
    """Selected components whose cost is the model's own estimate.

    Web search found no source, so the number is a plausible Swedish installed
    price rather than a quoted one. It belongs in the report because a reader
    cannot otherwise tell it apart from a searched market price, and the two
    deserve different amounts of trust.
    """
    return [
        {"komponent": c.get("name", ""),
         "alternativ": c.get("valt_alternativ", ""),
         "kostnad": c.get("kostnad_sek", 0)}
        for c in components
        if c.get("prisunderlag") == "llm_estimate"
    ]


def _estimated_price_prompt_block(caveats: list[dict]) -> str:
    if not caveats:
        return ""
    lines = "\n".join(
        f"- {c['komponent']}: \"{c['alternativ']}\", {c['kostnad']:,.0f} SEK"
        for c in caveats
    )
    return (
        "\nPriser utan källa (viktigt, ta upp under kostnadsbedömning och "
        "osäkerheter):\n"
        f"{lines}\n"
        "För dessa hittade webbsökningen ingen priskälla, så beloppet är "
        "språkmodellens egen uppskattning av ett typiskt installerat pris. "
        "Skriv detta rakt ut och blanda inte ihop dem med sökta marknadspriser.\n"
    )


def render_estimated_price_caveats(caveats: list[dict]) -> str:
    """The appendix that always lands, regardless of what the model wrote."""
    rows = "\n".join(
        f"| {cell(c['komponent'])} | {cell(c['alternativ'])} | {c['kostnad']:,.0f} |"
        for c in caveats
    )
    return (
        "## Priser utan källa\n\n"
        "Kostnaderna nedan är språkmodellens egen uppskattning av ett typiskt "
        "installerat pris på den svenska byggmarknaden. Webbsökningen hittade "
        "ingen priskälla för dem. De redovisas hellre än utelämnas, eftersom en "
        "tom kostnadskolumn gör alternativen omöjliga att väga mot varandra, "
        "men de är svagare underlag än de sökta marknadspriserna och behöver "
        "kontrolleras mot offert innan de används i upphandling.\n\n"
        "| Komponent | Valt alternativ | Uppskattad kostnad (SEK) |\n"
        "|---|---|---|\n"
        f"{rows}\n"
    )


def build_gwp_basis_caveats(components: list[dict]) -> list[dict]:
    """Selected components whose climate figure rests on GWP-GHG.

    Used where an EPD's own components did not add up and GWP-fossil was
    unusable. Close to the fossil basis in magnitude but not the same
    indicator, so it gets named rather than folded in silently.
    """
    return [
        {"komponent": c.get("name", ""), "alternativ": c.get("valt_alternativ", "")}
        for c in components
        if c.get("gwp_underlag") == "ghg"
    ]


def render_gwp_basis_caveats(caveats: list[dict]) -> str:
    rows = "\n".join(
        f"| {cell(c['komponent'])} | {cell(c['alternativ'])} |" for c in caveats
    )
    return (
        "## Avvikande klimatunderlag\n\n"
        "Klimatsiffrorna bygger på GWP-fossil för skedena A1-A3, enligt "
        "Boverkets metod. För posterna nedan gick det inte: produktens egen "
        "miljödeklaration är internt motsägelsefull, det redovisade "
        "fossilvärdet stämmer inte med deklarationens egen totalsumma. För dem "
        "används i stället GWP-GHG, alltså totala växthusgasutsläpp exklusive "
        "biogent kol. Det ligger nära GWP-fossil i storleksordning men är inte "
        "samma indikator, och skillnaden bör nämnas om siffran förs vidare.\n\n"
        "| Komponent | Valt alternativ |\n|---|---|\n"
        f"{rows}\n"
    )


def render_override_caveats(rows: list[dict]) -> str:
    """The figures in this report that are the user's, not Aida's.

    Deterministic and appended after the model's text, like every other caveat
    here, for the reason PR #550 set out: the marking is the condition, not a
    nicety, so it must not depend on whether the model chose to repeat it. This
    one matters more than the others, because an overridden figure is the only
    number in the document that no calculation of ours stands behind.
    """
    body = "\n".join(
        f"| {cell(r['komponent'])} | {cell(r['fält'])} | {r['värde']:,.0f} "
        f"| {cell(r['anteckning'])} |"
        for r in rows
    )
    return (
        "## Manuellt angivna värden\n\n"
        "Posterna nedan har skrivits över av den som gjort analysen, och siffran "
        "kommer alltså inte från Aidas beräkning utan från underlaget i "
        "anteckningen. Aidas eget värde finns kvar i verktyget och visas igen om "
        "överskrivningen tas bort. Övriga siffror i rapporten är beräknade enligt "
        "metoden ovan.\n\n"
        "| Komponent | Fält | Angivet värde | Anteckning |\n|---|---|---|---|\n"
        f"{body}\n"
    )


def _override_prompt_block(rows: list[dict]) -> str:
    """Tell the model the same facts, so its own text is not written in ignorance
    of what the appendix will say."""
    if not rows:
        return ""
    lines = "\n".join(
        f"- {r['komponent']}, {r['fält']}: {r['värde']:,.0f} ({r['anteckning']})"
        for r in rows
    )
    return (
        "\nManuellt angivna värden (satta av användaren, inte beräknade av dig):\n"
        f"{lines}\n"
        "Dessa siffror ingår redan i summorna ovan. Nämn i texten att de är "
        "manuellt angivna om du refererar till dem. Hitta inte på egna värden "
        "för dem.\n"
    )


def _caveat_prompt_block(caveats: list[dict]) -> str:
    """Give the model the same facts, so its own limitations section is not
    written in ignorance of what the appendix will say."""
    if not caveats:
        return ""
    lines = "\n".join(
        f"- {c['komponent']}: valt återbruk \"{c['alternativ']}\" har "
        f"{c['tillgangligt']} artiklar i lager mot ett behov på {c['behov']:.0f}."
        for c in caveats
    )
    return (
        "\nTillgång på återbruk (viktigt, ta upp under osäkerheter):\n"
        f"{lines}\n"
        "Kostnad och klimatnytta ovan är beräknade på hela behovet, alltså som "
        "om resten går att få tag på begagnat senare. Skriv detta rakt ut.\n"
    )


def render_stock_caveats(caveats: list[dict]) -> str:
    """The appendix that always lands, regardless of what the model wrote."""
    rows = "\n".join(
        f"| {cell(c['komponent'])} | {cell(c['alternativ'])} "
        f"| {cell(c['tillgangligt'])} | {c['behov']:.0f} |"
        for c in caveats
    )
    return (
        "## Antaganden om tillgång till återbruk\n\n"
        "Kostnad och klimatnytta för återbruk är beräknade på hela behovet, även "
        "där Palats hade färre artiklar när analysen kördes. Det är avsiktligt: "
        "Aida används i ett tidigt planeringsskede och marknadsplatsens lager "
        "omsätts innan något handlas upp. Siffrorna förutsätter alltså att "
        "resten går att få tag på begagnat, vilket behöver stämmas av innan de "
        "används som underlag för upphandling.\n\n"
        "| Komponent | Valt återbruksalternativ | I lager vid analys | Behov |\n"
        "|---|---|---|---|\n"
        f"{rows}\n"
    )


def _fmt(value, digits=0) -> str:
    """A number as a Swedish reader writes one: hard space for thousands, comma
    for the decimal, and no decimal at all when there is nothing after it.

    "112.0 m2" in a klimatredovisning reads as a foreign document, and worse,
    "1 447.0" mixes a Swedish thousands separator with an English decimal
    point inside one figure. The trailing zero is its own small untruth: it
    claims a tenth of a kilo of precision on a number that came from an
    estimated quantity times a declared average.
    """
    if value is None:
        return "-"
    out = f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")
    if "," in out:
        out = out.rstrip("0").rstrip(",")
    return out


def _signed(value) -> str:
    """A deviation keeps its sign. "+1 200" and "1 200" are different findings,
    and the second one reads as an amount rather than as an overrun."""
    return f"{value or 0:+,.0f}".replace(",", " ")


def render_followup_report(project: dict, result: dict, overrides=None,
                           property_ref: str = "") -> str:
    """The klimatredovisning: what the building actually cost the climate.

    Written without the model, unlike every other report here. Three reasons,
    in order of weight.

    Every figure in this document already exists. `followup.compute` produced
    the rows, the totals and the list of uncertainties; there is no judgement
    left to make, only a document to lay out. A model between the numbers and
    the page can only add wording, and wording is the thing that goes wrong: the
    place a summary rounds off is exactly the sentence saying the totals cover
    three of five components, because that sentence is the awkward one.

    It is also a document of record rather than a piece of advice. Someone puts
    this in a redovisning to a nämnd, and two runs on identical state must
    produce identical text, or the difference between them becomes a question
    nobody can answer.

    And it returns instantly, which matters on Vercel's ten seconds.
    """
    rows = result.get("rows") or []
    totals = result.get("totals") or {}
    uncertain = result.get("uncertainties") or []
    counted = totals.get("rows_counted", 0)
    total_rows = totals.get("rows_total", 0)

    head = f"# Klimatredovisning: {project.get('building_type') or 'ombyggnation'}\n\n"
    if property_ref:
        head += f"Avser {property_ref}. "
    head += f"Upprättad {date.today().isoformat()}.\n\n"

    if not counted:
        # Not an error: a follow-up that has begun but has nothing bound yet is
        # a normal state, and it deserves a document that says which rows are
        # missing rather than a total of zero.
        return (
            head
            + "## Utfall\n\nIngen byggdel har ännu ett utfall som går att räkna. "
            + (f"De {total_rows} byggdelarna saknar installerad mängd, bunden "
               "miljödeklaration, eller båda. " if total_rows else "")
            + "Redovisningen nedan listar vad som saknas.\n\n"
            + _followup_uncertainty_section(uncertain)
            + FOLLOWUP_METHOD
        )

    coverage = (
        f"Summorna gäller {counted} av {total_rows} byggdelar. Utanför står "
        f"{', '.join(totals.get('uncounted_names') or [])}. Baslinje och plan är "
        "räknade över samma rader som utfallet, så jämförelsen gäller, men den "
        "gäller inte hela projektet.\n\n"
    ) if counted != total_rows else (
        "Summorna gäller projektets enda byggdel.\n\n" if total_rows == 1
        else f"Summorna gäller samtliga {total_rows} byggdelar.\n\n"
    )

    body = (
        "## Utfall\n\n"
        f"Klimatpåverkan från det som faktiskt installerades: "
        f"{_fmt(totals.get('outcome_co2e_kg'))} kg CO2e (GWP-fossil, skedena A1-A3). "
        f"Baslinjen för samma byggdelar var {_fmt(totals.get('baseline_co2e_kg'))} kg CO2e, "
        f"alltså {_fmt(totals.get('avoided_vs_baseline_kg'))} kg CO2e undveks. "
        f"Mot den plan som valdes i analysen är avvikelsen "
        f"{_signed(totals.get('deviation_vs_plan_kg'))} kg CO2e.\n\n"
        + coverage
    )

    if totals.get("cost_rows_counted"):
        n = totals["cost_rows_counted"]
        # "de 1 byggdelar" is what a template written only for the plural case
        # produces, and one priced component is a common state early in a
        # follow-up. A document that leaves the tool should not read as generated.
        scope = ("den byggdel som har ett pris" if n == 1
                 else f"de {n} byggdelar som har ett pris")
        body += (
            f"Verklig kostnad för {scope}: "
            f"{_fmt(totals.get('actual_cost_sek'))} SEK mot planerade "
            f"{_fmt(totals.get('planned_cost_sek'))} SEK, alltså "
            f"{_signed(totals.get('cost_difference_sek'))} SEK.\n\n"
        )
    if totals.get("cost_uncounted_names"):
        body += (
            "Har verkligt pris men inget planerat, så kostnaden jämförs inte: "
            f"{', '.join(totals['cost_uncounted_names'])}.\n\n"
        )

    body += (
        "## Per byggdel\n\n"
        "| Byggdel | Installerat | Mängd | Underlag | Utfall (kg CO2e) "
        "| Baslinje (kg) | Planerat (kg) |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    for r in rows:
        quantity = (f"{_fmt(r['installed_quantity'], 1)} {r.get('installed_unit') or ''}".strip()
                    if r.get("installed_quantity") is not None else "-")
        outcome = (_fmt(r["outcome_co2e_kg"], 1) if r.get("outcome_co2e_kg") is not None
                   else "Räknas inte")
        body += (
            f"| {cell(r.get('name', ''))} "
            f"| {cell(r.get('installed_name') or '-')} "
            f"| {cell(quantity)} "
            f"| {cell(_followup_source_label(r))} "
            f"| {cell(outcome)} "
            f"| {_fmt(r.get('baseline_co2e_kg'), 1) if r.get('baseline_co2e_kg') is not None else '-'} "
            f"| {_fmt(r.get('planned_co2e_kg'), 1) if r.get('planned_co2e_kg') is not None else '-'} |\n"
        )
    body += "\n"

    out = head + body + _followup_uncertainty_section(uncertain)

    ghg = [r for r in rows
           if isinstance(r.get("epd"), dict) and r["epd"].get("gwp_basis") == "ghg"]
    if ghg:
        out += (
            "## Avvikande klimatunderlag\n\n"
            "För byggdelarna nedan gick GWP-fossil inte att använda: "
            "miljödeklarationens egna delposter stämmer inte med dess totalsumma. "
            "I stället används GWP-GHG, alltså totala växthusgasutsläpp exklusive "
            "biogent kol. Det ligger nära i storleksordning men är inte samma "
            "indikator.\n\n"
            "| Byggdel | Miljödeklaration |\n|---|---|\n"
            + "".join(f"| {cell(r.get('name', ''))} | {cell(r['epd'].get('name') or r['epd'].get('id'))} |\n"
                      for r in ghg)
            + "\n"
        )

    # Same principle as the section above, one step weaker: not a different
    # indicator but a different chain of custody. The figure comes from a real
    # published declaration, named by its registration number in the table, but
    # it was inferred from that document rather than read out of it. A reader who
    # checks the row against the source will find a number that does not match to
    # the decimal, and the honest place to say so is here and not afterwards.
    estimated = [r for r in rows
                 if isinstance(r.get("epd"), dict)
                 and r["epd"].get("gwp_source") == "estimated"]
    if estimated:
        out += (
            "## Uppskattat ur miljödeklaration\n\n"
            "Byggdelarna nedan är bundna till en miljödeklaration som Aida inte "
            "hämtar maskinellt, utan som är inlagd för hand. För just de här "
            "raderna är siffran uppskattad ur deklarationen i stället för avläst "
            "ur den, till exempel för en variant som deklarationen inte redovisar "
            "separat. Deklarationen är riktig och går att slå upp på sitt "
            "registreringsnummer. Talet är en tolkning av den.\n\n"
            "| Byggdel | Miljödeklaration |\n|---|---|\n"
            + "".join(f"| {cell(r.get('name', ''))} | {cell(r['epd'].get('name') or r['epd'].get('id'))} |\n"
                      for r in estimated)
            + "\n"
        )

    override_rows = overrides_mod.listing(project, overrides)
    if override_rows:
        out += render_override_caveats(override_rows) + "\n"

    return out + FOLLOWUP_METHOD


def _followup_source_label(row: dict) -> str:
    """What stands behind this row's figure, in the cell rather than in a footnote.

    A reader comparing two rows needs to know that one is the installed product's
    own declaration and the other is a category average, and a legend at the
    bottom of the page is not where that comparison happens.
    """
    from aida.followup import QUALITY_LABELS

    label = QUALITY_LABELS.get(row.get("match_quality") or "none", "Ingen träff")
    epd = row.get("epd")
    if isinstance(epd, dict) and epd.get("reg_no"):
        return f"{label} ({epd['reg_no']})"
    return label


def _followup_uncertainty_section(uncertain: list[dict]) -> str:
    """Always present, even when empty.

    An absent section reads as an oversight; a section saying every row rests on
    the installed product's own declaration is a claim, and it is one this
    document should have to make explicitly.
    """
    if not uncertain:
        return (
            "## Osäkerheter\n\nVarje redovisad byggdel har en miljödeklaration "
            "bunden till den produkt som faktiskt installerades, och en "
            "installerad mängd i deklarationens egen enhet.\n\n"
        )
    return (
        "## Osäkerheter\n\n"
        "Byggdelarna nedan vilar inte på den installerade produktens egen "
        "deklaration. De redovisas hellre än utelämnas, men de bär inte samma "
        "vikt som raderna ovan.\n\n"
        "| Byggdel | Underlag | Varför |\n|---|---|---|\n"
        + "".join(f"| {cell(u['komponent'])} | {cell(u['underlag'])} | {cell(u['varför'])} |\n"
                  for u in uncertain)
        + "\n"
    )


FOLLOWUP_METHOD = (
    "## Metod och avgränsning\n\n"
    "Utfallet är beräknat som installerad mängd gånger miljödeklarationens "
    "GWP-fossil för skedena A1-A3, per byggdel. En byggdel utan bunden "
    "deklaration, utan registrerad mängd, eller där deklarationen är angiven i "
    "en annan enhet än mängden, räknas inte in i summan. Den räknas alltså inte "
    "som noll, utan står utanför, och namnges ovan.\n\n"
    "Återbruk redovisas som noll i A1-A3, eftersom ingen ny produktion skett. "
    "Transporten av det återbrukade materialet är registrerad men inte omräknad "
    "till utsläpp, eftersom omräkningen kräver en massa per byggdel som inte "
    "finns i underlaget. Utfallet för återbruksrader är därför ett golv.\n\n"
    "Baslinjen följer NollCO2-metoden och beskriver utsläppen om allt hade "
    "byggts nytt utan miljöambitioner. Kolumnen Planerat är det alternativ som "
    "valdes i analysen innan arbetet utfördes.\n\n"
    "Detta är beslutsunderlag, inte en certifierad klimatdeklaration enligt "
    "Boverkets föreskrifter.\n"
)


def generate_report_pdf(project: Project, selections: Selections, output_path: str) -> str:
    """Generate a PDF report. Falls back to markdown if PDF generation fails."""
    markdown = generate_report_markdown(project, selections)

    try:
        import subprocess
        md_path = output_path.replace(".pdf", ".md")
        with open(md_path, "w") as f:
            f.write(markdown)

        result = subprocess.run(
            ["pandoc", md_path, "-o", output_path, "--pdf-engine=xelatex"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            return output_path
    except (FileNotFoundError, subprocess.SubprocessError):
        pass

    # Fallback: save as markdown
    md_path = output_path.replace(".pdf", ".md")
    with open(md_path, "w") as f:
        f.write(markdown)
    return md_path


def main():
    """CLI entry point for report."""
    args = sys.argv[1:]

    project_path = None
    selections_path = None
    fmt = "markdown"
    output_path = None

    i = 0
    while i < len(args):
        if args[i] == "--project" and i + 1 < len(args):
            project_path = args[i + 1]
            i += 2
        elif args[i] == "--selections" and i + 1 < len(args):
            selections_path = args[i + 1]
            i += 2
        elif args[i] == "--format" and i + 1 < len(args):
            fmt = args[i + 1]
            i += 2
        elif args[i] == "--output" and i + 1 < len(args):
            output_path = args[i + 1]
            i += 2
        else:
            i += 1

    if not project_path or not selections_path:
        print("Usage: python -m aida.agents.report --project <project.json> --selections <selections.json> [--format markdown|pdf] [--output path]", file=sys.stderr)
        sys.exit(1)

    try:
        project = Project.from_json_file(project_path)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Fel: Kunde inte läsa projektfilen: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        selections = Selections.from_json_file(selections_path)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Fel: Kunde inte läsa urvalsfilen: {e}", file=sys.stderr)
        sys.exit(1)

    if not selections.components:
        print("Fel: Inga komponenter valda. Kan inte generera rapport.", file=sys.stderr)
        sys.exit(1)

    if fmt == "pdf" and output_path:
        path = generate_report_pdf(project, selections, output_path)
        print(f"Rapport sparad: {path}", file=sys.stderr)
        # generate_report_pdf returns the .pdf path on success, or a .md path on
        # pandoc fallback. Only the markdown is safe to read as text — opening a
        # binary PDF in text mode raises UnicodeDecodeError on the success path.
        if path.endswith(".md"):
            with open(path) as f:
                print(f.read())
    else:
        report = generate_report_markdown(project, selections)
        if output_path:
            with open(output_path, "w") as f:
                f.write(report)
            print(f"Rapport sparad: {output_path}", file=sys.stderr)
        print(report)


if __name__ == "__main__":
    main()
