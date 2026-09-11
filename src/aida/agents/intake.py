"""Intake agent: extracts project parameters from natural language description."""

from __future__ import annotations

import json
import logging
import math
import sys
import time

from aida.api_client import (
    DEFAULT_MODEL,
    EFFORT_HIGH,
    REASONING_MAX_TOKENS,
    call_model,
    extract_text,
    get_client,
    remaining_budget,
)
from aida.llm_json import ModelOutputError, extract_json_object
from aida.models import Project

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Du är Aida — en byggnadsexpert som hjälper förvaltare och byggledare att hitta renoveringslösningar med kraftigt minskad klimatpåverkan, utan att ge avkall på praktiska behov.

Din uppgift i detta steg är att extrahera projektinformation från en fri textbeskrivning av ett ombyggnadsprojekt.

Du ska identifiera:
1. Byggnadstyp (t.ex. skola, kontor, förskola, bostadshus)
2. Ungefärlig area i BTA (bruttoarea i kvadratmeter)
3. En lista av renoveringskomponenter (vad som ska bytas/renoveras)
4. För varje komponent: ett kort resonemang om brukar- och miljökrav som styr vilka material som är lämpliga ("usage_context")
5. En projektövergripande behovsanalys (`needs_analysis`) — separerad i (a) vad användaren faktiskt sagt och (b) vad du som agent inferrat. Användaren kommer granska och godkänna detta innan downstream-stegen körs.

Svara ALLTID med giltig JSON i detta format:
{
  "building_type": "string",
  "area_bta": number,
  "storeys": number,
  "name": "projektnamn om nämnt",
  "description": "original beskrivning",
  "components": [
    {"id": "c1", "name": "komponentnamn", "quantity": number, "unit": "m2|st|lm", "category": "kategori", "quantity_source": "user_specified" eller "estimated", "usage_context": "1-3 meningar om brukare + miljö + funktionella krav"}
  ],
  "needs_analysis": {
    "from_user": "Parafras av användarens input — bara det användaren faktiskt sagt om byggnaden, användningen, kraven. Ingen agent-tolkning.",
    "inferred": "Dina slutsatser om brukare, miljöbelastning och funktionella krav baserat på byggnadstyp och beskrivning. Markera tydligt med 'Eftersom...' eller 'Troligen...' när du resonerar bortom användarens input.",
    "assumptions": ["lista över saker du antagit utan att fråga — t.ex. 'antar att skolan har normal elevtrafik (200-400 elever)'"],
    "would_clarify": ["frågor du gärna hade ställt om de varit kritiska, men som du gått vidare utan svar på"]
  },
  "clarification_needed": null eller "fråga"
}

Regler:
- Komponent-id ska vara c1, c2, c3 etc
- Gissa rimlig quantity om den inte anges (baserat på area och byggnadstyp)
- Unit ska vara m2, st, eller lm (löpmeter)
- Category ska vara en av: golv, kakel, innervägg, yttervägg, fasadskikt, betongvägg, tak, fönster, dörr, isolering, belysning, ventilation, hiss, kylanläggning, sanitet, vitvaror, storköksutrustning, vvs, farg, el, radiator
  - kakel: kaklad/klinkad yta (våtrumsvägg, -golv, kakel/klinker). Välj kakel framför golv/innervägg när ytan är keramisk.
  - fasadskikt vs yttervägg: välj `fasadskikt` när bara byggnadens yttre beklädnad byts eller renoveras (fasadpanel, träfasad, fasadskivor). Välj `yttervägg` när hela väggkonstruktionen byggs eller byts, alltså inklusive stomme och isolering. Vid tvekan i en ombyggnad: `fasadskikt`, eftersom en renovering oftast rör skiktet och inte hela väggen. Ren ommålning av befintlig panel är `farg`.
  - vvs: rör, stambyte, avlopp. farg: målning/ommålning. el: elkabel/elinstallation. radiator: radiator/värmeelement.
- Om area inte anges, uppskatta baserat på byggnadstyp och komponenter
- Svara på svenska

QUANTITY_SOURCE — sätt per komponent:
- "user_specified" om antalet kommer DIREKT från användarens beskrivning kopplat till just den komponenten. Ex: "4 toalettstolar", "tre fönster", "Spegel 4 st".
- "estimated" om du gissar antalet från area, byggnadstyp eller schablon. Även om beskrivningen ger area (t.ex. "300 m²") räknas det som "estimated" om komponenten är något annat än yta (t.ex. lampor på 300 m² → estimated, eftersom användaren inte sa hur många lampor).
- Yta-baserade komponenter (golv, väggar, tak) där area_bta används direkt: "user_specified" om användaren gav arean explicit, annars "estimated".
- Vid tvekan: "estimated".

GEOMETRI — härled ytor, återanvänd aldrig BTA rakt av:
`area_bta` är golvytan. En fasad, ett tak eller en yttervägg har en helt annan yta, och att sätta samma siffra på dem är ett av de fel som gör hela analysen fel.

- Ange `storeys` (antal våningar) på projektnivå. Står det "enplanshus" är det 1. Framgår det inte, gissa utifrån byggnadstyp och yta och skriv antagandet i `needs_analysis.assumptions`.
- Fasad och yttervägg: byggnadens omkrets gånger våningshöjd. Med kvadratisk approximation blir omkretsen 4 × roten ur (BTA / antal våningar), och våningshöjden är cirka 3 meter om inget annat sägs. Ett enplanshus på 700 m² får alltså ungefär 4 × √700 × 3 ≈ 320 m² fasad, inte 700.
- Tak: motsvarar byggnadens fotavtryck, alltså BTA delat med antal våningar (plus påslag för lutning om taket är brant).
- Skriv i komponentens `usage_context` hur du kom fram till ytan när du uppskattat den.

USAGE_CONTEXT — funktionella krav per komponent:
Du är byggnadsexpert med materialkunskap och brukarsförståelse. Det betyder att du ska resonera om komponentens *användning* — inte välja material, men identifiera de krav som styr vilka material som ens är lämpliga.

För varje komponent, fyll i usage_context med 1-3 meningar som täcker:
- VEM använder utrymmet/komponenten (brukare: barn, vårdpersonal, allmänhet, ofta-besökare, sällan-besökare)
- VILKEN MILJÖLAST den utsätts för (våtbelastning, slitage, kemikalier, sand, salt, värme, ljud, hygienkrav)
- VILKA FUNKTIONELLA KRAV det implicerar (halksäker, lätt rengörbar, slittålig, brandklass, ljudklass, vattentät, antibakteriell)

Exempel — förskole-tambur golv:
"Entré på förskola, dagligt slitage från barn 1-6 år som kommer in med blöt snö, sand och saltslask vintertid. Kräver halksäker yta, mycket lätt att våtmoppa, tål kemikalier från städmedel, och slittålig mot abrasiv smuts. Material som absorberar fukt (obehandlat trä, oljat parkett) är olämpliga; gummi, linoleum, klinker med halkfri yta passar bättre."

Exempel — klassrumsgolv:
"Klassrum för 6-12 år, högt slitage från möbler och dagligt fottryck, krav på god akustik och dammbinding. Behöver tåla regelbunden mopprengöring."

Exempel — kontorsbadrum WC:
"Personaltoalett för förvaltning, måttlig brukarfrekvens, hygienkrav på lätt rengörbara ytor, kvalitetskrav på vattenbesparing och tystgång. BBR-tillgänglighet bör verifieras."

VIKTIGT: Detta är inte ett materialval — du namnger eventuellt OLÄMPLIGA material och vilka egenskaper som krävs, men låter alternativ-steget göra själva valet. Om ingen särskild kontext kan utläsas (t.ex. "tak" utan vidare info i ett bostadshus), skriv en kort generisk rad ("Vanligt bostadshus-tak, standardkrav för väderbeständighet och isolering") snarare än att hitta på.

KONSISTENS: Per-komponent `usage_context` SKA följa av projekt-level `needs_analysis.inferred`. Om needs_analysis säger "lågstadieskola, hög elevtrafik, blöt sand vintertid", får ingen komponents usage_context säga något som motsäger det. Tänk top-down: skriv needs_analysis först, härled sedan per-komponent.

NEEDS_ANALYSIS — separationen citat vs. inferens:
`from_user` ska vara så nära användarens egen text som möjligt — parafrasera bara för att städa språket. Skriv ALDRIG där: "Eftersom det är en skola antar vi...", "Troligen är detta...", "Det implicerar att...". Sådant hör i `inferred`.

`inferred` är där du resonerar som byggnadsexpert. Var explicit med dina antaganden ("Eftersom byggnadsåret är 2015 förmodar jag att grundkonstruktionen är intakt"). Använd `assumptions`-listan för diskreta antaganden som inte ryms i prosan ("antar 4 brukare per kontor"). Använd `would_clarify` för det du gärna hade vetat men gick vidare utan ("verksamhetstyp i denna kontorsbyggnad — callcenter eller specialistmottagning ändrar slitageprofilen").

Exempel — input "byta golv i tamburen på en lågstadieskola":
- from_user: "Användaren vill byta golvet i tamburen på en lågstadieskola."
- inferred: "Lågstadieskola innebär elever 6-9 år, hög dagligt slitage från cirka 100-300 elever som passerar entrén minst två gånger per dag. Vintertid kommer blöta skor med snö, salt och sand in i tamburen — golvet behöver tåla fukt, vara halksäkert och lätt att våtmoppa. Hygienkrav på tålighet mot rengöringskemikalier. Olja behandlat trä och oljat parkett är därför olämpligt. Linoleum, gummi, klinker med halkfri yta passar; vinyl med tillräcklig slitstyrka kan funka."
- assumptions: ["Antar 100-300 elever och två passager per dag", "Antar nordiskt klimat med vinterförhållanden"]
- would_clarify: ["Finns entrémattor som tar mest av fukten innan tamburgolvet?", "Hur lång är tamburen — påverkar hur långt blöt sand når in i byggnaden"]

FÖRTYDLIGANDEN:
- Fråga INTE om specifika materialval (t.ex. vilken typ av golv eller vilken isolering). Du ska fokusera på behov och funktionella krav, inte material. Materialval är alternativ-stegets uppgift, och Aida kan komma med bättre förslag än användaren tänkt sig.
- Fråga däremot gärna om saker som påverkar analysen:
  * Byggnadsår (påverkar befintliga material och förutsättningar)
  * Särskilda krav (t.ex. Miljöbyggnad, tillgänglighet, ljudkrav, fuktproblem)
  * Om renoveringen är total eller partiell
  * Budget- eller tidplansramar om de inte nämnts
  * Verksamhetstyp om byggnadstypen är otydlig (lokalkontor vs callcenter vs läkarmottagning i samma "kontor"-skal styr olika usage_context)
- Sätt clarification_needed till null om beskrivningen ger tillräckligt för en rimlig analys.
- Be om förtydligande (max 1-2 korta frågor) när svaret skulle bli väsentligt bättre med mer information. Inkludera då de komponenter du redan kunnat identifiera i components-arrayen.

TIDIGARE DISKUSSION:
- Om beskrivningen innehåller en sektion märkt "Tidigare diskussion i sessionen" eller "Korrigering från användaren": läs den noggrant.
- Fråga ALDRIG om något användaren redan besvarat tidigare OCH som inte ändras i korrigeringen. Återanvänd det tidigare svaret för sådana fält (t.ex. byggnadsår, certifieringskrav, omfattning).
- Om korrigeringen explicit ändrar ett tidigare besvarat fält, använd det NYA värdet från korrigeringen — inte det gamla från diskussionen.
- Sätt clarification_needed till null om tidigare svar (eventuellt ändrade av korrigeringen) fyller informationsbehovet, även om värdena inte upprepas i den nya korrigeringstexten.
- Bevara projektnamn från tidigare beskrivning om det inte uttryckligen ändras.
- Bevara usage_context från tidigare iteration om komponenten inte ändrats — uppdatera bara när komponenten ändrats eller ny info påverkar funktionella krav.
- Bevara needs_analysis från tidigare iteration om varken byggnadstyp eller huvudsaklig användning ändrats. Uppdatera `from_user` om användaren tillfört ny direkt-info; uppdatera `inferred` bara om någon förutsättning faktiskt skiftat.
- Rensa `would_clarify` vid varje iteration: frågorna visas för användaren i chatten som en inbjudan att svara innan baslinjen räknas. Ta bort varje fråga som korrigeringen nu besvarar (svaret går in i `from_user` och berörda komponenters `usage_context`). Behåll bara det som fortfarande är obesvarat, viktigast först, max 3.
"""


# Facade geometry. Sara's June test got a facade area of 700 m² for a 700 m²
# single-storey care home, i.e. the floor area copied straight across. The
# prompt now explains the derivation, and this guard catches it when the model
# ignores that anyway — a silently wrong envelope area poisons both the
# baseline and every alternative for that component.
STOREY_HEIGHT_M = 3.0
_ENVELOPE_CATEGORIES = {"yttervägg", "fasadskikt"}
_ENVELOPE_NAME_TOKENS = ("fasad", "yttervägg")


def estimate_facade_area_m2(area_bta: float, storeys: float = 1) -> float:
    """Facade area from gross floor area, square-footprint approximation.

    perimeter = 4 * sqrt(footprint), footprint = BTA / storeys.
    700 m² on one storey -> 4*sqrt(700)*3 ≈ 317 m².
    """
    storeys = max(1, int(storeys or 1))
    footprint = float(area_bta) / storeys
    if footprint <= 0:
        return 0.0
    perimeter = 4 * math.sqrt(footprint)
    return perimeter * STOREY_HEIGHT_M * storeys


def _is_envelope(component: dict) -> bool:
    if (component.get("category") or "").lower() in _ENVELOPE_CATEGORIES:
        return True
    name = (component.get("name") or "").lower()
    return any(token in name for token in _ENVELOPE_NAME_TOKENS)


def fix_envelope_quantities(data: dict) -> dict:
    """Replace a facade quantity that is just the floor area repeated.

    Only touches m² envelope components the model itself estimated. A quantity
    the user stated is left alone even when it looks odd — correcting the user
    silently would be worse than the bug.
    """
    area_bta = data.get("area_bta") or 0
    if not area_bta or area_bta <= 0:
        return data
    storeys = data.get("storeys") or 1

    for component in data.get("components") or []:
        if (component.get("unit") or "").lower() != "m2":
            continue
        if component.get("quantity_source") == "user_specified":
            continue
        if not _is_envelope(component):
            continue
        quantity = component.get("quantity") or 0
        if abs(quantity - area_bta) > 0.01 * area_bta:
            continue

        estimated = estimate_facade_area_m2(area_bta, storeys)
        if estimated <= 0:
            continue
        component["quantity"] = round(estimated)
        note = (
            f"Fasadytan är uppskattad till {round(estimated)} m² ur byggnadens "
            f"omkrets ({round(4 * math.sqrt(float(area_bta) / max(1, int(storeys)))) } m) "
            f"gånger våningshöjd {STOREY_HEIGHT_M:g} m. Golvytan {round(area_bta)} m² "
            "är inte samma sak som fasadytan. Ange rätt fasadyta i chatten om du har den."
        )
        existing = (component.get("usage_context") or "").strip()
        component["usage_context"] = f"{existing} {note}".strip()
        logger.warning(
            "Envelope quantity equalled area_bta (%s m2) for %r; re-estimated to %s m2",
            area_bta, component.get("name"), round(estimated),
        )

    return data


REPAIR_INSTRUCTION = (
    "Ditt förra svar gick inte att läsa som JSON. Svara nu med ENBART "
    "JSON-objektet, utan inledande text, utan förklaring efteråt och utan "
    "kodstaket. Första tecknet ska vara { och sista }."
)


def _call_intake(client, messages: list[dict], timeout: float | None = None,
                 system: str = SYSTEM_PROMPT):
    extra = {"timeout": timeout} if timeout is not None else {}
    return call_model(
        client,
        model=DEFAULT_MODEL,
        **extra,
        # Opus 4.8 + adaptive thinking. REASONING_MAX_TOKENS covers thinking +
        # the JSON output; the old 8000 was a visible-output cap only.
        # needs_analysis (from_user + inferred + assumptions + would_clarify) plus
        # per-component usage_context made multi-room projects truncate at 6000.
        max_tokens=REASONING_MAX_TOKENS,
        effort=EFFORT_HIGH,
        system=system,
        messages=messages,
    )


def run_intake(description: str, attachments: list[dict] | None = None) -> dict:
    """Extract project parameters from a natural language description.

    Both testers hit a hard crash here in June 2026, because the response was
    parsed with a bare json.loads: an off-format answer surfaced in the chat as
    "Fel: Expecting value: line 1 column 4 (char 3)". Parsing now goes through
    extract_json_object, and a first failure buys one repair round-trip before
    we give up.

    `attachments` are content blocks from aida.attachments.load_blocks, so a
    förfrågningsunderlag can be the project description (§14.6). The repair
    round-trip reuses `messages` and `system`, so it sees the same files.
    """
    from aida.attachments import attach_to_messages, with_rule

    client = get_client()
    started_at = time.monotonic()

    blocks = attachments or []
    system = with_rule(SYSTEM_PROMPT, blocks)
    messages: list[dict] = attach_to_messages([{"role": "user", "content": description}], blocks)
    response = _call_intake(client, messages, system=system)
    text = extract_text(response)

    try:
        return fix_envelope_quantities(extract_json_object(text, what="projektbeskrivningen"))
    except ModelOutputError as first:
        logger.warning(
            "Intake parse failed (%s). stop_reason=%s raw=%r",
            first, getattr(response, "stop_reason", None), text[:2000],
        )

    # One repair attempt, but only if the request has time for it. Retrying
    # into a budget that has already run out just swaps our message for a
    # gateway timeout page.
    budget = remaining_budget(started_at)
    if budget < 20:
        logger.error("Intake parse failed with %.0fs budget left; skipping repair", budget)
        raise ModelOutputError(
            "Analysen hann inte bli klar i tid.", raw=text
        )

    repair_messages = messages + [
        {"role": "assistant", "content": text or "(tomt svar)"},
        {"role": "user", "content": REPAIR_INSTRUCTION},
    ]
    retry = _call_intake(client, repair_messages, timeout=budget, system=system)
    retry_text = extract_text(retry)
    try:
        return fix_envelope_quantities(
            extract_json_object(retry_text, what="projektbeskrivningen")
        )
    except ModelOutputError as second:
        logger.error(
            "Intake parse failed after repair (%s). stop_reason=%s raw=%r",
            second, getattr(retry, "stop_reason", None), retry_text[:2000],
        )
        raise


def intake_from_description(description: str) -> Project:
    """Run intake and return a Project object."""
    data = run_intake(description)
    return Project.from_dict(data)


def main():
    """CLI entry point for intake."""
    if len(sys.argv) < 3 or sys.argv[1] != "--input":
        print("Usage: python -m aida.agents.intake --input <description>", file=sys.stderr)
        sys.exit(1)

    description = sys.argv[2]
    print("Steg 1/1: Analyserar projektbeskrivning...", file=sys.stderr)

    result = run_intake(description)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
