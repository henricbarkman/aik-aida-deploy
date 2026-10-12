"""Intake agent: extracts project parameters from natural language description."""

from __future__ import annotations

import json
import logging
import math
import re
import sys
import time

from aida.api_client import (
    DEFAULT_MODEL,
    EFFORT_HIGH,
    EFFORT_MEDIUM,
    REASONING_MAX_TOKENS,
    call_model,
    extract_text,
    get_client,
    remaining_budget,
)
from aida.data.climate_data import canonical_category
from aida.llm_json import ModelOutputError, extract_json_object
from aida.models import construction_year

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Du är Aida — en byggnadsexpert som hjälper förvaltare och byggledare att hitta renoveringslösningar med kraftigt minskad klimatpåverkan, utan att ge avkall på praktiska behov.

Din uppgift i detta steg är att extrahera projektinformation från en fri textbeskrivning av ett ombyggnadsprojekt.

Du ska identifiera:
1. Byggnadstyp (t.ex. skola, kontor, förskola, bostadshus)
2. Ungefärlig area i BTA (bruttoarea i kvadratmeter)
3. En lista av renoveringskomponenter (vad som ska bytas/renoveras)
4. En projektövergripande behovsanalys (`needs_analysis`) — separerad i (a) vad användaren faktiskt sagt och (b) vad du som agent inferrat. Användaren kommer granska och godkänna detta innan downstream-stegen körs.

Svara ALLTID med giltig JSON i detta format:
{
  "building_type": "string",
  "area_bta": number,
  "storeys": number,
  "construction_year": number eller null,
  "name": "projektnamn om nämnt",
  "components": [
    {"id": "c1", "name": "komponentnamn", "quantity": number, "unit": "m2|st|lm|kg", "category": "kategori", "quantity_source": "user_specified" eller "estimated"}
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
- Unit ska vara m2, st, eller lm (löpmeter). kg bara för avjämning och tätskikt, och bara när användaren själv angett vikten (till exempel antal säckar gånger säckvikt); räkna aldrig om kvadratmeter till kg.
- Category ska vara en av: golv, kakel, innervägg, yttervägg, fasadskikt, betongvägg, tak, undertak, tätskikt, fönster, dörr, isolering, belysning, ventilation, hiss, kylanläggning, sanitet, vitvaror, storköksutrustning, vvs, farg, el, radiator, fast_inredning, los_inredning, stomme
  - tak vs undertak: tak är yttertaket (takpapp, takpannor, plåttak). undertak är innertak och akustik: undertaksplattor, akustikplattor, väggabsorbenter, bafflar och undertakets bärverk (T24-profiler). Namnge sorten ("Akustikplattor", "Väggabsorbenter", "Bafflar", "Bärverk T24", "Gipstak", "Putsat innertak"), eftersom sorterna jämförs var för sig. Plattor, absorbenter, bafflar och gipstak i m2, bärverk i lm. Säger beskrivningen bara "innertak" eller "himling": fråga i clarification_needed om det är undertaksplattor i bärverk, gips eller puts, och lägg inte till plattor och bärverk på egen hand.
  - tätskikt: tätskikt i våtrum (rollat eller flytande tätskikt, tätskiktsmembran, tätskiktsduk) under kakel. Egen komponent, skild från kaklet. Yttertakets tätskikt (papp, takduk) är tak.
  - Avjämning (flytspackel, avjämningsmassa) är golv, men en egen komponent med "Avjämning" i namnet, skild från golvbeläggningen ovanpå. I m2: skriv skiktets tjocklek i namnet ("Avjämning 10 mm"). Saknas tjockleken: fråga efter den i clarification_needed, och gissa den inte.
  - Glaspartier och glasväggar är innervägg, med "Glasparti" i namnet.
  - stomme: bärande stomme, alltså balkar, pelare och bjälklag i stål, limträ, KL-trä eller betong. Bärande väggar är inte stomme.
    Reglar (trä eller stål), läkt, syll, konstruktionsvirke och konstruktionsskivor (OSB, plywood, spånskiva) är också stomme, även när de sitter i en innervägg. Skriv dimensionen i namnet ("Reglar 45x95", "Stålreglar 70 mm", "OSB-skiva 12 mm") och räkna reglar i lm (antal × längd), skivor i m2. Saknas dimension eller längd för reglar: fråga efter den i clarification_needed, och gissa den inte.
    Stålbalkar och stålpelare: skriv profilen i namnet när underlaget anger den ("Stålbalk HEA 200", "Stålpelare VKR 100x100x5") och räkna i lm. Vikten per meter tas då ur standardtabeller. Hitta inte på en profil.
    Betongbjälklag: skriv sorten och tjockleken i namnet ("Håldäck 200 mm", "Massivbjälklag 200 mm", "Plattbärlag 50 mm") och räkna i m2. Betongbalkar och betongpelare: tvärsnittet i namnet ("Betongbalk 300x500"), räkna i lm. Står håldäckets vikt per m² i underlaget, skriv den också ("Håldäck 200 mm, 290 kg/m2"). Hitta inte på tjocklek eller vikt.
  - kakel: kaklad/klinkad yta (våtrumsvägg, -golv, kakel/klinker). Välj kakel framför golv/innervägg när ytan är keramisk.
  - fasadskikt vs yttervägg: välj `fasadskikt` när bara byggnadens yttre beklädnad byts eller renoveras (fasadpanel, träfasad, fasadskivor). Välj `yttervägg` när hela väggkonstruktionen byggs eller byts, alltså inklusive stomme och isolering. Vid tvekan i en ombyggnad: `fasadskikt`, eftersom en renovering oftast rör skiktet och inte hela väggen. Ren ommålning av befintlig panel är `farg`.
  - vvs: rör, stambyte, avlopp. farg: målning/ommålning. el: elkabel/elinstallation. radiator: radiator/värmeelement.
  - fast_inredning: inredning som är fast monterad i byggnaden, alltså köksskåp, köksluckor, bänkskivor, diskbänkar, spegelskåp och badrumsskåp. Namnge komponenten efter vad den är ("Köksluckor", "Spegelskåp"), inte bara "Inredning". Platsbyggd eller inbyggd förvaring (en platsbyggd garderob, inbyggda skåp, fast monterade hyllor) är också fast_inredning; skriv det då i namnet ("Platsbyggd garderob", "Inbyggda skåp"), för det är namnet som avgör. Förvaring utan ett sådant ord räknas som lös.
  - los_inredning: lösa möbler och lös inredning, alltså stolar, kontorsstolar, bord, skrivbord, förvaringsskåp, hyllor, garderober, soffor, fåtöljer, fristående skärmar, gardiner och mattor. Ta med dem bara när användaren själv nämner dem, och hitta inte på möblering till en byggnad. Namnge varje sort för sig ("Elevstol", "Kontorsstol", "Skrivbord", "Förvaringsskåp", "Soffa", "Golvskärm"), aldrig bara "Möbler", och räkna i st. En lös matta är los_inredning, en golvbeläggning (plastmatta, heltäckningsmatta, entrématta) är golv. Mattor och gardiner jämförs per m²: skriv storleken med enhet i namnet när användaren anger den ("Matta 2x3 m", "Gardin 140x250 cm") och räkna i st, eller ange ytan i m² om det är den som är känd. Hitta inte på en storlek.
  - Installationer: tar beskrivningen upp dem, ta med var och en som egen komponent i sin kategori (ventilation för kanaler, don och aggregat; vvs; el; radiator; belysning; kylanläggning; hiss), även när resten av beskrivningen handlar om ytskikt.
  - Ventilationsaggregat (FTX-aggregat, från- och tilluftsaggregat, luftbehandlingsaggregat) räknas i st. Anger beskrivningen aggregatets luftflöde, skriv det i namnet ("FTX-aggregat 3 000 m³/h"; l/s och m³/s går också bra). Framgår det att aggregatet sitter i varje lägenhet, eller att det betjänar hela byggnaden, skriv det också i namnet ("FTX-aggregat, ett per lägenhet", "Centralt FTX-aggregat") och likaså en systembeteckning ("Luftbehandlingsaggregat LB01"). Anges inget av det: gissa inget flöde och fråga inte efter det, lägg bara till aggregatet.
- Om area inte anges, uppskatta baserat på byggnadstyp och komponenter
- `construction_year` är året byggnaden uppfördes, som ett heltal, när beskrivningen anger det ("byggår 1974", "byggd 1974", "uppförd 1962"). Anges det inte: null, och gissa inte. Ett årtal för en tidigare renovering eller ett material ("plastgolv från 2003") är inte byggåret.
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
- Skriv i `needs_analysis.assumptions` hur du kom fram till ytan när du uppskattat den.

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
- Bevara needs_analysis från tidigare iteration om varken byggnadstyp eller huvudsaklig användning ändrats. Uppdatera `from_user` om användaren tillfört ny direkt-info; uppdatera `inferred` bara om någon förutsättning faktiskt skiftat.
- Rensa `would_clarify` vid varje iteration: frågorna visas för användaren i chatten som en inbjudan att svara innan baslinjen räknas. Ta bort varje fråga som korrigeringen nu besvarar (svaret går in i `from_user`). Behåll bara det som fortfarande är obesvarat, viktigast först, max 3.
- "Svar på Aidas frågor" är det användaren svarat i tidigare rundor, med frågan före. "Aidas frågor" före en korrigering är de frågor Aida senast ställde, och korrigeringen kan vara svaret på dem. Varje svar är uppgifter om projektet: för in dem i `from_user`, och ställ aldrig samma fråga igen, inte heller med andra ord. "Vet inte" är också ett svar; gör då ett antagande och skriv det i `assumptions`.
- "Komponenter som redan identifierats" är det en tidigare runda hittade. Varje komponent i den listan ska finnas med i `components`, med samma namn och mängd om inget ändrats (en mängd du härlett ur BTA eller antal våningar räknar du om när de ändras), om inte användaren uttryckligen tagit bort den eller sagt att den inte ingår. Två komponenter som är samma sak får slås ihop (sprinklerrör och sprinklerhuvuden till ett sprinklersystem), men ingen får falla bort i tysthet. Lägg sedan till det svaret tillför.
"""


# Second step of intake: the per-component usage_context, written in parallel
# chunks once the component list is known. Until 2026-10-12 one call wrote both,
# and on large projects the answer outgrew max_tokens: Patrik's gruppboende
# (about 40 components) was cut off after 16 000 tokens and 172 s, the repair
# took another 81 s, and in production that ran into Vercel's 300 s as
# "Analysen tog för lång tid". Without usage_context the list took 130 s.
USAGE_SYSTEM_PROMPT = """Du är Aida — en byggnadsexpert som hjälper förvaltare och byggledare att hitta renoveringslösningar med kraftigt minskad klimatpåverkan, utan att ge avkall på praktiska behov.

Projektets komponenter är redan identifierade. Din uppgift i detta steg är att skriva `usage_context` för var och en av komponenterna du får: de brukar- och miljökrav som styr vilka material som är lämpliga.

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

KONSISTENS: Per-komponent `usage_context` SKA följa av projekt-level `needs_analysis.inferred`. Om needs_analysis säger "lågstadieskola, hög elevtrafik, blöt sand vintertid", får ingen komponents usage_context säga något som motsäger det. needs_analysis är redan skriven och står i underlaget: härled varje komponent ur den.

Svara ENBART med ett JSON-objekt där varje nyckel är ett komponent-id du fått och värdet är dess usage_context, till exempel {"c1": "...", "c2": "..."}. Ta med alla id:n du fått och inga andra. Svara på svenska."""


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


def canonicalize_categories(data: dict) -> dict:
    """Rewrite each component's category to its canonical key (fonster -> fönster).

    Runs before anything reads the field, so the stored analysis carries the
    spelling every exact lookup downstream expects (the facade guard above,
    the EPD category sets), not only resolve_category.
    """
    for component in data.get("components") or []:
        if component.get("category"):
            component["category"] = canonical_category(component["category"])
    return data


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


_YEAR = r"(?:ca\.?\s*|cirka\s*|omkring\s*|år\s+)?(1[0-9]{3}|20[0-9]{2})\b"
_CONSTRUCTION_YEAR_RES = (
    # "byggår 1974", "byggnadsåret är 1974", "Byggår: 1974"
    re.compile(r"\bbygg(?:nads)?år(?:et)?\s*(?:är\s+|var\s+|:\s*|=\s*)?" + _YEAR, re.I),
    # "byggd 1974", "byggdes år 1974", "uppförd 1962", "byggt i 1974"
    re.compile(r"\b(?:byggd(?:es)?|byggt|uppförd(?:es)?|uppfört)\s+(?:i\s+|på\s+)?" + _YEAR, re.I),
)


def construction_year_from_text(text: str) -> int | None:
    """The building year the user wrote, read with a pattern, or None.

    The model is asked for the year, but Sara's "byggår 1974." came back as an
    unknown year in October 2026, and that sentence is not a judgement call. So
    an explicit mention is read here too and fills in when the model left the
    field empty. Only explicit forms count: "plastgolv från 2003" is the year of
    a floor, not of the building.

    "byggår" beats "byggdes": the second can be about any part ("huset
    byggdes 1920, garaget byggdes 2015", "tillbyggnad uppförd 2015, huset
    byggt 1960"), and the pattern cannot tell the building from an extension
    or a correction, so "byggdes" mentions that disagree give None: a wrong
    year is worse than "inte angivet". Among "byggår" mentions the last wins,
    since a correction is written after what it corrects. Aida's own questions
    in the intake log ("Aida: Stämmer byggår 1962?") are not the user's answer
    and are skipped.
    """
    text = "\n".join(line for line in (text or "").split("\n")
                     if not line.lstrip().startswith("Aida:"))
    explicit, built = (
        [construction_year(int(m.group(1))) for m in pattern.finditer(text)]
        for pattern in _CONSTRUCTION_YEAR_RES
    )
    explicit = [y for y in explicit if y]
    built = [y for y in built if y]
    if explicit:
        return explicit[-1]
    return built[0] if built and len(set(built)) == 1 else None


def _finish(data: dict, description: str) -> dict:
    """The list as the rest of Aida expects it, before usage_context is added.

    usage_context is dropped if the model wrote one anyway: it is the second
    step's, and the facade guard below writes its note into the same field."""
    for component in data.get("components") or []:
        if isinstance(component, dict):
            component.pop("usage_context", None)
    # Not asked of the model: copying the input back cost a thousand tokens of
    # answer per round on the gruppboende. The browser replaces it with the
    # user's own words (intake_log); this is the fallback for other callers.
    data.setdefault("description", description)
    # The model's year first: after a correction ("byggåret är 1980, inte 1974")
    # it has the new one, and the text still holds the old.
    data["construction_year"] = (construction_year(data.get("construction_year"))
                                 or construction_year_from_text(description))
    return fix_envelope_quantities(canonicalize_categories(data))


def _extract(client, description: str, blocks: list[dict], started_at: float) -> dict:
    """The project and its component list, without usage_context.

    Both testers hit a hard crash here in June 2026, because the response was
    parsed with a bare json.loads: an off-format answer surfaced in the chat as
    "Fel: Expecting value: line 1 column 4 (char 3)". Parsing now goes through
    extract_json_object, and a first failure buys one repair round-trip before
    we give up. The repair round-trip reuses `messages` and `system`, so it
    sees the same files.
    """
    from aida.attachments import attach_to_messages, with_rule

    system = with_rule(SYSTEM_PROMPT, blocks)
    messages: list[dict] = attach_to_messages([{"role": "user", "content": description}], blocks)
    response = _call_intake(client, messages, system=system)
    text = extract_text(response)

    try:
        return _finish(extract_json_object(text, what="projektbeskrivningen"), description)
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
        return _finish(extract_json_object(retry_text, what="projektbeskrivningen"), description)
    except ModelOutputError as second:
        logger.error(
            "Intake parse failed after repair (%s). stop_reason=%s raw=%r",
            second, getattr(retry, "stop_reason", None), retry_text[:2000],
        )
        raise


# Twelve components per call, up to four calls at once: the gruppboende's 40
# become four chunks of ten that each answer in well under a minute.
USAGE_CHUNK_SIZE = 12
USAGE_MAX_PARALLEL = 4
USAGE_MAX_TOKENS = 8000
# The list is the correctness step and keeps high effort; describing use for a
# known component is shorter work.
USAGE_EFFORT = EFFORT_MEDIUM
# Below this many seconds left there is no point starting the second step:
# the list is worth more to the user than a timeout.
USAGE_MIN_BUDGET = 15


def _usage_message(data: dict, description: str, chunk: list[dict]) -> str:
    needs = json.dumps(data.get("needs_analysis") or {}, ensure_ascii=False)
    lines = [
        f"Byggnadstyp: {data.get('building_type', '')}, {data.get('area_bta', '')} m² BTA, "
        f"{data.get('storeys') or 1} våning(ar).",
        "",
        "Användarens beskrivning:",
        description,
        "",
        f"Behovsanalys (needs_analysis): {needs}",
        "",
        "Komponenter att skriva usage_context för:",
    ]
    lines += [
        f"- {c['id']}: {c.get('name', '')} ({c.get('quantity', '')} {c.get('unit', '')}, "
        f"kategori {c.get('category', '')})"
        for c in chunk
    ]
    return "\n".join(lines)


def _usage_chunk(data: dict, description: str, chunk: list[dict], started_at: float) -> dict:
    """usage_context per component id for one chunk; {} when the call fails.

    A failed chunk leaves its components without usage_context rather than
    failing the intake: the list is what the user is waiting for, and every
    step downstream already handles an empty usage_context.

    The budget is read when the chunk starts, not when the step does: past
    USAGE_MAX_PARALLEL chunks run in a second wave, which must not get the
    whole budget again."""
    timeout = remaining_budget(started_at)
    if timeout < USAGE_MIN_BUDGET:
        logger.warning("No time left for a usage_context chunk (%.0fs)", timeout)
        return {}
    try:
        response = call_model(
            get_client(),
            model=DEFAULT_MODEL,
            timeout=timeout,
            max_tokens=USAGE_MAX_TOKENS,
            effort=USAGE_EFFORT,
            system=USAGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _usage_message(data, description, chunk)}],
        )
        found = extract_json_object(extract_text(response), what="användningsbeskrivningarna")
    except Exception:
        logger.exception("usage_context chunk failed (%d components)", len(chunk))
        return {}
    return {k: v.strip() for k, v in found.items() if isinstance(v, str) and v.strip()}


def add_usage_context(data: dict, description: str, started_at: float) -> dict:
    """Write usage_context for every component, in parallel chunks.

    Not run when intake asks a clarifying question: only the component names
    are shown then, and the next round writes the list again."""
    components = [c for c in data.get("components") or [] if isinstance(c, dict) and c.get("id")]
    if not components:
        return data
    budget = remaining_budget(started_at)
    if budget < USAGE_MIN_BUDGET:
        logger.warning("No time left for usage_context (%.0fs); %d components without it",
                       budget, len(components))
        return data

    n_chunks = -(-len(components) // USAGE_CHUNK_SIZE)
    size = -(-len(components) // n_chunks)  # balanced: 40 -> 10, 10, 10, 10
    chunks = [components[i:i + size] for i in range(0, len(components), size)]

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(USAGE_MAX_PARALLEL, len(chunks))) as ex:
        results = list(ex.map(lambda chunk: _usage_chunk(data, description, chunk, started_at), chunks))

    written: dict[str, str] = {}
    for found in results:
        written.update(found)
    for component in components:
        text = written.get(component["id"])
        if text:
            # The facade guard's note stays, after the description of use.
            note = (component.get("usage_context") or "").strip()
            component["usage_context"] = f"{text} {note}".strip()
    missing = [c["id"] for c in components if c["id"] not in written]
    if missing:
        logger.warning("usage_context missing for %d of %d components: %s",
                       len(missing), len(components), ", ".join(missing))
    return data


def run_intake(description: str, attachments: list[dict] | None = None) -> dict:
    """Extract project parameters from a natural language description.

    Two steps since 2026-10-12: the project and its component list in one call,
    then usage_context per component in parallel chunks (see
    USAGE_SYSTEM_PROMPT for why).

    `attachments` are content blocks from aida.attachments.load_blocks, so a
    förfrågningsunderlag can be the project description (§14.6). Only the first
    step reads them; the second works from the needs_analysis the first wrote
    from them, so a long PDF is not paid for once per chunk.
    """
    client = get_client()
    started_at = time.monotonic()
    data = _extract(client, description, attachments or [], started_at)
    if data.get("clarification_needed"):
        return data
    return add_usage_context(data, description, started_at)


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
