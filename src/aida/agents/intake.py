"""Intake agent: extracts project parameters from natural language description."""

from __future__ import annotations

import json
import sys

from aida.api_client import (
    DEFAULT_MODEL,
    THINKING_LOW,
    extract_text,
    get_client,
    thinking_config,
)
from aida.models import Project

SYSTEM_PROMPT = """Du är AIda — en byggnadsexpert som hjälper förvaltare och byggledare att hitta renoveringslösningar med kraftigt minskad klimatpåverkan, utan att ge avkall på praktiska behov.

Din uppgift i detta steg är att extrahera projektinformation från en fri textbeskrivning av ett ombyggnadsprojekt.

Du ska identifiera:
1. Byggnadstyp (t.ex. skola, kontor, förskola, bostadshus)
2. Ungefärlig area i BTA (bruttoarea i kvadratmeter)
3. En lista av renoveringskomponenter (vad som ska bytas/renoveras)

Svara ALLTID med giltig JSON i detta format:
{
  "building_type": "string",
  "area_bta": number,
  "name": "projektnamn om nämnt",
  "description": "original beskrivning",
  "components": [
    {"id": "c1", "name": "komponentnamn", "quantity": number, "unit": "m2|st|lm", "category": "kategori"}
  ],
  "clarification_needed": null eller "fråga"
}

Regler:
- Komponent-id ska vara c1, c2, c3 etc
- Gissa rimlig quantity om den inte anges (baserat på area och byggnadstyp)
- Unit ska vara m2, st, eller lm (löpmeter)
- Category ska vara en av: golv, innervägg, yttervägg, betongvägg, tak, fönster, dörr, isolering, belysning, ventilation, hiss, kylanläggning, sanitet, vitvaror, storköksutrustning
- Om area inte anges, uppskatta baserat på byggnadstyp och komponenter
- Svara på svenska

FÖRTYDLIGANDEN:
- Fråga INTE om specifika materialval (t.ex. vilken typ av golv eller vilken isolering). Du ska fokusera på behov, inte material. Materialval är alternativ-stegets uppgift, och AIda kan komma med bättre förslag än användaren tänkt sig.
- Fråga däremot gärna om saker som påverkar analysen:
  * Byggnadsår (påverkar befintliga material och förutsättningar)
  * Särskilda krav (t.ex. Miljöbyggnad, tillgänglighet, ljudkrav, fuktproblem)
  * Om renoveringen är total eller partiell
  * Budget- eller tidplansramar om de inte nämnts
- Sätt clarification_needed till null om beskrivningen ger tillräckligt för en rimlig analys.
- Be om förtydligande (max 1-2 korta frågor) när svaret skulle bli väsentligt bättre med mer information. Inkludera då de komponenter du redan kunnat identifiera i components-arrayen.

TIDIGARE DISKUSSION:
- Om beskrivningen innehåller en sektion märkt "Tidigare diskussion i sessionen" eller "Korrigering från användaren": läs den noggrant.
- Fråga ALDRIG om något användaren redan besvarat tidigare OCH som inte ändras i korrigeringen. Återanvänd det tidigare svaret för sådana fält (t.ex. byggnadsår, certifieringskrav, omfattning).
- Om korrigeringen explicit ändrar ett tidigare besvarat fält, använd det NYA värdet från korrigeringen — inte det gamla från diskussionen.
- Sätt clarification_needed till null om tidigare svar (eventuellt ändrade av korrigeringen) fyller informationsbehovet, även om värdena inte upprepas i den nya korrigeringstexten.
- Bevara projektnamn från tidigare beskrivning om det inte uttryckligen ändras.
"""


def run_intake(description: str) -> dict:
    """Extract project parameters from a natural language description."""
    client = get_client()

    response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=2000 + THINKING_LOW,
        thinking=thinking_config(THINKING_LOW),
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": description}
        ],
    )

    text = extract_text(response)

    # Extract JSON from response (handle markdown code blocks)
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    return json.loads(text.strip())


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
