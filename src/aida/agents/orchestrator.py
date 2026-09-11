"""Orchestrator: intent router + advisory branch.

Increment 1 of the orchestration redesign (docs/orchestration-redesign.md).

Today the flow is steered by a frontend state-machine (sendMessage in app.py):
in `idle`, ANY text falls through to runIntake, so a question gets parsed as a
project description and the intake LLM crashes on non-JSON
("Expecting value: line 1 column 1"). There is no "question" branch.

This module adds the missing front door, server-side:
  - classify_intent: is this message a new project, an advisory question, or a
    flow action the existing machinery already handles?
  - answer_advisory: answer questions WITHOUT mutating analysis state, grounded
    in real EPD data via a lookup tool so numbers have a source.

Strangler scope: the frontend calls /api/route first. Advisory questions are
answered here and stop. Everything else falls through to the existing flow
untouched. Later increments pull more intents (correct/select/rerun) into a real
server-side orchestrator that owns AnalysisState.
"""

from __future__ import annotations

import logging

from aida import knowledge
from aida.agents.alternatives import _format_epd_list, _load_epd_alternatives
from aida.agents.chat_agent import _format_state, _sanitize_history
from aida.api_client import DEFAULT_MODEL, extract_text, get_client
from aida.data.climate_data import normalize_component_name

logger = logging.getLogger(__name__)

# Intent classification runs on every routed message (hot path), and is a simple
# discriminated choice — Haiku 4.5 handles it accurately (verified against the
# crash inputs + mutation-imperative probe) at lower latency/cost than Sonnet.
# Advisory answering needs grounded synthesis, so it stays on the default (Sonnet).
CLASSIFIER_MODEL = "anthropic/claude-haiku-4.5"
ADVISORY_MODEL = DEFAULT_MODEL

# Intent taxonomy for increment 1. Deliberately coarse: we only need to peel off
# the two intents the current flow handles badly (new_project crashes on
# questions; advisory has no branch at all). Everything else is flow_action,
# which the existing sendMessage switch + chat_agent already handle correctly.
INTENT_NEW_PROJECT = "new_project"
INTENT_ADVISORY = "advisory_question"
INTENT_FLOW_ACTION = "flow_action"
_VALID_INTENTS = {INTENT_NEW_PROJECT, INTENT_ADVISORY, INTENT_FLOW_ACTION}

_CLASSIFY_TOOL = [{
    "name": "classify",
    "description": "Klassificera användarens meddelande i exakt en intent.",
    "input_schema": {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": [INTENT_NEW_PROJECT, INTENT_ADVISORY, INTENT_FLOW_ACTION],
                "description": (
                    "new_project: en projektbeskrivning att analysera (byggnadstyp, "
                    "yta, åtgärder), eller en uttrycklig begäran att börja om. "
                    "advisory_question: en fråga som söker kunskap eller råd "
                    "(material, klimat, metod, byggriktlinjerna, hur verktyget fungerar) och som kan "
                    "besvaras utan att ändra projektets state. "
                    "flow_action: en korrigering, ett val, en borttagning, en "
                    "omkörningsbegäran eller ett 'gå vidare'-kommando som rör "
                    "det pågående projektet."
                ),
            },
            "reason": {"type": "string", "description": "Kort motivering (en mening)."},
        },
        "required": ["intent"],
    },
}]

_CLASSIFY_SYSTEM = """Du är intent-routern för Aida, ett klimatkalkylverktyg för ombyggnationer.

Din enda uppgift: klassificera användarens meddelande i exakt en intent genom att anropa verktyget `classify`. Svara aldrig med fritext.

Vägledning:
- Om INGET projekt finns ännu och meddelandet beskriver ett bygge ("renovera matsal 100 m2, nya golv och fönster") → new_project.
- Om INGET projekt finns och meddelandet är en fråga ("vilket golv har lägst klimatpåverkan?", "vad betyder GWP?", "vad säger byggriktlinjerna om plastmatta?") → advisory_question. Den ska INTE tolkas som en projektbeskrivning.
- Om ett projekt finns: en fråga om resonemang, material eller metod ("varför är betong sämre?", "vilket av alternativen är bäst för fukt?") → advisory_question.
- Om ett projekt finns: en instruktion som ändrar projektet ("ta bort fönstren", "välj Tarkett", "byt golvet till parkett", "räkna om", "kör vidare", "tänk bredare på golv") → flow_action.
- Tveksamt mellan advisory och flow_action: om meddelandet ber Aida GÖRA något med projektet → flow_action. Om det ber om KUNSKAP → advisory_question.
"""

_ADVISORY_SYSTEM = """Du är Aida, en byggnadsexpert som hjälper förvaltare och byggledare att hitta renoveringslösningar med kraftigt minskad klimatpåverkan utan att ge avkall på praktiska behov.

Användaren ställer en RÅDGIVNINGSFRÅGA. Du ska svara, inte ändra något i projektet. Du har inga verktyg som muterar state.

PRINCIPER:
- Svara på svenska, kortfattat och konkret.
- Varje siffra ska ha en källa. Använd verktyget `lookup_materials` för att hämta verkliga EPD-värden (kg CO2e per enhet) när användaren frågar om ett materialslag (golv, innervägg, fönster, dörr, isolering, tak, belysning, ventilation, sanitet m.fl.). Fabricera aldrig siffror.
- Frågor om Karlstads byggriktlinjer eller om hur Aida räknar: sök med `search_knowledge` innan du svarar. Ange dokument, utgåva och avsnitt, till exempel "Riktlinje Bygg, utgåva 8, Plastmattor". Citera bara hela meningar, ordagrant ur utdragen du fått och inom citattecken, och tillskriv aldrig riktlinjerna ett krav som inte står där.
- Om frågan rör ett pågående projekt: använd projektets state (komponenter, baslinje, alternativ, val) i ditt svar.
- Klimatmåttet är GWP-fossil för skedena A1-A3 (produktskedet), samma som i Boverkets klimatdatabas och i Aidas beräkningar.
- Om du jämför material: nämn att lägre kg CO2e/enhet är bättre, men att praktiska krav (fukt, slitage, ljud, tillgänglighet) kan utesluta det klimatbästa.
- Om frågan inte går att besvara med tillgänglig data: säg det ärligt och föreslå hur användaren kan komma vidare (t.ex. starta ett projekt så att Aida kan räkna baslinje och alternativ).
- Avsluta gärna med en kort öppning till nästa steg ("Vill du att jag startar en analys för det här?") när det är relevant, men ställ inga onödiga frågor.
"""

_LOOKUP_TOOL = [{
    "name": "lookup_materials",
    "description": (
        "Hämta verkliga EPD-alternativ (produkt + kg CO2e per enhet) för ett "
        "materialslag, så att rådgivningen blir grundad i data. Ange materialslaget "
        "i fritext (t.ex. 'golv', 'innervägg', 'fönster')."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": "Materialslag i fritext, t.ex. 'golv', 'innervägg', 'fönster', 'dörr'.",
            },
        },
        "required": ["category"],
    },
}]

# A guideline question can take two searches and a material lookup before the
# answer, so four turns ran out on questions that were going fine.
_MAX_ADVISORY_TURNS = 6
_ADVISORY_TOOLS = _LOOKUP_TOOL + [knowledge.SEARCH_TOOL]


def classify_intent(
    message: str,
    has_project: bool,
    history: list[dict] | None = None,
    attachment_note: str = "",
) -> dict:
    """Classify a user message into one intent. Returns {intent, reason}.

    Fails safe to flow_action: if the classifier errors or returns an unknown
    value, we defer to the existing flow rather than risk routing a real project
    edit into the advisory dead-end.

    `attachment_note` names attached files (§14.4). The classifier gets names,
    never the files: it only has to see that "vad står i ritningen?" is a
    question about something that exists, and Haiku reading a 40-page PDF on
    every message would cost more than the answer.
    """
    client = get_client()
    clean = _sanitize_history(history or [])
    state_note = (
        "Ett projekt finns redan i sessionen."
        if has_project
        else "Inget projekt finns ännu i sessionen."
    )
    if attachment_note:
        state_note += " " + attachment_note + "."
    # Same boundary handling as run_chat_agent. _sanitize_history guarantees
    # internal alternation but not the edges, and Anthropic rejects a leading
    # assistant turn or two user turns in a row. This was survivable while the
    # history died on reload; from increment 4 it persists, so a single failed
    # chat turn would otherwise leave a permanently unroutable history.
    recent = list(clean[-6:])
    if recent and recent[0]["role"] == "assistant":
        recent = recent[1:]
    if recent and recent[-1]["role"] == "user":
        recent = recent[:-1]
    messages = recent + [{"role": "user", "content": message}]

    try:
        response = client.messages.create(
            model=CLASSIFIER_MODEL,
            max_tokens=300,
            system=_CLASSIFY_SYSTEM + "\n\n" + state_note,
            tools=_CLASSIFY_TOOL,
            tool_choice={"type": "tool", "name": "classify"},
            messages=messages,
        )
    except Exception as e:
        logger.exception("classify_intent failed")
        return {"intent": INTENT_FLOW_ACTION, "reason": f"classifier-fel: {e}", "error": True}

    for block in response.content or []:
        if getattr(block, "type", None) == "tool_use" and block.name == "classify":
            intent = (block.input or {}).get("intent")
            if intent in _VALID_INTENTS:
                return {"intent": intent, "reason": (block.input or {}).get("reason", "")}
            break

    logger.warning("classify_intent: no valid intent in response, defaulting to flow_action")
    return {"intent": INTENT_FLOW_ACTION, "reason": "ingen giltig intent", "error": True}


def _lookup_materials(category_query: str) -> str:
    """Resolve a free-text material name to a category and return formatted EPD data.

    On a miss, name the categories that actually have data so the advisory model
    can retry with a recognized term instead of dead-ending (or fabricating).
    """
    epd_data = _load_epd_alternatives()
    available = ", ".join(sorted(epd_data.keys())) or "(inga)"
    key = normalize_component_name(category_query or "")
    if not key:
        return (
            f"Hittade inget materialslag som matchar '{category_query}'. "
            f"Tillgängliga kategorier: {available}."
        )
    epds = epd_data.get(key, [])
    if not epds:
        return (
            f"Inga EPD-alternativ finns förkategoriserade för '{category_query}' "
            f"(kategori: {key}). Tillgängliga kategorier: {available}."
        )
    return f"EPD-alternativ för kategori '{key}':\n" + _format_epd_list(epds)


def answer_advisory(
    message: str,
    history: list[dict] | None = None,
    project: dict | None = None,
    baseline: dict | None = None,
    alternatives: dict | None = None,
    selections: dict | None = None,
    attachments: list[dict] | None = None,
) -> dict:
    """Answer an advisory question without mutating state. Returns {reply, tool_calls}.

    `attachments` are content blocks from aida.attachments.load_blocks. This is
    the branch "vad står det om ventilationen?" lands in, so it is the one that
    most needs the file.
    """
    from aida.attachments import attach_to_messages, with_rule

    blocks = attachments or []
    client = get_client()
    clean = _sanitize_history(history or [])

    system_prompt = with_rule(_ADVISORY_SYSTEM, blocks)
    if project:
        state_block = _format_state(project, baseline, alternatives, selections or {})
        system_prompt += "\n\nNUVARANDE PROJEKT-STATE:\n" + state_block

    # Same edge handling as classify_intent and run_chat_agent. With files the
    # first turn has to be the user's, because that is where they go, and the
    # tail must not collide with the message appended here.
    recent = list(clean[-8:])
    if recent and recent[0]["role"] == "assistant":
        recent = recent[1:]
    if recent and recent[-1]["role"] == "user":
        recent = recent[:-1]
    messages: list[dict] = attach_to_messages(
        recent + [{"role": "user", "content": message}], blocks)
    tool_calls: list[dict] = []

    # An exception here must NOT propagate: route() would re-raise, api_route would
    # 500, and the frontend's fail-safe fall-through would run runIntake on the
    # question — re-creating the exact idle crash this feature exists to kill. So
    # an API error returns a graceful advisory reply instead.
    try:
        for _ in range(_MAX_ADVISORY_TURNS):
            response = client.messages.create(
                model=ADVISORY_MODEL,
                max_tokens=1200,
                system=system_prompt,
                tools=_ADVISORY_TOOLS,
                messages=messages,
            )

            if response.stop_reason != "tool_use":
                return {"reply": (extract_text(response) or "").strip(), "tool_calls": tool_calls}

            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                if block.name == "lookup_materials":
                    category = (block.input or {}).get("category", "")
                    result_text = _lookup_materials(category)
                    tool_calls.append({"name": "lookup_materials", "input": block.input})
                elif block.name == "search_knowledge":
                    result_text = knowledge.run_search(block.input)
                    tool_calls.append({"name": "search_knowledge", "input": block.input})
                else:
                    result_text = f"Okänt verktyg: {block.name}"
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })
            messages.append({"role": "user", "content": tool_results})
    except Exception as e:
        logger.exception("answer_advisory failed")
        return {
            "reply": "Jag kunde inte hämta ett svar just nu. Försök igen om en stund.",
            "tool_calls": tool_calls,
            "error": str(e),
        }

    logger.warning("answer_advisory hit max turns")
    return {
        "reply": "Jag hann inte färdigt med svaret. Försök formulera om frågan.",
        "tool_calls": tool_calls,
    }


def route(
    message: str,
    history: list[dict] | None = None,
    project: dict | None = None,
    baseline: dict | None = None,
    alternatives: dict | None = None,
    selections: dict | None = None,
    attachments: list[dict] | None = None,
    attachment_note: str = "",
) -> dict:
    """Classify, and answer in the same call when the intent is advisory.

    The classifier gets `attachment_note` (file names), the advisory answer gets
    `attachments` (the files). See §14.4 for why they are split that way.

    Returns:
      {intent: 'advisory_question', reply: str, ...}  — frontend renders + stops
      {intent: 'new_project' | 'flow_action'}         — frontend falls through to existing flow
    """
    has_project = bool(project and project.get("components"))
    classification = classify_intent(message, has_project=has_project, history=history,
                                     attachment_note=attachment_note)
    intent = classification["intent"]

    if intent == INTENT_ADVISORY:
        answer = answer_advisory(
            message, history=history, project=project, baseline=baseline,
            alternatives=alternatives, selections=selections, attachments=attachments,
        )
        return {
            "intent": intent,
            "reply": answer["reply"],
            "reason": classification.get("reason", ""),
            "tool_calls": answer.get("tool_calls", []),
        }

    return {"intent": intent, "reason": classification.get("reason", "")}
