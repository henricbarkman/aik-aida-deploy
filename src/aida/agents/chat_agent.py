"""Chat agent with tool-use: surgical state mutations via conversation.

Tools that edit project components and selections, plus scoped rerun requests
the frontend executes.

add_component was missing until 2026-08-20, and the absence did not surface as
an error. Asked to add a component mid-analysis, the model improvised a
workaround that does not exist ("do it in the project view"), so the user's only
real option was to start the whole analysis over. Worth remembering when adding
a capability here: a tool the agent lacks becomes advice the agent invents.
"""

from __future__ import annotations

import copy
import logging

from aida import knowledge
from aida import sheet as sheet_mod
from aida.api_client import DEFAULT_MODEL, extract_text, get_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Du är Aida, en byggnadsexpert som hjälper förvaltare och byggledare att hitta renoveringslösningar med kraftigt minskad klimatpåverkan utan att ge avkall på praktiska behov.

Du ser projektets nuvarande state (komponenter, baslinje, alternativ, val) och har verktyg för att korrigera state och trigga om-körningar baserat på användarens input.

VERKTYG:
- update_component: korrigera material, mängd, enhet eller kategori för en komponent ("det är linoleum, inte vinyl", "500 m² blev 700").
- add_component: lägg till en komponent som saknas ("vi ska byta innerdörrarna också"). Går att göra när som helst, även efter att baslinjen är beräknad och alternativ valda.
- select_alternative: välj ett alternativ för en komponent ("välj Tarkett iQ för golvet").
- remove_component: ta bort en komponent ("vi byter inte fönstren ändå").
- rerun_baseline: begär att baslinjen räknas om för specifika komponenter (component_ids=['c1']) eller hela analysen (component_ids=[]). Frontend kör själva omberäkningen.
- rerun_alternatives: begär att alternativen körs om, eventuellt med user_feedback som styrning ("fokusera på ljudmiljö", "bara svenska tillverkare").
- set_as_built: registrera vad som faktiskt installerades, i uppföljningsläget ("vi la iQ Granit, 100 kvadrat, fakturan gick på 41 000").
- bind_epd: bind en miljödeklaration till det installerade ("ja, det är iQ Granit SD"). Kräver ett epd_id ur kandidatlistan.

UPPFÖLJNING:
Uppföljningen handlar om vad som blev, inte vad som planerades. Registrera bara det användaren faktiskt uppger. Saknas mängd, pris eller produktnamn: fråga, fyll aldrig i själv. En rad utan bunden deklaration är ett giltigt utfall och redovisas som uppskattad, så det är alltid bättre att lämna den tom än att binda en deklaration du inte vet stämmer.

NÄR DU SKA ANVÄNDA VERKTYG:
- Använd verktyg när användaren ger en konkret korrigering, ett val eller en begäran som går att genomföra direkt.
- Använd INTE verktyg för rena frågor ("varför är betong sämre?"). Svara bara med text.
- Om användaren är tvetydig: fråga först, använd verktyg sen.

OBLIGATORISKT RERUN-MÖNSTER VID MATERIAL- ELLER KATEGORIBYTE:
När du anropar update_component och ändringen rör name, category eller unit (alltså inte ENBART quantity), ska du i SAMMA tur också anropa rerun_baseline och rerun_alternatives med component_ids=[id på komponenten]. Användaren ska aldrig behöva klicka en knapp eller säga "räkna om" för att få ut nya värdet. Skala-tricket (linjär skalning vid quantity-only) gäller bara mängd, inte material.

OBLIGATORISKT MÖNSTER NÄR EN KOMPONENT LÄGGS TILL:
En ny komponent har varken baslinje eller alternativ förrän de körts. Anropa därför i SAMMA tur add_component, sedan rerun_baseline och rerun_alternatives med component_ids=[nya id:t]. Aldrig tom lista: en full omkörning skulle räkna om allt annat i onödan och kräva en extra bekräftelse av användaren. Övriga komponenters val överlever en sådan scopad omkörning, så det finns ingen anledning att börja om.

Saknas mängd i det användaren sagt: fråga efter den först, anropa inte add_component med ett påhittat antal. Vet du enheten men inte antalet, fråga efter antalet.

Säg aldrig till användaren att hen ska lägga till komponenten någon annanstans, i projektvyn eller genom att börja om. Du kan göra det härifrån.

KONFIRMATION VID FULL OMKÖRNING:
Om användaren ber om "kör om hela analysen", "börja om" eller liknande som leder till rerun_baseline eller rerun_alternatives med tom component_ids: bekräfta först i text vad det innebär (alla nuvarande val och beräkningar görs om) och vänta på explicit ja innan du anropar verktyget.

UNDVIK SPAMMA RERUNS:
- Anropa rerun_X bara när användaren faktiskt ändrat något som påverkar värdet, eller explicit bett om en uppdatering.
- Anropa aldrig samma rerun_X med samma component_ids två gånger i samma tur. Systemet returnerar fel om du försöker, men du ska inte ens försöka.
- Vid en fråga om "varför ser alternativ X dyrare ut?": svara med resonemang från state, kör inte rerun_alternatives.

EFTER EN MUTERING ELLER BEGÄRD RERUN:
- Bekräfta kort vad som ändrades och vad du har begärt att räknas om.
- Om ENDAST mängd ändrades: klimatvärdena skalas automatiskt (linjärt). Säg det kort utan att begära rerun.
- Om material/kategori ändrades eller komponent togs bort: nämn att du har begärt rerun_baseline och rerun_alternatives för den komponenten. Frontend hanterar exekveringen och visar nya värden.
- Om det är ett val (select_alternative): nämn den nya totala besparingen om baslinje och alla val finns.

PRINCIPER:
- Priser avser installerat pris (material + arbete) i SEK exkl moms.
- Svara på svenska, kortfattat och konkret.
- Siffror hämtar du från state, fabricera aldrig.

BASLINJENS DATAKÄLLOR (viktigt — använd när användaren undrar över baseline-värdena):
Baslinjen bygger på två källor, och varje komponent visar vilken som använts:
- "Boverkets klimatdatabas": komponentens standardmaterial finns direkt i Boverket (t.ex.
  gipsskiva, betong, mineralull, stål). Mest precist.
- "Environdec EPD-typvärde": Boverket är organiserad efter materialtyp (~200 generiska
  produkter) och saknar vissa komponenttyper helt (golvbeläggning, sanitetsporslin, vitvaror,
  belysning). För dem använder vi istället ett kategori-aggregat: medianen av den övre
  (sämsta) halvan av Environdec-EPD:erna i kategorin. Övre halvan för att approximera ett
  konventionellt standardval utan klimathänsyn (EPD-databaser lutar mot klimatmedvetna
  tillverkare, så hela medianen hade underskattat).

Om en användare undrar varför ett golv inte har en Boverket-produkt: förklara att Boverket
saknar golv som kategori, så vi använder ett EPD-typvärde istället för att låna en orelaterad
Boverket-produkt. Det är design, inte bugg. Använd description-fältet i baseline-state — det
innehåller Aidas resonemang.

Om baslinjevärdet uppenbart inte matchar materialet användaren beskriver: be hen bekräfta
materialet och kör om baslinjen.
"""


TOOLS = [
    {
        "name": "update_component",
        "description": (
            "Uppdatera en komponents egenskaper (namn, mängd, enhet, kategori). "
            "Använd när användaren korrigerar ett material eller en mängd. "
            "Inkludera bara de fält som faktiskt ska ändras."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "component_id": {
                    "type": "string",
                    "description": "ID från projektets komponentlista (c1, c2, etc)",
                },
                "name": {"type": "string"},
                "quantity": {"type": "number"},
                "unit": {"type": "string", "enum": ["m2", "st", "lm"]},
                "category": {
                    "type": "string",
                    "enum": [
                        "golv", "innervägg", "yttervägg", "betongvägg", "tak",
                        "fönster", "dörr", "isolering", "belysning", "ventilation",
                        "hiss", "kylanläggning", "sanitet", "vitvaror", "storköksutrustning",
                    ],
                },
            },
            "required": ["component_id"],
        },
    },
    {
        "name": "add_component",
        "description": (
            "Lägg till en ny komponent i projektet ('vi ska byta innerdörrarna också'). "
            "Fungerar när som helst, även efter att baslinjen är beräknad och alternativ "
            "valda: den nya komponenten får en egen baslinje och egna alternativ, och "
            "befintliga val påverkas inte. Be användaren om mängd om hen inte angett "
            "någon, hitta aldrig på ett antal."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Komponentens namn som användaren beskriver den, t.ex. 'Innerdörrar' eller 'Linoleumgolv'.",
                },
                "quantity": {"type": "number"},
                "unit": {"type": "string", "enum": ["m2", "st", "lm"]},
                "category": {
                    "type": "string",
                    "enum": [
                        "golv", "innervägg", "yttervägg", "betongvägg", "tak",
                        "fönster", "dörr", "isolering", "belysning", "ventilation",
                        "hiss", "kylanläggning", "sanitet", "vitvaror", "storköksutrustning",
                    ],
                },
                "quantity_source": {
                    "type": "string",
                    "enum": ["user_specified", "estimated"],
                    "description": (
                        "'user_specified' bara när användaren själv angav antalet. "
                        "Uppskattade du det, sätt 'estimated' så visas det som en "
                        "uppskattning i tabellen."
                    ),
                },
                "usage_context": {
                    "type": "string",
                    "description": (
                        "Valfritt. Brukare, miljö och funktionella krav om det framgår "
                        "av samtalet. Styr vilka alternativ som är lämpliga."
                    ),
                },
            },
            "required": ["name", "quantity", "unit"],
        },
    },
    {
        "name": "select_alternative",
        "description": (
            "Välj ett av de befintliga alternativen för en komponent. "
            "Matcha fuzzy på produktnamn mot alternatives-listan i state. "
            "Om användaren vill välja baslinjen istället, använd alternative_name='baslinje'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "component_id": {"type": "string"},
                "alternative_name": {
                    "type": "string",
                    "description": "Produktnamn eller del av namn (fuzzy-matchas), eller 'baslinje' för baslinjevalet.",
                },
            },
            "required": ["component_id", "alternative_name"],
        },
    },
    {
        "name": "set_as_built",
        "description": (
            "Registrera vad som faktiskt installerades för en komponent, i "
            "uppföljningsläget ('vi la iQ Granit, 100 kvadrat, fakturan gick på "
            "41 000'). Skicka bara de fält användaren har angett: uppgifterna "
            "fylls på över tid och en tom mängd raderar inte den som redan finns. "
            "Hitta aldrig på en mängd, ett pris eller ett produktnamn."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "component_id": {"type": "string"},
                "installed_name": {"type": "string", "description": "Produkten som installerades"},
                "quantity": {"type": "number"},
                "unit": {"type": "string", "enum": ["m2", "st", "lm", "kg"]},
                "actual_cost": {"type": "number", "description": "Verklig kostnad i SEK exkl moms"},
                "cost_source": {"type": "string", "description": "Var priset kommer ifrån, t.ex. fakturanummer"},
                "transport_km": {"type": "number"},
                "match_quality": {
                    "type": "string",
                    "enum": ["product", "generic", "typvarde", "reuse", "none"],
                    "description": "Sätt 'reuse' när komponenten är återbrukad.",
                },
            },
            "required": ["component_id"],
        },
    },
    {
        "name": "bind_epd",
        "description": (
            "Bind en miljödeklaration till det som installerades ('ja, det är "
            "iQ Granit SD'). Samma sak som att klicka en rad i kandidatlistan. "
            "Ange epd_id från kandidatlistan. Skicka epd_id=null för att ta bort "
            "bindningen. Gissa aldrig ett id: har du inga kandidater, be "
            "användaren söka fram produkten i arket först."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "component_id": {"type": "string"},
                "epd_id": {"type": ["string", "null"], "description": "uuid från kandidatlistan, eller null"},
                "match_quality": {
                    "type": "string",
                    "enum": ["product", "generic", "typvarde", "reuse", "none"],
                    "description": "product när deklarationen gäller just den produkten, generic när den är leverantörens allmänna.",
                },
            },
            "required": ["component_id"],
        },
    },
    {
        "name": "remove_component",
        "description": "Ta bort en komponent helt från projektet. Baslinje, alternativ och val för komponenten rensas också.",
        "input_schema": {
            "type": "object",
            "properties": {
                "component_id": {"type": "string"},
            },
            "required": ["component_id"],
        },
    },
    {
        "name": "rerun_baseline",
        "description": (
            "Begär att baslinjen räknas om. Använd vid material- eller kategori-ändring, "
            "eller om användaren ber om en uppdaterad baslinje. Ange component_ids=['c1','c3'] "
            "för partiell omkörning av specifika komponenter, eller tom lista [] för komplett "
            "omkörning av hela baslinjen. Vid komplett omkörning: bekräfta med användaren först "
            "i ett tidigare textmeddelande innan du anropar verktyget."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "component_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Lista med komponent-id (c1, c2, ...). Tom lista = omberäkna alla.",
                },
                "reason": {
                    "type": "string",
                    "description": "Kort förklaring till varför baslinjen ska räknas om (visas för användaren).",
                },
            },
            "required": ["component_ids", "reason"],
        },
    },
    {
        "name": "rerun_alternatives",
        "description": (
            "Begär att alternativen körs om. Använd vid material/kategori-ändring eller när "
            "användaren vill se nya förslag (eventuellt med ett särskilt önskemål, t.ex. "
            "'fokusera på ljudmiljö' eller 'bara svenska tillverkare'). Partiell via component_ids "
            "eller komplett via tom lista. Vid komplett omkörning: bekräfta med användaren först."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "component_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Lista med komponent-id. Tom lista = kör om alla.",
                },
                "user_feedback": {
                    "type": "string",
                    "description": "Frivilligt önskemål som ges som extra instruktion till alternatives-LLM:en.",
                },
                "reason": {
                    "type": "string",
                    "description": "Kort förklaring till varför alternativen ska räknas om.",
                },
            },
            "required": ["component_ids", "reason"],
        },
    },
]


def _format_state(project, baseline, alternatives, selections) -> str:
    """Compact, LLM-readable snapshot of current state."""
    lines = []
    if project:
        lines.append(f"PROJEKT: {project.get('building_type', '?')}, {project.get('area_bta', '?')} m² BTA")
        if project.get("name"):
            lines.append(f"Namn: {project['name']}")
        lines.append("KOMPONENTER:")
        for c in project.get("components", []):
            lines.append(
                f"  {c.get('id')}: {c.get('name')} — {c.get('quantity')} {c.get('unit')} [{c.get('category', '?')}]"
            )
    else:
        lines.append("PROJEKT: (inget projekt än)")

    if baseline and baseline.get("components"):
        total_co2 = sum(c.get("co2e_kg", 0) for c in baseline["components"])
        total_cost = sum(c.get("cost_sek", 0) for c in baseline["components"])
        lines.append(f"\nBASLINJE: {round(total_co2):,} kg CO₂e, {round(total_cost):,} SEK totalt")
        for c in baseline["components"]:
            lines.append(
                f"  {c.get('component_id')}: {c.get('component_name')} — "
                f"{round(c.get('co2e_kg', 0))} kg CO₂e, {round(c.get('cost_sek', 0))} SEK"
            )

    if alternatives and alternatives.get("components"):
        lines.append("\nALTERNATIV:")
        for c in alternatives["components"]:
            alts = c.get("alternatives", [])
            lines.append(f"  {c.get('component_id')}: {c.get('component_name')} — {len(alts)} alternativ")
            for a in alts[:5]:
                lines.append(
                    f"    • {a.get('name')}: {round(a.get('co2e_kg', 0))} kg CO₂e, {round(a.get('cost_sek', 0))} SEK"
                )
            if len(alts) > 5:
                lines.append(f"    ... +{len(alts) - 5} till")

    if selections:
        sel_entries = [(cid, s) for cid, s in selections.items() if s]
        if sel_entries:
            lines.append("\nVAL:")
            for cid, s in sel_entries:
                sel = s.get("selected_alternative", {})
                lines.append(
                    f"  {cid}: {s.get('name')} → {sel.get('name')} "
                    f"({round(sel.get('co2e_kg', 0))} kg, {round(sel.get('cost_sek', 0))} SEK)"
                )

    return "\n".join(lines)


from aida.mutations import (  # noqa: F401  (re-exported for existing importers)
    AS_BUILT_HANDLERS as _AS_BUILT_HANDLERS,
)
from aida.mutations import (  # noqa: F401  (re-exported for existing importers)
    HANDLERS as _HANDLERS,
)
from aida.mutations import (
    _already_requested,
    _apply_add_component,
    _apply_remove_component,
    _apply_rerun_alternatives,
    _apply_rerun_baseline,
    _apply_select_alternative,
    _apply_update_component,
    _find_component,
    _find_component_alternatives,
    _next_component_id,
    _scale_component_values,
    _validate_component_ids,
)
from aida.mutations import (
    build_state_updates as _build_state_updates,
)
from aida.mutations import (
    run_handler as _run_handler,
)


def _resolve_bind_input(inp, resolver):
    """Turn an `epd_id` from the model into the declaration itself.

    Resolution is a network call, so it cannot live in `mutations.py`, which is
    pure by design and has to stay movable behind /api/turn. It cannot live in
    the model's hands either: the GWP figure has to come from the register, not
    from what a language model remembers about a product. So the caller injects
    a resolver and the pure handler only ever sees a resolved declaration.

    Returns (input, error) with exactly one meaningful.
    """
    inp = dict(inp or {})
    if "epd" in inp:
        return inp, ""
    epd_id = inp.pop("epd_id", None)
    if not epd_id:
        # An explicit null is how the model unbinds, and that needs no lookup.
        inp["epd"] = None
        return inp, ""
    if resolver is None:
        return inp, "Kan inte slå upp deklarationer just nu."
    try:
        found = resolver(epd_id)
    except Exception:
        logger.exception("EPD-uppslag misslyckades för %s", epd_id)
        return inp, "Uppslaget mot EPD-registret misslyckades."
    if not found:
        return inp, (f"Hittade ingen deklaration med id {epd_id}. Be användaren "
                     f"söka fram produkten i arket i stället för att gissa.")
    inp["epd"] = found
    return inp, ""


def _sanitize_history(history: list) -> list[dict]:
    """Filter history to a shape Anthropic accepts: only {role, content} entries
    with role in {user, assistant} and content as a non-empty string. Collapses
    consecutive same-role turns by dropping the earlier one — we never want
    two user or two assistant turns in a row."""
    clean: list[dict] = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        content = entry.get("content")
        if role not in ("user", "assistant"):
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        if clean and clean[-1]["role"] == role:
            clean[-1] = {"role": role, "content": content}
        else:
            clean.append({"role": role, "content": content})
    return clean


def run_chat_agent(
    message: str,
    history: list[dict] | None = None,
    project: dict | None = None,
    baseline: dict | None = None,
    alternatives: dict | None = None,
    selections: dict | None = None,
    max_turns: int = 5,
    overrides: dict | None = None,
    as_built: dict | None = None,
    epd_resolver=None,
    attachments: list[dict] | None = None,
    sheet: dict | None = None,
) -> dict:
    """Run chat with tool-use loop.

    `attachments` are content blocks from aida.attachments.load_blocks. They go
    first in the first user turn and stay there through the tool loop, so the
    second to fifth call of a turn read them from the prompt cache instead of
    paying for the whole PDF again.

    `sheet` is the open sheet in Chatt (§13.7), None in the other modes. With it
    the agent also gets the block tools and search_knowledge, and a changed
    sheet comes back in state_updates.

    Returns dict with:
      - reply: str — assistant's final text reply
      - state_updates: dict — {project?, selections?} with changed objects
      - tool_calls: list — trace of tool invocations (for debug/UI)
    """
    from aida.attachments import attach_to_messages, with_rule

    blocks = attachments or []
    client = get_client()
    history = _sanitize_history(history or [])

    # Work on copies so we can diff at the end.
    project = copy.deepcopy(project) if project else None
    baseline = copy.deepcopy(baseline) if baseline else None
    alternatives = copy.deepcopy(alternatives) if alternatives else None
    selections = copy.deepcopy(selections) if selections else {}
    overrides = copy.deepcopy(overrides) if overrides else {}
    as_built = copy.deepcopy(as_built) if as_built else {}
    # Normalizing also copies, so the caller's sheet is never written to.
    sheet = sheet_mod.normalize_sheet(sheet) if sheet is not None else None
    tools = TOOLS + sheet_mod.SHEET_TOOLS + [knowledge.SEARCH_TOOL] if sheet is not None else TOOLS
    max_tokens = 1500
    if sheet is not None:
        # An answer on the sheet is several writes, and a table is long.
        max_turns, max_tokens = max_turns + sheet_mod.EXTRA_TURNS, 8000

    touched_bags: set[str] = set()
    tool_calls: list[dict] = []
    pending_actions: list[dict] = []

    state_block = _format_state(project, baseline, alternatives, selections)
    system_prompt = with_rule(SYSTEM_PROMPT, blocks) + "\n\nNUVARANDE STATE:\n" + state_block
    if sheet is not None:
        system_prompt += "\n\n" + sheet_mod.prompt(sheet)

    # Anthropic requires the first message to be 'user' and forbids two same-role
    # turns in a row. _sanitize_history guarantees internal alternation but not
    # the boundaries: drop a leading assistant turn (e.g. a stored UI greeting)
    # and a trailing user turn that would collide with the message we append.
    recent = list(history[-10:])
    if recent and recent[0]["role"] == "assistant":
        recent = recent[1:]
    if recent and recent[-1]["role"] == "user":
        recent = recent[:-1]
    messages: list[dict] = attach_to_messages(
        recent + [{"role": "user", "content": message}], blocks)

    for _ in range(max_turns):
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=max_tokens,
            system=system_prompt,
            tools=tools,
            messages=messages,
        )

        if sheet_mod.cut_off(response):
            logger.warning("chat_agent: reply cut off inside a tool call")
            sheet_mod.nudge(messages)
            continue
        if response.stop_reason != "tool_use":
            reply = extract_text(response) or ""
            if not reply.strip() and "sheet" in touched_bags:
                reply = sheet_mod.DONE
            return {
                "reply": reply.strip(),
                "state_updates": _build_state_updates(
                    touched_bags, project, baseline, alternatives, selections,
                    pending_actions, overrides=overrides, as_built=as_built, sheet=sheet,
                ),
                "tool_calls": tool_calls,
            }

        # Accumulate assistant turn (text + tool_use blocks)
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            # Both maps, not just HANDLERS. The as-built tools live in their own
            # registry because they take a different bag, and a dispatch that
            # only knew the first one would reject them here as unknown - before
            # ever reaching the seam that does know them.
            # A lookup, not a mutation, so it does not go through the seam.
            if block.name == "search_knowledge" and sheet is not None:
                result_text = knowledge.run_search(block.input)
                tool_calls.append({"name": block.name, "input": block.input, "ok": True})
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result_text})
                continue
            known = (block.name in _HANDLERS or block.name in _AS_BUILT_HANDLERS
                     or (sheet is not None and block.name in sheet_mod.SHEET_HANDLERS))
            if not known:
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": f"Okänt verktyg: {block.name}",
                    "is_error": True,
                })
                tool_calls.append({"name": block.name, "input": block.input, "ok": False})
                continue

            tool_input = block.input
            if block.name == "bind_epd":
                tool_input, resolve_error = _resolve_bind_input(tool_input, epd_resolver)
                if resolve_error:
                    tool_results.append({
                        "type": "tool_result", "tool_use_id": block.id,
                        "content": resolve_error, "is_error": True,
                    })
                    tool_calls.append({"name": block.name, "input": block.input,
                                       "ok": False, "result": resolve_error})
                    continue

            try:
                # Through the shared seam, not straight to the handler: the
                # override lifecycle lives there, and a material change made by
                # the chat has to drop a stale manual figure exactly as a cell
                # edit does.
                result_text, ok, handler_touched = _run_handler(
                    block.name, tool_input, project, baseline, alternatives,
                    selections, pending_actions, overrides=overrides,
                    as_built=as_built, sheet=sheet,
                )
            except Exception as e:
                logger.exception("Tool %s failed", block.name)
                result_text = f"Fel vid {block.name}: {e}"
                ok = False
                handler_touched = set()

            if ok:
                touched_bags |= handler_touched

            tool_calls.append({
                "name": block.name,
                "input": tool_input,
                "ok": ok,
                "result": result_text,
            })
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_text,
                **({"is_error": True} if not ok else {}),
            })

        messages.append({"role": "user", "content": tool_results})

    # Exhausted turns without a stop — force a final reply.
    logger.warning("chat_agent hit max_turns=%d", max_turns)
    return {
        "reply": (sheet_mod.UNFINISHED if "sheet" in touched_bags else
                  "Jag fastnade i en loop. Försök formulera om, eller använd knapparna för att köra om stegen."),
        "state_updates": _build_state_updates(
            touched_bags, project, baseline, alternatives, selections,
            pending_actions, overrides=overrides, as_built=as_built, sheet=sheet,
        ),
        "tool_calls": tool_calls,
    }
