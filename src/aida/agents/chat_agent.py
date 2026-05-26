"""Chat agent with tool-use: surgical state mutations via conversation.

Phase 1 scope: three tools that edit project components and selections.
Heavier operations (rerun baseline/alternatives) stay on the button flow —
the agent suggests them in text when appropriate.
"""

from __future__ import annotations

import copy
import logging

from aida.api_client import DEFAULT_MODEL, extract_text, get_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Du är AIda, en byggnadsexpert som hjälper förvaltare och byggledare att hitta renoveringslösningar med kraftigt minskad klimatpåverkan utan att ge avkall på praktiska behov.

Du ser projektets nuvarande state (komponenter, baslinje, alternativ, val) och har verktyg för att korrigera state och trigga om-körningar baserat på användarens input.

VERKTYG:
- update_component: korrigera material, mängd, enhet eller kategori för en komponent ("det är linoleum, inte vinyl", "500 m² blev 700").
- select_alternative: välj ett alternativ för en komponent ("välj Tarkett iQ för golvet").
- remove_component: ta bort en komponent ("vi byter inte fönstren ändå").
- rerun_baseline: begär att baslinjen räknas om för specifika komponenter (component_ids=['c1']) eller hela analysen (component_ids=[]). Frontend kör själva omberäkningen.
- rerun_alternatives: begär att alternativen körs om, eventuellt med user_feedback som styrning ("fokusera på ljudmiljö", "bara svenska tillverkare").

NÄR DU SKA ANVÄNDA VERKTYG:
- Använd verktyg när användaren ger en konkret korrigering, ett val eller en begäran som går att genomföra direkt.
- Använd INTE verktyg för rena frågor ("varför är betong sämre?"). Svara bara med text.
- Om användaren är tvetydig: fråga först, använd verktyg sen.

OBLIGATORISKT RERUN-MÖNSTER VID MATERIAL- ELLER KATEGORIBYTE:
När du anropar update_component och ändringen rör name, category eller unit (alltså inte ENBART quantity), ska du i SAMMA tur också anropa rerun_baseline och rerun_alternatives med component_ids=[id på komponenten]. Användaren ska aldrig behöva klicka en knapp eller säga "räkna om" för att få ut nya värdet. Skala-tricket (linjär skalning vid quantity-only) gäller bara mängd, inte material.

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

BOVERKET-PROXY (viktigt — använd när användaren undrar över "konstiga" baseline-produkter):
Boverkets klimatdatabas är organiserad efter MATERIALTYP (~200 generiska byggprodukter), inte byggnadsfunktion. Den saknar därför kategorier för t.ex. "golvbeläggning", "sanitetsprodukter" och liknande. Baslinje-agenten matchar istället efter MATERIALSAMMANSÄTTNING:
- Vinylgolv (PVC) → "Takduk, PVC" är *samma basmaterial* (PVC), justerat för tjocklek/densitet
- Gipsskiva på innervägg → "Gipsskiva, standardskiva"
- Stålreglar → "Lättreglar av stål"

Om en användare frågar "varför står Takduk på golvet?" eller liknande: förklara att det är en *materialproxy*, inte en felmatchning — Boverket har inte golv som kategori, så vi använder PVC-baserade takduken som referens eftersom basmaterialet är detsamma. Det är design, inte bugg. Använd description-fältet i baseline-state om det finns där — det innehåller AIdas resonemang om proxy-valet.

Om proxyn är uppenbart fel (t.ex. trägolv mappat till PVC): be användaren bekräfta materialet och kör om baslinjen.
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


def _find_component(project, component_id):
    if not project:
        return None
    for c in project.get("components", []):
        if c.get("id") == component_id:
            return c
    return None


def _find_component_alternatives(alternatives, component_id):
    if not alternatives:
        return None
    for c in alternatives.get("components", []):
        if c.get("component_id") == component_id:
            return c
    return None


def _scale_component_values(cid: str, factor: float, baseline, alternatives, selections) -> set[str]:
    """Scale all cached CO₂e and cost values for a single component by `factor`.

    Returns a set naming which state bags were touched. Per-unit climate and price
    are linear in quantity under NollCO2; scaling avoids a full rerun when only
    quantity changes.
    """
    touched: set[str] = set()
    if factor == 1.0:
        return touched

    if baseline and baseline.get("components"):
        for c in baseline["components"]:
            if c.get("component_id") == cid:
                c["co2e_kg"] = c.get("co2e_kg", 0) * factor
                c["cost_sek"] = c.get("cost_sek", 0) * factor
                touched.add("baseline")

    if alternatives and alternatives.get("components"):
        for c in alternatives["components"]:
            if c.get("component_id") != cid:
                continue
            if "baseline_co2e_kg" in c:
                c["baseline_co2e_kg"] = c.get("baseline_co2e_kg", 0) * factor
            if "baseline_cost_sek" in c:
                c["baseline_cost_sek"] = c.get("baseline_cost_sek", 0) * factor
            for a in c.get("alternatives", []):
                a["co2e_kg"] = a.get("co2e_kg", 0) * factor
                a["cost_sek"] = a.get("cost_sek", 0) * factor
            touched.add("alternatives")

    if selections and cid in selections:
        sel = selections[cid]
        sel["baseline_co2e_kg"] = sel.get("baseline_co2e_kg", 0) * factor
        sel["baseline_cost_sek"] = sel.get("baseline_cost_sek", 0) * factor
        chosen = sel.get("selected_alternative") or {}
        if chosen:
            chosen["co2e_kg"] = chosen.get("co2e_kg", 0) * factor
            chosen["cost_sek"] = chosen.get("cost_sek", 0) * factor
        touched.add("selections")

    return touched


def _apply_update_component(inp, project, baseline, alternatives, selections, pending_actions):
    cid = inp.get("component_id")
    target = _find_component(project, cid)
    if not target:
        return f"Komponent {cid} finns inte i projektet.", False, set()

    changed = {}
    old_quantity = target.get("quantity")
    for key in ("name", "quantity", "unit", "category"):
        if key in inp and inp[key] is not None:
            target[key] = inp[key]
            changed[key] = inp[key]
    if not changed:
        return f"Ingen ändring angiven för {cid}.", False, set()

    # If material identity changed (name/category), prior usage_context may no
    # longer match — better to clear it than carry stale functional requirements
    # into the next alternatives search. Pure quantity/unit changes preserve it.
    # Chat-agent has no tool to set usage_context directly; rerun intake to
    # generate a fresh one when needed.
    if "name" in changed or "category" in changed:
        target["usage_context"] = ""

    touched: set[str] = {"project"}

    quantity_only = set(changed.keys()) == {"quantity"}
    if quantity_only:
        new_quantity = target["quantity"]
        try:
            old_q = float(old_quantity)
            new_q = float(new_quantity)
        except (TypeError, ValueError):
            old_q = new_q = 0.0
        if old_q > 0 and new_q > 0 and old_q != new_q:
            factor = new_q / old_q
            touched |= _scale_component_values(cid, factor, baseline, alternatives, selections)
            return (
                f"Uppdaterade {cid}: mängd {old_q:g} → {new_q:g} {target.get('unit', '')}. "
                f"Baslinje och alternativ skalade automatiskt — ingen omräkning behövs."
            ), True, touched
        if old_q == new_q:
            return f"Ingen ändring: {cid} är redan {new_q:g}.", False, set()

    return (
        f"Uppdaterade komponent {cid}: {changed}. "
        f"OBS: baslinjen och alternativen för denna komponent är nu inaktuella — kör om dem."
    ), True, touched


def _apply_remove_component(inp, project, baseline, alternatives, selections, pending_actions):
    cid = inp.get("component_id")
    target = _find_component(project, cid)
    if not target:
        return f"Komponent {cid} finns inte i projektet.", False, set()

    project["components"] = [c for c in project.get("components", []) if c.get("id") != cid]
    touched: set[str] = {"project"}

    if baseline and baseline.get("components"):
        before = len(baseline["components"])
        baseline["components"] = [c for c in baseline["components"] if c.get("component_id") != cid]
        if len(baseline["components"]) != before:
            touched.add("baseline")

    if alternatives and alternatives.get("components"):
        before = len(alternatives["components"])
        alternatives["components"] = [
            c for c in alternatives["components"] if c.get("component_id") != cid
        ]
        if len(alternatives["components"]) != before:
            touched.add("alternatives")

    if selections and cid in selections:
        del selections[cid]
        touched.add("selections")

    return f"Komponenten {cid} ({target.get('name')}) borttagen.", True, touched


def _apply_select_alternative(inp, project, baseline, alternatives, selections, pending_actions):
    cid = inp.get("component_id")
    alt_query = (inp.get("alternative_name") or "").strip().lower()
    comp_alts = _find_component_alternatives(alternatives, cid)
    if not comp_alts:
        return f"Inga alternativ finns för {cid}.", False, set()

    if alt_query == "baslinje":
        selections[cid] = {
            "id": cid,
            "name": comp_alts.get("component_name", ""),
            "selected_alternative": {
                "name": "Baslinje",
                "co2e_kg": comp_alts.get("baseline_co2e_kg", 0),
                "cost_sek": comp_alts.get("baseline_cost_sek", 0),
                "source": "NollCO2",
            },
            "baseline_co2e_kg": comp_alts.get("baseline_co2e_kg", 0),
            "baseline_cost_sek": comp_alts.get("baseline_cost_sek", 0),
        }
        return f"Valde baslinjen för {comp_alts.get('component_name', cid)}.", True, {"selections"}

    match = None
    for a in comp_alts.get("alternatives", []):
        if alt_query in (a.get("name") or "").lower():
            match = a
            break

    if not match:
        names = [a.get("name", "") for a in comp_alts.get("alternatives", [])]
        return (
            f"Hittade inget alternativ som matchar '{inp.get('alternative_name')}' för {cid}. "
            f"Tillgängliga: {', '.join(names)}"
        ), False, set()

    selections[cid] = {
        "id": cid,
        "name": comp_alts.get("component_name", ""),
        "selected_alternative": {
            "name": match.get("name", ""),
            "co2e_kg": match.get("co2e_kg", 0),
            "cost_sek": match.get("cost_sek", 0),
            "source": match.get("source", ""),
        },
        "baseline_co2e_kg": comp_alts.get("baseline_co2e_kg", 0),
        "baseline_cost_sek": comp_alts.get("baseline_cost_sek", 0),
    }
    return (
        f"Valde '{match.get('name')}' för {comp_alts.get('component_name', cid)} "
        f"({round(match.get('co2e_kg', 0))} kg CO₂e, {round(match.get('cost_sek', 0))} SEK)."
    ), True, {"selections"}


def _validate_component_ids(cids: list[str], project) -> tuple[list[str], list[str]]:
    """Split component_ids into (known, unknown) based on the project."""
    if not project:
        return [], cids
    known_set = {c.get("id") for c in project.get("components", [])}
    known = [c for c in cids if c in known_set]
    unknown = [c for c in cids if c not in known_set]
    return known, unknown


def _already_requested(pending_actions: list, action_type: str, cids: list[str]) -> bool:
    """True if this exact (type, sorted-component-ids) combo is already queued this turn."""
    target = tuple(sorted(cids))
    for pa in pending_actions:
        if pa.get("type") == action_type and tuple(sorted(pa.get("component_ids") or [])) == target:
            return True
    return False


_REASON_MAX = 500
_FEEDBACK_MAX = 500


def _apply_rerun_baseline(inp, project, baseline, alternatives, selections, pending_actions):
    raw_cids = inp.get("component_ids")
    cids = list(raw_cids) if isinstance(raw_cids, list) else []
    # Cap length so an oversized LLM-emitted reason cannot flood the next prompt
    # or the chat UI. The reason is shown to the user and stored in pending_actions.
    reason = (inp.get("reason") or "").strip()[:_REASON_MAX]

    if not reason:
        return "Saknar reason. Varför ska baslinjen räknas om?", False, set()

    if cids:
        known, unknown = _validate_component_ids(cids, project)
        if unknown:
            return f"Okänt komponent-id i rerun_baseline: {unknown}.", False, set()
        cids = known

    if _already_requested(pending_actions, "rerun_baseline", cids):
        return "Baslinje-omkörning är redan begärd denna tur.", False, set()

    pending_actions.append({
        "type": "rerun_baseline",
        "component_ids": cids,
        "reason": reason,
    })

    scope = "alla komponenter" if not cids else f"komponent {', '.join(cids)}"
    return f"Begärt: räkna om baslinjen för {scope}. Orsak: {reason}", True, set()


def _apply_rerun_alternatives(inp, project, baseline, alternatives, selections, pending_actions):
    raw_cids = inp.get("component_ids")
    cids = list(raw_cids) if isinstance(raw_cids, list) else []
    reason = (inp.get("reason") or "").strip()[:_REASON_MAX]
    # user_feedback flows into the alternatives-LLM prompt as an extra instruction.
    # Cap to limit prompt-injection blast radius from a manipulated LLM emission.
    user_feedback = (inp.get("user_feedback") or "").strip()[:_FEEDBACK_MAX]

    if not reason:
        return "Saknar reason. Varför ska alternativen räknas om?", False, set()

    if cids:
        known, unknown = _validate_component_ids(cids, project)
        if unknown:
            return f"Okänt komponent-id i rerun_alternatives: {unknown}.", False, set()
        cids = known

    if _already_requested(pending_actions, "rerun_alternatives", cids):
        return "Alternativ-omkörning är redan begärd denna tur.", False, set()

    action = {
        "type": "rerun_alternatives",
        "component_ids": cids,
        "reason": reason,
    }
    if user_feedback:
        action["user_feedback"] = user_feedback
    pending_actions.append(action)

    scope = "alla komponenter" if not cids else f"komponent {', '.join(cids)}"
    feedback_note = f" Önskemål: {user_feedback}." if user_feedback else ""
    return f"Begärt: kör om alternativen för {scope}. Orsak: {reason}.{feedback_note}", True, set()


_HANDLERS = {
    "update_component": _apply_update_component,
    "select_alternative": _apply_select_alternative,
    "remove_component": _apply_remove_component,
    "rerun_baseline": _apply_rerun_baseline,
    "rerun_alternatives": _apply_rerun_alternatives,
}


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
) -> dict:
    """Run chat with tool-use loop.

    Returns dict with:
      - reply: str — assistant's final text reply
      - state_updates: dict — {project?, selections?} with changed objects
      - tool_calls: list — trace of tool invocations (for debug/UI)
    """
    client = get_client()
    history = _sanitize_history(history or [])

    # Work on copies so we can diff at the end.
    project = copy.deepcopy(project) if project else None
    baseline = copy.deepcopy(baseline) if baseline else None
    alternatives = copy.deepcopy(alternatives) if alternatives else None
    selections = copy.deepcopy(selections) if selections else {}

    touched_bags: set[str] = set()
    tool_calls: list[dict] = []
    pending_actions: list[dict] = []

    state_block = _format_state(project, baseline, alternatives, selections)
    system_prompt = SYSTEM_PROMPT + "\n\nNUVARANDE STATE:\n" + state_block

    messages: list[dict] = list(history[-10:]) + [{"role": "user", "content": message}]

    for _ in range(max_turns):
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=1500,
            system=system_prompt,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            reply = extract_text(response) or ""
            return {
                "reply": reply.strip(),
                "state_updates": _build_state_updates(
                    touched_bags, project, baseline, alternatives, selections,
                    pending_actions,
                ),
                "tool_calls": tool_calls,
            }

        # Accumulate assistant turn (text + tool_use blocks)
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            handler = _HANDLERS.get(block.name)
            if not handler:
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": f"Okänt verktyg: {block.name}",
                    "is_error": True,
                })
                tool_calls.append({"name": block.name, "input": block.input, "ok": False})
                continue

            try:
                result_text, ok, handler_touched = handler(
                    block.input, project, baseline, alternatives, selections, pending_actions,
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
                "input": block.input,
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
        "reply": "Jag fastnade i en loop. Försök formulera om, eller använd knapparna för att köra om stegen.",
        "state_updates": _build_state_updates(
            touched_bags, project, baseline, alternatives, selections,
            pending_actions,
        ),
        "tool_calls": tool_calls,
    }


def _build_state_updates(
    touched: set[str], project, baseline, alternatives, selections,
    pending_actions: list[dict] | None = None,
) -> dict:
    updates: dict = {}
    if "project" in touched:
        updates["project"] = project
    if "baseline" in touched:
        updates["baseline"] = baseline
    if "alternatives" in touched:
        updates["alternatives"] = alternatives
    if "selections" in touched:
        updates["selections"] = selections
    if pending_actions:
        updates["pending_actions"] = pending_actions
    return updates
