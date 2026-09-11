"""Every state mutation Aida can make, as pure functions.

These used to live inside chat_agent.py, reachable only by the model. That was
fine while the chat was the only way to change anything. It stopped being fine
when cells in the sheet became editable: a cell that edited state its own way
would be a second set of rules for what goes stale, and the two would drift.

So the rules live here instead, and both doors open onto them. The chat agent
dispatches tool calls through HANDLERS; /api/mutate dispatches a cell edit
through apply_mutation. Same functions, same invariants, one owner.

No LLM, no HTTP, no DOM. Every handler takes the four state bags plus a list of
pending actions, mutates them in place, and returns (message, ok, touched_bags).
That signature is what lets this module move behind /api/turn (design §3)
unchanged when the orchestrator arrives.

Overrides (§12.5) are a fifth bag, and they are not like the other four: they are
not computed, they are claimed. `run_handler` is where that difference is
enforced - it is the seam both doors pass through, so a manual figure cannot
survive a change to the component it was a claim about, whichever door made the
change.
"""

from __future__ import annotations

from aida import followup as followup_mod
from aida import overrides as overrides_mod


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
            # Unparseable stored/new quantity — don't collapse to "0 == 0" and
            # report a false no-change; fall through to the generic update so the
            # user's edit (already written to the component) is kept.
            old_q = new_q = None
        if old_q is not None and new_q is not None:
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


def _next_component_id(project) -> str:
    """First free cN, so a new component never collides with an existing id.

    Not len(components)+1: after a removal that reuses a live id, and the whole
    state model keys baseline, alternatives and selections on it. A collision
    would silently attach the new component's figures to the old one.
    """
    taken = {c.get("id") for c in project.get("components", [])}
    n = 1
    while f"c{n}" in taken:
        n += 1
    return f"c{n}"


def _apply_add_component(inp, project, baseline, alternatives, selections, pending_actions):
    name = (inp.get("name") or "").strip()
    if not name:
        return "Komponenten behöver ett namn.", False, set()

    quantity = inp.get("quantity")
    try:
        quantity = float(quantity)
    except (TypeError, ValueError):
        return f"Mängd saknas eller går inte att tolka för {name}.", False, set()
    if quantity <= 0:
        return f"Mängden för {name} måste vara större än noll.", False, set()

    unit = inp.get("unit") or ""
    if unit not in ("m2", "st", "lm"):
        return f"Enheten för {name} måste vara m2, st eller lm.", False, set()

    # A duplicate name is more likely a second attempt at the same thing than a
    # genuine second component, and two rows with the same name are impossible
    # to tell apart in the comparison table.
    existing = [c for c in project.get("components", [])
                if (c.get("name") or "").strip().lower() == name.lower()]
    if existing:
        return (
            f"Det finns redan en komponent som heter {name} ({existing[0].get('id')}). "
            "Vill du ändra mängden på den i stället, eller ska den nya heta något annat?"
        ), False, set()

    cid = _next_component_id(project)
    source = inp.get("quantity_source")
    project.setdefault("components", []).append({
        "id": cid,
        "name": name,
        "quantity": quantity,
        "unit": unit,
        "category": inp.get("category") or "",
        # Component.__post_init__ normalises anything unexpected to "estimated",
        # so a wrong guess here degrades to the honest label rather than a lie.
        "quantity_source": source if source in ("user_specified", "estimated") else "estimated",
        "usage_context": inp.get("usage_context") or "",
    })

    return (
        f"La till {cid}: {name}, {quantity:g} {unit}. "
        "Den saknar baslinje och alternativ tills de körts för just den komponenten. "
        "Övriga komponenter och deras val rörs inte."
    ), True, {"project"}


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
        # Skip "info" pseudo-alternatives ("Inget tillgängligt i Palats" etc.):
        # they carry co2e=0/cost=0 and selecting one would understate the total.
        if a.get("alternative_type") == "info":
            continue
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


def _apply_set_override(inp, project, overrides):
    """Pin a derived figure to one the user has a source for (§12.5)."""
    cid = inp.get("component_id")
    comp = _find_component(project, cid)
    if not comp:
        return f"Komponent {cid} finns inte i projektet.", False, set()

    field = inp.get("field")
    one, err = overrides_mod.normalize(field, inp.get("value"), inp.get("note"))
    if err:
        return f"Ingen överskrivning: {err}", False, set()

    overrides.setdefault(cid, {})[field] = one
    label = overrides_mod._FIELD_LABELS[field]
    return (
        f"{label} för {comp.get('name')} är satt till {one['value']:,.0f} manuellt. "
        f"Anteckning: {one['note']}. Det beräknade värdet finns kvar och visas igen "
        f"om du tar bort överskrivningen.",
        True,
        {"overrides"},
    )


def _apply_clear_override(inp, project, overrides):
    """Lift an override. The computed figure was never overwritten, so it returns."""
    cid = inp.get("component_id")
    comp = _find_component(project, cid)
    name = comp.get("name") if comp else cid
    field = inp.get("field")

    if field:
        if field not in overrides_mod.OVERRIDE_FIELDS:
            return f"Okänt fält: {field}", False, set()
        entry = overrides.get(cid) or {}
        if field not in entry:
            return f"{name} har ingen överskrivning på {field}.", False, set()
        del entry[field]
        if not entry:
            overrides.pop(cid, None)
        label = overrides_mod._FIELD_LABELS[field]
        return f"Överskrivningen av {label.lower()} för {name} är borttagen. Aidas beräknade värde gäller igen.", True, {"overrides"}

    dropped = overrides_mod.drop_for_component(overrides, cid)
    if not dropped:
        return f"{name} har inga överskrivningar.", False, set()
    return (
        f"Överskrivningarna för {name} är borttagna ({', '.join(dropped).lower()}). "
        f"Aidas beräknade värden gäller igen.",
        True,
        {"overrides"},
    )


HANDLERS = {
    "update_component": _apply_update_component,
    "add_component": _apply_add_component,
    "select_alternative": _apply_select_alternative,
    "remove_component": _apply_remove_component,
    "rerun_baseline": _apply_rerun_baseline,
    "rerun_alternatives": _apply_rerun_alternatives,
}

# Kept apart because they take the overrides bag instead of the four computed
# ones. Everything else about them is the same, and `run_handler` below is what
# both doors actually call, so the split is not visible to callers.
OVERRIDE_HANDLERS = {
    "set_override": _apply_set_override,
    "clear_override": _apply_clear_override,
}


def _apply_set_as_built(inp, project, as_built):
    """Record what was actually installed for a component (§12.6).

    Merges rather than replaces, because the installed row is filled in over
    time and from two directions: a quantity typed in the sheet, an invoice
    price added a week later, an EPD bound from the candidate list. A write that
    replaced the record would make the last field entered the only one kept.
    """
    cid = inp.get("component_id")
    comp = _find_component(project, cid)
    if not comp:
        return f"Komponent {cid} finns inte i projektet.", False, set()

    merged = dict(as_built.get(cid) or {})
    for key in ("installed_name", "quantity", "unit", "match_quality",
                "transport_km", "actual_cost", "cost_source"):
        if key in inp:
            merged[key] = inp[key]
    if "epd" in inp:
        merged["epd"] = inp["epd"]
    merged.pop("at", None)

    record, err = followup_mod.normalize_as_built(merged)
    if err:
        return f"Inget installerat registrerat: {err}", False, set()

    as_built[cid] = record
    what = record["installed_name"] or comp.get("name", cid)
    amount = (f"{record['quantity']:,.0f} {record['unit']}".replace(",", " ")
              if record["quantity"] is not None else "mängd ej angiven")
    return (
        f"Installerat för {comp.get('name')}: {what}, {amount}.",
        True,
        {"as_built"},
    )


def _apply_bind_epd(inp, project, as_built):
    """Bind a declaration to what was installed, or lift the binding.

    The same act as clicking a row in the candidate list, which is why it is a
    tool and not a bit of endpoint logic: "ja, det är iQ Granit SD" typed in the
    chat has to land in exactly the same place as the click (§12.6).
    """
    cid = inp.get("component_id")
    comp = _find_component(project, cid)
    if not comp:
        return f"Komponent {cid} finns inte i projektet.", False, set()

    merged = dict(as_built.get(cid) or {})
    epd = inp.get("epd")
    quality = inp.get("match_quality")

    if not epd:
        # Unbinding is not the same as never having bound: the quality has to
        # come down with the declaration, or the row keeps claiming a product
        # match it no longer has behind it.
        merged["epd"] = None
        merged["match_quality"] = quality if quality in followup_mod.MATCH_QUALITIES else "none"
    else:
        merged["epd"] = epd
        merged["match_quality"] = quality if quality in followup_mod.MATCH_QUALITIES else "product"
    merged.pop("at", None)

    record, err = followup_mod.normalize_as_built(merged)
    if err:
        return f"Ingen bindning: {err}", False, set()
    if epd and not record["epd"]:
        return "Ingen bindning: deklarationen saknar id.", False, set()

    as_built[cid] = record
    if not record["epd"]:
        return (f"Bindningen för {comp.get('name')} är borttagen. "
                f"Underlaget är nu {followup_mod.QUALITY_LABELS[record['match_quality']].lower()}.",
                True, {"as_built"})

    label = followup_mod.QUALITY_LABELS[record["match_quality"]].lower()
    gwp = record["epd"]["gwp_per_unit"]
    tail = (f" {gwp:g} kg CO₂e per {record['epd']['unit'] or 'enhet'}."
            if gwp is not None else
            " Deklarationen saknar GWP-värde, så utfallet går inte att räkna än.")
    return (
        f"{comp.get('name')} är bunden till {record['epd']['name'] or record['epd']['id']} "
        f"({label}).{tail}",
        True,
        {"as_built"},
    )


# The sixth bag. Like overrides it is claimed rather than computed, so it goes
# through the same seam and the same staleness rules.
AS_BUILT_HANDLERS = {
    "set_as_built": _apply_set_as_built,
    "bind_epd": _apply_bind_epd,
}

# A mutation that changes what a component IS, or removes it, makes any manual
# figure for it a claim about something that no longer exists. select_alternative
# is not in the list: it changes what was chosen, not what the baseline is.
_OVERRIDE_INVALIDATING = ("update_component", "remove_component")


def _drop_stale_overrides(tool, inp, overrides) -> str:
    """Forget overrides whose component just changed underneath them.

    Loud rather than silent, and dropped rather than rescaled. A förvaltare who
    pinned 213 kg from a supplier EPD pinned it for that material at that
    quantity; scaling it would put a number they never wrote under their own
    source, which is worse than asking them to enter it again.
    """
    if tool not in _OVERRIDE_INVALIDATING:
        return ""
    dropped = overrides_mod.drop_for_component(overrides, inp.get("component_id"))
    if not dropped:
        return ""
    return (
        f"Överskrivningen av {', '.join(dropped).lower()} togs bort, "
        f"eftersom den gällde komponenten som den såg ut innan."
    )


def _drop_stale_as_built(tool, inp, as_built) -> str:
    """Forget an installed record whose component no longer matches it.

    Narrower than the override rule on purpose. An override is a claim about a
    component's derived value, so any change to the component invalidates it. An
    as-built record is a claim about physical reality: renaming the planned
    component does not un-install what was installed. What does break it is the
    component disappearing, and the unit changing, because the recorded quantity
    is only meaningful next to its unit.

    Passing `unit` unchanged also drops it. Over-eager rather than under-eager:
    the cost of the false positive is retyping a row, the cost of the false
    negative is a quantity read in the wrong unit inside a climate declaration.
    """
    if tool == "remove_component":
        gone = followup_mod.drop_for_component(as_built, inp.get("component_id"))
        return "Det installerade som var registrerat på komponenten togs bort." if gone else ""
    if tool == "update_component" and "unit" in inp:
        gone = followup_mod.drop_for_component(as_built, inp.get("component_id"))
        return ("Det installerade som var registrerat togs bort, eftersom enheten "
                "ändrades och den registrerade mängden gällde den gamla.") if gone else ""
    return ""


def run_handler(tool, inp, project, baseline, alternatives, selections,
                pending_actions, overrides=None, as_built=None, sheet=None):
    """The one seam both doors pass through.

    The chat loop needs the per-call touched set and cannot go through
    `apply_mutation`, so without this the override lifecycle would live on the
    cell side only and the two doors would disagree about when a manual figure
    stops being true. That is the exact failure the extraction in step 3 was for.

    The sheet tools (§13) take the open sheet and nothing else. The sheet only
    exists in Chatt, so without one they refuse rather than write somewhere
    nobody sees.
    """
    from aida.sheet import SHEET_HANDLERS

    if tool in SHEET_HANDLERS:
        if sheet is None:
            return f"{tool} kräver bladet, som bara finns i läget Chatt.", False, set()
        return SHEET_HANDLERS[tool](inp, sheet)

    if tool in OVERRIDE_HANDLERS:
        if overrides is None:
            return f"{tool} kräver överskrivningar i anropet.", False, set()
        return OVERRIDE_HANDLERS[tool](inp, project, overrides)

    if tool in AS_BUILT_HANDLERS:
        if as_built is None:
            return f"{tool} kräver uppföljningsdata i anropet.", False, set()
        return AS_BUILT_HANDLERS[tool](inp, project, as_built)

    handler = HANDLERS.get(tool)
    if handler is None:
        return f"Okänt verktyg: {tool}", False, set()

    message, ok, touched = handler(
        inp, project, baseline, alternatives, selections, pending_actions,
    )
    if ok and overrides is not None:
        note = _drop_stale_overrides(tool, inp, overrides)
        if note:
            message = f"{message} {note}"
            touched = set(touched) | {"overrides"}
    if ok and as_built is not None:
        note = _drop_stale_as_built(tool, inp, as_built)
        if note:
            message = f"{message} {note}"
            touched = set(touched) | {"as_built"}
    return message, ok, touched


def build_state_updates(
    touched: set[str], project, baseline, alternatives, selections,
    pending_actions: list[dict] | None = None, overrides=None, as_built=None, sheet=None,
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
    if "overrides" in touched:
        updates["overrides"] = overrides if overrides is not None else {}
    # Presence, not truthiness, for the same reason as overrides: clearing the
    # last record leaves {}, and a falsy check would send nothing and let the
    # client keep showing what the server just dropped.
    if "as_built" in touched:
        updates["as_built"] = as_built if as_built is not None else {}
    # Touched only by a sheet tool that succeeded, so the sheet is there. Sent
    # whole even when Aida removed her last block: the empty sheet is the edit.
    if "sheet" in touched and sheet is not None:
        updates["sheet"] = sheet
    if pending_actions:
        updates["pending_actions"] = pending_actions
    return updates


# Fields whose change makes the cached climate and cost figures wrong rather than
# merely scaled. Quantity is linear under NollCO2 and is handled by scaling; these
# are not, so they need the numbers computed again.
_MATERIAL_FIELDS = ("name", "category", "unit")


def _component_ids(project) -> set[str]:
    return {c.get("id") for c in (project or {}).get("components", [])}


def _queue_reruns(tool, inp, project, before_ids, before_component, pending_actions):
    """Queue the scoped reruns the chat agent's prompt asks the model to make.

    The chat path states these two rules in the system prompt and trusts the model
    to follow them. A cell edit has no model to trust, so the same rules are
    applied here in code. That is the better place for them: a rule in a prompt is
    a request, a rule here is a guarantee, and the two doors now cannot disagree
    about when a component's numbers stopped being true.
    """
    if tool == "add_component":
        cids = sorted(_component_ids(project) - before_ids)
        reason = "Ny komponent saknar baslinje och alternativ."
    elif tool == "update_component":
        changed = [f for f in _MATERIAL_FIELDS
                   if f in inp and inp[f] is not None
                   and (before_component or {}).get(f) != inp[f]]
        if not changed:
            return  # quantity only: the handler already scaled, nothing is stale
        cids = [inp.get("component_id")]
        reason = f"Ändrat {', '.join(changed)} kräver nya värden."
    else:
        return

    if not cids:
        return

    # Scoped, never the empty list. Empty means "recompute everything", which
    # would throw away every other component's choices to fix one cell.
    for action_type in ("rerun_baseline", "rerun_alternatives"):
        if not _already_requested(pending_actions, action_type, cids):
            pending_actions.append({
                "type": action_type,
                "component_ids": cids,
                "reason": reason,
            })


def apply_mutation(tool, inp, project, baseline, alternatives, selections,
                   auto_rerun: bool = False, overrides=None, as_built=None) -> dict:
    """Run one mutation and report what changed. The door /api/mutate comes in by.

    `auto_rerun` is what separates a cell edit from a chat tool call. The model is
    told to follow a material change with scoped reruns; a cell has nobody to tell,
    so it asks for them here.
    """
    if tool not in HANDLERS and tool not in OVERRIDE_HANDLERS and tool not in AS_BUILT_HANDLERS:
        return {"ok": False, "message": f"Okänt verktyg: {tool}", "state_updates": {}}

    before_ids = _component_ids(project)
    before_component = None
    if tool == "update_component":
        found = _find_component(project, inp.get("component_id"))
        before_component = dict(found) if found else None

    pending_actions: list[dict] = []
    message, ok, touched = run_handler(
        tool, inp, project, baseline, alternatives, selections, pending_actions,
        overrides=overrides, as_built=as_built,
    )

    if ok and auto_rerun:
        _queue_reruns(tool, inp, project, before_ids, before_component, pending_actions)

    return {
        "ok": ok,
        "message": message,
        "state_updates": build_state_updates(
            touched, project, baseline, alternatives, selections, pending_actions,
            overrides=overrides, as_built=as_built,
        ),
    }
