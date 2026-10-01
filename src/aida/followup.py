"""Uppföljning: what was actually installed, against baseline and against plan.

(orchestration-redesign §12.6)

The analysis says what a renovation would cost the climate. This module answers
the question afterwards: what did it actually cost. That is the number a
förvaltare can put in a klimatredovisning, and it is the only number in the tool
that is about a building rather than about a plan.

Three rules shape the whole module.

**A row without a usable EPD contributes nothing, and is named.** Same rule the
aggregate already applies to prices: a missing price is not zero kronor, and a
missing climate figure is not zero emissions. Summing an unmatched row as zero
would report a saving that came out of a data gap, and it would do it in the one
document that leaves the tool. So `totals` sums only the rows it can compute and
carries the names of the rest, for whoever renders it to say out loud.

**An EPD declared in a different unit is not a number.** A declaration per kg
cannot be multiplied by a quantity in square metres. Note that this is a
*stricter* test than the alternatives side's `_unit_class`, and deliberately so:
that one asks whether two rows may appear in the same ranked list, and it puts
kg and st in one class on purpose. Multiplying a per-kg figure by a count of
pieces is not a ranking question, it is an order-of-magnitude error with no
marking on it. So the rule here is equality after spelling is normalised, not
class membership.

**Transport is recorded, not converted.** Reuse has no A1-A3 production, so its
outcome is zero plus transport. Converting kilometres to kilograms needs a mass
per component that Aida does not have, so this module stores the distance,
marks the row, and leaves the figure as a floor rather than inventing a factor.
A stated floor is honest; a fabricated total is not.
"""

from __future__ import annotations

from datetime import UTC, datetime

from aida import overrides as overrides_mod
from aida.agents.aggregate import article_price

# Spelling variants of one unit, not classes of comparable units. Every entry
# here means "the same physical quantity written differently"; nothing that
# needs a conversion factor belongs in this map.
_UNIT_ALIASES = {
    "m²": "m2", "kvm": "m2", "kvadratmeter": "m2",
    "m³": "m3", "kubikmeter": "m3",
    "pcs": "st", "styck": "st", "stycken": "st",
    "m": "lm", "meter": "lm", "löpmeter": "lm", "lpm": "lm",
    "kilogram": "kg",
}

# What the match produced. An outcome, not a setting: the user picks an EPD, the
# quality follows from what kind of EPD it turned out to be.
MATCH_QUALITIES = ("product", "generic", "typvarde", "reuse", "none")

QUALITY_LABELS = {
    "product": "Produktspecifik EPD",
    "generic": "Generisk EPD",
    "typvarde": "Kategorins typvärde",
    "reuse": "Återbruk",
    "none": "Ingen träff",
}

# The two qualities whose figure is not about the product that was installed.
# The report has to name these rows; that is what makes the rest trustworthy.
UNCERTAIN_QUALITIES = ("typvarde", "none")

_MAX_NAME = 200
_MAX_SOURCE = 120


def _num(value):
    """A finite number, or None. Strings arrive from JSON payloads and forms."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def _text(value, limit):
    return str(value or "").strip()[:limit]


def normalize_epd(epd) -> dict | None:
    """One bound EPD, or None. `gwp_per_unit` may be absent: a declaration can be
    bound before its figure is fetched, and that is a match without an outcome
    rather than an error."""
    if not isinstance(epd, dict):
        return None
    ident = _text(epd.get("id") or epd.get("uuid"), _MAX_NAME)
    if not ident:
        return None
    gwp = _num(epd.get("gwp_per_unit"))
    return {
        "id": ident,
        "name": _text(epd.get("name"), _MAX_NAME),
        "gwp_per_unit": gwp if (gwp is not None and gwp >= 0) else None,
        "unit": _text(epd.get("unit"), 16),
        "gwp_basis": _text(epd.get("gwp_basis"), 16),
        "reg_no": _text(epd.get("reg_no"), _MAX_NAME),
        # "estimated" = the figure was inferred from a declaration rather than
        # read off one. Only that exact value marks the row, and absence means
        # "declared" on purpose: every EPD bound before this field existed came
        # from the Environdec API, and defaulting the other way would stamp
        # "uppskattad" across every klimatredovisning already saved. The
        # fail-closed default belongs in resolve_epd, where a missing field
        # means a new hand-entered row nobody classified.
        "gwp_source": ("estimated" if epd.get("gwp_source") == "estimated"
                       else "declared"),
    }


def normalize_as_built(entry) -> tuple[dict | None, str]:
    """Validate one as-built record. Returns (record, error), exactly one set.

    Called from both doors, so what a cell accepts and what the chat accepts
    cannot differ.
    """
    if not isinstance(entry, dict):
        return None, "Uppgiften måste vara ett objekt."

    quality = _text(entry.get("match_quality"), 16) or "none"
    if quality not in MATCH_QUALITIES:
        return None, f"Okänd matchkvalitet: {quality}"

    quantity = _num(entry.get("quantity"))
    if quantity is not None and quantity < 0:
        return None, "Mängden kan inte vara negativ."

    cost = _num(entry.get("actual_cost"))
    if cost is not None and cost < 0:
        return None, "Kostnaden kan inte vara negativ."

    transport = _num(entry.get("transport_km"))
    if transport is not None and transport < 0:
        return None, "Transportsträckan kan inte vara negativ."

    return {
        "installed_name": _text(entry.get("installed_name"), _MAX_NAME),
        "quantity": quantity,
        "unit": _text(entry.get("unit"), 16),
        "epd": normalize_epd(entry.get("epd")),
        "match_quality": quality,
        "transport_km": transport,
        "actual_cost": cost,
        "cost_source": _text(entry.get("cost_source"), _MAX_SOURCE),
        "at": _text(entry.get("at"), 40) or datetime.now(UTC).isoformat(timespec="seconds"),
    }, ""


def normalize_unit(unit: str) -> str:
    """One spelling per unit. Does not convert anything."""
    lowered = (unit or "").strip().lower()
    return _UNIT_ALIASES.get(lowered, lowered)


def units_comparable(epd_unit: str, component_unit: str) -> bool:
    """Can a declaration in one unit be multiplied by a quantity in the other?

    Only when they are the same unit. Getting this wrong does not raise: it
    produces a figure with the wrong order of magnitude and no marking on it,
    which is the worst failure this tool has. An unknown unit on either side is
    a no, because "we could not tell" must not read as "yes".
    """
    a = normalize_unit(epd_unit)
    b = normalize_unit(component_unit)
    return bool(a and b and a == b)


def _outcome_for(as_built: dict, component_unit: str) -> tuple[float | None, str]:
    """(kg CO2e, note). None means the row cannot be counted, and the note says why."""
    quality = as_built.get("match_quality") or "none"
    quantity = as_built.get("quantity")
    unit = as_built.get("unit") or component_unit

    # Reuse produces nothing in A1-A3. The transport that carried it here is
    # real but unconverted, so the figure is a floor and says so.
    if quality == "reuse":
        km = as_built.get("transport_km")
        note = ("Återbruk: A1-A3 är noll. Transporten "
                + (f"({km:.0f} km) " if km else "")
                + "är inte omräknad till utsläpp, så siffran är ett golv.")
        return 0.0, note

    epd = as_built.get("epd")
    if not epd:
        return None, "Ingen EPD bunden, så utfallet går inte att räkna."
    if epd.get("gwp_per_unit") is None:
        return None, "Den bundna EPD:n saknar GWP-värde för A1-A3."
    if quantity is None:
        return None, "Ingen installerad mängd angiven."
    if not units_comparable(epd.get("unit"), unit):
        return None, (f"EPD:n är deklarerad per {epd.get('unit') or 'okänd enhet'} "
                      f"och mängden är i {unit or 'okänd enhet'}, så de går inte att multiplicera.")

    return round(epd["gwp_per_unit"] * quantity, 1), ""


def _planned_for(cid: str, selections: dict, baseline_co2e: float | None):
    """(kg, is_baseline). A component nobody chose an alternative for is planned
    as its baseline, and the caller says so in the cell rather than presenting
    the baseline as if it had been a choice. A chosen alternative without a
    figure is an unknown plan, not the baseline's (same rule as the cost)."""
    sel = selections.get(cid) if isinstance(selections, dict) else None
    if isinstance(sel, dict):
        alt = sel.get("selected_alternative")
        if isinstance(alt, dict):
            value = _num(alt.get("co2e_kg"))
            if value is not None:
                return value, alt.get("name") == "Baslinje"
            if alt.get("name") != "Baslinje":
                return None, False
    return baseline_co2e, True


def _planned_cost_for(cid: str, selections: dict, baseline_cost: float | None):
    """The planned cost, or None when the plan has no price.

    A component nobody chose for is planned at its baseline cost. A chosen
    alternative without a price is an unknown plan, not the baseline's: until
    2026-10-01 it fell back to the baseline cost, and a per-article Palats price
    saved in cost_sek counted as the whole component's cost. An unpriced
    baseline (0) is no plan either: planned at 0 kr, its whole actual cost read
    as overrun and "estimated 0" went into the calibration facts.
    """
    sel = selections.get(cid) if isinstance(selections, dict) else None
    if isinstance(sel, dict):
        alt = sel.get("selected_alternative")
        if isinstance(alt, dict) and alt.get("name") != "Baslinje":
            if article_price(alt):
                return None
            value = _num(alt.get("cost_sek"))
            return value if value is not None and value > 0 else None
    value = _num(baseline_cost)
    return value if value is not None and value > 0 else None


def compute(project, baseline, selections, as_built, overrides=None) -> dict:
    """One row per project component, plus totals that only count what they can.

    Pure: nothing here writes into the state it was handed. Overrides are laid on
    first, so the baseline column is the one the rest of the tool shows.
    """
    project = project or {}
    as_built = as_built if isinstance(as_built, dict) else {}
    baseline, _, selections = overrides_mod.apply(
        baseline, None, selections if isinstance(selections, dict) else {}, overrides,
    )
    baseline = baseline or {}
    selections = selections or {}

    base_rows = {}
    for row in (baseline.get("components") or []):
        if isinstance(row, dict) and row.get("component_id"):
            base_rows[row["component_id"]] = row

    rows = []
    for comp in (project.get("components") or []):
        if not isinstance(comp, dict) or not comp.get("id"):
            continue
        cid = comp["id"]
        base = base_rows.get(cid) or {}
        entry = as_built.get(cid)
        entry = entry if isinstance(entry, dict) else {}

        baseline_co2e = _num(base.get("co2e_kg"))
        baseline_cost = _num(base.get("cost_sek"))
        planned_co2e, planned_is_baseline = _planned_for(cid, selections, baseline_co2e)
        outcome, note = _outcome_for(entry, comp.get("unit", "")) if entry else (
            None, "Inget installerat registrerat än.")

        rows.append({
            "component_id": cid,
            "name": comp.get("name", ""),
            "unit": comp.get("unit", ""),
            "planned_quantity": _num(comp.get("quantity")),
            "installed_name": entry.get("installed_name", ""),
            "installed_quantity": entry.get("quantity"),
            "installed_unit": entry.get("unit") or comp.get("unit", ""),
            "epd": entry.get("epd"),
            "match_quality": entry.get("match_quality") or "none",
            "transport_km": entry.get("transport_km"),
            "outcome_co2e_kg": outcome,
            "outcome_note": note,
            "baseline_co2e_kg": baseline_co2e,
            "baseline_co2e_override": base.get("co2e_override", ""),
            "planned_co2e_kg": planned_co2e,
            "planned_is_baseline": planned_is_baseline,
            "actual_cost_sek": entry.get("actual_cost"),
            "cost_source": entry.get("cost_source", ""),
            "planned_cost_sek": _planned_cost_for(cid, selections, baseline_cost),
            "has_as_built": bool(entry),
        })

    return {"rows": rows, "totals": _totals(rows),
            "uncertainties": uncertainties(rows)}


def _sum(values):
    known = [v for v in values if v is not None]
    return round(sum(known), 1) if known else 0.0


def _totals(rows) -> dict:
    """Only over rows that can be counted, and the rest by name.

    A total that silently skipped them would still be a total; it just would not
    be the one the reader thinks they are looking at.
    """
    # A row counts only with all three figures: the cards and the report say
    # baseline and plan cover the same rows as the outcome.
    counted = [r for r in rows if r["outcome_co2e_kg"] is not None
               and r["baseline_co2e_kg"] is not None and r["planned_co2e_kg"] is not None]
    uncounted = [r["name"] for r in rows if r not in counted]

    outcome = _sum(r["outcome_co2e_kg"] for r in counted)
    # Compared over the SAME rows, never over all of them: an outcome for three
    # components against a baseline for five is not a saving, it is a subtraction
    # of two different things.
    baseline = _sum(r["baseline_co2e_kg"] for r in counted)
    planned = _sum(r["planned_co2e_kg"] for r in counted)

    # Cost is compared over rows priced on BOTH sides, for the same reason as
    # above; a row with an actual cost but no planned one is named instead.
    cost_rows = [r for r in counted
                 if r["actual_cost_sek"] is not None and r["planned_cost_sek"] is not None]
    cost_uncounted = [r["name"] for r in counted
                      if r["actual_cost_sek"] is not None and r["planned_cost_sek"] is None]
    actual_cost = _sum(r["actual_cost_sek"] for r in cost_rows)
    planned_cost = _sum(r["planned_cost_sek"] for r in cost_rows)

    return {
        "outcome_co2e_kg": outcome,
        "baseline_co2e_kg": baseline,
        "planned_co2e_kg": planned,
        "avoided_vs_baseline_kg": round(baseline - outcome, 1),
        "deviation_vs_plan_kg": round(outcome - planned, 1),
        "rows_counted": len(counted),
        "rows_total": len(rows),
        "uncounted_names": uncounted,
        "actual_cost_sek": round(actual_cost),
        "planned_cost_sek": round(planned_cost),
        "cost_difference_sek": round(actual_cost - planned_cost),
        "cost_rows_counted": len(cost_rows),
        "cost_uncounted_names": cost_uncounted,
    }


def uncertainties(rows) -> list[dict]:
    """The rows whose figure is not about the product that was installed.

    Deterministic and built from state, like every other caveat in this tool, so
    it cannot go missing because a model chose not to repeat it.
    """
    out = []
    for r in rows:
        quality = r.get("match_quality") or "none"
        if quality in UNCERTAIN_QUALITIES or r.get("outcome_co2e_kg") is None:
            out.append({
                "komponent": r.get("name", ""),
                "underlag": QUALITY_LABELS.get(quality, quality),
                "varför": r.get("outcome_note") or
                          "Siffran kommer inte från den installerade produktens egen deklaration.",
            })
    return out


def facts(analysis_id: str, rows) -> list[dict]:
    """Rows for `follow_up_facts`: what was estimated against what turned out.

    Collected, not fed back. §12.6 is explicit that nothing here loops into
    pricing yet; first gather, then decide, because a correction applied from a
    handful of projects is a rumour with a number on it.
    """
    out = []
    for r in rows:
        cid = r.get("component_id")
        if r.get("actual_cost_sek") is not None and r.get("planned_cost_sek") is not None:
            out.append({
                "analysis_id": analysis_id, "component_id": cid, "field": "cost_sek",
                "estimated": r["planned_cost_sek"], "actual": r["actual_cost_sek"],
                "note": r.get("cost_source", ""),
            })
        if r.get("outcome_co2e_kg") is not None and r.get("planned_co2e_kg") is not None:
            out.append({
                "analysis_id": analysis_id, "component_id": cid, "field": "co2e_kg",
                "estimated": r["planned_co2e_kg"], "actual": r["outcome_co2e_kg"],
                "note": r.get("match_quality", ""),
            })
    return out


def drop_for_component(as_built, cid: str) -> bool:
    """Forget this component's as-built record. Returns whether anything went.

    Same rule as an override (§12.5): the record is a claim about a specific
    thing at a specific quantity, so once that thing changes, keeping it would
    attribute an installation to a component that is no longer the one described.
    """
    if not as_built or cid not in as_built:
        return False
    del as_built[cid]
    return True
