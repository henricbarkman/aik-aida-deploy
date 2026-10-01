"""Aggregate agent: computes totals from user selections."""

from __future__ import annotations

import json
import math
import sys

from aida.models import AggregateResult, Project, Selections


def _number(value) -> float | None:
    """A figure as a float, or None when there is no usable figure.

    bool is excluded on purpose: True is an int to Python and would sum as 1.
    NaN and ±inf are no figure either; inf made round() raise in the chat state.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def article_price(alt: dict | None) -> float:
    """The per-article asking price of a Palats row, or 0 for any other row.

    Such a row ("725 kr/st *") prices one listed article for a component counted
    in m2, lm or kg, so it is not the component's cost and must never be summed
    as one. Rows built since 2026-09-30 carry it in `article_price_sek` with
    cost_sek 0. Analyses saved before that hold it in cost_sek, and the only
    trace of what it is is the trailing " *" on a Palats row: read those the same
    way, so an old analysis stops reporting a 29 000 kr saving on a floor.
    """
    if not isinstance(alt, dict):
        return 0.0
    price = _number(alt.get("article_price_sek"))
    if price is not None and price > 0:
        return price
    legacy = (
        "article_price_sek" not in alt
        and str(alt.get("name") or "").rstrip().endswith("*")
        and str(alt.get("source") or "").startswith("[Palats]")
    )
    cost = _number(alt.get("cost_sek"))
    return cost if legacy and cost is not None and cost > 0 else 0.0


def compute_aggregate(project: Project, selections: Selections) -> AggregateResult:
    """Compute aggregate totals from component selections."""
    total_co2e = 0.0
    total_cost = 0.0
    baseline_co2e = 0.0
    baseline_cost = 0.0
    component_details = []
    # A selected alternative with no price contributes nothing to total_cost.
    # Until 2026-08-20 that was indistinguishable from contributing zero kronor,
    # so a basket where one of two components was unpriced reported a large
    # saving against a full baseline. Track which ones, so every presentation
    # site can say what the total leaves out, and keep a second pair of totals
    # over the subset priced on both sides so a percentage compares like with
    # like. The baseline side got the same rule on 2026-09-30.
    unpriced: list[str] = []
    baseline_unpriced: list[str] = []
    comparable_cost = 0.0
    comparable_baseline_cost = 0.0

    project_ids = [c.id for c in project.components]
    known_ids = set(project_ids)
    # Components the totals will not cover. Filled in below with every project
    # component that has no selection, or one without a usable climate figure.
    # A stderr line was all this used to produce, and on Vercel nobody reads it.
    counted: set[str] = set()

    for sel in selections.components:
        # Skip selections for components no longer in the project (e.g. a
        # component removed after it was selected). Summing these orphans would
        # silently inflate the totals with phantom components.
        if sel.id not in known_ids:
            print(f"Varning: hoppar över urval för okänd komponent: {sel.id}", file=sys.stderr)
            continue
        alt =sel.selected_alternative if isinstance(sel.selected_alternative, dict) else {}
        alt_co2e = _number(alt.get("co2e_kg"))
        bl_co2e = _number(sel.baseline_co2e_kg)
        if alt_co2e is None or bl_co2e is None:
            # A missing figure is not zero emissions: counted as 0 it became a
            # saving of the whole baseline, and None raised a TypeError that
            # took /api/report down with a 500. Named as "utan val" instead.
            print(f"Varning: urval utan CO2e-värde för {sel.id}, räknas inte", file=sys.stderr)
            continue
        counted.add(sel.id)
        # A per-article reuse price is not the component's cost (see
        # article_price), whichever shape the row was saved in.
        per_article = article_price(alt)
        alt_cost = 0.0 if per_article else max(_number(alt.get("cost_sek")) or 0.0, 0.0)
        bl_cost = max(_number(sel.baseline_cost_sek) or 0.0, 0.0)
        has_price = alt_cost > 0
        baseline_has_price = bl_cost > 0

        total_co2e += alt_co2e
        total_cost += alt_cost
        baseline_co2e += bl_co2e
        baseline_cost += bl_cost
        if has_price and baseline_has_price:
            comparable_cost += alt_cost
            comparable_baseline_cost += bl_cost
        if not has_price:
            unpriced.append(sel.name)
        if not baseline_has_price:
            baseline_unpriced.append(sel.name)

        component_details.append({
            "id": sel.id,
            "name": sel.name,
            "valt_alternativ": alt.get("name", ""),
            "co2e_kg": alt_co2e,
            "kostnad_sek": alt_cost,
            "baslinje_co2e_kg": bl_co2e,
            "baslinje_kostnad_sek": bl_cost,
            "co2e_besparing_kg": round(bl_co2e - alt_co2e, 1),
            # Explicit, so the report table can print "Pris saknas" instead of
            # formatting a zero that reads as free.
            "pris_saknas": not has_price,
            "baslinje_pris_saknas": not baseline_has_price,
            # The one listed article's price for a per-article reuse pick, so
            # the report can name it next to "Pris saknas" without it ever
            # entering a sum. 0 for every other pick.
            "artikelpris_sek": per_article,
            "källa": alt.get("source", ""),
            # Carried through so the report can state where a reuse figure
            # assumes more stock than Palats holds. None for non-reuse picks
            # and for analyses saved before 2026-08-15.
            "tillgangligt_antal": alt.get("available_quantity"),
            "behov_antal": next(
                (c.quantity for c in project.components if c.id == sel.id), None
            ),
            "prisunderlag": alt.get("price_basis", ""),
            # "ghg" when the figure rests on GWP-GHG instead of GWP-fossil, so
            # the report can say so in words rather than leaving "(GWP-GHG)"
            # sitting in the source column as jargon.
            "gwp_underlag": alt.get("gwp_basis", ""),
        })

    # Project order, so the report lists the gaps the way the table shows them.
    missing = [cid for cid in project_ids if cid not in counted]
    if missing:
        print(f"Varning: Komponenter saknar val: {missing}", file=sys.stderr)

    return AggregateResult(
        total_co2e_kg=round(total_co2e, 1),
        total_cost_sek=round(total_cost),
        baseline_total_co2e_kg=round(baseline_co2e, 1),
        baseline_total_cost_sek=round(baseline_cost),
        co2e_savings_kg=round(baseline_co2e - total_co2e, 1),
        cost_difference_sek=round(total_cost - baseline_cost),
        components=component_details,
        unpriced_components=unpriced,
        comparable_cost_sek=round(comparable_cost),
        comparable_baseline_cost_sek=round(comparable_baseline_cost),
        baseline_unpriced_components=baseline_unpriced,
        missing_selection_ids=missing,
    )


def main():
    """CLI entry point for aggregate."""
    if len(sys.argv) < 5 or sys.argv[1] != "--project" or sys.argv[3] != "--selections":
        print("Usage: python -m aida.agents.aggregate --project <project.json> --selections <selections.json>", file=sys.stderr)
        sys.exit(1)

    project_path = sys.argv[2]
    selections_path = sys.argv[4]

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

    # Validate that selections reference valid component IDs
    project_ids = {c.id for c in project.components}
    selection_ids = {c.id for c in selections.components}
    invalid_ids = selection_ids - project_ids
    if invalid_ids and not (selection_ids & project_ids):
        print(f"Fel: Komponent-ID i urval matchar inte projektet: {invalid_ids}", file=sys.stderr)
        sys.exit(1)

    result = compute_aggregate(project, selections)
    print(result.to_json())


if __name__ == "__main__":
    main()
