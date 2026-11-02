"""Transport, building site and demolition, as a choice on top of the main figure.

DECISIONS 19. Aida's main figure is manufacturing (A1-A3), the module where the
choice of material decides most. A klimatdeklaration or a comparison with another
tool counts more. This module counts what can be counted per building part, and the
report shows it as its own section when the user ticks it. It never changes the
totals: a number that means different things on different days compares with
nothing.

Scope and sources, IVL's Anvisningar för LCA-beräkning av byggprojekt, version
2025-02, anvisning 5.1 for renovation: A1-A5 and C2-C4.

- A4, transport to the site: weight x Boverket's A4 per kg.
- A5, waste on site: Boverket's A5.1 as a share of A1-A3, times the chosen figure.
  A smaller climate figure wastes less climate per kg.
- A5, energy on site: not counted. IVL's schablon is for a whole new building (S2)
  and anvisning S1 says schablons should be avoided for renovation. Per building
  part there is no source.
- C1, taking it down: not counted. IVL 5.1: "generellt sätt mycket låg" and need
  not be calculated.
- C2, carting the waste: tonnes x 50 km x 1,5 MJ/tonne-km (IVL 8.3) x Boverket's
  diesel (Diesel MK1, 0,0899 kg CO2e/MJ).
- C3-C4, treating the waste: weight x IVL's table 17 by product group.

The rules per choice:
- Baseline or a new product: all of it. The weight is the conventional one's: an
  EPD gives no weight, and a new floor weighs about what the floor it replaces does.
- Reused: A4 is already in its figure and there is no waste beyond it. The old part
  still has to go.
- Keep and repair: A4 and waste only for what the repair adds. Nothing is demolished.

Assumptions the section states: what is demolished weighs what the baseline's
material weighs, since Aida does not know what is there today. An unknown is named,
never counted as zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aida.data import bevara, massor

C2_KM = 50.0
C2_MJ_PER_TONNE_KM = 1.5
DIESEL_KG_PER_MJ = 0.0899  # Boverket: Diesel MK1, reduktionsplikt (2024)


def demolition_transport_per_kg() -> float:
    return C2_KM * C2_MJ_PER_TONNE_KM * DIESEL_KG_PER_MJ / 1000


def _kind(selected: dict | None) -> str:
    """baseline, reuse, preserve or new, also for selections saved before the
    alternative_type travelled with them: the source says it."""
    alt = selected if isinstance(selected, dict) else {}
    source = str(alt.get("source") or "")
    kind = str(alt.get("alternative_type") or "")
    if kind == "preserve" or source.startswith(bevara.SOURCE_PREFIX):
        return "preserve"
    if kind == "reuse" or source.startswith("[Palats]") or alt.get("picks"):
        return "reuse"
    if kind == "baseline" or alt.get("name") == "Baslinje" or source == "NollCO2":
        return "baseline"
    return "new"


_STAGE_NAMES = ("a4", "a5", "c2", "c34")


@dataclass
class Stages:
    a4: float | None = 0.0
    a5: float | None = 0.0
    c2: float | None = 0.0
    c34: float | None = 0.0

    def add(self, other: Stages) -> None:
        """A sum of what is known. Nothing known stays None, never 0."""
        for name in _STAGE_NAMES:
            mine, theirs = getattr(self, name), getattr(other, name)
            if mine is None or theirs is None:
                setattr(self, name, theirs if mine is None else mine)
            else:
                setattr(self, name, mine + theirs)

    @property
    def total(self) -> float:
        return sum(v or 0.0 for v in (self.a4, self.a5, self.c2, self.c34))


@dataclass
class Row:
    id: str
    name: str
    kind: str
    chosen: Stages
    baseline: Stages
    missing: list[str] = field(default_factory=list)
    estimated: str = ""  # the weight's arithmetic, when the weight is a typical value
    analog: str = ""  # the record standing in, when Boverket has none for the product


@dataclass
class LifeCycle:
    rows: list[Row]
    chosen: Stages
    baseline: Stages
    a1a3_chosen: float
    a1a3_baseline: float


def _spill(figure: float, material) -> float | None:
    if material is None or material.a5 is None or not material.a1a3:
        return None
    return figure * material.a5 / material.a1a3


def _stages_of_bundle(comp, row: dict, bundle) -> Row:
    """A recipe's accessories: every amount is known, per material."""
    quantity = comp.quantity or 0
    a4 = a5 = c2 = c34 = 0.0
    for item in bundle.items:
        material = massor.MATERIALS[item.material]
        kg = item.kg * quantity
        a4 += kg * material.a4
        a5 += kg * (material.a5 or 0.0)
        c2 += kg * demolition_transport_per_kg()
        c34 += kg * material.c34
    stages = Stages(a4, a5, c2, c34)
    kind = _kind(row.get("_selected"))
    chosen = Stages(0.0, 0.0, 0.0, 0.0) if kind == "preserve" else stages
    return Row(comp.id, comp.name, kind, chosen, stages)


def _stages_of_part(comp, row: dict, baseline: dict | None) -> Row:
    from aida.data.byggdelsrecept import BUNDLES
    from aida.data.climate_data import canonical_category, normalize_component_name

    bundle = BUNDLES.get(getattr(comp, "recipe_bundle", ""))
    if bundle is not None:
        return _stages_of_bundle(comp, row, bundle)
    bl = baseline or {}
    text = " ".join(str(p) for p in (comp.name, bl.get("assumed_material"),
                                     bl.get("boverket_product")) if p)
    category = canonical_category(comp.category) or normalize_component_name(comp.name)
    mass = massor.mass_per_unit(category, text, comp.unit)
    quantity = comp.quantity or 0
    material = mass.material
    missing: list[str] = []

    kg = mass.kg * quantity if mass.kg is not None else None
    a4 = kg * material.a4 if kg is not None and material else None
    c2 = kg * demolition_transport_per_kg() if kg is not None else None
    c34 = kg * material.c34 if kg is not None and material else None
    bl_figure = row.get("baslinje_co2e_kg") or 0.0
    bl_stages = Stages(a4, _spill(bl_figure, material), c2, c34)

    kind = _kind(row.get("_selected"))
    if kind == "preserve":
        key = str((row.get("_selected") or {}).get("source") or "").removeprefix(
            bevara.SOURCE_PREFIX).strip()
        repair = bevara.BY_KEY.get(key)
        if key == bevara.KEPT_LAYER:
            chosen = Stages(0.0, 0.0, 0.0, 0.0)
        elif repair:
            chosen = Stages(repair.a4_per_unit * quantity, repair.a5_per_unit * quantity, 0.0, 0.0)
        else:
            chosen = Stages(None, None, 0.0, 0.0)
            missing.append("lagningen går inte att slå upp, så transport och spill kan inte räknas")
    elif kind == "reuse":
        chosen = Stages(0.0, 0.0, c2, c34)
    elif kind == "baseline":
        chosen = Stages(a4, bl_stages.a5, c2, c34)
    else:
        chosen = Stages(a4, _spill(row.get("co2e_kg") or 0.0, material), c2, c34)

    if mass.kg is None:
        missing.append(mass.how)
    elif material is None:
        missing.append("materialet framgår inte, så transport, spill och omhändertagande "
                       "kan inte räknas")
    elif material.a5 is None:
        missing.append("spillet saknas i källan")
    return Row(
        comp.id, comp.name, kind, chosen, bl_stages, missing,
        mass.how if mass.kg is not None and mass.estimated else "",
        material.record if material is not None and material.analog else "",
    )


def compute(project, aggregate, selections, baseline: dict | None) -> LifeCycle:
    """The stages for every counted component, chosen and baseline side by side."""
    by_id = {c.id: c for c in project.components}
    selected = {s.id: s.selected_alternative for s in selections.components}
    baselines = {b.get("component_id"): b for b in (baseline or {}).get("components") or []
                 if isinstance(b, dict)}
    rows: list[Row] = []
    chosen, base = Stages(None, None, None, None), Stages(None, None, None, None)
    for detail in aggregate.components:
        comp = by_id.get(detail.get("id"))
        if comp is None:
            continue
        row = _stages_of_part(comp, {**detail, "_selected": selected.get(comp.id)},
                              baselines.get(comp.id))
        rows.append(row)
        chosen.add(row.chosen)
        base.add(row.baseline)
    return LifeCycle(rows, chosen, base, aggregate.total_co2e_kg,
                     aggregate.baseline_total_co2e_kg)


def _kg(value: float | None) -> str:
    if value is None:
        return "saknas"
    text = f"{value:,.0f}" if abs(value) >= 10 else f"{value:,.1f}"
    return text.replace(",", " ").replace(".", ",")


def _gaps(life: LifeCycle, side: str) -> set[str]:
    """The stages where at least one part's figure is unknown, on one side."""
    return {name for name in _STAGE_NAMES
            if any(getattr(getattr(r, side), name) is None for r in life.rows)}


def _at_least(text: str, short: bool) -> str:
    """A sum with something missing is a floor, and says so."""
    return f"minst {text}" if short else text


def _cell_kg(stages: Stages, name: str, gaps: set[str]) -> str:
    value = getattr(stages, name)
    return _at_least(_kg(value), name in gaps and value is not None)


def _cell(text) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_markdown(life: LifeCycle) -> str:
    """The section, deterministic: Word gets exactly this."""
    chosen_all = life.a1a3_chosen + life.chosen.total
    baseline_all = life.a1a3_baseline + life.baseline.total
    gaps_chosen, gaps_baseline = _gaps(life, "chosen"), _gaps(life, "baseline")
    def row(label: str, stage: str) -> str:
        return (f"| {label} | {_cell_kg(life.chosen, stage, gaps_chosen)} "
                f"| {_cell_kg(life.baseline, stage, gaps_baseline)} |")
    lines = [
        "## Transport, byggplats och rivning",
        "",
        "Aidas huvudsiffra är tillverkningen av materialen (A1-A3). Det här avsnittet "
        "lägger till det som går att räkna per byggdel. Huvudsiffran och besparingen i "
        "resten av rapporten är oförändrade.",
        "",
        "| Del | Valda alternativ (kg CO2e) | Baslinje (kg CO2e) |",
        "|---|---|---|",
        f"| Tillverkning (A1-A3), huvudsiffran | {_kg(life.a1a3_chosen)} | {_kg(life.a1a3_baseline)} |",
        row("Transport till byggplatsen (A4)", "a4"),
        row("Spill på byggplatsen (A5)", "a5"),
        row("Transport av rivningsavfall (C2)", "c2"),
        row("Omhändertagande av rivningsavfall (C3-C4)", "c34"),
        f"| **Summa** | **{_at_least(_kg(chosen_all), bool(gaps_chosen))}** "
        f"| **{_at_least(_kg(baseline_all), bool(gaps_baseline))}** |",
        "",
        "Så är det räknat: vikten per byggdel gånger Boverkets värden för transport och "
        "spill (klimatdatabasen, version 02.07.000). Rivningsavfallet körs 50 km med "
        "1,5 MJ per ton och km på Boverkets diesel och behandlas med IVL:s värden per "
        "produktgrupp (Anvisningar för LCA-beräkning av byggprojekt, version 2025-02). "
        "Det som rivs antas väga lika mycket som baslinjens material, eftersom Aida inte "
        "vet vad som finns i dag. Återbruk har sin transport i huvudsiffran och inget "
        "spill. Det som bevaras rivs inte, och där räknas bara det som tillförs.",
        "",
        "Räknas inte: energin på byggplatsen (A5), för IVL:s schablon gäller en hel "
        "nybyggd byggnad och det finns inget värde per byggdel. Demontering (C1), som "
        "enligt IVL är mycket låg och inte behöver räknas vid renovering.",
    ]
    missing = [r for r in life.rows if r.missing]
    if missing:
        lines += ["", "**Saknas i summan**, för att vikt eller material inte är känt. "
                      "Summan är alltså för låg med det som de här byggdelarna står för:", ""]
        lines += [f"- {_cell(r.name)}: {_cell(r.missing[0])}" for r in missing]
    estimated = [r for r in life.rows if r.estimated]
    if estimated:
        lines += ["", "**Uppskattad vikt:**", ""]
        lines += [f"- {_cell(r.name)}: {_cell(r.estimated)}" for r in estimated]
    analog = [r for r in life.rows if r.analog]
    if analog:
        lines += ["", "**Saknar egen post hos Boverket**, räknat med närmaste post:", ""]
        lines += [f"- {_cell(r.name)}: {_cell(r.analog)}" for r in analog]
    return "\n".join(lines)
