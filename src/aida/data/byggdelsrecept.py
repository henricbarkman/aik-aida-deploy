"""What comes with a building part: its layers, in the project list (DECISIONS 21).

Tiles are laid in adhesive on a waterproofing on a board. Counting only the tile
leaves out a large part of a bathroom, and a part left out of the baseline is left
out of every comparison. So intake expands a part that matches a recipe, and the
user sees the layers as rows of their own and can remove them.

Layers come in two kinds:

- A layer Aida can compare (waterproofing, levelling compound, a plasterboard,
  studs, insulation) is an ordinary component. It goes through the baseline and the
  alternatives like any other, under a name the rest of Aida already reads.
- What has no alternatives worth comparing (adhesive and grout, battens, underlay,
  fixings) is one bundle per part: `recipe_bundle` on the component. Its figure is
  deterministic, each amount times Boverket's value, no model and no ranking.

A layer already named in the project is not added again, so nothing counts twice.
Every amount is a typical one and marked as an estimate, from Boverket's conversion
factors in massor and ordinary product data sheets. Nothing here is read from
another tool's catalog.
"""

from __future__ import annotations

from dataclasses import dataclass

from aida.data.bevara import DOES_NOT_EXIST, is_new_wall, is_wood
from aida.data.massor import MATERIALS

# The figures per kg live in massor.MATERIALS, checked against Boverket's file.
_BATTEN_KG_M = 0.025 * 0.048 * MATERIALS["sågat"].density_kg_m3


@dataclass(frozen=True)
class Item:
    material: str  # key in massor.MATERIALS
    kg: float  # per unit of the component
    how: str  # the amount, in words


@dataclass(frozen=True)
class Bundle:
    key: str
    name: str  # the component's name in the project list
    unit: str
    category: str
    items: tuple[Item, ...]

    @property
    def per_unit(self) -> float:
        return sum(i.kg * MATERIALS[i.material].a1a3 for i in self.items)


def _sv(x: float) -> str:
    return f"{round(x, 2):g}".replace(".", ",")


def _battens(metres: float, spacing: str) -> Item:
    return Item("sågat", metres * _BATTEN_KG_M,
                f"läkt 25x48, {_sv(metres)} m per m² ({spacing}), {_sv(_BATTEN_KG_M)} kg/m")


_TILE_FIXING = Bundle(
    "kakel-fastmassa", "Fästmassa och fog till kakel", "m2", "kakel",
    (Item("murbruk", 4.0, "fästmassa, 4 kg/m², cementbaserad, räknad som murbruk"),
     Item("murbruk", 0.5, "fog, 0,5 kg/m², cementbaserad, räknad som murbruk")))
_WINDOW_KIT = Bundle(
    "fonster-tillbehor", "Montage och foder till fönster", "st", "fönster",
    (Item("fogmassa", 0.3, "fogmassa, 0,3 kg"),
     Item("glasull", 0.2, "drevning, 0,2 kg"),
     Item("hyvlat", 2.4, "foder runt om på båda sidor, 2,4 kg"),
     Item("plåt", 1.4, "fönsterbleck, 1,4 kg")))
_ROOF_TILES = Bundle(
    "tak-pannor", "Underlagspapp, läkt och infästning till takpannor", "m2", "tak",
    (Item("underlagspapp", 4.0, "underlagspapp, 4 kg/m²"),
     _battens(3.6, "bärläkt c360 och ströläkt c1200"),
     Item("spik", 0.1, "spik och clips, 0,1 kg/m²")))
_ROOF_SHEET = Bundle(
    "tak-underlag", "Underlagspapp och infästning till yttertak", "m2", "tak",
    (Item("underlagspapp", 4.0, "underlagspapp, 4 kg/m²"),
     Item("spik", 0.1, "spik och infästning, 0,1 kg/m²")))
_FACADE_PANEL = Bundle(
    "fasad-laekt", "Läkt och infästning till fasadbeklädnad", "m2", "fasadskikt",
    (_battens(1.67, "c600"), Item("spik", 0.05, "spik eller skruv, 0,05 kg/m²")))
_FACADE_BRICK = Bundle(
    "fasad-bruk", "Murbruk till tegelfasad", "m2", "fasadskikt",
    (Item("murbruk", 35.0, "murbruk, 35 kg/m²"),))

BUNDLES: dict[str, Bundle] = {b.key: b for b in (
    _TILE_FIXING, _WINDOW_KIT, _ROOF_TILES, _ROOF_SHEET, _FACADE_PANEL, _FACADE_BRICK)}
_BY_NAME = {b.name.lower(): b.key for b in BUNDLES.values()}

_AREA = ("m2", "m²", "kvm")


def bundle_text(bundle: Bundle, unit: str) -> str:
    """The baseline row's text: every amount and what it costs."""
    parts = "; ".join(
        f"{i.how} × {_sv(MATERIALS[i.material].a1a3)} kg CO2e/kg ({MATERIALS[i.material].record})"
        for i in bundle.items)
    return (f"Lager som följer med byggdelen. {parts} = {_sv(bundle.per_unit)} kg CO2e/{unit}. "
            f"Mängderna är uppskattningar av typisk åtgång, värdena är Boverkets. "
            f"Aida jämför inga alternativ för de här lagren.")


def _names(components: list[dict]) -> str:
    return " | ".join(str(c.get("name") or "").lower() for c in components)


def _has(text: str, *words: str) -> bool:
    return any(w in text for w in words)


def _category(comp: dict) -> str:
    return str(comp.get("category") or "").strip().lower()


def _name(comp: dict) -> str:
    return str(comp.get("name") or "").lower()


# Whether a layer is already a row depends on what the row belongs to: a roof's
# membrane is not the bathroom's, and a door's trim is not the window's.
def _on_roof(comp: dict) -> bool:
    return _category(comp) == "tak" or "tak" in _name(comp)


def _on_facade(comp: dict) -> bool:
    return _category(comp) in ("fasadskikt", "yttervägg") or _has(_name(comp), "fasad", "yttervägg")


def _indoors(components: list[dict]) -> str:
    return _names([c for c in components if not _on_roof(c) and not _on_facade(c)])


def _wall_boards_named(components: list[dict]) -> bool:
    """Is the wet room's board already a row? Only a wall row or one that says wet
    room counts: a plasterboard ceiling or a facade board is not the board behind
    the tile, and must not take it away."""
    for comp in components:
        # A worktop or a floor board in the bathroom is no wall board, and a new wall
        # counted as its board is the wall itself, wherever it stands.
        name = _name(comp).replace("bänkskiv", "").replace("golvskiv", "")
        if name.startswith("gipsskiva 13 mm,"):
            continue
        if _category(comp) == "innervägg" or _has(name, "våtrum", "badrum", "dusch"):
            if _has(name, "gips", "skiva", "skivor"):
                return True
    return False


def _wet_room_layers(comp: dict, all_names: str, floor: bool, board_named: bool) -> list[dict]:
    name = str(comp.get("name") or "").lower()
    if _has(name, "kök", "stänk"):  # a kitchen splashback has no waterproofing behind it
        return [_bundle(_TILE_FIXING, comp)] if "fästmassa" not in all_names else []
    layers = []
    if not _has(all_names, "tätskikt", "tätskiktsduk", "membran"):
        layers.append(_plain(comp, "Tätskikt", "tätskikt"))
    if floor:
        if not _has(all_names, "avjämn", "flytspackel", "golvspackel"):
            layers.append(_plain(comp, "Avjämning 10 mm", "golv"))
    elif not board_named:
        layers.append(_plain(comp, "Våtrumsgips 13 mm", "innervägg"))
    if "fästmassa" not in all_names:
        layers.append(_bundle(_TILE_FIXING, comp))
    return layers


def _plain(parent: dict, name: str, category: str, quantity: float | None = None,
           unit: str | None = None) -> dict:
    return {"name": name, "category": category, "unit": unit or parent.get("unit"),
            "quantity": parent.get("quantity") if quantity is None else quantity,
            "_parent": parent}


def _bundle(bundle: Bundle, parent: dict) -> dict:
    return {**_plain(parent, bundle.name, bundle.category), "recipe_bundle": bundle.key}


def _layers_of(comp: dict, components: list[dict]) -> list[dict]:
    name = str(comp.get("name") or "").lower()
    unit = str(comp.get("unit") or "").strip().lower()
    category = str(comp.get("category") or "").strip().lower()
    rest = [c for c in components if c is not comp]
    others = _names(rest)
    if unit in _AREA and (category == "kakel" or "klinker" in name):
        if _has(name, "fog", "fästmassa"):
            return []
        return _wet_room_layers(comp, _indoors(rest), _has(name, "klinker", "golv"),
                                _wall_boards_named(rest))
    if category == "fönster" and unit == "st":
        windows = _names([c for c in rest if _category(c) == "fönster"])
        if _has(name, "ruta", "renover", "målning", "måla", "foder", "bleck", "bänk") or (
                _has(windows, "foder", "bleck")
                or _has(others, "fönsterbleck", "fönsterfoder", "drevning")):
            return []
        return [_bundle(_WINDOW_KIT, comp)]
    if category == "tak" and unit in _AREA:
        roof = _names([c for c in rest if _on_roof(c)])
        if _has(name, "målning", "måla", "underlag", "läkt", "råspont") or (
                _has(roof, "läkt") or _has(others, "underlagspapp", "takpapp under")):
            return []
        if _has(name, "panna", "pannor", "takpann"):
            return [_bundle(_ROOF_TILES, comp)]
        return [_bundle(_ROOF_SHEET, comp)]
    if category == "fasadskikt" and unit in _AREA:
        facade = _names([c for c in rest if _on_facade(c)])
        if _has(name, "målning", "måla", "läkt") or _has(facade, "läkt"):
            return []
        if "tegel" in name:
            # Repointing or repairing a brick wall does not lay it again.
            if _has(name, "omfog", "fog", "laga", "renover"):
                return []
            return [] if _has(others, "murbruk") else [_bundle(_FACADE_BRICK, comp)]
        if _has(name, "panel", "fibercement", "skiv") or is_wood(name):
            return [_bundle(_FACADE_PANEL, comp)]
        return []
    if category == "innervägg" and unit in _AREA:
        walls = _names([c for c in rest if not _on_roof(c) and not _on_facade(c)
                        and not _has(_name(c), "vind", "bjälklag", "golv")])
        return _new_wall_layers(comp, name, walls)
    return []


def _new_wall_layers(comp: dict, name: str, others: str) -> list[dict]:
    """A new wall only: on an existing one, a wall row is a surface. The row is
    renamed to the one board it is counted as, so the whole wall is the sum."""
    if not is_new_wall(name, comp.get("existing_condition") or ""):
        return []
    # Nothing stands there to keep, and the layers inherit that.
    comp["existing_condition"] = DOES_NOT_EXIST
    if _has(name, "glasparti", "betong", "tegel", "regel", "isoler", "målning"):
        return []
    if _has(others, "regel", "reglar", "gipsskiva", "gips", "isolering", "mineralull"):
        return []
    quantity = comp.get("quantity") or 0
    # "ena sidan" in the name: without it the model at times read the row as both
    # sides of the wall, and the other side is a row of its own (164 kg against 78 on
    # two runs of the same project).
    comp["name"] = f"Gipsskiva 13 mm, ena sidan, {comp.get('name')}"
    return [_plain(comp, "Gipsskiva 13 mm, andra sidan av väggen", "innervägg"),
            _plain(comp, "Stålreglar 70 mm", "stomme", quantity=round(quantity * 3, 1), unit="lm"),
            _plain(comp, "Isolering mineralull 70 mm", "isolering")]


def _same_unit(a, b) -> bool:
    a, b = str(a or "").strip().lower(), str(b or "").strip().lower()
    return a == b or (a in _AREA and b in _AREA)


def expand(data: dict) -> dict:
    """Add the layers to an intake result, in place. Runs once per intake; a row a
    previous round already added is recognised by its name and not added again."""
    components = [c for c in data.get("components") or [] if isinstance(c, dict)]
    for comp in components:
        key = _BY_NAME.get(str(comp.get("name") or "").strip().lower())
        # In another unit than the recipe's the amounts per unit do not apply.
        if key and _same_unit(BUNDLES[key].unit, comp.get("unit")):
            comp["recipe_bundle"] = key
    added: list[dict] = []
    merged: dict[str, dict] = {}
    for comp in list(components):
        if comp.get("recipe_bundle"):
            continue
        for layer in _layers_of(comp, components):
            same = merged.get(layer["name"])
            if same is not None:
                same["quantity"] = (same.get("quantity") or 0) + (layer.get("quantity") or 0)
                same["usage_context"] += f" {comp.get('name')}."
                # Several parts can share a layer, so taking one away must not take it.
                same["recipe_parent"] = ",".join(
                    p for p in (same.get("recipe_parent"), comp.get("id")) if p)
                continue
            parent = layer.pop("_parent")
            layer.update(
                quantity_source="estimated",
                recipe_parent=parent.get("id") or "",
                existing_condition=parent.get("existing_condition") or "",
                usage_context=f"Lager som följer med: {parent.get('name')}.",
            )
            merged[layer["name"]] = layer
            added.append(layer)
    if not added:
        return data
    used = [int(str(c.get("id"))[1:]) for c in components
            if str(c.get("id") or "")[1:].isdigit()]
    next_id = max(used, default=0) + 1
    for layer in added:
        layer["id"] = f"c{next_id}"
        next_id += 1
    data["components"] = components + added
    needs = data.setdefault("needs_analysis", {})
    if isinstance(needs, dict):
        names = ", ".join(layer["name"] for layer in added)
        assumptions = needs.setdefault("assumptions", [])
        if isinstance(assumptions, list):
            assumptions.append(
                f"Aida har lagt till lagren som följer med byggdelarna: {names}. Mängderna är "
                f"uppskattade. Ta bort en rad om lagret inte ska göras om; ändrar du mängden "
                f"på en byggdel följer lagren inte med av sig själva.")
    return data
