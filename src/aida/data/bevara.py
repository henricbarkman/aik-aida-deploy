"""Keep and repair: an alternative of its own, before reuse and new (DECISIONS 20).

The climate best thing to do with a building part that still works is to leave it
there. Until 2026-11-02 Aida had no way to say so: every component compared new
products and reuse with a new conventional one. The first thing a property owner asks
("does it need replacing at all?") had no row.

A repair has a real figure, and it is counted the same way as everything else: the
material the repair adds times Boverket's value per kg.

    kg per component unit that the repair adds  x  Boverket's A1-A3 per kg

The amounts are typical ones and marked as estimates; a window's new pane has its
own Boverket weight, a coat of paint is the spread rate unit_conversion already
uses for the baseline. A kind of part with no repair we can put material on (a
toilet, a lift, a lamp) has no row: "keep it" with no material would be zero, and
a zero in the table reads as a calculated figure.

The row says what it assumes: that the part is there and can be repaired. Intake
asks about the condition, and a component the user says must be replaced, or that
does not exist yet, gets no row at all.

Boverket 02.07.000, read 2026-11-02. Boverket has no indoor paint and no floor
treatment: "Utomhusfärg, vattenburen akryl" is the proxy for both, and the row names
it. Source for the spread rate of paint: unit_conversion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from aida.data.massor import BOVERKET_VERSION, DOOR_AREA_M2, WINDOW_AREA_M2

SOURCE_PREFIX = "[Bevara]"
KEPT_LAYER = "lager"  # a recipe's layer that stays as it is (DECISIONS 21)


@dataclass(frozen=True)
class Added:
    """One material the repair adds, per component unit."""

    record: str  # Boverket's name
    a1a3: float
    a4: float
    a5: float
    amount_kg: float
    how: str  # the amount, in words


@dataclass(frozen=True)
class Repair:
    key: str  # goes in the alternative's source, so a saved selection finds it again
    name: str
    unit: tuple[str, ...]  # the component units it applies to
    added: tuple[Added, ...]
    assumes: str

    @property
    def per_unit(self) -> float:
        return sum(a.amount_kg * a.a1a3 for a in self.added)

    @property
    def a4_per_unit(self) -> float:
        return sum(a.amount_kg * a.a4 for a in self.added)

    @property
    def a5_per_unit(self) -> float:
        return sum(a.amount_kg * a.a5 for a in self.added)


# Records from Boverket 02.07.000: (name, A1-A3, A4, A5.1).
_PAINT = ("Utomhusfärg, vattenburen akryl", 2.5, 0.045, 0.127)
_SILICATE = ("Silikatfärg", 1.2, 0.045, 0.0618)
_MORTAR = ("Murbruk A (CS IV)", 0.199, 0.045, 0.0147)
_RENDER = ("Mur- och putsbruk B (CS III)", 0.167, 0.045, 0.0127)
_PANE_2 = ("Isolerruta, dubbelglas (4-16-4)", 1.56, 0.144, 0.0)
_UNDERLAY = ("Underlagspapp", 0.68, 0.045, 0.0448)
_TOPCOAT = ("Ytpapp", 0.56, 0.045, 0.0373)

# Spread rates, kg/m². Two coats outdoors and indoors: unit_conversion's midpoint
# is 0,45 and its range is 0,5-0,7 outdoors and 0,28-0,40 indoors.
_OUTDOOR_PAINT = 0.6
_INDOOR_PAINT = 0.34
_FLOOR_TREATMENT = 0.1  # one coat of oil or lacquer, typical data sheet rates
_PANE_SHARE = 0.8  # the pane's share of a window's area, typical


def _sv(x: float) -> str:
    return f"{round(x, 3):g}".replace(".", ",")


def _added(record: tuple, amount: float, how: str) -> Added:
    name, a1a3, a4, a5 = record
    return Added(name, a1a3, a4, a5, amount, how)


_AREA = ("m2", "m²", "kvm")
_COUNT = ("st", "styck", "stk", "pcs")

_WINDOW_PANE_2 = Repair(
    "fonster-ruta-2", "Behåll bågen, byt till ny isolerruta med två glas", _COUNT,
    (_added(_PANE_2, WINDOW_AREA_M2 * _PANE_SHARE * 20.0,
            f"antagen ruta {_sv(WINDOW_AREA_M2 * _PANE_SHARE)} m² × 20 kg/m² (Boverket)"),),
    "bågar och karmar är friska och tål den nya rutans vikt")
_WINDOW_PAINT = Repair(
    "fonster-mala", "Renovera och måla om befintligt fönster", _COUNT,
    (_added(_PAINT, WINDOW_AREA_M2 * 0.7 * _OUTDOOR_PAINT,
            f"antagen målad yta {_sv(WINDOW_AREA_M2 * 0.7)} m² × {_sv(_OUTDOOR_PAINT)} kg/m²"),),
    "karmar och bågar går att laga och glaset behålls")
_DOOR_PAINT = Repair(
    "dorr-mala", "Renovera och måla om befintlig dörr", _COUNT,
    (_added(_PAINT, DOOR_AREA_M2 * 2 * _INDOOR_PAINT,
            f"två sidor, {_sv(DOOR_AREA_M2)} m² per sida × {_sv(_INDOOR_PAINT)} kg/m²"),),
    "dörren uppfyller kraven på ljud och brand där den sitter")
_FACADE_PAINT = Repair(
    "fasad-mala", "Laga och måla om befintlig träfasad", _AREA,
    (_added(_PAINT, _OUTDOOR_PAINT, f"två strykningar, {_sv(_OUTDOOR_PAINT)} kg/m²"),),
    "panelen är frisk utom enstaka bräder")
_FACADE_RENDER = Repair(
    "fasad-puts", "Laga puts och avfärga befintlig fasad", _AREA,
    (_added(_RENDER, 2.0, "lagning av tio procent av ytan, 2 kg/m² räknat på hela fasaden"),
     _added(_SILICATE, _OUTDOOR_PAINT, f"två strykningar, {_sv(_OUTDOOR_PAINT)} kg/m²")),
    "putsen sitter fast utom på mindre partier")
_BRICK_REPOINT = Repair(
    "fasad-fogar", "Foga om befintlig tegelfasad", _AREA,
    (_added(_MORTAR, 5.0, "fogbruk, 5 kg/m²"),),
    "tegelstenen är hel och det är fogarna som behöver lagas")
_WALL_PAINT = Repair(
    "vagg-mala", "Spackla och måla om befintlig vägg", _AREA,
    (_added(_PAINT, _INDOOR_PAINT, f"två strykningar, {_sv(_INDOOR_PAINT)} kg/m²"),),
    "väggen står kvar och klarar kraven på ljud, brand och fukt")
_FLOOR_SAND = Repair(
    "golv-slipa", "Slipa och behandla befintligt trägolv", _AREA,
    (_added(_PAINT, _FLOOR_TREATMENT, f"olja eller lack, {_sv(_FLOOR_TREATMENT)} kg/m²"),),
    "golvet har kvar slitskikt nog för en slipning")
_FLOOR_CARE = Repair(
    "golv-laga", "Laga och behandla befintligt golv", _AREA,
    (_added(_PAINT, _FLOOR_TREATMENT, f"ytbehandling, {_sv(_FLOOR_TREATMENT)} kg/m²"),),
    "mattan sitter fast och är hel utom mindre skador")
_TILE_REGROUT = Repair(
    "kakel-fogar", "Foga om och laga befintligt kakel eller klinker", _AREA,
    (_added(_MORTAR, 0.5, "fogbruk, 0,5 kg/m²"),),
    "plattorna sitter fast och tätskiktet bakom är friskt")
_ROOF_UNDERLAY = Repair(
    "tak-lagga-om", "Lägg om befintliga takpannor på ny underlagspapp", _AREA,
    (_added(_UNDERLAY, 4.0, "underlagspapp, 4 kg/m² (Boverket)"),),
    "pannorna är hela och går att lägga tillbaka")
_ROOF_TOPCOAT = Repair(
    "tak-papp", "Lägg ny ytpapp på befintligt papptak", _AREA,
    (_added(_TOPCOAT, 5.0, "ytpapp, 5 kg/m² (Boverket)"),),
    "underlaget är torrt och friskt")
_ROOF_PAINT = Repair(
    "tak-mala", "Tvätta och måla om befintligt plåttak", _AREA,
    (_added(_PAINT, _OUTDOOR_PAINT, f"två strykningar, {_sv(_OUTDOOR_PAINT)} kg/m²"),),
    "plåten är tät och utan genomrostning")
_CEILING_PAINT = Repair(
    "undertak-mala", "Behåll befintligt undertak och måla om det", _AREA,
    (_added(_PAINT, _INDOOR_PAINT, f"två strykningar, {_sv(_INDOOR_PAINT)} kg/m²"),),
    "plattorna är hela och klarar kraven på ljud och brand")
_RADIATOR_PAINT = Repair(
    "radiator-mala", "Behåll befintlig radiator, rengör och måla om", _COUNT,
    (_added(_PAINT, 2.0 * _INDOOR_PAINT, f"antagen yta 2 m² × {_sv(_INDOOR_PAINT)} kg/m²"),),
    "radiatorn är tät och passar den värme huset ska ha")
_KITCHEN_PAINT = Repair(
    "inredning-mala", "Behåll inredningen och måla om luckorna", _COUNT,
    (_added(_PAINT, 1.0 * _INDOOR_PAINT, f"antagen yta 1 m² × {_sv(_INDOOR_PAINT)} kg/m²"),),
    "stommen i skåpen är frisk")

_REPAIRS = (_WINDOW_PANE_2, _WINDOW_PAINT, _DOOR_PAINT, _FACADE_PAINT, _FACADE_RENDER,
            _BRICK_REPOINT, _WALL_PAINT, _FLOOR_SAND, _FLOOR_CARE, _TILE_REGROUT,
            _ROOF_UNDERLAY, _ROOF_TOPCOAT, _ROOF_PAINT, _CEILING_PAINT, _RADIATOR_PAINT,
            _KITCHEN_PAINT)
BY_KEY: dict[str, Repair] = {r.key: r for r in _REPAIRS}


def _has(text: str, *words: str) -> bool:
    return any(w in text for w in words)


# A row called this is new whatever the condition says. Not "Nya fönster" or "Nytt
# tak": those replace something that is there, and that is the choice this module is for.
_NEW_WALL = ("ny vägg", "nya vägg", "ny innervägg", "nya innervägg")


def is_new_wall(name: str, condition: str = "") -> bool:
    return ((condition or "").strip().lower() == DOES_NOT_EXIST
            or _has((name or "").lower(), *_NEW_WALL))


def is_wood(text: str) -> bool:
    """"Träpanel" and "lärkträ", not "Fasadputs, sträv"."""
    return re.search(r"(?<!s)trä", text or "") is not None


def repair_for(category: str, name: str, unit: str) -> Repair | None:
    """The repair that fits a component, or None when none fits.

    `name` is the component's name with the baseline's assumed material appended.
    What decides is what the part is. A category that holds several kinds of part
    takes its repair from the name, and no match gives no row: painting a concrete
    wall is not repairing a brick one.
    """
    text = (name or "").lower()
    unit = (unit or "").strip().lower()
    repair: Repair | None = None
    if category == "fönster":
        repair = _WINDOW_PAINT if _has(text, "renover", "måla") else _WINDOW_PANE_2
    elif category == "dörr":
        repair = None if _has(text, "ytterdörr", "entré", "brand") else _DOOR_PAINT
    elif category == "fasadskikt":
        # The baseline's material is wood whatever the facade is (DECISIONS 24), so
        # what the row itself names goes first: fibre cement is no wooden facade.
        if _has(text, "fibercement", "eternit", "plåt", "skiv"):
            repair = None
        elif _has(text, "puts"):
            repair = _FACADE_RENDER
        elif _has(text, "tegel"):
            repair = _BRICK_REPOINT
        elif _has(text, "panel") or is_wood(text):
            repair = _FACADE_PAINT
    elif category == "yttervägg":
        if _has(text, "tegel"):
            repair = _BRICK_REPOINT
        elif _has(text, "puts"):
            repair = _FACADE_RENDER
    elif category == "innervägg":
        repair = None if is_new_wall(text) or _has(text, "tegel", "betong") else _WALL_PAINT
    elif category == "golv":
        if _has(text, "parkett", "trägolv", "massivgolv", "plank"):
            repair = _FLOOR_SAND
        elif _has(text, "linoleum", "vinyl", "plastmatta", "plastgolv", "pvc", "gummi"):
            repair = _FLOOR_CARE
        elif _has(text, "klinker"):
            repair = _TILE_REGROUT
    elif category == "kakel":
        repair = _TILE_REGROUT
    elif category == "tak":
        if _has(text, "plåt", "falsat"):
            repair = _ROOF_PAINT
        elif _has(text, "takpann", "panna", "pannor", "tegel"):
            repair = _ROOF_UNDERLAY
        elif _has(text, "papp"):
            repair = _ROOF_TOPCOAT
    elif category == "undertak":
        repair = _CEILING_PAINT
    elif category == "radiator":
        repair = _RADIATOR_PAINT
    elif category == "fast_inredning":
        repair = _KITCHEN_PAINT if _has(text, "kök", "skåp", "lucka", "luckor") else None
    if repair is None or unit not in repair.unit:
        return None
    return repair


# existing_condition on a Component, as intake reads it from the user's words.
CAN_BE_REPAIRED = "går att laga"
MUST_BE_REPLACED = "måste bytas"
DOES_NOT_EXIST = "finns inte"
CONDITIONS = (CAN_BE_REPAIRED, MUST_BE_REPLACED, DOES_NOT_EXIST)


def offered(condition: str) -> bool:
    """Whether the row is shown. Not when the user said the part must go, or that
    there is nothing there. An unstated condition is shown with its assumption, so
    the option is on the table in the phase when it can still be taken."""
    return (condition or "").strip().lower() not in (MUST_BE_REPLACED, DOES_NOT_EXIST)


def reasoning(repair: Repair, condition: str, unit: str) -> str:
    """The row's text: what is done, the arithmetic, and what it assumes."""
    parts = "; ".join(
        f"{a.how} × {_sv(a.a1a3)} kg CO2e/kg ({a.record})" for a in repair.added)
    told = (condition or "").strip().lower() == CAN_BE_REPAIRED
    assume = (f"Du har sagt att det går att laga. Det förutsätter att {repair.assumes}."
              if told else
              f"Förutsätter att delen finns kvar och att {repair.assumes}. Skicket är "
              f"inte angivet, så kontrollera det innan siffran används.")
    return (f"Befintlig del behålls och lagas. Klimatvärdet är det som tillförs: {parts} "
            f"= {_sv(repair.per_unit)} kg CO2e/{unit} (Boverkets klimatdatabas "
            f"{BOVERKET_VERSION}; åtgången är en uppskattning). Arbetet och rivningen av "
            f"det gamla ingår inte. {assume}")
