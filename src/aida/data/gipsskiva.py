"""Plasterboard thickness: read it, and compare boards only within one thickness.

HENRIC-3399. A gypsum ceiling (HENRIC-3370) and a plasterboard wall met all 63
boards in the catalog, 6.4 to 18 mm, in one pool. On production "Nytt innertak
av gips 40 m2" got Boverket's standard 12.5 mm board as baseline and a 9.5 mm
board as its first alternative, so part of the "saving" was a thinner board
with a different strength and fire class, not a cleaner factory.

Rules:
- A board's thickness is read from its name, the way the maker states it
  ("12.5mm", "12,5 mm", "DURAGIPS A 9,5", "BA13", "Habito 13"), and only when
  the name states none from the EPD's own ILCD data (layer thickness, read from
  data.environdec.com 2026-10-02 and kept in EPD_LAYER_THICKNESS_MM with its
  registration number). A board with neither is not compared: an unknown
  thickness is not the standard one.
- Thicknesses are compared as nominal classes, because makers state the same
  board in three ways: 12.5 mm (EN 520), 12.7 mm (1/2", ASTM C1396) and 13 mm
  (the trade name: Swedish "GN 13", French "BA13", Greek and Spanish "Habito
  13", whose own EPD data gives 12.5 mm). Likewise 15/15.9/16.
- A component with a thickness in its name meets boards of that class. One
  without meets the standard class, 12.5 to 13 mm, and the row says so.
"""

from __future__ import annotations

import re

# Nominal classes, mm. Each is (label, low, high): a board is in a class when
# its stated thickness lies within [low, high]. The gaps between classes are
# deliberate: a 11 mm or a 14 mm board is no standard size and meets nothing.
_CLASSES: tuple[tuple[str, float, float], ...] = (
    ("6,5", 6.0, 6.5),
    ("9,5", 9.0, 10.0),
    ("12,5", 12.0, 13.0),
    ("15", 15.0, 16.0),
    ("18", 18.0, 18.0),
    ("25", 25.0, 25.4),
)
STANDARD_CLASS = "12,5"
STANDARD_TEXT = "12,5 till 13 mm"

# Thickness from the EPD's ILCD dataset (processInformation ... materialProperties,
# "layer thickness"), for boards whose name states none. Read 2026-10-02 from
# https://data.environdec.com/resource/processes/<uuid>?format=json&view=extended.
# Keyed by registration number. Boards the dataset gives no thickness for
# (Gyproc Robust, Vindtæt, Studio, ErgoLite SE, Normal FI, Rigips Fonic,
# Gyprock) are not listed and are not compared.
EPD_LAYER_THICKNESS_MM: dict[str, float] = {
    "EPD-IES-0027337:002": 12.5,             # Chapa de Drywall Hardboard (Knauf)
    "EPD-IES-0029248:001": 12.5,             # Gyproc ErgoLite (Saint-Gobain Finland)
    "EPD-IES-0000429:003 (S-P-00429)": 12.5,  # Gyproc Loftplader (Denmark)
    "EPD-IES-0000428:004 (S-P-00428)": 12.5,  # Gyproc Normal (Denmark)
}

_MM_RE = re.compile(r"(?<![\d.,])(\d{1,2}(?:[.,]\d{1,2})?)\s*mm\b")
# Standard decimal sizes stated without "mm" ("DURAGIPS A 12,5", "RINOVA A 9,5").
_DECIMAL_RE = re.compile(r"(?<![\d.,])(6[.,]4|6[.,]5|9[.,]5|12[.,]5|12[.,]7|15[.,]9)(?![\d.,])")
_BA_RE = re.compile(r"\bba\s?(13|15|18|25)\b")
# A bare whole number in a board's name is its nominal thickness ("Habito 13",
# "Fireline 15", "Gipsskiva 2x13"). Only 6 to 25: "2x" before it is the number
# of layers, and a registration number is never a free-standing word.
_BARE_RE = re.compile(r"(?:(?<=\dx)|(?<![\w.,-]))(6|9|13|15|16|18|25)(?![\w.,-]|\s*x\b)")


def _to_mm(text: str) -> float:
    return float(text.replace(",", "."))


def thickness_from_name(name: str) -> float | None:
    """The board thickness a name states, in mm, or None."""
    text = (name or "").lower()
    for rx, scale in ((_MM_RE, None), (_DECIMAL_RE, None), (_BA_RE, "ba"), (_BARE_RE, None)):
        for m in rx.finditer(text):
            mm = _to_mm(m.group(1))
            if scale == "ba":
                mm = {13: 12.5}.get(int(mm), mm)
            if thickness_class(mm):
                return mm
    return None


def board_thickness(row: dict) -> tuple[float | None, str]:
    """(mm, source) for a catalog row: the name first, then the EPD's own data."""
    mm = thickness_from_name(str(row.get("name") or ""))
    if mm is not None:
        return mm, "enligt produktnamnet"
    mm = EPD_LAYER_THICKNESS_MM.get(str(row.get("reg_no") or ""))
    if mm is not None:
        return mm, "enligt EPD:ns data"
    return None, ""


def thickness_class(mm: float | None) -> str:
    """The nominal class of a thickness ("12,5"), or "" outside every class."""
    if mm is None:
        return ""
    for label, low, high in _CLASSES:
        if low <= mm <= high:
            return label
    return ""


def component_class(name: str) -> tuple[str, bool]:
    """(class, stated) for a component: the class its name states, or the
    standard class and stated=False. A wall thickness ("Gipsvägg 95 mm") is no
    board class and counts as unstated."""
    cls = thickness_class(thickness_from_name(name))
    if cls:
        return cls, True
    return STANDARD_CLASS, False


def format_mm(mm: float) -> str:
    return f"{mm:g}".replace(".", ",")
