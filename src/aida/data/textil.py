"""Rugs and curtains: the area one piece covers, read from its name (HENRIC-3366).

The EPDs for both are declared per m² (carpet by the roll, curtain fabric), and
a rug or a curtain is bought per piece. "Matta 2x3 m" is one rug of 6 m², so a
per-m² figure times 6 is that rug's figure. The size is read from the name and
only there, the way a stud's section is (unit_conversion.member_volume_per_unit):
no stated size, no area, and the caller asks for one. Nothing is assumed about a
typical rug or a typical curtain.

What the size means. For a rug it is the rug. For a curtain it is the curtain's
own width and drop as sold ("Gardin 140x250 cm" is a panel 140 cm wide), which is
the fabric in it apart from hems. It is not the window: a curtain gathered across
a window holds more fabric than the window's area, and how much more depends on
the heading, so a window size is not turned into fabric here.
"""

from __future__ import annotations

import math
import re

# The two kinds of loose textile (climate_data.furniture_subcategory).
TEXTILE_KINDS = frozenset({"matta", "gardin"})

_NUM = r"(\d+(?:[.,]\d+)?)"
_UNIT = r"(mm|cm|meter|m)\b"
# "2x3 m", "2 x 3 m", "200×300 cm", "1,4x2,5m". The unit after the second
# number is the unit of both, which is how sizes are written in Swedish shops.
# A unit after each number ("2 m x 3 m") is read too.
_RECT_RE = re.compile(_NUM + r"\s*(mm|cm|meter|m)?\s*[x×*]\s*" + _NUM + r"\s*" + _UNIT,
                      re.IGNORECASE)
# A round rug: "ø 200 cm", "Ø200 cm", "diameter 2 m", "diam. 160 cm".
_ROUND_RE = re.compile(r"(?:ø|Ø|⌀|diameter|diam\.?)\s*" + _NUM + r"\s*" + _UNIT,
                       re.IGNORECASE)
_TO_M = {"mm": 0.001, "cm": 0.01, "m": 1.0, "meter": 1.0}
# A size with no unit ("Matta 200x300"): 200 what? Asked, not guessed.
_BARE_PAIR_RE = re.compile(_NUM + r"\s*[x×*]\s*" + _NUM + r"(?!\s*(?:mm|cm|meter|m)\b)")

# Outside these a reading is more likely a typo or another number than a rug or
# a curtain: a side under 10 cm or over 20 m, an area over 100 m². A wall-to-
# wall carpet of 120 m² is a floor (golv), not a loose rug.
_MIN_SIDE_M = 0.1
_MAX_SIDE_M = 20.0
_MAX_AREA_M2 = 100.0


def _num(text: str) -> float:
    return float(text.replace(",", "."))


def _fmt(x: float) -> str:
    """2.0 -> "2", 1.4 -> "1,4": the Swedish way, without trailing zeros."""
    return f"{x:g}".replace(".", ",")


def piece_area_m2(name: str) -> tuple[float | None, str]:
    """(m² per piece, label) from the size the name states, or (None, why).

    `why` is "" when the name states no size at all, and a Swedish sentence
    when it states one that cannot be used (two sizes, a size outside what a
    rug or a curtain can be). The label says where the area came from:
    "2×3 m = 6 m² per styck".
    """
    text = name or ""
    rects = _RECT_RE.findall(text)
    rounds = _ROUND_RE.findall(text)
    if len(rects) + len(rounds) > 1:
        return None, ("Namnet anger flera mått, så det går inte att avgöra hur stor "
                      "en styck är.")
    if rects:
        a, unit_a, b, unit_b = rects[0]
        side_a = _num(a) * _TO_M[(unit_a or unit_b).lower()]
        side_b = _num(b) * _TO_M[unit_b.lower()]
        area = side_a * side_b
        sides = (side_a, side_b)
        label = f"{_fmt(side_a)}×{_fmt(side_b)} m = {_fmt(round(area, 2))} m² per styck"
    elif rounds:
        d, unit = rounds[0]
        diameter = _num(d) * _TO_M[unit.lower()]
        area = math.pi * (diameter / 2) ** 2
        sides = (diameter,)
        label = f"ø {_fmt(diameter)} m = {_fmt(round(area, 2))} m² per styck"
    elif _BARE_PAIR_RE.search(text):
        return None, ("Måttet i namnet saknar enhet. Skriv det med m, cm eller mm, "
                      "till exempel \"2x3 m\" eller \"200x300 cm\".")
    else:
        return None, ""
    if any(not (_MIN_SIDE_M <= s <= _MAX_SIDE_M) for s in sides) or area > _MAX_AREA_M2:
        return None, (f"Måttet i namnet ({label.split(' = ')[0]}) är inte en rimlig "
                      f"storlek för en matta eller en gardin.")
    return round(area, 4), label
