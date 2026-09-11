"""Claims: where every number on the open sheet comes from (orchestration-redesign §13.3).

Henric, 2026-09-11: numbers should preferably have a source, estimates are
allowed, but it has to be stated which is which. This module is where that rule
lives, so that it holds whether or not a prompt remembers it.

A claim is one number with its origin attached. Block text never carries a bare
climate figure, price, lifespan or share; it carries a reference, `{{c1}}`, and
the claim with that id says what the number is and what it rests on:

    source    a lookup in data Aida ships (EPD, Boverket, NollCO2, the
              guidelines, the method description, Palats, a web price)
    estimate  no source, and `grounds` says what the figure rests on instead
    user      the user wrote it
    derived   computed HERE from other claims, never by the model

Three rules follow, and they are the reason this is a module and not a
paragraph in a prompt:

1. An estimate without grounds is rejected. "30 år" is not a claim until it
   says why thirty.
2. A derived number inherits its weakest input. 4.4 kg divided by an estimated
   30 years is an estimate, even though 4.4 has a source. The value and the
   marking are both computed in `resolve`, so a marking cannot fall off at one
   step of a chain, and the model cannot state a result the inputs do not give.
3. A number with a unit written straight into the text is rejected at write
   time. `scan_unmarked` finds it and the tool returns an error naming it, so
   the model rewrites. Rendering it with a warning instead would teach readers
   to ignore the warning.

What the scan cannot see, and nothing here pretends otherwise: numbers written
as words ("fem år"), and units outside `_UNITS`. For mul and div the result
unit is the caller's label and is not checked dimensionally; add and sub do
require the same unit, because adding kilograms to years gives a figure with no
meaning and no error.

Error messages are Swedish because the model reads them and rewrites in Swedish.
"""

from __future__ import annotations

import math
import re

from aida.followup import normalize_unit

BASES = ("source", "estimate", "user", "derived")
SOURCE_KINDS = ("epd", "boverket", "nollco2", "riktlinje", "metod", "palats", "web")
OPS = ("add", "sub", "mul", "div")

# Inheritance order, weakest first. A user's own figure sits between an estimate
# and a source: it is a known origin (they wrote it and can answer for it), but
# not one Aida can show anyone else.
_STRENGTH = {"estimate": 0, "user": 1, "source": 2}

MAX_CLAIMS = 200
UNIT_MAX = 40
REF_MAX = 200
GROUNDS_MAX = 240
EXCERPT_MAX = 800

_ID_RE = re.compile(r"^c\d{1,3}$")
REF_RE = re.compile(r"\{\{\s*(c\d{1,3})\s*\}\}")
_ANY_TOKEN_RE = re.compile(r"\{\{[^{}]*\}\}")

# A number: optional sign, then either a digit group with thousand separators
# (space, no-break space, narrow no-break space) or plain digits, then an
# optional decimal part; or a bare decimal part, ",5". The lookbehind keeps it
# from starting inside a word or another number, so "A1", "S-P-01234"'s tail
# after a letter, and the "0" of "2,0" do not count as numbers of their own.
_NUM = (r"(?<![\w.,])[-−+]?(?:"
        r"(?:\d{1,3}(?:[   ]\d{3})+|\d+)"
        r"(?:[.,]\d+)?|[.,]\d+)")

# Longer spellings first, so the alternation does not stop at a prefix. Every
# entry is something §13.3 counts as a claim: climate, price, lifespan, share,
# area, energy.
_UNITS = (
    r"kg\s*co[2₂]e", r"kg",
    r"tco2e", r"ton(?:s)?",
    r"kronor", r"tkr", r"mkr", r"kr", r"sek",
    r"år(?:s|en)?", r"månad(?:er|en)?",
    r"procent", r"%",
    r"kvadratmeter", r"kvm", r"m²", r"m2",
    r"kwh", r"mwh", r"gwh",
)
_SCALE = r"(?:(?:miljoner|miljarder|tusen)\s+)?"

# A range is one hit, so the error names both ends ("20-30 år") and not only
# the end next to the unit, which the model would then fix alone.
_RANGE = r"(?:\s*[-–—−]\s*|\s+(?:till|à|och|eller)\s+)"
_UNMARKED_RE = re.compile(
    _NUM + r"(?:" + _RANGE + _NUM + r")?\s*" + _SCALE
    + r"(?:" + "|".join(_UNITS) + r")(?!\w)",
    re.IGNORECASE,
)

# What a reference becomes during the scan: not a word character and never in
# model text, so it cannot join a number or a unit.
_REF_MARK = "\x00"
# The other half of the range problem: "20-{{c2}} år" renders as a range whose
# first end has no claim. "och" only counts after "mellan", since "klass 2 och
# {{c3}}" is not a range.
_RANGE_TO_REF_RE = re.compile(
    r"(" + _NUM + r")(?:\s*[-–—−]\s*|\s+(?:till|à)\s+)" + _REF_MARK
    + r"|mellan\s+(" + _NUM + r")\s+och\s+" + _REF_MARK,
    re.IGNORECASE,
)

# Emphasis, inline code and tags between a number and its unit ("**30** år")
# would hide the pair. Blanked rather than removed, so numbers on either side
# of a marker never run together.
_MARKUP_RE = re.compile(r"[*_`~]|<[^<>]{1,40}>")


def _finite_number(value) -> float | None:
    # bool is an int in Python; True is not a figure anyone wrote.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _text(value, limit: int) -> str | None:
    """A stripped string within `limit`, or None if absent or too long."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text) > limit:
        return None
    return text


def normalize_claim(raw) -> tuple[dict | None, str]:
    """Validate one claim. Returns (claim, error) with exactly one set.

    Keys that do not belong to the claim's basis are dropped, so a stored claim
    only ever says one thing about where it came from. A derived claim's `value`
    is dropped too: `resolve` computes it.
    """
    if not isinstance(raw, dict):
        return None, "Ett påstående måste vara ett objekt."

    cid = raw.get("id")
    if not isinstance(cid, str) or not _ID_RE.match(cid):
        return None, f"Ogiltigt id {cid!r}: använd c1, c2, c3 och så vidare."

    basis = raw.get("basis")
    if basis not in BASES:
        return None, f"{cid}: basis måste vara en av {', '.join(BASES)}."

    unit = raw.get("unit", "")
    if not isinstance(unit, str) or len(unit.strip()) > UNIT_MAX:
        return None, f"{cid}: enheten måste vara text, högst {UNIT_MAX} tecken."
    claim: dict = {"id": cid, "basis": basis, "unit": unit.strip()}

    if basis == "derived":
        derived = raw.get("derived")
        if not isinstance(derived, dict):
            return None, f"{cid}: ett härlett tal behöver derived med from och op."
        op = derived.get("op")
        if op not in OPS:
            return None, f"{cid}: op måste vara en av {', '.join(OPS)}."
        operands = derived.get("from")
        if (not isinstance(operands, list) or len(operands) < 2
                or not all(isinstance(o, str) and _ID_RE.match(o) for o in operands)):
            return None, f"{cid}: from måste lista minst två id:n, till exempel [\"c1\", \"c2\"]."
        if op in ("sub", "div") and len(operands) != 2:
            return None, f"{cid}: {op} tar exakt två tal."
        # c1 + c1 is 2 × c1 while the source list names c1 once.
        if len(set(operands)) != len(operands):
            return None, (f"{cid}: samma id står två gånger i from. Gånger ett antal "
                          f"skrivs som mul med ett eget påstående för antalet.")
        claim["derived"] = {"op": op, "from": list(operands)}
        return claim, ""

    value = _finite_number(raw.get("value"))
    if value is None:
        return None, f"{cid}: value måste vara ett tal."
    claim["value"] = value

    if basis == "source":
        source = raw.get("source")
        if not isinstance(source, dict):
            return None, f"{cid}: ett tal med källa behöver source med kind och ref."
        kind = source.get("kind")
        if kind not in SOURCE_KINDS:
            return None, f"{cid}: källans kind måste vara en av {', '.join(SOURCE_KINDS)}."
        ref = _text(source.get("ref"), REF_MAX)
        if ref is None:
            return None, f"{cid}: källan behöver en ref, högst {REF_MAX} tecken."
        normalized = {"kind": kind, "ref": ref}
        if source.get("excerpt") is not None:
            excerpt = _text(source.get("excerpt"), EXCERPT_MAX)
            if excerpt is None:
                return None, f"{cid}: utdraget måste vara text, högst {EXCERPT_MAX} tecken."
            normalized["excerpt"] = excerpt
        claim["source"] = normalized

    elif basis == "estimate":
        estimate = raw.get("estimate")
        grounds = _text(estimate.get("grounds"), GROUNDS_MAX) if isinstance(estimate, dict) else None
        if grounds is None:
            return None, (f"{cid}: en uppskattning måste säga vad den bygger på "
                          f"(estimate.grounds, högst {GROUNDS_MAX} tecken).")
        claim["estimate"] = {"grounds": grounds}

    return claim, ""


def _compute(op: str, values: list[float]) -> float | None:
    if op == "add":
        return math.fsum(values)
    if op == "sub":
        return values[0] - values[1]
    if op == "mul":
        return math.prod(values)
    if values[1] == 0:
        return None
    return values[0] / values[1]


def _unique(items) -> list:
    return list(dict.fromkeys(items))


def resolve(raw_claims) -> tuple[list[dict], list[str]]:
    """Validate a block's claims and compute every derived one.

    Returns (resolved, errors). Each resolved claim carries three computed keys
    besides its own:

        effective  'source' | 'user' | 'estimate', what the number rests on
                   after inheritance. For anything but derived it is the basis.
        grounds    every estimate's grounds the number depends on, in order
        leaves     the non-derived claims it ultimately comes from

    A claim that depends on an invalid, unknown or circular one fails with its
    own error instead of being computed from what is left. The caller decides
    what an error means; the block tools reject the whole write on any.
    """
    if not isinstance(raw_claims, list):
        return [], ["Påståendena måste vara en lista."]
    if len(raw_claims) > MAX_CLAIMS:
        return [], [f"Högst {MAX_CLAIMS} påståenden per block."]

    errors: list[str] = []
    claims: dict[str, dict] = {}
    order: list[str] = []
    invalid: set[str] = set()

    for raw in raw_claims:
        claim, err = normalize_claim(raw)
        if err:
            errors.append(err)
            rid = raw.get("id") if isinstance(raw, dict) else None
            if isinstance(rid, str):
                invalid.add(rid)
            continue
        cid = claim["id"]
        if cid in claims or cid in invalid:
            # Neither copy is usable: a reference to c2 would mean whichever one
            # happened to be read last.
            errors.append(f"{cid}: id:t används två gånger.")
            claims.pop(cid, None)
            invalid.add(cid)
            continue
        claims[cid] = claim
        order.append(cid)

    resolved: dict[str, dict] = {}
    failed: set[str] = set()
    visiting: set[str] = set()

    def visit(cid: str) -> dict | None:
        if cid in resolved:
            return resolved[cid]
        if cid in failed:
            return None
        claim = claims[cid]

        if claim["basis"] != "derived":
            out = dict(claim)
            out["effective"] = claim["basis"]
            out["grounds"] = [claim["estimate"]["grounds"]] if claim["basis"] == "estimate" else []
            out["leaves"] = [cid]
            resolved[cid] = out
            return out

        visiting.add(cid)
        operands: list[dict] = []
        problem = ""
        for oid in claim["derived"]["from"]:
            if oid == cid or oid in visiting:
                problem = f"{cid}: räknas ur sig självt via {oid}."
                break
            if oid in invalid:
                problem = f"{cid}: bygger på {oid}, som är ogiltigt."
                break
            if oid not in claims:
                problem = f"{cid}: bygger på {oid}, som inte finns."
                break
            operand = visit(oid)
            if operand is None:
                problem = f"{cid}: bygger på {oid}, som inte gick att räkna."
                break
            operands.append(operand)
        visiting.discard(cid)

        if not problem:
            op = claim["derived"]["op"]
            units = [normalize_unit(o["unit"]) for o in operands]
            if op in ("add", "sub") and len(set(units)) != 1:
                shown = " och ".join(_unique(o["unit"] or "(ingen enhet)" for o in operands))
                problem = f"{cid}: kan inte räkna {op} på olika enheter: {shown}."
            else:
                value = _compute(op, [o["value"] for o in operands])
                if value is None:
                    problem = f"{cid}: division med noll."
                elif not math.isfinite(value):
                    problem = f"{cid}: resultatet är inte ett ändligt tal."

        if problem:
            errors.append(problem)
            failed.add(cid)
            return None

        if op in ("add", "sub"):
            unit = operands[0]["unit"]
        elif claim["unit"]:
            unit = claim["unit"]
        else:
            joiner = "/" if op == "div" else "·"
            unit = joiner.join(o["unit"] for o in operands if o["unit"])

        out = dict(claim)
        out["value"] = value
        out["unit"] = unit
        out["effective"] = min((o["effective"] for o in operands), key=_STRENGTH.__getitem__)
        out["grounds"] = _unique(g for o in operands for g in o["grounds"])
        out["leaves"] = _unique(leaf for o in operands for leaf in o["leaves"])
        resolved[cid] = out
        return out

    # `order` still holds ids that a later duplicate knocked out of `claims`;
    # they are already reported and must not be visited.
    order = [cid for cid in order if cid in claims]
    for cid in order:
        visit(cid)

    return [resolved[cid] for cid in order if cid in resolved], errors


def refs_in(text: str) -> list[str]:
    """The claim ids a text refers to, in order of first appearance."""
    return _unique(REF_RE.findall(text or ""))


def scan_unmarked(text: str) -> list[str]:
    """Numbers with a claim unit written straight into the text.

    Markup is blanked first, so bold around a figure does not hide it.
    References (`{{c1}}`) then become a marker, so a number the renderer will
    put there is never mistaken for one the model wrote, while a bare number
    opening a range into one ("20-{{c2}} år") is still caught. Only well-formed
    references are replaced: braces around a bare figure (`{{30 år}}`) would
    otherwise hide it from the check.
    """
    marked = REF_RE.sub(f" {_REF_MARK} ", _MARKUP_RE.sub(" ", text or ""))
    hits = [(m.start(), m.group(0)) for m in _UNMARKED_RE.finditer(marked)]
    hits += [(m.start(), m.group(1) or m.group(2)) for m in _RANGE_TO_REF_RE.finditer(marked)]
    return [" ".join(hit.split()) for _, hit in sorted(hits)]


def check_text(text: str, known_ids) -> list[str]:
    """Everything wrong with one piece of block text, as messages for the model."""
    known = set(known_ids)
    malformed = [t for t in _unique(_ANY_TOKEN_RE.findall(text or "")) if not REF_RE.fullmatch(t)]
    errors = [f"{token} är ingen giltig hänvisning. Skriv {{{{cN}}}}, en hänvisning "
              f"per påstående." for token in malformed]
    errors += [f"Hänvisningen {{{{{ref}}}}} finns inte bland påståendena."
               for ref in refs_in(text) if ref not in known]
    errors += [f"Talet \"{hit}\" saknar ursprung. Skriv det som ett påstående "
               f"({{{{cN}}}}) med källa, eller som en uppskattning med grund."
               for hit in scan_unmarked(text)]
    return errors


def format_value(value: float, unit: str = "") -> str:
    """Swedish display: decimal comma, space between thousands, sensible precision.

    Precision follows size so a derived 0.1466 shows as 0,15 and a baseline of
    3290.4 as 3 290. Below 1 the decimals grow with the magnitude, so a small
    figure never rounds to a zero that would read as "no emissions".
    """
    if value == 0:
        value = 0.0  # -0.0 would otherwise print as "−0"
    magnitude = abs(value)
    if magnitude >= 100:
        decimals = 0
    elif magnitude >= 10:
        decimals = 1
    elif magnitude >= 1 or magnitude == 0:
        decimals = 2
    else:
        decimals = max(2, 1 - math.floor(math.log10(magnitude)))
    whole, _, frac = f"{value:,.{decimals}f}".partition(".")
    frac = frac.rstrip("0")
    text = whole.replace(",", " ") + ("," + frac if frac else "")
    text = text.replace("-", "−")
    return f"{text} {unit}" if unit else text
