"""The open sheet as a document (orchestration-redesign §13.8).

Word gets the sheet in order. A reader of the attached document never saw the
figures' marks on screen, so the marking has to be in the text: every figure is
written as it is shown, followed by a number in brackets. The numbers point to
two lists at the end. "Källor" says where each sourced figure comes from.
"Uppskattningar" holds every estimate with what it rests on, so it is visible in
a document attached to a procurement which figures have no source.

A figure computed from others numbers those too, so a source is never lost
because only the result was cited. The calculation's sections are written from
the state the browser shows, overrides included, and the report goes in as it is.
A suggestion is not part of the sheet until the user accepts it, so it is left out.

Pure: markdown in, markdown out, and it does not raise on what a browser sends.
aida.docx_export makes the Word file.
"""

from __future__ import annotations

from aida import claims as claims_mod
from aida import sheet as sheet_mod

KIND_LABELS = {"epd": "EPD", "boverket": "Boverkets klimatdatabas", "nollco2": "NollCO2",
               "riktlinje": "Riktlinje", "metod": "Aidas metod", "palats": "Palats", "web": "Webben"}
OPS = {"add": "plus", "sub": "minus", "mul": "gånger", "div": "delat med"}
SOURCES, ESTIMATES = "Källor", "Uppskattningar"


def _bare(text) -> str:
    return str(text or "").strip().rstrip(".!?:;,")


def _cell(text) -> str:
    """One table cell: on one line, with a pipe that cannot split the row."""
    return " ".join(str(text if text is not None else "").split()).replace("|", "\\|")


def _dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _dicts(value) -> list[dict]:
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


def _figure(value, unit: str) -> str:
    try:
        return claims_mod.format_value(float(value), unit)
    except (TypeError, ValueError, OverflowError):
        return ""


class _Figures:
    """One numbering for the document. A figure that reads the same and rests on
    the same thing keeps its number wherever it is cited."""

    def __init__(self):
        self.entries: list[tuple[str, str]] = []  # (heading, line)
        self._numbers: dict[tuple[str, str], int] = {}

    def number(self, claim: dict, by_id: dict, memo: dict, seen: frozenset = frozenset()) -> int:
        # `memo` holds the numbers already given in this block. Without it a
        # figure computed from the two before it, again and again, is explained
        # once per path through its operands: exponential in the number of
        # claims, and the sheet comes from the browser.
        cid = claim.get("id")
        if cid in memo:
            return memo[cid]
        heading, text = self._explain(claim, by_id, memo, seen | {cid})
        key = (heading, f"{claim.get('display', '')}: {text}")
        if key not in self._numbers:
            self.entries.append(key)
            self._numbers[key] = len(self.entries)
        memo[cid] = self._numbers[key]
        return memo[cid]

    def _explain(self, claim: dict, by_id: dict, memo: dict, seen: frozenset) -> tuple[str, str]:
        basis = claim.get("basis")
        if basis == "source":
            src = _dict(claim.get("source"))
            return SOURCES, KIND_LABELS.get(src.get("kind"), "Källa") + (f", {src['ref']}" if src.get("ref") else "") + "."
        if basis == "user":
            return SOURCES, "Uppgift från användaren."
        if basis == "estimate":
            return ESTIMATES, f"Uppskattning. Bygger på: {_bare(_dict(claim.get('estimate')).get('grounds'))}."
        derived = _dict(claim.get("derived"))
        parts = []
        for oid in derived.get("from", []):
            operand = by_id.get(oid)
            if operand is None or oid in seen:
                parts.append(str(oid))
            else:
                parts.append(f"{operand.get('display', oid)} [{self.number(operand, by_id, memo, seen)}]")
        text = "Räknat: " + f" {OPS.get(derived.get('op'), derived.get('op', ''))} ".join(parts) + "."
        if claim.get("effective") == "estimate":
            grounds = "; ".join(_bare(g) for g in claim.get("grounds", []))
            return ESTIMATES, f"{text} En uppskattning, eftersom det bygger på: {grounds}."
        return SOURCES, text

    def cite(self, text: str, by_id: dict, memo: dict) -> str:
        """Every {{cN}} in `text` as its figure and number."""
        def one(match):
            claim = by_id.get(match.group(1))
            if claim is None:
                return "[saknas]"
            return f"{claim.get('display', '')} [{self.number(claim, by_id, memo)}]"
        return claims_mod.REF_RE.sub(one, text or "")

    def markdown(self) -> str:
        out = []
        for heading in (SOURCES, ESTIMATES):
            lines = [f"- [{n}] {line}" for n, (h, line) in enumerate(self.entries, 1) if h == heading]
            if lines:
                out.append(f"## {heading}\n\n" + "\n".join(lines))
        return "\n\n".join(out)


def _table(columns: list, rows: list[list]) -> str:
    head = "| " + " | ".join(_cell(c) for c in columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    return "\n".join([head, rule] + ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows])


def _projekt(st: dict) -> str:
    project = _dict(st.get("project"))
    if not project:
        return ""
    head = f"{project.get('building_type') or 'Byggnad'}, {_figure(project.get('area_bta'), 'm² BTA')}"
    if project.get("name"):
        head += f" ({project['name']})"
    parts = [f"## {sheet_mod.MODEL_SECTIONS['projekt']}", head]
    if project.get("description"):
        parts.append(str(project["description"]))
    comps = _dicts(project.get("components"))
    if comps:
        parts.append(_table(["Komponent", "Mängd"],
                            [[c.get("name", ""), _figure(c.get("quantity"), str(c.get("unit") or ""))] for c in comps]))
    return "\n\n".join(parts)


def _source(comp: dict) -> str:
    """A figure the user replaced is theirs, with their note, not the source's."""
    if comp.get("co2e_override"):
        return f"Ändrat av användaren: {_bare(comp['co2e_override'])}"
    return str(comp.get("source") or "")


def _baslinje(st: dict) -> str:
    comps = _dicts(_dict(st.get("baseline")).get("components"))
    if not comps:
        return ""
    rows = [[c.get("component_name", ""), _figure(c.get("co2e_kg"), "kg CO2e"), _source(c)] for c in comps]
    total = 0.0
    for c in comps:
        try:
            total += float(c.get("co2e_kg") or 0)
        except (TypeError, ValueError):
            pass
    return "\n\n".join([f"## {sheet_mod.MODEL_SECTIONS['baslinje']}",
                        _table(["Komponent", "Klimatpåverkan", "Källa"], rows),
                        f"Totalt: {_figure(total, 'kg CO2e')}."])


def _alternativ(st: dict) -> str:
    comps = _dicts(_dict(st.get("alternatives")).get("components"))
    if not comps:
        return ""
    selections = _dict(st.get("selections"))
    rows = []
    for c in comps:
        cid = c.get("component_id")
        chosen = _dict(_dict(selections.get(cid) if isinstance(cid, str) else None).get("selected_alternative"))
        baseline = _figure(c.get("baseline_co2e_kg"), "kg CO2e")
        if c.get("baseline_co2e_override"):
            baseline += " (ändrat av användaren)"
        rows.append([c.get("component_name", ""), chosen.get("name") or "Inget val ännu",
                     _figure(chosen.get("co2e_kg"), "kg CO2e") if chosen else "", baseline])
    return "\n\n".join([f"## {sheet_mod.MODEL_SECTIONS['alternativ']}",
                        _table(["Komponent", "Valt", "Klimatpåverkan", "Baslinje"], rows)])


def _rapport(st: dict) -> str:
    report = st.get("reportMarkdown")
    return report.strip() if isinstance(report, str) else ""


_MODEL = {"projekt": _projekt, "baslinje": _baslinje, "alternativ": _alternativ, "rapport": _rapport}


def sheet_markdown(raw_sheet, state=None) -> str:
    """The sheet in order as markdown, with its figures' sources and estimates last."""
    sheet = sheet_mod.normalize_sheet(raw_sheet)
    st = _dict(state)
    figures = _Figures()
    title = sheet["title"] or str(_dict(st.get("project")).get("name") or "") or "Blad från Aida"
    parts = [f"# {title}"]
    for block in sheet["blocks"]:
        btype, content = block["type"], block["content"]
        by_id, memo = {c["id"]: c for c in block["claims"]}, {}
        if btype == "text":
            parts.append(figures.cite(content["markdown"], by_id, memo))
        elif btype == "table":
            parts.append(_table([figures.cite(c, by_id, memo) for c in content["columns"]],
                                [[figures.cite(c, by_id, memo) for c in row] for row in content["rows"]]))
        elif btype == "quote":
            parts.append(f"> {' '.join(content['text'].split())}\n> Källa: {content['label']}")
        elif btype == "note":
            parts.append(f"#### Anteckning\n\n{content['markdown']}")
        elif btype == "model":
            parts.append(_MODEL[content["section"]](st))
    parts.append(figures.markdown())
    return "\n\n".join(p for p in parts if p.strip()) + "\n"
