"""The open sheet in Chatt (§13.2, §13.3, §13.5): blocks Aida and the user write together.

    sheet = {"title": str, "seq": int, "blocks": [block, ...]}
    block = {"id": "b3", "type": "text" | "table" | "quote" | "note",
             "author": "aida" | "user", "user_edited": bool,
             "content": {...}, "claims": [resolved claim, ...], "at": ISO time}

Content per type:

    text   {"markdown": str}                              numbers as {{cN}} claims
    table  {"columns": [str], "rows": [[str, ...], ...]}  every cell under the same rule
    quote  {"text": str, "section": id, "label": str}     whole sentences from the section
    note   {"markdown": str}                              the user's own, never scanned

The block tools are the one way Aida writes to the sheet, and every write is
checked here instead of trusted to the prompt. A number with a unit outside
{{cN}} is rejected with the number named, so the model writes it again with a
source or as an estimate with grounds (claims.check_text). A quote has to be
whole sentences from the section it cites (knowledge.quote_in_source). A block
the user wrote or edited is not Aida's to change or remove. A rejected write
leaves the sheet as it was: the model gets every error at once and nothing is
half applied.

Handlers take (inp, sheet), change `sheet` in place and return
(message, ok, touched), the shape of the handlers in mutations.py.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

from aida import claims as claims_mod
from aida import knowledge

BLOCK_TYPES = ("text", "table", "quote", "note")
# Notes are the user's. Aida answers in text, tables and quotes.
AIDA_TYPES = ("text", "table", "quote")

MAX_BLOCKS = 80
TITLE_MAX = 160
TEXT_MAX = 8000
QUOTE_MAX = 2000
MAX_COLUMNS = 10
MAX_ROWS = 60
CELL_MAX = 400

_BLOCK_ID_RE = re.compile(r"^b\d{1,5}$")
# A cell that is only a figure, however hedged: "30", "ca 30", "~30", "20-30",
# "mellan 20 och 30", "30 st". The unit usually sits in the column heading, where
# the text scan cannot see it, so "4,4" under "kg CO2e/m²" has to be a claim as
# surely as "4,4 kg" in running text.
_FIGURE = r"[-−+]?\s*(?:\d[\d\s  ]*(?:[.,]\d+)?|[.,]\d+)"
_BARE_FIGURE_RE = re.compile(
    r"^(?:(?:ca\.?|cirka|ungefär|omkring|runt|kring|drygt|knappt|nästan|minst|högst|max\.?|min\.?"
    r"|över|under|upp\s+till|mellan)\s+|[~≈<>≤≥]\s*)?"
    + _FIGURE + r"(?:\s*(?:[-–—]|till|och)\s*" + _FIGURE + r")?\s*(?:%|st|ggr|gånger)?\.?$",
    re.IGNORECASE,
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def empty_sheet(title: str = "") -> dict:
    return {"title": title, "seq": 0, "blocks": []}


def _text(value, limit: int) -> str:
    return value.strip()[:limit] if isinstance(value, str) else ""


def _resolved(raw_claims) -> tuple[list[dict], list[str]]:
    """claims.resolve, plus the Swedish display string the renderer shows."""
    resolved, errors = claims_mod.resolve([] if raw_claims is None else raw_claims)
    for claim in resolved:
        claim["display"] = claims_mod.format_value(claim["value"], claim["unit"])
    return resolved, errors


def _raw_ids(raw_claims) -> list[str]:
    """Every id the model declared, valid or not, so a reference to a claim that
    failed validation is reported once, as the claim's own error."""
    if not isinstance(raw_claims, list):
        return []
    return [c["id"] for c in raw_claims if isinstance(c, dict) and isinstance(c.get("id"), str)]


def _check_table(raw: dict, known: list[str]) -> tuple[dict | None, list[str]]:
    columns, rows = raw.get("columns"), raw.get("rows")
    if (not isinstance(columns, list) or not 1 <= len(columns) <= MAX_COLUMNS
            or not all(isinstance(c, str) and c.strip() for c in columns)):
        return None, [f"En tabell behöver columns: 1 till {MAX_COLUMNS} rubriker som text."]
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_ROWS:
        return None, [f"En tabell behöver rows: 1 till {MAX_ROWS} rader."]
    columns = [c.strip()[:CELL_MAX] for c in columns]
    errors = [f"Rubriken \"{c}\": {e}" for c in columns for e in claims_mod.check_text(c, known)]
    clean: list[list[str]] = []
    for i, row in enumerate(rows, 1):
        if not isinstance(row, list) or len(row) != len(columns):
            errors.append(f"Rad {i} ska ha {len(columns)} celler, en per kolumn.")
            continue
        cells = []
        for column, cell in zip(columns, row):
            where = f"Rad {i}, {column}"
            if isinstance(cell, (int, float)) and not isinstance(cell, bool):
                cell = str(cell)
            if not isinstance(cell, str):
                errors.append(f"{where}: en cell är text.")
                continue
            cell = cell.strip()
            if len(cell) > CELL_MAX:
                errors.append(f"{where}: högst {CELL_MAX} tecken i en cell.")
            if _BARE_FIGURE_RE.match(cell):
                errors.append(f"{where}: talet \"{cell}\" saknar ursprung. Skriv det som ett påstående "
                              f"({{{{cN}}}}) med källa, eller som en uppskattning med grund.")
            errors += [f"{where}: {e}" for e in claims_mod.check_text(cell, known)]
            cells.append(cell)
        clean.append(cells)
    return {"columns": columns, "rows": clean}, errors


def _check_content(btype: str, raw, known: list[str]) -> tuple[dict | None, list[str]]:
    """Validate a block's content. Returns (content, errors) for the model."""
    if not isinstance(raw, dict):
        return None, ["content måste vara ett objekt."]
    if btype == "text":
        markdown = raw.get("markdown")
        if not isinstance(markdown, str) or not markdown.strip():
            return None, ["En textruta behöver content.markdown."]
        if len(markdown) > TEXT_MAX:
            return None, [f"Högst {TEXT_MAX} tecken i en textruta. Dela upp svaret i flera block."]
        return {"markdown": markdown.strip()}, claims_mod.check_text(markdown, known)
    if btype == "table":
        return _check_table(raw, known)
    if btype == "quote":
        text, sid = raw.get("text"), raw.get("section")
        found = knowledge.section(sid)
        if found is None:
            return None, [f"Avsnittet {sid!r} finns inte. Sök med search_knowledge och ange id:t ur träffen."]
        if not isinstance(text, str) or len(text) > QUOTE_MAX or not knowledge.quote_in_source(text, sid):
            return None, [f"Citatet står inte ordagrant i {knowledge.label(sid)}. Citera en eller flera hela "
                          f"meningar ur samma stycke, utan utelämningar, eller skriv med egna ord i en textruta."]
        return {"text": text.strip(), "section": sid, "label": knowledge.label(sid)}, []
    return None, [f"type måste vara en av {', '.join(AIDA_TYPES)}."]


def _build(btype: str, content, raw_claims) -> tuple[dict | None, list[dict], list[str]]:
    if btype == "quote":
        # A quote carries no claims: it is checked against its source instead,
        # and a figure inside it is the source's own.
        raw_claims = []
    resolved, claim_errors = _resolved(raw_claims)
    content, content_errors = _check_content(btype, content, _raw_ids(raw_claims))
    return content, resolved, claim_errors + content_errors


def _rejected(errors: list[str]) -> tuple[str, bool, set]:
    return "Blocket sparades inte:\n- " + "\n- ".join(errors), False, set()


def _find(sheet: dict, bid) -> tuple[int, dict | None]:
    for i, block in enumerate(sheet["blocks"]):
        if block["id"] == bid:
            return i, block
    return -1, None


def _users(block: dict) -> bool:
    return block["author"] == "user" or block["user_edited"]


def add_block(inp, sheet: dict) -> tuple[str, bool, set]:
    inp = inp if isinstance(inp, dict) else {}
    btype = inp.get("type")
    if btype not in AIDA_TYPES:
        return _rejected([f"type måste vara en av {', '.join(AIDA_TYPES)}. Anteckningar skriver användaren."])
    if len(sheet["blocks"]) >= MAX_BLOCKS:
        return _rejected([f"Bladet har redan {MAX_BLOCKS} block. Ändra eller ta bort ett i stället."])
    position = len(sheet["blocks"])
    after = inp.get("after")
    if after is not None:
        index, _ = _find(sheet, after)
        if index < 0:
            return _rejected([f"Blocket {after!r} finns inte, så det går inte att lägga något efter det."])
        position = index + 1
    content, resolved, errors = _build(btype, inp.get("content"), inp.get("claims"))
    if errors:
        return _rejected(errors)
    sheet["seq"] = max(sheet.get("seq", 0), _max_id(sheet)) + 1
    bid = f"b{sheet['seq']}"
    sheet["blocks"].insert(position, {"id": bid, "type": btype, "author": "aida", "user_edited": False,
                                      "content": content, "claims": resolved, "at": _now()})
    return f"Lade till {bid}.", True, {"sheet"}


def update_block(inp, sheet: dict) -> tuple[str, bool, set]:
    inp = inp if isinstance(inp, dict) else {}
    bid = inp.get("id")
    index, block = _find(sheet, bid)
    if block is None:
        return _rejected([f"Blocket {bid!r} finns inte."])
    if _users(block):
        return _rejected([f"{bid} har användaren skrivit eller ändrat, och det ändrar inte Aida. "
                          f"Lägg ett nytt block efter det i stället."])
    content = inp["content"] if "content" in inp else block["content"]
    # Claims left out keep the block's own. Resolved claims go back through
    # resolve unchanged, since normalize_claim drops the computed keys.
    raw_claims = inp["claims"] if "claims" in inp else block["claims"]
    content, resolved, errors = _build(block["type"], content, raw_claims)
    if errors:
        return _rejected(errors)
    sheet["blocks"][index] = {**block, "content": content, "claims": resolved, "at": _now()}
    return f"Ändrade {bid}.", True, {"sheet"}


def remove_block(inp, sheet: dict) -> tuple[str, bool, set]:
    inp = inp if isinstance(inp, dict) else {}
    bid = inp.get("id")
    index, block = _find(sheet, bid)
    if block is None:
        return _rejected([f"Blocket {bid!r} finns inte."])
    if _users(block):
        return _rejected([f"{bid} har användaren skrivit eller ändrat, och det tar inte Aida bort."])
    del sheet["blocks"][index]
    return f"Tog bort {bid}.", True, {"sheet"}


SHEET_HANDLERS = {
    "add_block": add_block,
    "update_block": update_block,
    "remove_block": remove_block,
}

# Each block is a tool call and a rejected one is redone, so an answer on the
# sheet takes turns a chat answer does not. The live questions in §13.11 used
# eight to eleven.
EXTRA_TURNS = 6
# When the budget runs out after something reached the sheet, the chat line has
# to say so: "try rephrasing" next to a half-written answer reads as nothing done.
UNFINISHED = "Jag skrev det jag hann på bladet men blev inte klar. Be mig fortsätta om något saknas."
# The chat line when the answer went on the sheet and the model ended its turn
# without writing one.
DONE = "Svaret ligger på bladet."
# Said to the model when the token limit cut its reply inside a tool call.
CUT_OFF = ("Ditt förra svar blev för långt och klipptes mitt i ett verktygsanrop, så inget av det "
           "sparades. Lägg ett eller två block åt gången.")


def cut_off(response) -> bool:
    """Whether the token limit stopped a response while it was writing a tool call.

    Such a call has a truncated input, so none of the response may run. Taken as
    the final answer, as any other stop is, it ends the turn with an empty reply
    and nothing on the sheet: a comparison written as four blocks at once did.
    """
    return (getattr(response, "stop_reason", None) == "max_tokens"
            and any(getattr(b, "type", None) == "tool_use" for b in response.content))


def nudge(messages: list[dict]) -> None:
    """Drop a cut-off reply by saying so in the turn it answered."""
    last = messages[-1]
    content = last["content"]
    blocks = [{"type": "text", "text": content}] if isinstance(content, str) else list(content)
    messages[-1] = {**last, "content": blocks + [{"type": "text", "text": CUT_OFF}]}

PROMPT = """BLADET (läget Chatt):
Användaren har ett blad bredvid chatten, och i det här läget hör svaret hemma där. Lägg det med add_block, och skriv i chatten bara en kort rad om vad du lade på bladet, till exempel "Jag lade en jämförelse av linoleum och vinyl på bladet."
- Formen följer frågan: ett kort svar och citat för en riktlinjefråga, en tabell och ett resonemang för en avvägning, en förklaring steg för steg för en metodfråga. Dela gärna upp svaret i flera block, och lägg gärna två eller tre av dem i samma tur i stället för ett i taget.
- Varje klimattal, pris, livslängd, mängd och andel är ett påstående i blockets claims, och texten hänvisar till det med {{c1}}. Helst med källa. Finns ingen källa, ge hellre en uppskattning med grund än inget svar. Ett tal som räknas ur andra, som klimatpåverkan per år, är derived: servern räknar ut det och märker det som uppskattning om något led är det.
- Ett tal med enhet som står direkt i texten avvisas, liksom en tabellcell med bara ett tal. Rätta det verktyget pekar ut och försök igen.
- Citat läggs som quote med avsnittets id ur search_knowledge, och är hela meningar ordagrant.
- Block som användaren skrivit eller ändrat rör du inte. Vill du komplettera ett sådant, lägg ett nytt block efter det.
- Rätta ett eget block med update_block i stället för att lägga en ny version bredvid."""


def describe(sheet: dict) -> str:
    """The sheet as the model reads it: one line per block, so it can name one."""
    if not sheet["blocks"]:
        return "Bladet är tomt."
    lines = []
    for block in sheet["blocks"]:
        who = "användaren" if _users(block) else "Aida"
        content = block["content"]
        if block["type"] == "table":
            columns = ", ".join(content.get("columns", []))
            what = f"tabell med kolumnerna {columns}, {len(content.get('rows', []))} rader"
        elif block["type"] == "quote":
            what = f"citat ur {content.get('label', '')}"
        else:
            what = " ".join(content.get("markdown", "").split())
            what = what if len(what) <= 200 else what[:200] + " …"
        lines.append(f"{block['id']} {block['type']} ({who}): {what}")
    return "\n".join(lines)


def prompt(sheet: dict) -> str:
    """The rules for writing to the sheet, and the sheet as it stands."""
    return PROMPT + "\n\nNUVARANDE BLAD:\n" + describe(sheet)


def _max_id(sheet: dict) -> int:
    return max((int(b["id"][1:]) for b in sheet["blocks"]), default=0)


def _clean_content(btype: str, raw: dict) -> dict:
    """Stored content in its type's shape, without judging it: what the user wrote
    stays, and what Aida wrote was judged when it was written."""
    if btype in ("text", "note"):
        return {"markdown": _text(raw.get("markdown"), TEXT_MAX)}
    if btype == "table":
        columns = [_text(c, CELL_MAX) for c in (raw.get("columns") or [])[:MAX_COLUMNS]
                   if isinstance(raw.get("columns"), list)]
        rows = [[_text(c, CELL_MAX) for c in row[:len(columns)]] + [""] * max(0, len(columns) - len(row))
                for row in (raw.get("rows") or [])[:MAX_ROWS] if isinstance(row, list)]
        return {"columns": columns, "rows": rows}
    sid = raw.get("section") if isinstance(raw.get("section"), str) else ""
    found = knowledge.section(sid)
    return {"text": _text(raw.get("text"), QUOTE_MAX), "section": sid,
            "label": knowledge.label(sid) if found else _text(raw.get("label"), 300)}


def normalize_sheet(raw) -> dict:
    """The sheet a client sent, in the one shape the tools and renderer expect.

    Never raises. A block that cannot be read is dropped rather than repaired,
    and claims are resolved again, so a display value always matches its claim.
    """
    if not isinstance(raw, dict):
        return empty_sheet()
    blocks: list[dict] = []
    seen: set[str] = set()
    for b in raw.get("blocks") if isinstance(raw.get("blocks"), list) else []:
        if len(blocks) >= MAX_BLOCKS:
            break
        if not isinstance(b, dict):
            continue
        bid, btype = b.get("id"), b.get("type")
        if not isinstance(bid, str) or not _BLOCK_ID_RE.match(bid) or bid in seen or btype not in BLOCK_TYPES:
            continue
        seen.add(bid)
        content = b.get("content") if isinstance(b.get("content"), dict) else {}
        resolved, _ = _resolved(b.get("claims") if btype in ("text", "table") else [])
        blocks.append({
            "id": bid, "type": btype,
            "author": "user" if b.get("author") == "user" or btype == "note" else "aida",
            "user_edited": b.get("user_edited") is True,
            "content": _clean_content(btype, content),
            "claims": resolved,
            "at": _text(b.get("at"), 40),
        })
    seq = raw.get("seq") if isinstance(raw.get("seq"), int) and not isinstance(raw.get("seq"), bool) else 0
    sheet = {"title": _text(raw.get("title"), TITLE_MAX), "seq": 0, "blocks": blocks}
    sheet["seq"] = max(seq, _max_id(sheet))
    return sheet


_CLAIMS_SCHEMA = {
    "type": "array",
    "description": (
        "Varje klimattal, pris, livslängd, mängd och andel i blocket. Texten hänvisar till dem med "
        "{{c1}}, {{c2}}, och servern skriver ut talet med enhet och ursprung. basis 'source' kräver "
        "source med kind och ref (ref: EPD-nummer, avsnittets id eller webbadress). basis 'estimate' "
        "kräver estimate.grounds, en kort mening om vad uppskattningen bygger på. basis 'user' är ett "
        "tal användaren själv gett. basis 'derived' räknas av servern ur andra påståenden, till exempel "
        "{\"from\": [\"c1\", \"c2\"], \"op\": \"div\"}, och får inget value. Ett härlett tal ärver "
        "sitt svagaste underlag."
    ),
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "c1, c2, c3 och så vidare, unikt i blocket."},
            "value": {"type": "number"},
            "unit": {"type": "string", "description": "Till exempel 'kg CO2e/m²', 'år', 'kr', '%'."},
            "basis": {"type": "string", "enum": list(claims_mod.BASES)},
            "source": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": list(claims_mod.SOURCE_KINDS)},
                    "ref": {"type": "string"},
                    "excerpt": {"type": "string"},
                },
            },
            "estimate": {"type": "object", "properties": {"grounds": {"type": "string"}}},
            "derived": {
                "type": "object",
                "properties": {
                    "from": {"type": "array", "items": {"type": "string"}},
                    "op": {"type": "string", "enum": list(claims_mod.OPS)},
                },
            },
        },
        "required": ["id", "basis"],
    },
}

_CONTENT_SCHEMA = {
    "type": "object",
    "description": (
        "text: {\"markdown\": \"...\"} med rubriker, stycken, listor och fetstil. "
        "table: {\"columns\": [\"Material\", \"Klimat\"], \"rows\": [[\"Linoleum\", \"{{c1}}\"]]}, "
        "där varje cell är text och varje tal en hänvisning. "
        "quote: {\"text\": \"...\", \"section\": \"<id ur search_knowledge>\"}, en eller flera hela "
        "meningar ordagrant ur avsnittet."
    ),
}

SHEET_TOOLS = [
    {
        "name": "add_block",
        "description": (
            "Lägg ett block på bladet: text, table eller quote. Välj form efter frågan: ett kort svar och "
            "ett citat för en riktlinjefråga, en tabell och ett resonemang för en avvägning. Ett tal med "
            "enhet som står direkt i texten avvisas; skriv det som ett påstående och hänvisa med {{cN}}."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": list(AIDA_TYPES)},
                "content": _CONTENT_SCHEMA,
                "claims": _CLAIMS_SCHEMA,
                "after": {"type": "string", "description": "Blockets id att lägga det efter. Utelämna för sist."},
            },
            "required": ["type", "content"],
        },
    },
    {
        "name": "update_block",
        "description": (
            "Ändra ett block Aida skrivit. Utelämnade fält behåller sitt värde. Block som användaren skrivit "
            "eller ändrat går inte att ändra; lägg ett nytt block efter i stället."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "content": _CONTENT_SCHEMA,
                "claims": _CLAIMS_SCHEMA,
            },
            "required": ["id"],
        },
    },
    {
        "name": "remove_block",
        "description": "Ta bort ett block Aida skrivit. Användarens block går inte att ta bort.",
        "input_schema": {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        },
    },
]
