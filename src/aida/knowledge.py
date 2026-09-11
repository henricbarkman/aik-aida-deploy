"""Citable knowledge: Karlstad's building guidelines and Aida's method (§13.4).

Two corpora, both inside src/aida/ because the deploy ships nothing else:

    riktlinjer  the byggriktlinjer, split into sections by
                scripts/bygg_kunskap.py, each carrying document, edition and
                heading path
    metod       data/knowledge/metod.md, Aida's method written for property
                managers, split here at its ## and ### headings

`search` is keyword search. Sources and questions are both Swedish, so the
language gap that made the Environdec search hard does not arise, and
stripping a few suffixes is enough for "plastmatta" to find "Plastmattor".

`quote_in_source` is the check a quote must pass before it can stand on the
sheet: whole sentences, word for word, from the section it cites, so neither a
paraphrase nor a fragment that drops the qualifier can pass as a citation.
"""

from __future__ import annotations

import functools
import json
import math
import re
import unicodedata
from pathlib import Path

KNOWLEDGE_DIR = Path(__file__).resolve().parent / "data" / "knowledge"
CORPORA = ("riktlinjer", "metod")
METHOD_LABEL = "Aidas metodbeskrivning"

DEFAULT_LIMIT = 4
# Long enough to answer from, short enough that four hits stay a small tool
# result. A longer section is cut to the paragraphs around the matches.
EXCERPT_MAX = 1200

_WORD_RE = re.compile(r"[0-9a-zåäöéü]+")
_STOPWORDS = frozenset("""
    och att det som en ett är på för av med till i om vad hur ska skall bör vi våra vårt
    vår kan den de har inte eller men så från vid när man mot under över efter enligt
    säger står finns får gäller jag du mig oss någon något några alla detta dessa
    riktlinje riktlinjer riktlinjerna byggriktlinje byggriktlinjer byggriktlinjerna
    aida aidas
""".split())
# Longest first, and never cut a word below four letters, so "golv" survives
# whole while "plastmattor" and "plastmatta" meet at "plastmatt".
_SUFFIXES = ("arnas", "ernas", "ornas", "orna", "arna", "erna", "ande", "ende",
             "or", "ar", "er", "en", "et", "na", "an", "a", "e", "s")
# What differs between a stored sentence and the same sentence quoted by a
# model or read on screen: curly quotes and no-break spaces, bullets, markdown
# emphasis and list markers, table pipes.
_TYPOGRAPHY = str.maketrans({"”": '"', "“": '"', "„": '"', "»": '"', "«": '"', "’": "'",
                             " ": " ", " ": " "})
_LIST_MARK_RE = re.compile(r"^\s*(?:[•●▪◦]|[-+*](?=\s)|\d+\.(?=\s))\s*")
_EMPHASIS_RE = re.compile(r"[*_`]")
# A sentence ends at . ! or ? before a capital, so "t.ex. vinyl" stays one.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÅÄÖ\"(])")
_MD_HEADING_RE = re.compile(r"^(#{2,3})\s+(.*\S)\s*$")

SEARCH_TOOL = {
    "name": "search_knowledge",
    "description": (
        "Sök i Karlstads byggriktlinjer och i Aidas metodbeskrivning. Returnerar avsnitt "
        "med dokument, utgåva, rubrik och text. Använd det för frågor om vad riktlinjerna "
        "kräver och om hur Aida räknar. Sök med sakord, till exempel 'plastmatta', "
        "'radon' eller 'baslinje'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Sakord att söka på."},
            "corpus": {
                "type": "string",
                "enum": list(CORPORA),
                "description": "Begränsa till riktlinjerna eller metoden. Utelämna för att söka i båda.",
            },
        },
        "required": ["query"],
    },
}


def slugify(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")


def section_id(prefix: str, path: list[str], taken: set[str]) -> str:
    """A readable id that survives a new edition as long as the headings do.

    'bygg/invandiga-ytskikt/ytskikt-pa-golv/plastmattor'. A path that repeats
    within one document gets a number, and the id is added to `taken`.
    """
    parts = [slugify(h) for h in path]
    base = "/".join([prefix] + [p for p in parts if p]) if any(parts) else f"{prefix}/inledning"
    sid, n = base, 1
    while sid in taken:
        n += 1
        sid = f"{base}-{n}"
    taken.add(sid)
    return sid


def _markdown_sections(text: str, prefix: str, label: str) -> list[dict]:
    """Split markdown at ## and ### headings. The # title names the document."""
    out: list[dict] = []
    taken: set[str] = set()
    path: list[str] = []
    lines: list[str] = []

    def flush():
        body = "\n".join(lines).strip()
        if body:
            out.append({"id": section_id(prefix, path, taken), "corpus": prefix,
                        "source": label, "path": list(path) or ["Inledning"], "text": body})
        lines.clear()

    for line in text.splitlines():
        match = _MD_HEADING_RE.match(line)
        if match:
            flush()
            depth = len(match.group(1)) - 1
            del path[depth - 1:]
            path.append(match.group(2))
        elif not line.startswith("# "):
            lines.append(line)
    flush()
    return out


@functools.cache
def method_markdown() -> str:
    """metod.md as written, for the "Om verktyget" dialog.

    The dialog and the answers read the same file, so they cannot say different
    things about how Aida works.
    """
    path = KNOWLEDGE_DIR / "metod.md"
    return path.read_text(encoding="utf-8") if path.exists() else ""


@functools.cache
def sections() -> tuple[dict, ...]:
    """Every section of both corpora, in document order."""
    out: list[dict] = []
    for path in sorted((KNOWLEDGE_DIR / "riktlinjer").glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        label = f"{doc['title']}, utgåva {doc['edition']}"
        for s in doc["sections"]:
            out.append({"id": s["id"], "corpus": "riktlinjer", "source": label,
                        "path": s["path"], "part": s.get("part"), "text": s["text"]})
    method = KNOWLEDGE_DIR / "metod.md"
    if method.exists():
        out += _markdown_sections(method.read_text(encoding="utf-8"), "metod", METHOD_LABEL)
    return tuple(out)


@functools.cache
def _index() -> tuple[tuple[dict, str, str], ...]:
    return tuple((s, " ".join(s["path"]).lower(), s["text"].lower()) for s in sections())


@functools.cache
def _by_id() -> dict[str, dict]:
    return {s["id"]: s for s in sections()}


def section(sid: str) -> dict | None:
    return _by_id().get(sid) if isinstance(sid, str) else None


def _stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[:-len(suffix)]
    return word


def _terms(query: str) -> list[str]:
    words = _WORD_RE.findall(query.lower()) if isinstance(query, str) else []
    return list(dict.fromkeys(_stem(w) for w in words if len(w) >= 3 and w not in _STOPWORDS))


def _label(s: dict) -> str:
    heading = " › ".join(s["path"]) + (f" (del {s['part']})" if s.get("part") else "")
    return f"{s['source']} › {heading}"


def label(sid: str) -> str:
    """How a section is named where it is cited: document, edition and heading path."""
    s = section(sid)
    return _label(s) if s else ""


def _excerpt(text: str, terms: list[str]) -> str:
    """The whole section, or the paragraphs around the best match with gaps marked."""
    if len(text) <= EXCERPT_MAX:
        return text
    paragraphs = text.split("\n")
    weights = [sum(p.lower().count(t) for t in terms) for p in paragraphs]
    lo = hi = max(range(len(paragraphs)), key=weights.__getitem__)
    size = len(paragraphs[lo])
    grew = True
    while grew:
        grew = False
        for j in (hi + 1, lo - 1):
            if 0 <= j < len(paragraphs) and size + len(paragraphs[j]) + 1 <= EXCERPT_MAX:
                size += len(paragraphs[j]) + 1
                lo, hi = min(lo, j), max(hi, j)
                grew = True
    window = paragraphs[lo:hi + 1]
    return "\n".join((["[…]"] if lo > 0 else []) + window + (["[…]"] if hi < len(paragraphs) - 1 else []))


def search(query: str, corpus: str | None = None, limit: int = DEFAULT_LIMIT) -> list[dict]:
    """The sections that best match `query`, best first.

    A term in a heading weighs three times a term in the text, rarer terms
    weigh more, and a section matching more of the terms beats one matching a
    single term many times.
    """
    terms = _terms(query)
    pool = [entry for entry in _index() if corpus is None or entry[0]["corpus"] == corpus]
    if not terms or not pool:
        return []
    df = {t: sum(1 for _, head, body in pool if t in head or t in body) for t in terms}
    scored = []
    for s, head, body in pool:
        score, matched = 0.0, 0
        for t in terms:
            if not df[t] or (t not in head and t not in body):
                continue
            matched += 1
            score += math.log(1 + len(pool) / df[t]) * (3 * (t in head) + min(body.count(t), 3))
        if matched:
            scored.append((score * matched / len(terms), s))
    scored.sort(key=lambda pair: -pair[0])
    return [{"id": s["id"], "corpus": s["corpus"], "label": _label(s),
             "excerpt": _excerpt(s["text"], terms)} for _, s in scored[:limit]]


def format_hits(hits: list[dict], query: str) -> str:
    """Search hits as the tool result the model reads."""
    if not hits:
        titles = sorted({s["source"] for s in sections() if s["corpus"] == "riktlinjer"})
        return (f"Inget avsnitt matchar \"{query}\". Pröva andra sakord, till exempel byggdelen "
                f"eller materialet. Källorna är {', '.join(titles)} och {METHOD_LABEL}.")
    blocks = [f"[{h['id']}] {h['label']}\n{h['excerpt']}" for h in hits]
    return ("Citera bara hela meningar, ordagrant ur texten nedan, och ange avsnittets id "
            "inom hakparentes.\n\n"
            + "\n\n".join(blocks))


def run_search(tool_input) -> str:
    """The search_knowledge tool: validate the model's input and search."""
    tool_input = tool_input if isinstance(tool_input, dict) else {}
    query = tool_input.get("query") if isinstance(tool_input.get("query"), str) else ""
    corpus = tool_input.get("corpus") if tool_input.get("corpus") in CORPORA else None
    return format_hits(search(query, corpus), query)


def _clean(text: str) -> str:
    text = _LIST_MARK_RE.sub("", text.translate(_TYPOGRAPHY))
    return " ".join(_EMPHASIS_RE.sub("", text).replace("|", " ").split())


def _core(text: str) -> str:
    """What two renderings of one sentence share: no outer quote marks, no closing
    stop, and the first letter in either case, since a quote woven into a
    sentence of one's own starts in lower case."""
    text = text.strip()
    for _ in range(2):
        text = text.strip("\"'").rstrip(".!?:;,").strip()
    return text[:1].lower() + text[1:]


def quote_in_source(quote: str, sid: str) -> bool:
    """Is `quote` one or more whole sentences, word for word, from section `sid`?

    Whole sentences from one paragraph or bullet, with nothing left out. An
    ellipsis or a fragment can drop the one word that carries the requirement:
    "Dessa ska kopplas direkt mot Citect och […] via DDC." would pass as a quote
    of "... och inte via DDC", and a sentence cut before "får inte förekomma"
    states the opposite of the rule. Crossing a bullet would splice two
    requirements into one. What may differ is what a reader cannot see:
    whitespace, bullets and markdown markers, curly against straight quotes,
    and the closing full stop.
    """
    s = section(sid)
    if s is None or not isinstance(quote, str):
        return False
    wanted = _core(_clean(quote))
    if not wanted:
        return False
    for paragraph in s["text"].split("\n"):
        sentences = _SENTENCE_SPLIT_RE.split(_clean(paragraph))
        for i in range(len(sentences)):
            for j in range(i, len(sentences)):
                if _core(" ".join(sentences[i:j + 1])) == wanted:
                    return True
    return False
