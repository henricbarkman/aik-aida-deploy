"""Is this EPD from a supplier a Swedish förvaltare can actually order from?

The question this file answers is NOT the one `geo` answers, and conflating the
two is the mistake that made this work necessary.

`geo` is the declaration's geographic validity: the region the LCA's electricity
mix, transport distances and waste scenarios were modelled for. It says nothing
about where the product can be bought. Ahlsell AB, the largest Nordic wholesaler,
declares GLO. Elitfönster AB declares RER. Bolon AB declares GLO. Dahl Sverige AB
declares GLO and RER. A filter on `geo in {SE, NORD, DK, NO, FI}` would throw away
precisely the rows a förvaltare can pick up the phone and order, and keep a
Turkish tile with an RER declaration.

What the catalog does carry is `owner`, the company that registered the EPD. A
company registered in a Nordic country has a Nordic sales organisation; that is a
proxy for availability rather than availability itself, and it is stated as a
proxy everywhere it surfaces. It is right about the direction and silent about
the details: it cannot tell whether a specific article is stocked this week.

Measured on the 1335-row catalog 2026-09-06: the legal-form signal alone matches
88 of 468 owners and 279 rows, against 11% for a `geo` filter. The curated list
below adds the Nordic companies whose names carry no legal form at all.
"""

from __future__ import annotations

import re

# Legal forms and country words. `AB`/`Ab` Swedish and Åland, `A/S`/`ApS` Danish
# and Norwegian, `AS`/`ASA` Norwegian, `Oy`/`Oyj` Finnish. ASA added
# 2026-10-02 (HENRIC-3392): Moelven Industrier ASA, nine facade rows, read as
# a foreign supplier.
_LEGAL_FORM = re.compile(
    r"(\bAB\b|\bA/S\b|\bAS\b|\bASA\b|\bOy\b|\bOyj\b|\bApS\b|\bAb\b"
    r"|Sweden|Denmark|Norway|Finland|Iceland"
    r"|Sverige|Danmark|Norge|Suomi|Ísland|Norden|Nordic)",
    re.I,
)

# Nordic suppliers whose registered name carries no legal form, so the pattern
# above cannot see them. Matched as a case-insensitive substring of `owner`.
# Every entry is a company a Swedish förvaltare or their wholesaler orders from.
_KNOWN_NORDIC = (
    "fm mattsson",      # Mora, SE
    "oras",             # Rauma, FI
    "kone",             # Espoo, FI
    "teknos",           # Helsingfors, FI
    "swerock",          # Peab, SE
    "daloc",            # Töreboda, SE
    "extena",           # Åstorp, SE
    "tarkett",          # HQ i Frankrike men egen svensk säljorg och fabrik i Ronneby
    "stora enso",
    "ahlsell",
    "beijer byggmaterial",
    "optimera",
    "byggmax",
    "essve",
    "isover",           # Saint-Gobain Sweden
    "gyproc",
    "weber",            # Saint-Gobain Weber, ej "Weber Turkey" — se _NOT_NORDIC
    "paroc",            # FI
    "rockwool",         # DK
    "kingspan",         # ej nordiskt bolag, men egen nordisk säljorg
    "icopal",
    "mataki",
    "nordiska",
    "flügger",
    "hempel",
    "jotun",            # NO
    "beckers",
    "alcro",
    "nordsjö",
    # Added 2026-09-06 after sweeping all 512 owner strings in the supplemented
    # catalog for Nordic-looking words the pattern had not matched. These three
    # were the entire yield, which is the useful part of the answer: the signal
    # is not systematically leaky, it just cannot read a name with no legal form
    # and no country word.
    "sveden trä",       # Sveden Trä, träfasadpanel, SE
    "etri fönster",     # SSC Etri Fönster, SE
    "byggevarer",       # Saint-Gobain Byggevarer, NO
    # 2026-10-02 (HENRIC-3392): the AIda stage probe offered Vedum's Ilse front
    # as "utländsk leverantör". Environdec registers the owner without "AB".
    "vedum",            # Vedum Kök & Bad, SE
)

# Companies the legal-form pattern matches for the wrong reason. `AS` is also the
# Estonian *aktsiaselts* and turns up in Romanian and Turkish group names, and
# `A.S.`/`A.Ş.` is Turkish *Anonim Şirketi* (that one does not match, because the
# dots break the word boundary, but the near-miss is worth naming here so nobody
# "fixes" the pattern by allowing dots).
#
# Kept deliberately short. A long denylist would mean the signal is wrong and
# should be replaced, not patched.
_NOT_NORDIC = (
    "halla a.s.",           # HALLA a.s., tjeckisk armaturtillverkare
    "technomar adrem",      # rumänsk grupp, AS läses fel
    "saint-gobain weber turkey",
    "saint-gobain construction products romania",
    "saint-gobain hellas",
)


def nordic_supplier(entry: dict) -> bool:
    """True when the EPD's owner is a company with a Nordic sales organisation.

    A proxy for "a förvaltare can order this", not a guarantee. See module
    docstring for why `geo` is not used and must not be added here.
    """
    owner = (entry.get("owner") or "").strip()
    if not owner:
        return False
    lowered = owner.lower()
    if any(bad in lowered for bad in _NOT_NORDIC):
        return False
    if any(known in lowered for known in _KNOWN_NORDIC):
        return True
    return bool(_LEGAL_FORM.search(owner))


def availability_label(entry: dict) -> str:
    """Short Swedish label for the report and the model prompt.

    Deliberately hedged. "Nordisk leverantör" is a claim about the company, and
    the alternative wording ("finns i Sverige") would be a claim about stock that
    the data cannot support.
    """
    return "nordisk leverantör" if nordic_supplier(entry) else "utländsk leverantör"
