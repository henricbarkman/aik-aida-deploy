"""Who is behind an EPD, counted as a company group rather than an owner string.

The typvärde is meant to say what is normal for a kind of product. When one
company's declarations make up most of the rows behind it, it says what that
company declares instead, and nothing about the number looks wrong (see
scripts/test_typvarde_dominans.py for how that was found). Counting that needs
one answer to "is this the same company?", and the owner field cannot give it
on its own: Saint-Gobain declares as Weber-Sodamco, Gyproc AB (Sweden),
Saint-Gobain Ecophon AB and some forty other names, so until 2026-10-01 a
levelling compound bucket where the group declared 48 of 60 rows measured 0.18
(HENRIC-3368).

Two things live here, both used by epd_baseline_medians for the published
typvärde and by the dominance test that checks it:

- `owner_group`: the group an owner belongs to. A curated map for groups whose
  subsidiaries carry other names, each entry with the source that establishes
  the ownership, and for everything else the owner string with its legal form
  and country stripped, so "Saint-Gobain Sweden AB" and "Saint-Gobain Finland
  Oy" meet without a map entry.
- `collapse_plants`: one product made at several plants (Mapelastic Zero is
  declared once per factory, nine times) counted once, so a product's weight in
  a median does not depend on how many factories its maker has.

Over-merging is the safe error for the map: two unrelated companies wrongly
joined make a share look larger, which withholds a typvärde and says so.
Splitting one group hides a dominance and says nothing. Even so, an entry goes
in only with a source; a likely-but-unverified link stays out.
"""

from __future__ import annotations

import re
from statistics import median

# Each entry: group name shown to the user, lowercase substrings of `owner`
# that belong to it, and where the ownership was verified (2026-10-01).
# Order matters only where patterns could overlap; none do today.
GROUPS: list[tuple[str, tuple[str, ...], str]] = [
    ("Saint-Gobain",
     # Specific enough not to catch an unrelated "Weber" or "Glava": every
     # other Weber company in the catalog declares as "Saint-Gobain ... Weber".
     ("saint-gobain", "saint gobain", "ecophon", "gyproc", "isover", "glava as",
      "rigips", "placo", "british gypsum", "weber-sodamco", "dahl sverige",
      "optimera", "kimmco", "vetrotech"),
     "Ecophon: ecophon.com/en/about-us ('Ecophon is part of the global "
     "Saint-Gobain Group'). Gyproc, Isover, Weber, Dalapro: Saint-Gobain Sweden "
     "AB press release on mynewsdesk ('företaget bakom varumärkena Dalapro, "
     "Gyproc, ISOVER och Weber'). Glava: byggmesteren.as 2017-10-11, "
     "Saint-Gobain eier Glava. Rigips: rigips.de ('part of the Saint-Gobain "
     "Group'). Optimera: optimera.se/om-oss/historia ('Optimera ägs idag av "
     "Saint-Gobain'). Dahl: Saint-Gobain still owns it; binding agreement "
     "2026-06-15 to sell Dahl Sweden/Norway/Denmark to Kesko, closing expected "
     "by early 2027 -- move 'dahl sverige' out of this entry then. Kimmco-Isover: "
     "joint venture consolidated in Saint-Gobain's accounts (Fox Business, "
     "2018-01-11). Vetrotech: vetrotech.com/history. Saint-Gobain 2025 Universal "
     "Registration Document, note 16, fully consolidated: Saint-Gobain Denmark "
     "A/S (registered names include Gyproc A/S, CVR 59983016), Placoplatre SA, "
     "Saint-Gobain Placo Iberica, Saint-Gobain Construction Products United "
     "Kingdom Ltd (British Gypsum), Saint-Gobain PAM Canalisation, Saint-Gobain "
     "Distribution Sweden AB. Weber-Sodamco: Saint-Gobain Weber completed the "
     "integration of Sodamco Holding (meconstructionnews.com, 2012-12-18)."),
    ("Rockwool",
     ("rockwool", "rockfon", "chicago metallic"),
     "rockwool.com/group/about-us lists Rockfon as a ROCKWOOL brand; ROCKWOOL "
     "Group acquired Chicago Metallic through its ROCKFON affiliate "
     "(GlobeNewswire 2013-08-19)."),
    ("Ballingslöv International",
     ("ballingslöv", "jke design", "kvik a/s"),
     "ballingslovinternational.se/en/about-us lists Ballingslöv, Multiform, "
     "Kvik, DANKÜCHEN, JKE Design, DFI-Geisler and KBV; /en/governance: "
     "'wholly owned by Stena Adactum AB'. Vedum is not in the group."),
    ("Lammhults Design Group",
     ("lammhults", "abstracta", "fora form", "ragnars"),
     "lammhultsdesigngroup.com/en/history: the group includes Abstracta "
     "(1999), Lammhults (1994), Fora Form and Ragnars."),
    ("Eczacıbaşı",
     ("eczacıbaşı", "eczacibasi", "vitra karo", "villeroy & boch tiles"),
     "eczacibasi.com.tr/en/field-of-activity/building-products: 'V&B Fliesen "
     "GmbH was acquired by the Eczacıbaşı Group in 2007'; VitrA Karo trades "
     "under the VitrA, Villeroy & Boch and engers brands."),
    ("Oras Group",
     ("oras group", "gustavsberg"),
     "Oras Group completed the acquisition of the Gustavsberg and Vatette "
     "businesses from Villeroy & Boch in 2025 (en.wikipedia.org/wiki/"
     "Oras_(company); announced 2025-07-11). Villeroy & Boch Gustavsberg AB is "
     "therefore Oras, not Villeroy & Boch."),
    ("Inwido",
     ("elitfönster", "diplomat dörrar"),
     "inwido.com lists Elitfönster (Scandinavia) and Diplomat. Svenska Fönster "
     "is VKR (svenskafonster.se/om-oss/agare-vkr)."),
    ("VKR",
     ("dovista", "svenska fönster"),
     "dovista.com/the-dovista-group/ejerforhold: 'DOVISTA A/S is 100% owned by "
     "VKR Holding A/S'; svenskafonster.se/om-oss/agare-vkr: 'ägs sedan år 2000 "
     "av VKR-gruppen'."),
    ("K-Svets Förvaltning",
     ("h-fönstret", "leiab"),
     "hfonstret.se/foretaget/ledning-o-agarstruktur: 'ingår i den familjeägda "
     "K-Svets Förvaltnings Aktiebolag där även LEIAB ingår'."),
    ("Mapei",
     ("mapei",),
     "Mapei 2025 consolidated financial statements (cdnmedia.mapei.com): '92 "
     "subsidiaries'; the group-company lists name Mapei As, Mapei Corp, Mapei "
     "UK, Mapei Gmbh (A), Mapei Hellas, Mapei Polska, Mapei Portugal, Mapei "
     "China, Mapei Malaysia, Mapei Far East, Mapei Vietnam, PT Mapei Indonesia."),
    ("Svedbergs Group",
     ("svedbergs", "macro design"),
     "Svedbergs Group year-end report 2025 (mfn.se, 2026-02-04) reports Macro "
     "Design as a group company; subsidiaries Svedbergs i Dalstorp AB and Macro "
     "Design AB (sv.wikipedia.org/wiki/Svedbergs_Group)."),
    ("Otis",
     ("otis elevator", "otis electric"),
     "Otis Worldwide 10-K FY2025, Exhibit 21 (sec.gov): Otis Electric Elevator "
     "Company Limited and Otis Elevator (China) Investment Company Limited."),
    ("Kingspan",
     ("kingspan", "tate global"),
     "Kingspan annual report 2025, principal subsidiaries: 'Czechia Kingspan AS "
     "100' and 'Tate Global Solutions Limited 100' (Companies House: formerly "
     "Kingspan Access Floors Limited)."),
    ("Knauf",
     ("knauf",),
     "Knauf Insulation: career.knauf.com/pages/our-businesses. Knauf Cyprus: "
     "'part of the global German Knauf Group' (greatplacetowork.com.cy). Knauf's "
     "Russian business (OOO Knauf) is still the group's after a failed sale "
     "(Reuters, 2025-10). Knauf Oy, Knauf UK/I and Yesos Knauf carry the group's "
     "name; the privately held group publishes no subsidiary list."),
    ("Kronospan",
     ("kronospan",),
     "kronospan.com/en_US/company/kronology and /en_TR/contacts list Kronospan "
     "Riga, Kronospan Tortosa, Kronospan Orman Urunleri and Kronospan GmbH."),
    ("CELSA",
     ("celsa",),
     "celsaho.com/en/celsa-huta-ostrowiec: 'companies that operate under the "
     "CELSA GROUP brand'; Compañía Española de Laminación is CELSA's Spanish "
     "company (suppliers.catalonia.com)."),
    ("Etex (URSA)",
     ("ursa ibérica", "ursa iberica", "ursa italy", "ursa poland"),
     "ursa.com/en/about-us/our-company: 'part of the Etex Group' (since 2022)."),
    ("Fibran",
     ("fibran",),
     "fibran.com/about-us lists FIBRAN SpA Italy and FIBRAN Bulgaria; Bulgaria "
     "is a joint venture with Fibran Greece (bsgroupofcompanies.com)."),
    ("Tarkett",
     ("tarkett",),
     "Tarkett consolidated financial statements 2024: 'Tarkett Industrial "
     "(Beijing) Co, Ltd China G 100% 100%'."),
    ("Soprema",
     ("soprema",),
     "soprema-international.com/en/history: Belgian BITAL renamed SOPREMA NV; "
     "Italian FLAG acquired."),
    ("isoplus",
     ("isoplus",),
     "isoplus.group locations page lists isoplus Fernwärmetechnik (AT) and "
     "isoplus Suomi Oy as group sites."),
    ("UPM",
     ("upm plywood", "upm timber"),
     "UPM half-year report 2026: Plywood is demerged into WISA Group Plc, "
     "planned 2026-10-31 -- take 'upm plywood' out of this entry then."),
    ("ArcelorMittal",
     ("arcelormittal",),
     "constructalia.arcelormittal.com 2025-10-14: 'ArcelorMittal Construction "
     "is now ArcelorMittal Building Solutions'; tubular.arcelormittal.com."),
    ("Boero",
     ("boero bartolomeo",),
     "attivacolori.it/il-gruppo: Attiva is operated by Boero Bartolomeo S.p.A."),
    ("Kerakoll",
     ("kerakoll",),
     "kerakollgroup.com/en/about-us; legal entity Kerakoll S.p.A."),
    ("Sealed Air",
     ("sealed air",),
     "Sealed Air FY2024 Exhibit 21 (sec.gov): 'Sealed Air S.r.l. Italy'; "
     "private under CD&R since 2026-04-09."),
]
# Checked and NOT joined (2026-10-01): Hunton Fiber AS (Skog Holding) and
# Huntonit AS (Byggma ASA); Optima Products Ltd (Optima Contracting) and RP
# Products / Planet Partitioning (Radii Planet); Villeroy & Boch Gustavsberg
# (Oras Group since 2025, not V&B; FM Mattsson and Oras are separate); Swedese
# (Patino Group) and Lammhults; Vedum and Ballingslöv International; Flexit
# (family-owned) and any other ventilation maker in the catalog. Chalkis LTD
# against CHALKIS S.A. could not be settled and is left apart.
# Air handling units, HENRIC-3386: Swegon Group AB is one of Investment AB
# Latour's wholly owned business areas (Latour year-end report 2025), S&P
# Sistemas de Ventilación is the Soler & Palau Ventilation Group (private;
# Pluggit, Exhausto, Fantech, Ventur among its brands), Acetec AB belongs to
# Svevik Industri AB (press release 2022-06-22). No other company of these
# groups declares in the catalog's ventilation rows, so no entry is needed;
# none of them is joined with Kampmann, Flexit, Salda or Zehnder.

_LEGAL_SUFFIXES = re.compile(
    r"\b(ab|a/s|as|oy|oyj|ltd|limited|gmbh|sa|inc|bv|nv|srl|spa|aps|plc|corp|co"
    r"|group|sweden|norge|denmark|finland)\b",
    re.IGNORECASE,
)


def _strip_legal(owner: str) -> str:
    # ​: "Saint-Gobain Distribution Sweden​" carries a zero-width
    # space that kept "sweden" from matching \b and the name from meeting its
    # siblings.
    cleaned = (owner.lower().replace("​", " ").replace(",", " ")
               .replace(".", " "))
    return " ".join(_LEGAL_SUFFIXES.sub(" ", cleaned).split()) or owner.lower().strip()


def owner_group(owner: str | None) -> str:
    """The company group behind an EPD owner string."""
    text = (owner or "?").lower().replace("​", " ")
    for name, patterns, _source in GROUPS:
        if any(p in text for p in patterns):
            return name
    return _strip_legal(owner or "?")


def owner_group_label(owner: str | None) -> str:
    """The group name as a user reads it: the map's name for a mapped group,
    otherwise the owner string as declared (stripped of trailing noise), since
    "AJ Produkter AB" is what someone can look up and "aj produkter" is not."""
    text = (owner or "?").lower().replace("​", " ")
    for name, patterns, _source in GROUPS:
        if any(p in text for p in patterns):
            return name
    return (owner or "?").replace("​", "").strip()


def is_mapped_group(owner: str | None) -> bool:
    text = (owner or "?").lower().replace("​", " ")
    return any(any(p in text for p in pats) for _n, pats, _s in GROUPS)


# ---------------------------------------------------------------------------
# One product, several plants
# ---------------------------------------------------------------------------

# A trailing parenthesis that names where the product was made rather than what
# it is: "(China Production)", "(Portugal production)", "(PL)", "(AU)". Not
# "(Plant-based)", and not a market ("(Nordic & Baltic market)" can be another
# formulation). Two
# letters only when they are a country code, so "(PB)" in "Particleboard (PB)"
# stays part of the name.
_COUNTRY_CODES = frozenset({
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU",
    "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES",
    "SE", "NO", "IS", "CH", "GB", "UK", "US", "CA", "MX", "BR", "AR", "CL", "AU",
    "NZ", "CN", "HK", "TW", "JP", "KR", "IN", "ID", "MY", "SG", "TH", "VN", "PH",
    "TR", "AE", "SA", "ZA", "EG",
})
_PLACE_MARKER = re.compile(
    r"\s*\(([^()]*\b(?:production|produced|plant(?!-)|factory|made in)\b[^()]*"
    r"|[A-Z]{2})\)\s*$",
    re.IGNORECASE,
)


def product_name(name: str) -> str:
    """The product's name with a trailing place-of-manufacture marker removed,
    lowercased. "Mapelastic Zero (China Production)" -> "mapelastic zero"."""
    n = (name or "").strip()
    m = _PLACE_MARKER.search(n)
    if m:
        inner = m.group(1).strip()
        if len(inner) != 2 or inner.upper() in _COUNTRY_CODES:
            n = n[:m.start()]
    return " ".join(n.lower().split())


def _site(row: dict) -> tuple[str, str]:
    return ((row.get("owner") or "").strip().lower(), (row.get("geo") or "").strip())


def collapse_plants(rows: list[tuple[float, dict]]) -> list[tuple[float, dict]]:
    """Rows with plant duplicates folded into one row per product.

    Two rows are one product made at two plants when they have the same group,
    the same product name (place marker removed) and come from different sites,
    a site being the declaring entity and the declared geography. Each site
    then contributes exactly one row. That condition is what tells a plant
    apart from the other reasons one name occurs twice:

    - the same name from one site in several values is a size range or a
      re-issued declaration ("Bathroom cabinet" four times from Sonas, 57 to
      81 kg/st), and stays as it is, because nothing in the row says which;
    - a group where some site has two rows and another one is mixed, and is
      also left alone rather than paired by guess.

    Exact duplicates (same group, name and value) are one row regardless of
    site: two registrations of the same figure for the same product.

    The folded product's value is the median of its plants, and its row is
    the first one's with `plants` set to how many it stands for. Rows without
    a name are never folded.
    """
    groups: dict[tuple[str, str], list[int]] = {}
    for i, (_gwp, row) in enumerate(rows):
        name = product_name(row.get("name", ""))
        if not name:
            continue
        groups.setdefault((owner_group(row.get("owner")), name), []).append(i)

    drop: set[int] = set()
    replace: dict[int, tuple[float, dict]] = {}
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        # Exact repeats first: same value -> one row.
        by_value: dict[float, list[int]] = {}
        for i in idxs:
            by_value.setdefault(round(rows[i][0], 6), []).append(i)
        kept = []
        for same in by_value.values():
            kept.append(same[0])
            drop.update(same[1:])
        if len(kept) < 2:
            if len(idxs) > 1:
                first = kept[0]
                replace[first] = (rows[first][0], {**rows[first][1], "plants": len(idxs)})
            continue
        sites = [_site(rows[i][1]) for i in kept]
        if len(set(sites)) != len(sites):
            continue  # a site with two values: sizes or versions, left alone
        first = kept[0]
        value = float(median(rows[i][0] for i in kept))
        replace[first] = (value, {**rows[first][1], "plants": len(idxs)})
        drop.update(kept[1:])
    return [replace.get(i, r) for i, r in enumerate(rows) if i not in drop]


# ---------------------------------------------------------------------------
# Concentration
# ---------------------------------------------------------------------------

# One group holding this share or more of a key's products makes the typvärde
# that group's range, and it is not published. The same 0.6 as OWNER_SHARE in
# scripts/audit_queue_fronts.py and the geo scope's fallback.
DOMINANCE_CEILING = 0.6


def concentration(rows: list[tuple[float, dict]]) -> dict:
    """How concentrated the population behind a typvärde is.

    Returns n (products), top and second (each {group, count, owners}, owners
    being the declared names behind the group, commonest first), top_share,
    top2_share, and two_suppliers: whether what is left after the largest
    group is itself dominated by one group, by the same ceiling. A key where
    that holds rests in practice on two companies' ranges. It is published,
    and the row says so; see metod.md and the dominance test for why that is
    a disclosure and not a second ceiling.
    """
    n = len(rows)
    counts: dict[str, int] = {}
    owners: dict[str, dict[str, int]] = {}
    for _, e in rows:
        g = owner_group(e.get("owner"))
        counts[g] = counts.get(g, 0) + 1
        raw = (e.get("owner") or "?").replace("​", "").strip()
        owners.setdefault(g, {})
        owners[g][raw] = owners[g].get(raw, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    def _entry(i: int) -> dict | None:
        if i >= len(ranked):
            return None
        g, c = ranked[i]
        names = [o for o, _ in sorted(owners[g].items(), key=lambda kv: (-kv[1], kv[0]))]
        first_owner = next(e.get("owner") for _, e in rows if owner_group(e.get("owner")) == g)
        return {"group": owner_group_label(first_owner), "count": c, "owners": names,
                "mapped": is_mapped_group(first_owner)}

    top, second = _entry(0), _entry(1)
    top_n = top["count"] if top else 0
    second_n = second["count"] if second else 0
    rest = n - top_n
    return {
        "n": n,
        "top": top,
        "second": second,
        "top_share": round(top_n / n, 3) if n else 0.0,
        "top2_share": round((top_n + second_n) / n, 3) if n else 0.0,
        "two_suppliers": bool(rest and second_n / rest >= DOMINANCE_CEILING),
    }


def describe_group(entry: dict) -> str:
    """'Saint-Gobain (Weber-Sodamco, Saint-Gobain Sweden AB m.fl.)' or the
    declared name alone when the group is one owner string."""
    owners = entry["owners"]
    if not entry.get("mapped"):
        return owners[0]
    # Semicolons: an owner string can itself hold a comma ("Saint-Gobain
    # Sweden AB, Weber floor"), and a comma list would split it in two.
    shown = "; ".join(owners[:2]) + (" m.fl." if len(owners) > 2 else "")
    return f"{entry['group']} ({shown})"


def dominance_reason(conc: dict) -> str:
    """Why a key is withheld, in the words the row shows, or '' if it is not."""
    top = conc.get("top")
    if not top or conc["top_share"] < DOMINANCE_CEILING:
        return ""
    kind = "koncern" if top.get("mapped") and len(top["owners"]) > 1 else "leverantör"
    return (f"{top['count']} av {conc['n']} produkter med EPD kommer från en och "
            f"samma {kind}, {describe_group(top)}, så ett typvärde vore deras "
            f"sortiment och inte ett typiskt val")


def concentration_note(conc: dict | None) -> str:
    """The sentence every row with a typvärde carries about who is behind it."""
    if not conc or not conc.get("top"):
        return ""
    top, second, n = conc["top"], conc.get("second"), conc["n"]
    text = f" Största leverantör: {describe_group(top)}, {top['count']} av {n} produkter."
    if second:
        text = (f" Största leverantörer: {describe_group(top)} {top['count']} och "
                f"{describe_group(second)} {second['count']} av {n} produkter.")
    if conc.get("two_suppliers"):
        text += " Typvärdet vilar i praktiken på dessa två leverantörers sortiment."
    return text
