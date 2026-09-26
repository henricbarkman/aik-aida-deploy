"""Utilities for component name normalization and reasoning templates.

Hardcoded climate data has been removed. Data sources:
- Baseline: Boverkets klimatdatabas (Typical A1-A3) + LLM estimation
- Alternatives: Environdec EPD:er (epd_alternatives.json)
- Reuse: Palats API (live)
"""

from __future__ import annotations

import re

# Reasoning templates per alternative type
REASONING = {
    "reuse": "Återbruk eliminerar nästan all tillverkningsrelaterad klimatpåverkan. Kvarvarande CO2e kommer främst från transport och eventuell renovering av materialet.",
    "climate_optimized": "Klimatoptimerat alternativ med lägre CO2e-avtryck jämfört med konventionell produkt, genom val av material med lägre inbyggd klimatpåverkan.",
    "conventional": "Konventionell nyproduktion utan särskild klimathänsyn. Representerar baslinjen: vad standardmaterial kostar klimatmässigt (Boverket Typical A1-A3).",
}


# Household fridges and freezers are vitvaror: in Swedish the word itself means
# "kyl, frys, spis, disk, tvätt". Until 2026-09-26 kylanläggning's bare "kyl"
# caught "Kylskåp" first, and kylanläggning is commercial refrigeration: its
# catalog rows are chillers and heat pumps and its price range starts at
# 30 000 kr, so a correctly searched 8 000 kr fridge was "corrected" to a
# 265 000 kr midpoint (Fable audit 2026-07-19, P2 #10).
#
# Read per word, because "kyl" is also the head of every commercial compound
# (kylrum, kylaggregat, kylbaffel, "kyl- och frysrum" in a storkök) and those
# stay kylanläggning. A word containing one of the stems is a household
# appliance; a bare "kyl"/"frys" word is one too, unless the same name also
# carries a commercial kyl/frys compound, which then decides.
_HOUSEHOLD_COLD_STEMS: tuple[str, ...] = (
    "kylskåp", "frysskåp", "kylfrys", "frysbox", "kombiskåp",
)
_HOUSEHOLD_COLD_WORDS = frozenset({
    "kyl", "kylen", "kylar", "kylarna", "frys", "frysen", "frysar", "frysarna",
})

# Beams and columns named without a material prefix. Bare substrings were the
# bug: "balk" is in "Balkongdörr" and "pelare" in "Duschpelare", so a balcony
# door and a shower tower were both priced as load-bearing frame (Fable audit
# 2026-07-19, P2 #10). Matched per word and from the word's start, which keeps
# "Balk", "Balkar", "Pelare" and "Pelarna" while "balkong*" and "*pelare"
# compounds fall through to their own categories. The frame compounds that do
# end in these words (stålbalk, betongpelare, takbalk...) are listed in the
# stomme row below.
_WORD_RE = re.compile(r"[a-zåäöéü]+")


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _names_bare_beam_or_column(text: str) -> bool:
    for w in _words(text):
        if w.startswith("balk") and not w.startswith("balkong"):
            return True
        if w.startswith("pelar"):
            return True
    return False


def names_household_cold_appliance(text: str) -> bool:
    """True when `text` names a household fridge or freezer ("Kylskåp",
    "Kyl/frys", "Ny kyl Electrolux"), False for commercial refrigeration
    ("Kylaggregat", "Kyl- och frysrum"). Shared with palats_client so a
    component and a listing are read the same way: reuse matching compares
    their categories."""
    household = False
    for w in _words(text):
        if any(s in w for s in _HOUSEHOLD_COLD_STEMS) or w in _HOUSEHOLD_COLD_WORDS:
            household = True
        elif "kyl" in w or "frys" in w:
            return False
    return household


# Checked before the substring table below. Plain substring matching cannot
# express "the primary noun wins", so a name carrying two category words lands
# wherever the table happens to look first. These two compounds are unambiguous
# on their own: both name a paint product, and without them yttervägg's bare
# "fasad" swallowed "fasadfärg" and priced a coat of paint as a wall. Keep the
# list to words that can only ever mean one thing.
_PRIORITY_TOKENS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("farg", ("fasadfärg", "fasadmålning")),
    # "armatur" is a lighting word in belysning below, but Swedish plumbing
    # uses it too: a "blandararmatur" is a tap. Belysning is checked before
    # sanitet, so without these the bare stem would price a mixer as a lamp.
    ("sanitet", ("blandararmatur", "sanitetsarmatur", "tvättställsarmatur",
                 "duscharmatur")),
    # Same trap the other way round: "tak" is checked before belysning, so a
    # "taklampa" landed in tak and got a roof typvärde per m2.
    ("belysning", ("taklampa", "takarmatur", "takbelysning")),
    # Fast inredning, added 2026-09-14 (Notion 3d6b0484). Every token names a
    # cabinet, front or worktop that is mounted to the building. They sit here
    # because the table below would misread them: "tvättställsskåp" contains
    # sanitet's "tvättställ", and "spegelskåp med belysning" belysning's word.
    # Loose furniture (förvaringsskåp, garderob, hyllor) is deliberately absent:
    # whether loose interior belongs in the baseline is still an open question.
    ("fast_inredning", ("spegelskåp", "tvättställsskåp", "badrumsskåp",
                        "badrumsinredning", "köksskåp", "kökslucka", "köksluckor",
                        "kökslåd", "köksinredning", "bänkskåp", "överskåp",
                        "underskåp", "bänkskiv", "fast inredning",
                        "fast_inredning")),
)


# Sanitary fixtures a cabinet is often named together with. Found in review of
# the fast_inredning change: "Tvättställ med underskåp" is a washbasin that has
# a cabinet, not a cabinet, and before the change it got sanitet/handfat.
_FIXTURE_WORDS = ("tvättställ", "handfat", "toalett", "wc", "dusch", "badkar",
                  "urinal", "blandare", "kran")


def fixture_named_besides_cabinet(text: str, cabinet_tokens) -> bool:
    """True when `text` still names a sanitary fixture once the fixed-interior
    words are removed. "Tvättställsskåp" is then a cabinet (nothing is left),
    "Tvättställ med underskåp" a washbasin (the fixture word remains).
    Shared with palats_client so a listing and a component read the same way.
    """
    rest = text.lower()
    for t in sorted(cabinet_tokens, key=len, reverse=True):
        rest = rest.replace(t, " ")
    return any(w in rest for w in _FIXTURE_WORDS)


def normalize_component_name(name: str) -> str:
    """Normalize a Swedish component name to match our data keys."""
    name_lower = name.lower().strip()

    for key, tokens in _PRIORITY_TOKENS:
        if any(t in name_lower for t in tokens):
            if key == "fast_inredning" and fixture_named_besides_cabinet(name_lower, tokens):
                continue
            return key

    # After the priority tokens, so "Överskåp kyl/frys" is still a cabinet, and
    # before the table, where kylanläggning's "kyl" would take it.
    if names_household_cold_appliance(name_lower):
        return "vitvaror"

    mappings = {
        # Keramik/kakel checked BEFORE golv: "golvklinker" contains "golv", so
        # golv would otherwise steal it. Ceramic wall+floor tile share one
        # material bucket; klinker (floor tile) moved here from golv 2026-06-17
        # so it gets a ceramic typvärde, not the vinyl/linoleum-dominated golv one.
        "kakel": ["kakel", "klinker", "keramik", "kakelplatt", "väggkakel",
                  "kakla", "ceramic tile"],
        # "matta" covers plastmatta, textilmatta and heltäckningsmatta by
        # substring; the compounds are listed anyway so the intent is legible.
        # Which floor a matta is (vinyl vs textil typvärde) is decided one
        # level down, by subtype_from_material in epd_baseline_medians.
        "golv": ["golv", "floor", "golvbeläggning", "vinylgolv",
                 "laminat", "parkett", "trägolv", "golvmaterial",
                 "matta", "mattor", "plastmatta", "textilmatta",
                 "heltäckningsmatta"],
        # Paint moved out to the dedicated "farg" category 2026-06-17 (it has its
        # own EPDs and a wrong unit basis vs gypsum). "ytskikt" stays here since
        # an unqualified surface layer on an interior wall is usually board, not paint.
        "innervägg": ["innervägg", "innerväggar", "interior wall", "gipsvägg",
                      "mellanvägg", "gipsskiva", "byggskivor", "byggskiva",
                      "ytskikt", "väggöverdraget", "väggöverdrag"],
        # Facade CLADDING, checked before yttervägg. Renovating the skin of a
        # building is not the same job as building a wall, and conflating the
        # two made both numbers wrong at once: Sara's 22mm wood panel was
        # measured against a typvärde for a complete ~200mm wall build-up, so
        # the baseline came out far too high and any alternative looked
        # spectacular. Same trap PROJECT.md notes for roof membranes vs whole
        # roofs. Bare "fasad" deliberately stays with yttervägg below, since
        # unqualified it usually means the wall; the compounds that name the
        # layer land here.
        # Stems, not full words: "fasadskiva" would miss "fasadskivor".
        "fasadskikt": ["fasadskikt", "fasadpanel", "fasadbeklädnad",
                       "fasadskiv", "fasadplatt", "fasadrenovering",
                       "fasadbyte", "träfasad", "träpanel", "panelbräd",
                       "wood cladding", "timber cladding", "facade cladding"],
        "yttervägg": ["yttervägg", "ytterväggar", "fasad", "exterior wall",
                      "puts", "bruk", "tegel", "tegelfasad"],
        # Bärande stomme (steel/timber/concrete frame). Placed BEFORE betongvägg
        # so a "betongbjälklag" (a floor slab is frame, not wall) routes here via
        # "bjälklag", while a plain "betongvägg" still falls through below. We
        # deliberately omit bare "bärande" (would steal load-bearing walls) and
        # bare "stål" (would steal ventilation's "stålkanal"). Bare "balk" and
        # "pelare" are not substrings here any more: see
        # _names_bare_beam_or_column, checked on this row in the loop below.
        "stomme": ["stomme", "stomsystem", "stålstomme", "stålbalk",
                   "stålpelare", "stålbjälke", "limträ", "limträbalk",
                   "kl-trä", "klträ", "korslimmat", "massivträ", "träbalk",
                   "träpelare", "betongbalk", "betongpelare", "bjälklag",
                   "håldäck", "takbalk", "bärbalk"],
        "betongvägg": ["betongvägg", "betong", "concrete"],
        "fönster": ["fönster", "window", "fönsterbyte", "energiglas"],
        "tak": ["tak", "roof", "takpannor", "takbeläggning", "yttertak",
                "takprodukter"],
        "isolering": ["isolering", "insulation", "tilläggsisolering",
                      "mineralull", "cellplast", "glasull", "stenull",
                      "cellulosa", "eps"],
        "storköksutrustning": ["storköksutrustning", "storkök", "diskmaskin",
                               "diskutrustning", "industrial kitchen"],
        # Heat pumps live here, not in radiator: a heat pump is a refrigeration
        # machine (compressor, refrigerant) and the catalog's three heat-pump
        # EPDs sit in kylanläggning, whose st typvärde is literally a
        # geothermal heat pump. A radiator is the emitter on the other end.
        "kylanläggning": ["kylanläggning", "kyl", "kylsystem", "refriger",
                          "cooling", "kylutrustning", "värmepump", "heat pump",
                          "bergvärme"],
        # Stem "armatur", not "armaturer": matching is substring, so the
        # plural never matched "LED-armatur". Plumbing armaturer are caught
        # by _PRIORITY_TOKENS above before this row is reached.
        "belysning": ["belysning", "ljus", "lighting", "lampor", "armatur",
                      "led-armatur"],
        "ventilation": ["ventilation", "ventilationskanal", "fläkt",
                        "stålkanal"],
        # Door subtypes (inner/ytter/brand/skjut) are one category here; the
        # split is made in palats_client.SUBCATEGORY_KEYWORDS["dörr"], the
        # same taxonomy the reuse search uses. "ytterport" has no "dörr" in it
        # and fell through to "" (LLM estimate). Bare "port" stays out: it is
        # inside transport, rapport and export.
        "dörr": ["dörr", "dörrar", "door", "innerdörr", "ytterdörr",
                 "branddörr", "entrédörr", "ytterport", "entréport"],
        "hiss": ["hiss", "elevator", "personhiss"],
        "sanitet": ["sanitet", "toalett", "wc", "handfat", "tvättställ",
                    "dusch", "badkar", "urinal", "blandare", "toilet",
                    "washbasin", "shower"],
        # After sanitet, not in the priority tokens: a "diskbänksblandare" is a
        # tap and must reach sanitet's "blandare" first.
        "fast_inredning": ["diskbänk", "diskho"],
        "vitvaror": ["vitvaror", "tvättmaskin", "torktumlare", "torkskåp",
                     "spis", "häll", "ugn", "mikrovåg", "köksfläkt",
                     "cooker hood", "washing machine"],
        # Renovation materials added 2026-06-17. Terms kept specific to avoid
        # stealing: no bare "el" (matches "element"/"elektronik"), no bare
        # "element" (matches "betongelement"), no bare "färg" (matches colours).
        # No bare "rör"/"stam": "rör" is too short and "stam" risks false hits.
        # Compounds below cover the real renovation cases (stambyte = pipe
        # replacement). "ventilationsrör" is already caught by ventilation above
        # ("ventilation" is a substring of it).
        "vvs": ["vvs", "stambyte", "stamledning", "avloppsrör", "vattenrör",
                "spillvatten", "tappvatten", "rörledning", "kopparrör",
                "dagvatten", "avlopp"],
        "farg": ["målning", "ommålning", "väggfärg", "fasadfärg",
                 "dispersionsfärg", "grundfärg", "målningsarbete"],
        "el": ["elkabel", "kabel", "kablage", "elinstallation", "elledning",
               "starkström", "elcentral"],
        "radiator": ["radiator", "radiatorer", "värmeelement", "värmepanel",
                     "handdukstork"],
    }

    for key, variants in mappings.items():
        for v in variants:
            if v in name_lower:
                return key
        if key == "stomme" and _names_bare_beam_or_column(name_lower):
            return key

    return ""


# Canonical category keys produced by normalize_component_name — also the valid
# values for a component's declared `category` field. Keep in sync with the
# mappings above, intake's category enum, and the EPD catalog categories.
VALID_CATEGORIES = {
    "kakel", "golv", "innervägg", "yttervägg", "fasadskikt", "stomme",
    "betongvägg", "fönster", "tak", "isolering", "storköksutrustning",
    "kylanläggning", "belysning", "ventilation", "dörr", "hiss", "sanitet",
    "vitvaror", "vvs", "farg", "el", "radiator", "fast_inredning",
}

_FOLD = str.maketrans("åäö", "aao")

# ASCII-folded spelling -> canonical key. Opus 5 writes declared categories
# without diacritics ("fonster", "dorr", "storkoksutrustning") in real intake
# runs, measured 2026-09-14. An exact set lookup rejected those, so the declared
# category was silently dropped and the component fell back to name guessing.
_FOLDED_CATEGORIES = {c.translate(_FOLD): c for c in VALID_CATEGORIES}


def canonical_category(raw: str | None) -> str:
    """Canonical key for a declared category, accepting the folded spelling.

    "fonster" -> "fönster", "Färg" -> "farg". Anything that is not a valid key
    in either spelling comes back stripped and lowercased but otherwise as
    given, so resolve_category can still reject it and fall back to the name.
    """
    cat = (raw or "").strip().lower()
    if cat in VALID_CATEGORIES:
        return cat
    return _FOLDED_CATEGORIES.get(cat.translate(_FOLD), cat)


def resolve_category(name: str, declared_category: str = "") -> str:
    """Resolve a component's EPD category.

    Honors the component's declared `category` (set by intake, which knows
    kakel/vvs/farg/el/radiator) when it is a known catalog category — so a
    tiled wall intake tagged `kakel` is treated as kakel by BOTH the baseline
    and the alternatives step, instead of being silently re-derived to
    innervägg from its name "Väggytskikt". Falls back to name-based
    normalization when the declared category is missing or unknown.
    """
    cat = canonical_category(declared_category)
    if cat in VALID_CATEGORIES:
        return cat
    return normalize_component_name(name)


