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
    # "taklampa" landed in tak and got a roof typvärde per m2. And "golv" took
    # "Golvlampa" into golv with 77 floor coverings as alternatives
    # (2026-09-28); a desk lamp would reach the furniture reader's "bord".
    ("belysning", ("taklampa", "takarmatur", "takbelysning", "golvlampa",
                   "golvarmatur", "bordslampa", "bordsarmatur", "läslampa",
                   "vägglampa", "fönsterlampa")),
    # Fast inredning, added 2026-09-14 (Notion 3d6b0484). Every token names a
    # cabinet, front or worktop that is mounted to the building. They sit here
    # because the table below would misread them: "tvättställsskåp" contains
    # sanitet's "tvättställ", and "spegelskåp med belysning" belysning's word.
    # Loose furniture (förvaringsskåp, garderob, hyllor) is los_inredning, read
    # by furniture_subcategory below. "Platsbyggd" makes anything fixed: a
    # "platsbyggd garderob" is joinery, not a wardrobe you can move.
    ("fast_inredning", ("spegelskåp", "tvättställsskåp", "badrumsskåp",
                        "badrumsinredning", "köksskåp", "kökslucka", "köksluckor",
                        "kökslåd", "köksinredning", "bänkskåp", "överskåp",
                        "underskåp", "bänkskiv", "fast inredning",
                        "fast_inredning", "platsbyggd")),
    # Loose interior as a category word, HENRIC-3290 del 2. After fast
    # inredning, so "platsbyggda möbler" stays fixed. A name that says only
    # "Möbler" lands here without a subcategory, and the alternatives step
    # asks which kind rather than comparing chairs with sofas.
    ("los_inredning", ("lös inredning", "lösa inredning", "los_inredning",
                       "lösa möbler", "möbler", "möblering")),
    # Air handling units, added 2026-09-28. They sit here and not on the
    # ventilation row below because kylanläggning is checked first and its bare
    # "kyl" would take "Luftbehandlingsaggregat med kylbatteri". "FTX-aggregat"
    # had no category at all before. A plain "Kylaggregat" names none of these
    # and stays kylanläggning. Bare "ftx" is safe as a substring in a Swedish
    # component name; it is NOT safe in EPD names (Jotun's "Ultra One D FTX" is
    # a paint), which is why the catalog side matches it as a word.
    ("ventilation", ("ftx", "ventilationsaggregat", "luftbehandlingsaggregat",
                     "luftbehandlingsenhet")),
)


# Lightweight expanded clay (lättklinker, Leca) is a masonry and frame
# material, not a tile, but "klinker" is a substring of it, so "Lättklinkerbalk"
# and "Lättklinkerblock" were priced as kakel per m2 (2026-09-28). Read before
# the table: a beam or a slab is stomme, everything else (blocks, walls) is
# betongvägg, the nearest mineral-wall bucket. "Golvklinker" and "klinkergolv"
# carry neither stem and stay kakel.
# Studs (reglar), HENRIC-3290. "Reglar", "Stålreglar" and "Träreglar" matched
# nothing, so the component had no category and no alternatives at all. Read
# per word and from the END of the word, because "regel" is also the head of
# "regelbunden" and "regelverk" (the Swedish for "regular" and "regulations"),
# which already occur in example texts.
_STUD_ENDINGS = ("regel", "regeln", "reglar", "reglarna")


def names_stud(text: str) -> bool:
    """True when `text` names a stud or studs ("Stålreglar 70", "Träregel")."""
    return any(w.endswith(_STUD_ENDINGS) for w in _words(text))


# Frame members named as the thing itself, 2026-09-28 (HENRIC-3290, second
# round). The first round sent a bare stud to innervägg, the wall it usually
# stands in, and that is where it met no comparable row: innervägg's rows are
# plasterboard per m2 and a handful of steel profiles per kg, while a stud is
# bought per löpmeter at a stated section. Sawn timber, glulam and structural
# board all live in stomme, so a stud does too, and a board or a batten with it.
#
# What decides is the HEAD of the name, the part before the first "på", "med",
# "i" or comma, and within it the first word that says what the thing is.
# "Reglar innervägg" and "Nya reglar i vägg" are studs; "Innervägg, gips på
# stålreglar" and "Gipsvägg med stålreglar" are walls that happen to stand on
# studs, and keep going to innervägg. A name whose head says nothing either way
# ("Gips på reglar") falls through to the table and, failing that, to the stud
# fallback at the very end, which still sends it to innervägg.
_HEAD_SPLIT_RE = re.compile(
    r"[,(/;:]| (?:på|med|i|till|för|av|och|mot|inkl|samt|under|bakom) ")
# Words keep their hyphens here and lose them before matching, so "OSB-skiva"
# is read as the one compound "osbskiva". A Swedish compound is named by its
# LAST part, which is why every pattern below is anchored at the end:
# "Golvspånskiva" is a board, "Plywoodbord" and "OSB-hylla" are furniture.
_COMPOUND_RE = re.compile(r"[a-zåäöéü]+(?:-[a-zåäöéü]+)*")
_FRAME_WORD_RE = re.compile(
    r"(?:"
    # studs, and a frame of them; light steel profiles and roof trusses
    r"regel|regeln|reglar|reglarna|regelstomme|regelstommen|regelstommar"
    # The one-letter profile names only at the start of the word: "aluprofil"
    # (window and glazing aluminium) ends in "uprofil" too.
    r"|(?:stål|hatt|^[cuz])profil(?:en|er|erna)?|takstol|takstolen|takstolar"
    # structural board. "Golvspånskiva" moves from golv on purpose: it is a
    # particleboard subfloor, and golv's typvärde is the vinyl or linoleum that
    # would be laid on it.
    r"|(?:plywood|spån|konstruktions|kryssfaner|kryssfanér|osb)skiv(?:a|an|or|orna)"
    r"|plywood|kryssfaner|kryssfanér|osb|råspont|råsponten"
    # sawn and engineered timber
    r"|virke|virket|konstruktionsträ|fanerträ|lvl|lvlbalk|lvlbalkar|kerto"
    # Solid timber as a material on its own. It was a substring on the stomme
    # row of the table below, where it also took "Massivträbord" and "Matbord
    # i massivträ" (tester, 2026-09-28), which then got asked for a stud's
    # section. As a word ending it is only the timber itself.
    r"|massivträ"
    # battens and sole plates. "Läktare" (a grandstand) ends in none of these,
    # and the lookbehind keeps "släkt" (relatives) and every "fläkt" (a fan:
    # köksfläkt, takfläkt) out.
    r"|(?<![sf])(?:läkt|läkten|läkter|läkterna)"
    r"|glespanel|glespanelen|syll|syllen|syllar|syllarna"
    r")$")
# Words that make the head a surface or a wall rather than the member.
_SURFACE_ENDINGS = ("vägg", "väggen", "väggar", "väggarna", "skiva", "skivan",
                    "skivor", "skivorna")
# Compounds that end in a frame word and are not framing. "Fasadvirke" is
# cladding. A "dörregel", "fönsterregel" or "skjutregel" is a bolt, door and
# window hardware (tester, 2026-09-28: a Palats "Dörregel mässing" was offered
# as reuse for timber studs). Firewood and packaging timber are not structural.
_NOT_FRAME_PREFIXES = ("fasad", "dörr", "dör", "fönster", "skjut", "kolv",
                       "slag", "bom", "lås", "grind", "port", "luck",
                       "bränsle", "emballage")
# An elided compound, "Vägg- och golvregel": the first word lends its tail to
# the second and names nothing on its own. Without this the head split at "och"
# left "vägg-", read as a wall (review, 2026-09-28).
_ELIDED_RE = re.compile(r"[a-zåäöéü]+-\s*(?:och|eller|samt|&|,|/)\s*")


def names_frame_member(text: str) -> bool:
    """True when the head of `text` names a stud, a batten, sawn timber or a
    structural board ("Reglar 45x95", "OSB-skiva 12 mm", "Takläkt"), False
    when it names a wall or a surface that merely stands on one ("Gipsvägg med
    stålreglar")."""
    text = _ELIDED_RE.sub("", text.lower().strip())
    head = _HEAD_SPLIT_RE.split(" " + text + " ", maxsplit=1)[0]
    for compound in _COMPOUND_RE.findall(head):
        w = compound.replace("-", "")
        if _FRAME_WORD_RE.search(w) and not w.startswith(_NOT_FRAME_PREFIXES):
            return True
        if w.endswith(_SURFACE_ENDINGS) or w.startswith("gips"):
            return False
    return False


def light_clinker_category(text: str) -> str:
    """'stomme' or 'betongvägg' for a lättklinker/Leca name, '' otherwise.

    "leca" is matched from the start of a word ("Leca-block", "Lecabalk"), so
    it cannot fire inside an unrelated word."""
    words = _words(text)
    if not any("lättklinker" in w or w.startswith("leca") for w in words):
        return ""
    for w in words:
        if ("balk" in w and "balkong" not in w) or "bjälklag" in w:
            return "stomme"
    return "betongvägg"


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


# Swedish inflection endings a compound noun can carry. Used to match a term
# as the HEAD of a compound ("halvmånebord" ends in "bord") without the false
# positives a bare substring test gives. Moved here from palats_client
# 2026-09-28 so a listing and a component read furniture with one matcher.
INFLECTIONS = ("", "a", "s", "n", "t", "an", "en", "et", "ar", "er", "or",
               "arna", "erna", "orna", "na")


def compound_tail(word: str, term: str) -> bool:
    """True when `word` is `term`, or a Swedish compound ending in `term`.

    This is the shape Swedish compounding actually needs. Neither simple form
    works on its own: `"stol" in word` also matches "toalettstol", while a
    word-boundary regex misses "kontorsstol". Matching on the compound TAIL
    catches kontorsstol/elevstol/mötesstol and leaves toalettstol to an
    exception set, which is checked first and with the same matcher, so an
    exception written as a stem covers its inflections too ("spiskåp" has to
    cover both "spiskåpa" and "spiskåpor").
    """
    return any(word.endswith(term + suffix) for suffix in INFLECTIONS)


def compound_units(title: str) -> list[str]:
    """Words in a title, plus a de-hyphenated form of each hyphenated word.

    Swedish writes plenty of compounds with a hyphen, especially after an
    initialism: "WC-stol", "LED-lampa". Splitting on the hyphen alone leaves a
    bare "stol". Emitting the joined form as well lets an exception set see
    the whole compound.
    """
    tokens = [t for t in re.split(r"[^\w-]+", title.lower()) if t.strip("-")]
    units: list[str] = []
    for token in tokens:
        parts = [p for p in token.split("-") if p]
        units.extend(parts)
        if len(parts) > 1:
            units.append("".join(parts))
    return units


# Loose furniture, HENRIC-3290 del 2. The subcategories match the catalog's
# (build_epd_alternatives.EPD_SUBCATEGORY_KEYWORDS["los_inredning"]) and
# Palats', so a Swedish "Elevstol", a Palats "Mötesstolar" and an English
# "Student Chair" land in one bucket. Read as compound tails: a Swedish
# compound is named by its last part, which is why "Skrivbordsstol" is a chair,
# "Bordsskärm" a screen and "Soffbord" a table.
#
# Order is checked per word, first match wins, so it only matters inside one
# compound: "skärmvägg" must reach akustik before anything reads "vägg".
# Written as stems where the plural drops a vowel ("hyll" for hylla/hyllor,
# "soff" for soffa/soffor).
#
# akustik is the loose kind only (desk, floor and free-standing screens). A
# wall absorber or an acoustic ceiling panel is fixed to the building and is
# not in this category. textil has no EPD rows today (the catalog's "curtain"
# hits are curtain walls and shutters), so a curtain gets the honest "no
# comparable products" rather than somebody else's number.
_FURNITURE_TAILS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("akustik", ("skärmvägg", "skärm", "rumsavdelare")),
    ("soffa", ("soff", "fåtölj", "schäslong", "divan")),
    # "bänk" only in compounds that name furniture: a bare "bänk" is as often a
    # diskbänk or a workbench. An "elevbänk" is a pupil's desk.
    ("förvaring", ("skåp", "hyll", "hyllsystem", "byrå", "garderob", "hurts",
                   "sideboard", "skänk", "reol", "klädfack", "förvaring",
                   "tvbänk")),
    ("bord", ("bord", "elevbänk", "skolbänk")),
    # A task chair, with castors and a gas lift, is its own kind: the EPDs put
    # it at 60 to 180 kg CO2e against 3 to 55 for a four-legged chair or a
    # stool, and one typvärde over both was the Aeron's 86.7 for every
    # elevstol. Before "stol", which every one of these also ends in.
    ("kontorsstol", ("kontorsstol", "kontorstol", "arbetsstol",
                     "skrivbordsstol", "datorstol")),
    ("stol", ("stol", "pall", "taburett", "sittbänk")),
    ("textil", ("gardin", "draperi")),
)

# Compounds that END in a furniture word and are something else. Checked first,
# with the same tail matcher. Most are appliances or installations ("kylskåp",
# "elskåp", "spiskåpa", which ends in "skåp" plus "a" by spelling coincidence),
# sanitary ware ("toalettstol", "duschpall"), fixed interior with its own
# category ("köksskåp", "spegelskåp"), or not a building product at all
# ("bildskärm", "rullstol", "lastpall").
_FURNITURE_EXCEPTIONS = (
    "toalettstol", "wcstol", "duschstol", "duschpall", "badpall", "rullstol",
    # A roof truss. names_frame_member takes it first on both sides, but this
    # reader is shared and should not call a truss a chair on its own.
    "takstol",
    "lastpall", "europall", "eurpall", "engångspall", "plastpall",
    "kylskåp", "frysskåp", "torkskåp", "värmeskåp", "elskåp", "apparatskåp",
    "kopplingsskåp", "säkringsskåp", "fördelningsskåp", "mätarskåp",
    "centralskåp", "fläktskåp", "brandskåp", "slangskåp", "serverskåp",
    "rackskåp", "nätverksskåp", "spiskåp",
    "spegelskåp", "badrumsskåp", "tvättställsskåp", "köksskåp", "bänkskåp",
    "överskåp", "underskåp", "högskåp", "diskbänksskåp", "diskskåp",
    "duschskärm", "badkarsskärm", "solskärm", "bildskärm", "datorskärm",
    "tvskärm", "lampskärm", "stänkskärm", "vindskärm", "insynsskärm",
    "projektorskärm", "projektionsskärm", "radiatorskärm", "elementskärm",
    "brandskärm",
    # A dishwashing table is storköksutrustning. A roller blind (Palats'
    # "Myggnät rullgardin") belongs to the window, not with curtains.
    "diskbord", "rullgardin",
    # "Bord" is also a sawn board. These went to golv, yttervägg and tak before
    # furniture was read, and still should. A skötbord is a changing table,
    # normally fixed in the toilet room rather than moved like a desk.
    "golvbord", "panelbord", "fasadbord", "lockbord", "spontbord", "takbord",
    "formbord", "skötbord",
)

# A word that names a part or a holder stops the reading, wherever it stands
# in the head: "Slanghållare städskåp" is a hose holder for a cleaning cabinet,
# not a cabinet (Palats, 2026-09-28), and so is "Städskåp slanghållare".
# "Stolsdyna" and "Bordsskiva" already end in no furniture tail.
_FURNITURE_PART_TAILS = ("hållare", "fäste", "konsol", "underrede", "stativ",
                         "dyna", "klädsel", "skiva", "lucka", "dörr", "hjul")

# A rug, as opposed to a floor covering. Bare "matta" is a rug; the floor
# compounds (plastmatta, textilmatta, heltäckningsmatta, golvmatta,
# entrématta) keep going to golv, and so does a "matta" that intake declared
# as golv. Only these prefixes make a compound a rug.
_RUG_FORMS = ("matta", "mattan", "mattor", "mattorna")
_RUG_PREFIXES = ("", "ry", "rya", "tras", "gång", "dörr", "bad", "badrums",
                 "lek", "ull", "sisal", "jute", "bomulls", "plysch")


def furniture_subcategory(text: str) -> str:
    """The loose-furniture subcategory `text` names, or "".

    One of akustik, soffa, förvaring, bord, kontorsstol, stol, textil. Reads the head of
    the name the way names_frame_member does ("Kontorsstol med armstöd" is a
    chair, "Bord och stolar" a table), and returns "" for anything whose head
    is not furniture ("Bordsskiva", "Kylskåp", "Toalettstol", "Plastmatta").
    Shared with palats_client so a listing and a component read the same way.
    """
    text = _ELIDED_RE.sub("", text.lower().strip())
    head = _HEAD_SPLIT_RE.split(" " + text + " ", maxsplit=1)[0]
    words = [c.replace("-", "") for c in _COMPOUND_RE.findall(head)]
    # Exceptions and parts first, over the whole head: "Städskåp slanghållare"
    # is a hose holder as much as "Slanghållare städskåp" is, and reading the
    # words in order let the cabinet answer before the holder was seen.
    for w in words:
        if any(compound_tail(w, exc) for exc in _FURNITURE_EXCEPTIONS):
            return ""
        if any(compound_tail(w, part) for part in _FURNITURE_PART_TAILS):
            return ""
    for w in words:
        for form in _RUG_FORMS:
            if w.endswith(form) and w[:-len(form)] in _RUG_PREFIXES:
                return "textil"
        for sub, tails in _FURNITURE_TAILS:
            if any(compound_tail(w, t) for t in tails):
                return sub
    return ""


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

    clinker = light_clinker_category(name_lower)
    if clinker:
        return clinker

    # Before the table, whose "tak" would take "Takregel" and "Takläkt" and
    # whose "golv" would take "Golvreglar".
    if names_frame_member(name_lower):
        return "stomme"

    # After the frame reader, so "Massivträ" stays timber while "Massivträbord"
    # is a table, and before the table below, whose "golv" would take
    # "Golvskärm" and whose "matta" would take a rug.
    if furniture_subcategory(name_lower):
        return "los_inredning"

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
                   "kl-trä", "klträ", "korslimmat", "träbalk",
                   "träpelare", "betongbalk", "betongpelare", "bjälklag",
                   "håldäck", "takbalk", "bärbalk", "lättbalk", "masonitebalk",
                   "i-balk"],
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
        "belysning": ["belysning", "ljus", "lighting", "lampa", "lampor",
                      "armatur", "led-armatur"],
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

    # A name that mentions studs only after its head ("Gips på reglar") is the
    # surface on them, and the surface is the wall they stand in. A stud named
    # as the thing itself was taken by names_frame_member above.
    if names_stud(name_lower):
        return "innervägg"

    return ""


# Canonical category keys produced by normalize_component_name — also the valid
# values for a component's declared `category` field. Keep in sync with the
# mappings above, intake's category enum, and the EPD catalog categories.
VALID_CATEGORIES = {
    "kakel", "golv", "innervägg", "yttervägg", "fasadskikt", "stomme",
    "betongvägg", "fönster", "tak", "isolering", "storköksutrustning",
    "kylanläggning", "belysning", "ventilation", "dörr", "hiss", "sanitet",
    "vitvaror", "vvs", "farg", "el", "radiator", "fast_inredning",
    "los_inredning",
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
        # A stud or a structural board declared as the part it stands in. Intake
        # used to file "Reglar" under innervägg, and every saved project still
        # carries that: the wall is right as context and wrong as a material,
        # because the comparable rows (sawn timber, steel studs, OSB) are all in
        # stomme. A wall that merely stands on studs ("Gipsvägg med stålreglar")
        # is not a frame member and keeps its declared category.
        if cat in _FRAME_HOST_CATEGORIES and names_frame_member(name):
            return "stomme"
        # A chair or a desk declared as fixed interior, which is what an intake
        # did before los_inredning existed. It then met countertops and mirror
        # cabinets. Storage is left alone: a wardrobe or a cabinet can be
        # built in, and then the declared category is right. So is anything
        # the name itself calls built in ("Platsbyggd soffa"): the name-based
        # reading has to agree before the declaration is overruled.
        if (cat == "fast_inredning"
                and furniture_subcategory(name) in _NEVER_FIXED_KINDS
                and normalize_component_name(name) == "los_inredning"):
            return "los_inredning"
        return cat
    return normalize_component_name(name)


# Declared categories a frame member is commonly filed under by mistake.
_FRAME_HOST_CATEGORIES = {"innervägg", "yttervägg", "tak", "golv"}

# Furniture kinds that are never fixed interior (see resolve_category).
_NEVER_FIXED_KINDS = {"stol", "kontorsstol", "bord", "soffa", "akustik", "textil"}


