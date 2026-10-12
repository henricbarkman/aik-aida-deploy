"""What sort of material an alternative is, the group it is shown under.

Testgruppen, October 2026: "Safety flooring" says nothing to someone choosing
a floor, and the product list read as a shopping list when the tool's ambition
is a rough plan. Henric's decision (2026-10-12): the alternatives are shown by
material type first (Linoleum, Gummigolv, Vinylgolv), each with the range of
climate impact and cost across its products, and the products only when a type
is folded out.

Read from what the catalog already knows, never from a model call: the row's
own subcategory, `subtype_from_material` for floors, `facade_kind` for facade
cladding, the thickness class for plasterboard. A category with no finer
taxonomy (hiss, radiator) is its own type. In a category that has one, a row
nothing here recognises gets "", and the view puts it under "Övrigt" rather
than dropping it.
"""

from __future__ import annotations

from aida.data.climate_data import facade_kind
from aida.data.epd_baseline_medians import subtype_from_material

# Floor subtypes, as a heading. The keys are the catalog's and
# subtype_from_material's.
_FLOOR = {
    "linoleum": "Linoleum", "laminat": "Laminatgolv", "keramik": "Klinker och keramik",
    "trä": "Trägolv", "gummi": "Gummigolv", "epoxi": "Epoxi- och plastgjutet golv",
    "vinyl": "Vinylgolv", "textil": "Textilgolv", "avjämning": "Avjämningsmassa",
    "natursten": "Natursten", "installationsgolv": "Installationsgolv",
    "golvskiva": "Golvskiva",
}

# English names subtype_from_material does not read (it reads the Swedish the
# baseline agent writes). Most specific first, substring match.
_FLOOR_EN = (
    ("installationsgolv", ("access floor", "raised floor", "raised access")),
    ("natursten", ("limestone", "kalksten", "granit", "marble", "marmor", "skiffer")),
    ("golvskiva", ("floor board", "golvgips", "golvskiva")),
    ("linoleum", ("linoleum", "marmoleum")),
    ("gummi", ("rubber", "nora ", "noraplan", "norament")),
    ("epoxi", ("epoxy", "mapefloor", "polyurethane", "resin floor")),
    ("vinyl", ("safety flooring", "safetred", "vinyl", "pvc", "lvt")),
    ("textil", ("carpet", "tufted", "needle felt", "needlefelt", "flotex")),
    ("trä", ("parquet", "wood floor", "hardwood", "oak floor", "bamboo", "cork")),
    ("keramik", ("tile", "porcelain", "ceramic", "stoneware")),
)

_FACADE = {
    "trä": "Träpanel", "puts": "Puts", "skiva": "Fasadskiva", "metall": "Plåt och metall",
    "komposit": "Kompositpanel", "tegel": "Tegel", "lerputs": "Lerputs",
}

# Subcategories that already are a sort of material or product.
_SUBCATEGORY = {
    # undertak
    "akustikplatta": "Akustikplatta", "väggabsorbent": "Väggabsorbent",
    "bärverk": "Bärverk", "baffel": "Baffel", "metallundertak": "Metallundertak",
    # stomme
    "betong": "Betong", "konstruktionsstål": "Stål", "virke": "Konstruktionsvirke",
    "limträ": "Limträ", "stålregel": "Stålregel", "konstruktionsskiva": "Konstruktionsskiva",
    # innervägg
    "glasparti": "Glasparti",
    # inredning
    "bord": "Bord", "stol": "Stol", "kontorsstol": "Kontorsstol", "soffa": "Soffa",
    "förvaring": "Förvaring", "matta": "Matta", "gardin": "Gardin",
    "akustik": "Akustikabsorbent", "kökslucka": "Kökslucka", "köksskåp": "Köksskåp",
    "badrumsinredning": "Badrumsinredning", "bänkskiva": "Bänkskiva",
    # sanitet, belysning, vitvaror, ventilation
    "blandare": "Blandare", "dusch": "Dusch", "handfat": "Handfat", "badkar": "Badkar",
    "toalett": "Toalettstol", "armatur": "Armatur", "taklampa": "Taklampa",
    "köksfläkt": "Köksfläkt", "aggregat": "Ventilationsaggregat",
}

_INSULATION = (
    ("Glasull", ("glasull", "glass wool", "glasswool", "glass mineral wool", "glava", "isover")),
    ("Stenull", ("stenull", "stone wool", "stonewool", "rock wool", "rockwool", "paroc")),
    ("Cellulosaisolering", ("cellulosa", "cellulose", "ekovilla")),
    ("Träfiberisolering", ("träfiber", "wood fibre", "wood fiber", "woodfibre")),
    ("Hampaisolering", ("hampa", "hemp")),
    ("Cellplast", ("eps", "xps", "cellplast", "polystyren", "polystyrene", "sundolitt",
                   "fibran", "sopraxps")),
    ("PIR- och PUR-isolering", ("pir", "pur ", "polyuretan", "polyurethane", "polyiso")),
    ("Mineralull", ("mineralull", "mineral wool")),
)

_WINDOW = (
    ("Balkongräcke", ("railing",)),
    ("Balkongdörr", ("balcony door", "balkongdörr")),
    ("Isolerglas", ("insulating glass", "glazing unit", "igu", "isolerglas")),
    ("Trä-aluminiumfönster",("trä-alu", "träalu", "trä/alu", "wood-alu", "wood/alu",
                              "timber-alu", "alu-clad", "aluminium-clad", "aluminium clad")),
    ("PVC-fönster", ("pvc", "upvc")),
    ("Aluminiumfönster", ("aluminium", "aluminum")),
    ("Träfönster", ("trä", "wood", "timber")),
)

# Categories without a subcategory, read from the (mostly English) name.
_BY_NAME = {
    "tak": (
        ("Sedumtak", ("green roof", "sedum", "gröna tak")),
        ("Takpannor", ("roof tile", "ridge tile", "takpanna", "takpannor", "concrete tiles")),
        ("Takpapp och bitumen", ("bitumen", "shingle", "takpapp", "roofing felt")),
        ("Sandwichpanel", ("sandwich", "insulating panel", "insulated panel")),
        ("Takduk", ("waterproof", "membrane", "takduk", "alkorplan")),
        ("Plåttak", ("sheet", "steel", "aluminium", "aluminum", "plåt", "roof 10")),
    ),
    "dörr": (
        ("Duschdörr", ("shower door", "duschdörr")),
        ("Branddörr", ("fire door", "fire rated", "firepro", "branddörr")),
        ("Balkongdörr", ("balcony door", "balkongdörr")),
        ("Ståldörr", ("steel door", "ståldörr", "metal door")),
        ("Glasdörr", ("glass door", "glazed", "glasdörr")),
        ("Trädörr", ("wood", "wooden", "trädörr", "furu", "ek ")),
    ),
    "ventilation": (
        ("Ventilationskanal", ("duct", "kanal")),
        ("Brandspjäll och spjäll", ("damper", "spjäll")),
        ("Don och galler", ("diffuser", "grille", "galler", "don ")),
        ("Fläkt", ("fan", "fläkt")),
    ),
    "belysning": (
        ("Spotlight", ("spot", "track light")),
        ("Gatubelysning", ("street light", "gatubelysning")),
        ("Linjärarmatur", ("linear", "linjär")),
    ),
    "yttervägg": (
        ("Glasfasad",("curtain wall", "glasfasad")),
        ("Sandwichpanel", ("sandwich",)),
        ("Putsad tilläggsisolering", ("etics", "thermokapa")),
        ("Puts", ("render", "puts")),
        ("Fasadfärg", ("paint", "färg")),
        ("Tegel", ("brick", "tegel")),
        ("Betongelement", ("precast", "concrete cladding", "betong")),
        ("Fibercementskiva", ("fibre cement", "fiber cement", "fibercement")),
        ("Plåt och metall", ("aluminium", "aluminum", "steel", "metal")),
        ("Träpanel", ("wood", "timber", "trä")),
    ),
}

_CATEGORY = {
    "golv": "Golvbeläggning", "kakel": "Kakel och klinker", "innervägg": "Innervägg",
    "yttervägg": "Yttervägg", "fasadskikt": "Fasadbeklädnad", "betongvägg": "Betongvägg",
    "tak": "Taktäckning", "undertak": "Undertak", "tätskikt": "Tätskikt",
    "fönster": "Fönster", "dörr": "Dörr", "isolering": "Isolering",
    "belysning": "Belysning", "ventilation": "Ventilation", "hiss": "Hiss",
    "kylanläggning": "Kylanläggning", "sanitet": "Sanitet", "vitvaror": "Vitvara",
    "storköksutrustning": "Storköksutrustning", "vvs": "VVS", "farg": "Färg",
    "el": "Elinstallation", "radiator": "Radiator", "fast_inredning": "Fast inredning",
    "los_inredning": "Lös inredning", "stomme": "Stomme",
}


def _first(text: str, table) -> str:
    for label, words in table:
        if any(w in text for w in words):
            return label
    return ""


def _words_hit(text: str, words) -> bool:
    """Short words ("eps", "pir", "pvc") as words, so "steps" is not EPS."""
    import re
    return any(re.search(r"(?<![a-zåäö])" + re.escape(w) + r"(?![a-zåäö])", text)
               if len(w) <= 4 else w in text for w in words)


def _plasterboard(name: str) -> str:
    from aida.data.gipsskiva import thickness_class, thickness_from_name

    cls = thickness_class(thickness_from_name(name))
    return f"Gipsskiva {cls} mm" if cls else "Gipsskiva"


# Categories with no finer taxonomy: every product in them is the same type.
_SINGLE_TYPE = ("betongvägg", "el", "farg", "hiss", "kylanläggning", "radiator",
                "storköksutrustning", "tätskikt", "vvs")


def material_type(category: str, subcategory: str | None, name: str) -> str:
    """The material type of one alternative, or "" when it is not known.

    `category` and `subcategory` are the catalog row's (or the Palats listing's),
    `name` its product name or listing title.
    """
    category = (category or "").strip()
    sub = (subcategory or "").strip()
    text = (name or "").lower()
    if not category:
        return ""
    if category in _SINGLE_TYPE:
        return _CATEGORY.get(category, category.capitalize())

    if category == "golv":
        key = sub if sub in _FLOOR else (subtype_from_material("golv", name or "")
                                         or _first(text, _FLOOR_EN))
        return _FLOOR.get(key, "")
    if category == "fasadskikt":
        return _FACADE.get(facade_kind(name or ""), "")
    if sub in _SUBCATEGORY:
        return _SUBCATEGORY[sub]
    if category in ("innervägg", "undertak") and any(
            w in text for w in ("gips", "plasterboard", "drywall", "gypsum", "gyproc")):
        return _plasterboard(name or "")
    if category == "isolering":
        for label, words in _INSULATION:
            if _words_hit(text, words):
                return label
    if category == "fönster":
        return _first(text, _WINDOW)
    if category == "kakel":
        return "Klinker" if any(w in text for w in ("klinker", "floor tile", "golv")) else "Kakel"
    if category in _BY_NAME:
        return _first(text, _BY_NAME[category])
    return ""
