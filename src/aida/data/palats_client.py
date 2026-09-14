"""Palats API client — fetch reuse listings from Karlstads kommun's internal marketplace.

Palats (palats.app) is the reuse platform used by Karlstads kommun for
building materials and fixtures. This client uses the internal API with
cookie-based authentication.

NOTE: This is an unofficial/internal API — it may change without notice.
Felix (Palats) has given permission to use it for experimentation.

Auth flow (automatic, no manual cookie management needed):
1. Try remember_me token → POST /api/v2/auth/refresh → fresh JWT (15 min)
2. If remember_me expired → login with PALATS_USERNAME/PALATS_PASSWORD → new cookies
3. Cookies cached in-process for the session lifetime
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

PALATS_BASE_URL = "https://palats.app/api"

# Location ID → human-readable name and address (from Palats web UI, 2026-04-01)
LOCATION_NAMES: dict[int, dict[str, str]] = {
    2945: {"name": "Sola byggåterbruk", "address": "Östanvindsgatan 14, Karlstad"},
    4008: {"name": "Sola Möbelåterbruk", "address": ""},
    4448: {"name": "KCCC", "address": "Tage Erlandergatan 8, Karlstad"},
    4462: {"name": "Gamla Wermlandsbanken", "address": "Tingvallagatan 11, Karlstad"},
    5003: {"name": "Bibliotekshuset", "address": "Västra Torggatan 26, Karlstad"},
    5761: {"name": "Vänersnipan", "address": "Bogsprötsgatan 20, Karlstad"},
}

# Vendor id -> public shop slug. The public listing page lives at
# https://palats.app/shop/<slug>/listing/<id> and renders without a login
# (verified in a browser 2026-09-14 on listing 39999, "WC-stol"). The API does
# not expose the slug, so it is mapped here by hand. A vendor missing from this
# map gets the internal /web/listing/<id> URL, which requires a Palats account.
SHOP_SLUGS: dict[str, str] = {
    "ve_01ka8cx0n1fckb4pyzv43gk3mf": "solabyggaterbruk-byggmaterial",
}
PALATS_WEB_URL = "https://palats.app"


def listing_url(listing_id: str, vendor_id: str = "") -> str:
    """Deep link a reader can click straight through to the listing.

    Until 2026-09-14 Aida linked to /web/listing/<id>, which only works for a
    logged-in Palats user, and wrote "palats.app/listing/<id>" in the source
    column, which redirects to palats.io and 404s. Johanna had to type the
    article number into Sola's shop search by hand.
    """
    if not listing_id:
        return ""
    slug = SHOP_SLUGS.get(vendor_id or "")
    if slug:
        return f"{PALATS_WEB_URL}/shop/{slug}/listing/{listing_id}"
    return f"{PALATS_WEB_URL}/web/listing/{listing_id}"


# Cache listings for 10 minutes within a process
_listings_cache: list[dict] | None = None
_listings_cache_time: float = 0
_CACHE_TTL = 600

# Connection status — lets callers distinguish "no products" from "connection failed"
# Values: "ok", "no_credentials", "auth_failed", "api_error", ""
last_fetch_status: str = ""

# Auth state — cached in-process, auto-refreshed
_auth_cookies: dict[str, str] | None = None
_auth_time: float = 0
_AUTH_TTL = 840  # Refresh auth every 14 min (JWT lives 15 min)


@dataclass
class PalatsListing:
    """A reuse listing from Palats, normalized for Aida."""

    id: str
    title: str
    description: str
    price: float  # SEK, 0 if free/unknown
    quantity: int
    unit: str
    category: str  # Aida category key (golv, fönster, etc.) or ""
    subcategory: str  # Finer-grained key within category (e.g. "toalett" within "sanitet")
    image_url: str
    url: str  # Direct link to listing on palats.app
    location: str  # Human-readable location name

    @property
    def display_source(self) -> str:
        return f"[Palats] palats.app — {self.title}"


def _login() -> dict[str, str] | None:
    """Authenticate with username/password, return fresh cookies."""
    username = os.environ.get("PALATS_USERNAME")
    password = os.environ.get("PALATS_PASSWORD")
    if not username or not password:
        logger.warning("Palats login skipped: PALATS_USERNAME=%s PALATS_PASSWORD=%s",
                        "set" if username else "MISSING",
                        "set" if password else "MISSING")
        return None
    try:
        resp = requests.post(
            f"{PALATS_BASE_URL}/v2/auth/login",
            json={"username": username, "password": password},
            timeout=15,
        )
        resp.raise_for_status()
        cookies = {}
        for cookie in resp.cookies:
            cookies[cookie.name] = cookie.value
        if "palats_session" in cookies:
            logger.info("Palats login successful")
            return cookies
        logger.warning("Palats login response missing session cookie")
        return None
    except requests.RequestException as e:
        logger.warning("Palats login failed: %s", e)
        return None


def _refresh_with_remember_me(remember_me: str) -> dict[str, str] | None:
    """Use remember_me token to get a fresh JWT via the refresh endpoint."""
    try:
        resp = requests.post(
            f"{PALATS_BASE_URL}/v2/auth/refresh",
            cookies={"remember_me": remember_me},
            json={},
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        new_session = resp.cookies.get("palats_session")
        if not new_session:
            return None
        cookies = {"palats_session": new_session, "remember_me": remember_me}
        logger.debug("Palats session refreshed via remember_me")
        return cookies
    except requests.RequestException:
        return None


def _get_cookies() -> dict[str, str] | None:
    """Get valid Palats auth cookies, auto-refreshing as needed.

    Priority: cached session → refresh via remember_me → full login.
    """
    global _auth_cookies, _auth_time

    # Return cached cookies if still fresh
    if _auth_cookies and (time.time() - _auth_time) < _AUTH_TTL:
        return _auth_cookies

    # Try refresh with remember_me (from cache or env)
    remember_me = (
        (_auth_cookies or {}).get("remember_me")
        or os.environ.get("PALATS_REMEMBER_ME")
    )
    if remember_me:
        cookies = _refresh_with_remember_me(remember_me)
        if cookies:
            _auth_cookies = cookies
            _auth_time = time.time()
            return cookies
        logger.info("remember_me refresh failed, falling back to login")

    # Fallback: full login with credentials
    cookies = _login()
    if cookies:
        _auth_cookies = cookies
        _auth_time = time.time()
        return cookies

    # Last resort: try raw env var cookies (may be expired but worth a shot)
    session = os.environ.get("PALATS_SESSION")
    if session:
        logger.debug("Using raw PALATS_SESSION env var (may be expired)")
        return {"palats_session": session}

    global last_fetch_status
    last_fetch_status = "auth_failed"
    logger.warning("No Palats credentials available — reuse search disabled")
    return None


# Subcategory keywords within categories that bucket many distinct item types.
# Used to differentiate e.g. toalettstol vs handfat inside the sanitet category,
# so a search for "Toalettstol" doesn't drown in unrelated sanitet listings.
# Order matters per subcategory list (most specific first).
# Order within a category matters because matching is substring-based:
# subcategories with compound keywords (e.g. "duschblandare") must be
# checked before subcategories whose keywords would partially match those
# compounds (e.g. "dusch"). Rule of thumb — put modifiers/instruments
# before the surfaces they attach to.
SUBCATEGORY_KEYWORDS: dict[str, list[tuple[str, list[str]]]] = {
    "sanitet": [
        # Compound blandare-words come first so "tvättställsblandare" doesn't
        # get caught by the "tvättställ" keyword in handfat.
        ("blandare", ["duschblandare", "tvättställsblandare", "badkarsblandare",
                      "köksblandare", "tvättställsarmatur"]),
        # Seats split out before "toalett" subcat — "Toalettsits" contains
        # "toalett" and would otherwise be classified as a toilet bowl.
        ("toalettsits", ["toalettsits", "toilet seat", "wc-sits"]),
        # Toilet bowl only — no generic "toalett" keyword (it would catch
        # sits/lock/papper/etc that contain the word).
        ("toalett", ["toalettstol", "wc-stol", "wc stol", "wc-toalett",
                     "vägghängd toalett"]),
        ("handfat", ["handfat", "tvättställ", "washbasin"]),
        ("dusch", ["duschvägg", "duschdörr", "duschkabin", "dusch"]),
        ("badkar", ["badkar", "bathtub"]),
        # Generic blandare-words last so they only catch plain "Blandare Mora"
        # listings without surface context.
        ("blandare", ["blandare", "kran"]),
        ("urinal", ["urinal"]),
        ("spegel", ["spegel"]),
    ],
    "belysning": [
        ("skrivbordsbelysning", ["skrivbordslampa", "skrivbordsbelysning", "bordslampa"]),
        ("taklampa", ["taklampa", "takbelysning", "takarmatur", "spotlight"]),
        ("vägglampa", ["vägglampa", "vägglykta"]),
        ("armatur", ["armatur", "belysning", "lampa", "led-"]),
    ],
    "dörr": [
        ("innerdörr", ["innerdörr"]),
        ("ytterdörr", ["ytterdörr", "entrédörr", "ytterport", "entréport"]),
        ("branddörr", ["branddörr"]),
        ("skjutdörr", ["skjutdörr"]),
    ],
    "fönster": [
        ("energiglas", ["energiglas", "treglas", "isolerglas"]),
        ("fönsterbänk", ["fönsterbänk"]),
    ],
    "vitvaror": [
        ("tvättmaskin", ["tvättmaskin"]),
        ("torktumlare", ["torktumlare", "torkskåp"]),
        # Compounds before the bare words they contain, same rule as sanitet.
        # "Mikrovågsugn" ends in "ugn" and "Spiskåpa" starts with "spis", so
        # both landed in the spis bucket while their own subcategories sat
        # empty — and the vitvaror/köksfläkt typvärde (84 kg/st) was therefore
        # unreachable for every cooker hood in the inventory.
        ("mikro", ["mikrovågsugn", "mikrovåg", "mikro"]),
        ("köksfläkt", ["köksfläkt", "spisfläkt", "spiskåp", "fläktkåp"]),
        ("spis", ["spis", "häll", "ugn"]),
    ],
    # Same keys as EPD_SUBCATEGORY_KEYWORDS["fast_inredning"] in
    # build_epd_alternatives, so a Swedish component or listing and an English
    # EPD meet in one bucket. Cabinets before fronts: "Diskbänksskåp" is a
    # cabinet, and bare "lucka"/"låda" are safe only because the category is
    # already decided.
    "fast_inredning": [
        ("badrumsinredning", ["spegelskåp", "badrumsskåp", "tvättställsskåp",
                              "badrumsinredning", "badrumsmöbl", "kommod"]),
        ("köksskåp", ["diskbänksskåp", "köksskåp", "bänkskåp", "överskåp",
                      "underskåp", "högskåp", "köksinredning"]),
        ("kökslucka", ["kökslucka", "köksluckor", "kökslåd", "lucka", "luckor",
                       "låda", "lådor"]),
        ("diskbänk", ["diskbänk", "diskho"]),
        ("bänkskiva", ["bänkskiv"]),
    ],
}


def _normalize_to_aida_subcategory(category: str, text: str) -> str:
    """Map listing text to a finer subcategory within its Aida category.

    Returns '' if the category has no subcategories defined or no keyword matched.
    """
    subcats = SUBCATEGORY_KEYWORDS.get(category)
    if not subcats:
        return ""
    text_lower = text.lower()
    for subcat, keywords in subcats:
        for kw in keywords:
            if kw in text_lower:
                return subcat
    return ""


# Swedish inflection endings a compound noun can carry. Used to match a term
# as the HEAD of a compound ("halvmånebord" ends in "bord") without the
# false positives a bare substring test gives.
_INFLECTIONS = ("", "a", "s", "n", "t", "an", "en", "et", "ar", "er", "or",
                "arna", "erna", "orna", "na")


def _compound_tail(word: str, term: str) -> bool:
    """True when `word` is `term`, or a Swedish compound ending in `term`.

    This is the shape Swedish compounding actually needs. Neither simple form
    works on its own: `"stol" in word` also matches "toalettstol", while a
    word-boundary regex misses "kontorsstol". Matching on the compound TAIL
    catches kontorsstol/elevstol/mötesstol and leaves toalettstol to the
    exception set, which is checked first and with the same matcher — so an
    exception written as a stem covers its inflections too ("spiskåp" has to
    cover both "spiskåpa" and "spiskåpor").
    """
    return any(word.endswith(term + suffix) for suffix in _INFLECTIONS)


def _compound_units(title: str) -> list[str]:
    """Words in a title, plus a de-hyphenated form of each hyphenated word.

    Swedish writes plenty of compounds with a hyphen, especially after an
    initialism: "WC-stol", "LED-lampa". Splitting on the hyphen alone leaves a
    bare "stol", which the furniture guard would catch. Emitting the joined
    form as well lets the exception set see the whole compound.
    """
    tokens = [t for t in re.split(r"[^\w-]+", title.lower()) if t.strip("-")]
    units: list[str] = []
    for token in tokens:
        parts = [p for p in token.split("-") if p]
        units.extend(parts)
        if len(parts) > 1:
            units.append("".join(parts))
    return units


# Compounds that END in a guarded term but ARE building products or
# appliances. Checked before the guard, so "toalettstol" survives the "stol"
# rule and "kylskåp" survives the "skåp" rule. Written as stems: the matcher
# adds inflections.
_NON_BUILDING_EXCEPTIONS = (
    "toalettstol", "wcstol", "duschstol", "duschpall", "badpall",
    "kylskåp", "frysskåp", "kylfrysskåp", "torkskåp", "elskåp",
    "apparatskåp", "kopplingsskåp", "säkringsskåp", "fördelningsskåp",
    "duschskärm", "solskärm",
    # Cooker hoods are vitvaror. "spiskåpa"/"spiskåpor" both end in the
    # guarded "skåp", which is a pure spelling coincidence.
    "spiskåp", "fläktkåp", "imkåp",
    # Fixed interior, category fast_inredning since 2026-09-14. A mirror cabinet
    # or a kitchen base cabinet is mounted to the building; a förvaringsskåp or a
    # garderob is not, and still falls to the "skåp" tail. Worktops are named by
    # material: a bare "Övrig bänkskiva" from the furniture vendor stays out.
    "spegelskåp", "badrumsskåp", "tvättställsskåp", "köksskåp", "bänkskåp",
    "överskåp", "underskåp", "högskåp", "diskbänksskåp",
    "laminatbänkskiv", "granitbänkskiv", "träbänkskiv", "stenbänkskiv",
    "kompositbänkskiv", "köksbänkskiv",
)

# Furniture, loose inventory and workwear. Palats carries all three (48
# mötesstolar, 42 arbetskläder-underdelar), and until Aida has an inredning
# category they cannot substitute a building component. Matched as compound
# tails against the TITLE only.
#
# The guard makes furniture invisible, not visible. That is the right order:
# today a "Halvmånebord med rejält laminat" is offered as a floor, which is
# worse than not being offered at all. Offering it as furniture is a separate
# and larger job (roadmap C1).
_NON_BUILDING_TAILS = (
    # möbler. Written as stems where the plural drops a vowel: "hyllor" is not
    # "hylla" plus an ending, so a "hylla" tail silently misses every plural
    # listing ("Hyllor - barn", "Bokhyllor"). Same for soffa/soffor.
    "bord", "stol", "fåtölj", "soff", "schäslong", "pall", "hurts",
    "hyll", "skåp", "skärm", "byrå", "garderob", "madrass",
    "whiteboard", "tidningshållare", "klädfack", "bänkskiva",
    # arbetskläder
    "kläder", "skjorta", "kavaj", "byxor", "jacka", "skor", "tröja",
    "väst", "overall",
    # storköksmaskiner som inte är byggdelar
    "degblandare", "deggryta",
)

# Parts and accessories. They are building-adjacent but cannot replace the
# component they belong to, so offering a door frame as a reuse alternative
# to a door is the same error class as offering a table as a floor.
_ACCESSORY_PHRASES = (
    " till ", "tillbehör", "reservdel", "underrede", "stödskiva",
)


def _is_non_building(title: str) -> bool:
    """True when the title names furniture, workwear or a loose part."""
    units = _compound_units(title)
    if any(_compound_tail(u, exc) for u in units for exc in _NON_BUILDING_EXCEPTIONS):
        return False
    if any(_compound_tail(u, tail) for u in units for tail in _NON_BUILDING_TAILS):
        return True
    padded = f" {title.lower()} "
    return any(p in padded for p in _ACCESSORY_PHRASES)


# Listings that keyword-match a category but are the wrong product type for
# it. Same idea and same vocabulary as REJECT_PATTERNS in
# scripts/build_epd_alternatives.py: the keyword net is deliberately wide, and
# this is where the known catches get thrown back. Substring match on title.
CATEGORY_EXCLUSIONS: dict[str, tuple[str, ...]] = {
    # A glazed partition is not insulation, however much its title says
    # "ljudisolering". Closest real category is innervägg, but a glass wall is
    # not a gypsum wall either, so it gets no category rather than a wrong one.
    "isolering": ("glasparti",),
    # An awning shades a window, it does not replace one. A window sill has
    # its own subcategory and is not a window either.
    "fönster": ("markis", "persienn", "gardin"),
    # Frames, rails and hardware cannot substitute a door leaf.
    "dörr": ("karm", "skena", "handtag", "beslag", "trycke", "dörrstopp",
             "gångjärn", "tröskel"),
    # A drawer or a cabinet above the fridge is not a cooling appliance.
    "kylanläggning": ("låda", "överskåp", "underskåp"),
}


def _normalize_to_aida_category(title: str, description: str = "") -> str:
    """Map a Palats listing to an Aida component category using keywords.

    Classification reads the TITLE only. `description` is accepted for
    backwards compatibility and deliberately ignored: the field it carries is
    Palats' `articleConditionComment`, a note about wear and damage, not a
    product description. Measured against the full live inventory (701
    listings, 2026-08-31) it changed the outcome for exactly one listing, and
    that one was wrong — an "Omklädningsskåp" whose comment read "Rostigt
    golv" was classified as flooring and offered as a floor reuse option.

    Returns the Aida category key (e.g. 'golv', 'fönster') or '' if no match.
    """
    text = title.lower()

    # Structural frame elements (beams, columns, slabs) are load-bearing stomme,
    # not renovation finish materials. "Lättklinkerbalk" contains "klinker" and
    # would otherwise be miscategorized as golv/kakel and offered as a floor
    # reuse option. Skip them so they aren't matched to finish components. Uses
    # compound forms only — bare "balk" would hit "balkong" and bare "pelare"
    # would hit "duschpelare" (a shower tower, a legit sanitet fixture).
    _structural = ("klinkerbalk", "betongbalk", "stålbalk", "limträbalk",
                   "håldäck", "bjälklag", "armeringsjärn", "betongpelare",
                   "stålpelare", "limträpelare")
    if any(t in text for t in _structural):
        return ""

    if _is_non_building(title):
        return ""

    # Order matters — more specific matches first
    # Multi-word patterns checked before single-word to avoid false positives
    category_keywords: list[tuple[str, list[str]]] = [
        # First, because its words are compounds that end in other categories'
        # words or sit next to them: "Överskåp kyl/frys" would reach
        # kylanläggning's "kyl", "Spegelskåp med belysning" belysning, and
        # "Tvättställsskåp" sanitet. Measured on Sola 2026-09-14: 5 diskbänkar,
        # 4 diskhoar, 4 kökslådor, 4 köksluckor, 2 bänkskåp, 6 spegelskåp and 4
        # material-named bänkskivor, none of which had a category before.
        ("fast_inredning", [
            "spegelskåp", "badrumsskåp", "tvättställsskåp", "köksskåp",
            "bänkskåp", "överskåp", "underskåp", "högskåp", "kökslucka",
            "köksluckor", "kökslåd", "diskbänk", "diskho", "bänkskiv",
        ]),
        ("fönster", [
            "fönster", "fönsterbåge", "fönsterkassett", "fönsterbänk",
            "energiglas",
        ]),
        ("dörr", [
            "dörr", "dörrblad", "dörrkarm", "innerdörr", "ytterdörr",
            "branddörr", "skjutdörr", "entrédörr",
        ]),
        # Ceramic before golv, mirroring normalize_component_name in
        # climate_data: "golvklinker" contains "golv". Without its own key here
        # a Palats kakel listing was classified `golv` and could therefore
        # never match a `kakel` component, since matching compares category
        # keys across the two taxonomies.
        ("kakel", [
            "kakel", "klinker", "keramikplatt", "väggkakel", "golvklinker",
        ]),
        ("golv", [
            # "laminat" and bare "matta" were both dropped 2026-08-31. They are
            # the two keywords the live inventory proved too loose: "laminat"
            # matched a table and two countertops, "matta" matched two moisture
            # barriers. The compounds below keep every real floor listing.
            "golv", "parkett", "vinylgolv", "vinylmatta", "laminatgolv",
            "trägolv", "golvplatta", "golvmatta", "plastmatta",
            "heltäckningsmatta", "textilmatta", "entrématta", "linoleum",
        ]),
        ("tak", [
            "takpann", "takplåt", "takskiva", "yttertak", "undertak",
            "undertaksplatt", "takbrygga",
        ]),
        ("belysning", [
            "lampa", "armatur", "belysning", "spotlight",
            "taklampa", "takbelysning", "vägglampa", "led lampa",
            "skrivbordsbelysning",
        ]),
        ("isolering", [
            "isolering", "mineralull", "glasull", "stenull", "cellplast",
            "eps", "xps", "cellulosa", "ljudisolerande",
        ]),
        ("innervägg", [
            "gipsskiva", "gips", "väggskiva", "byggskiva",
            "reglar", "innervägg",
        ]),
        ("yttervägg", ["fasadskiva", "fasadplatta", "puts", "fasad"]),
        # Cooker hoods moved to vitvaror below, so this list must no longer
        # claim them. Bare "fläkt" took all ten of the live inventory's kitchen
        # hoods, which left the vitvaror/köksfläkt typvärde unreachable and
        # offered a cooker hood as a ventilation duct. Bare "don" went the same
        # way: it is three letters that sit inside plenty of unrelated words,
        # and every real use in this domain is a compound.
        ("ventilation", [
            "ventilation", "ventilationskanal", "ventilationsfläkt",
            "frånluftsfläkt", "tilluftsfläkt", "takfläkt", "kanalfläkt",
            "imkanal", "tilluftsdon", "frånluftsdon", "ventilationsdon",
            "uteluftsdon", "spjäll",
        ]),
        # Radiators before vvs and in their OWN key. The component side
        # (normalize_component_name) has had a `radiator` category all along,
        # and search_listings_for_component compares the two keys for equality,
        # so while radiators were filed as vvs here all eight live listings were
        # unreachable for a radiator component: Aida searched, found nothing,
        # and reported "inget på Palats just nu". Same class as the klinker
        # mismatch fixed 2026-08-31, found by scripts/audit_catalog.py on its
        # first run — and precisely the half a verifier in the request path
        # cannot see, because the defect shows up as an empty list.
        ("radiator", ["panelradiator", "sektionsradiator", "elradiator",
                      "radiator", "handdukstork"]),
        ("vvs", ["avloppsrör", "vattenrör", "stamrör", "golvbrunn"]),
        ("hiss", ["hiss", "elevator"]),
        ("storköksutrustning", ["diskmaskin", "storkök"]),
        ("sanitet", ["toalett", "wc", "handfat", "tvättställ", "dusch",
                     "badkar", "urinal", "blandare", "kran"]),
        # Before ventilation-adjacent words could claim them: a köksfläkt is a
        # white good in Aida's taxonomy and has its own typvärde.
        ("vitvaror", ["tvättmaskin", "torktumlare", "torkskåp", "spis",
                      "häll", "ugn", "mikrovåg", "köksfläkt", "spisfläkt",
                      "spiskåp", "fläktkåp"]),
        ("kylanläggning", ["kyl", "frys", "kylskåp", "kylanläggning"]),
    ]

    for category, keywords in category_keywords:
        excluded = CATEGORY_EXCLUSIONS.get(category, ())
        if any(x in text for x in excluded):
            continue
        # fast_inredning is checked first, so a title that names a fixture as
        # well ("Tvättställ med underskåp", "Diskbänksblandare") must fall
        # through to sanitet. Same rule as normalize_component_name.
        if category == "fast_inredning":
            from aida.data.climate_data import fixture_named_besides_cabinet
            if fixture_named_besides_cabinet(text, keywords):
                continue
        for kw in keywords:
            if kw in text:
                return category

    return ""


def fetch_listings(force_refresh: bool = False) -> list[dict]:
    """Fetch all published listings from Palats.

    Returns raw API response (only PUBLISHED), cached for 10 minutes.
    Returns empty list if no credentials or API error.
    Sets ``last_fetch_status`` so callers can distinguish failure from empty.
    """
    global _listings_cache, _listings_cache_time, last_fetch_status

    if (
        not force_refresh
        and _listings_cache is not None
        and (time.time() - _listings_cache_time) < _CACHE_TTL
    ):
        return _listings_cache

    cookies = _get_cookies()
    if not cookies:
        # _get_cookies already distinguishes "auth_failed" (login attempted but
        # rejected) from a plain absence of credentials. Don't clobber it.
        if last_fetch_status != "auth_failed":
            last_fetch_status = "no_credentials"
        logger.debug("No Palats credentials — skipping reuse search")
        return []

    try:
        resp = requests.get(
            f"{PALATS_BASE_URL}/v2/listings",
            cookies=cookies,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        # Handle both list and wrapped response formats
        if isinstance(data, list):
            listings = data
        elif isinstance(data, dict):
            listings = data.get("listings", data.get("data", data.get("items", [])))
        else:
            listings = []

        # Only keep published listings with available articles
        listings = [
            l for l in listings
            if l.get("listingStatus") == "PUBLISHED"
            and l.get("availableArticlesCount", 0) > 0
        ]

        _listings_cache = listings
        _listings_cache_time = time.time()
        last_fetch_status = "ok"
        logger.info("Fetched %d published listings from Palats", len(listings))
        return listings

    except requests.RequestException as e:
        last_fetch_status = "api_error"
        logger.warning("Palats API error: %s", e)
        return []


def _extract_listing(raw: dict) -> PalatsListing:
    """Extract a PalatsListing from raw API response.

    Mapped to actual Palats API v2 field names (verified 2026-03-31).
    """
    listing_id = str(raw.get("id", ""))
    title = raw.get("title", "")
    description = raw.get("articleConditionComment", "") or ""
    price = float(raw.get("price", 0) or 0)
    quantity = int(raw.get("availableArticlesCount", 0))
    unit = "st"

    # Thumbnail — use fullSizePath for best quality
    thumbnail = raw.get("thumbnail")
    image_url = ""
    if isinstance(thumbnail, dict):
        image_url = thumbnail.get("fullSizePath", thumbnail.get("path", ""))

    # Owner info for context
    owner = raw.get("owner", {})
    owner_name = owner.get("name", "") if isinstance(owner, dict) else ""

    # Both classifications read the title only. `description` here is Palats'
    # articleConditionComment (wear and damage), which names materials
    # incidentally: "Rostigt golv" on a locker, "repor i laminatet" on a desk.
    category = _normalize_to_aida_category(title)
    subcategory = _normalize_to_aida_subcategory(category, title)

    # Resolve location
    location_id = raw.get("locationId")
    loc_info = LOCATION_NAMES.get(location_id, {}) if location_id else {}
    location = loc_info.get("name", "")

    return PalatsListing(
        id=listing_id,
        title=title,
        description=f"{description} (kontakt: {owner_name})" if owner_name else description,
        price=price,
        quantity=quantity,
        unit=unit,
        category=category,
        subcategory=subcategory,
        image_url=image_url,
        url=listing_url(listing_id, str(raw.get("vendorId") or "")),
        location=location,
    )


def component_subcategory(component_name: str, category: str) -> str:
    """Infer the user's intended subcategory from the component name.

    Reuses SUBCATEGORY_KEYWORDS so listing-side and component-side
    classification stay in sync.
    """
    return _normalize_to_aida_subcategory(category, component_name)


def search_listings_for_component(
    component_name: str,
    all_listings: list[dict] | None = None,
    category: str | None = None,
) -> list[PalatsListing]:
    """Find Palats listings matching an Aida component.

    Two-stage relevance: listings whose subcategory matches the component's
    intended subcategory come first, then other listings in the same
    category. Lets a search for "Toalettstol" surface toilets ahead of
    handfat/dusch/etc. within the same sanitet bucket.

    Args:
        component_name: Aida component name (e.g. 'Toalettstol', 'Golv vinyl')
        all_listings: Pre-fetched raw listings (avoids re-fetching per component)
        category: The category the component was ROUTED to, when the caller
            has one. The alternatives agent routes by name + usage_context +
            directive (a tiled "Väggytskikt" goes to kakel), and until
            2026-09-14 the Palats side ignored that and re-derived the category
            from the bare name, so the EPD list and the reuse list for one
            component could come from two different materials. Falls back to
            the name-derived category when not given.

    Returns:
        Matched listings, ordered subcategory-match first.
    """
    from aida.data.climate_data import normalize_component_name

    target_category = category or normalize_component_name(component_name)
    if not target_category:
        return []

    if all_listings is None:
        all_listings = fetch_listings()

    if not all_listings:
        return []

    target_subcategory = component_subcategory(component_name, target_category)

    primary: list[PalatsListing] = []
    secondary: list[PalatsListing] = []
    for raw in all_listings:
        listing = _extract_listing(raw)
        if listing.category != target_category:
            continue
        if target_subcategory and listing.subcategory == target_subcategory:
            primary.append(listing)
        else:
            secondary.append(listing)

    return primary + secondary


# Reuse CO2e assumptions (kg CO2e per unit) — transport and minor refurbishment only
REUSE_CO2E_PER_UNIT: dict[str, float] = {
    "golv": 0.5,      # m2
    "radiator": 5.0,  # st — heavy steel, got its own listing-side category
                      # 2026-09-01 and would otherwise fall to the 2.0 default
                      # purely by omission. Between dörr (3.0) and kylanläggning
                      # (25.0): more transport mass than a door, no refrigerant.
    "kakel": 0.5,     # m2 — same handling as golv; got its own listing-side
                      # category 2026-08-31 and would otherwise fall to the
                      # 2.0 default purely by omission.
    "innervägg": 1.5,  # m2
    "yttervägg": 2.0,  # m2
    "fönster": 10.0,   # st — heavier, more transport impact
    "dörr": 3.0,       # st
    "tak": 1.0,        # m2
    "isolering": 0.5,  # m2
    "belysning": 1.0,  # st
    "ventilation": 0.5,  # lm
    "diskmaskin": 15.0,  # st
    "kylanläggning": 25.0,  # st
    "hiss": 500.0,     # st
}

# Default if category not in the dict above
_DEFAULT_REUSE_CO2E = 2.0
