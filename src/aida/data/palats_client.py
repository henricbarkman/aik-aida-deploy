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
    """A reuse listing from Palats, normalized for AIda."""

    id: str
    title: str
    description: str
    price: float  # SEK, 0 if free/unknown
    quantity: int
    unit: str
    category: str  # AIda category key (golv, fönster, etc.) or ""
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
        ("ytterdörr", ["ytterdörr", "entrédörr"]),
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
        ("spis", ["spis", "häll", "ugn"]),
        ("köksfläkt", ["köksfläkt", "fläktkåpa"]),
        ("mikro", ["mikrovåg", "mikro"]),
    ],
}


def _normalize_to_aida_subcategory(category: str, text: str) -> str:
    """Map listing text to a finer subcategory within its AIda category.

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


def _normalize_to_aida_category(title: str, description: str = "") -> str:
    """Map a Palats listing to an AIda component category using keywords.

    Returns the AIda category key (e.g. 'golv', 'fönster') or '' if no match.
    """
    text = f"{title} {description}".lower()

    # Order matters — more specific matches first
    # Multi-word patterns checked before single-word to avoid false positives
    category_keywords: list[tuple[str, list[str]]] = [
        ("fönster", [
            "fönster", "fönsterbåge", "fönsterkassett", "fönsterbänk",
            "energiglas",
        ]),
        ("dörr", [
            "dörr", "dörrblad", "dörrkarm", "innerdörr", "ytterdörr",
            "branddörr", "skjutdörr", "entrédörr",
        ]),
        ("golv", [
            "golv", "parkett", "klinker", "kakel", "vinylgolv", "vinylmatta",
            "laminat", "trägolv", "golvplatta", "matta", "linoleum",
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
        ("ventilation", [
            "ventilation", "fläkt", "kanal", "ventilationskanal",
            "don", "tilluftsdon", "frånluftsdon",
        ]),
        ("vvs", ["panelradiator", "radiator", "avloppsrör"]),
        ("hiss", ["hiss", "elevator"]),
        ("storköksutrustning", ["diskmaskin", "storkök"]),
        ("sanitet", ["toalett", "wc", "handfat", "tvättställ", "dusch",
                     "badkar", "urinal", "blandare", "kran"]),
        ("vitvaror", ["tvättmaskin", "torktumlare", "torkskåp", "spis",
                      "häll", "ugn", "mikrovåg", "köksfläkt"]),
        ("kylanläggning", ["kyl", "frys", "kylskåp", "kylanläggning"]),
    ]

    for category, keywords in category_keywords:
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

    category = _normalize_to_aida_category(title, description)
    subcategory = _normalize_to_aida_subcategory(category, f"{title} {description}")

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
        url=f"https://palats.app/web/listing/{listing_id}" if listing_id else "",
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
) -> list[PalatsListing]:
    """Find Palats listings matching an AIda component.

    Two-stage relevance: listings whose subcategory matches the component's
    intended subcategory come first, then other listings in the same
    category. Lets a search for "Toalettstol" surface toilets ahead of
    handfat/dusch/etc. within the same sanitet bucket.

    Args:
        component_name: AIda component name (e.g. 'Toalettstol', 'Golv vinyl')
        all_listings: Pre-fetched raw listings (avoids re-fetching per component)

    Returns:
        Matched listings, ordered subcategory-match first.
    """
    from aida.data.climate_data import normalize_component_name

    target_category = normalize_component_name(component_name)
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
