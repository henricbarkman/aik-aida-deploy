"""Climate data provider with layered fallback: Boverket API → Environdec → LLM estimate.

Usage:
    from aida.data.climate_provider import ClimateProvider
    provider = ClimateProvider()
    result = provider.lookup("betong")
    print(result.name, result.co2e_per_unit, result.source)

CLI:
    python -m aida.data.climate_provider --sync
    python -m aida.data.climate_provider --lookup "mineralull"
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass

from aida.data.climate_cache import CacheEntry, ClimateCache

logger = logging.getLogger(__name__)

# Maps Boverket category names (lowercase) → Aida component keys.
# Used for unit conversion inference and search filtering.
# Keys use common variations; matching is substring-based.
BOVERKET_TO_AIDA: dict[str, str] = {
    "isolering": "isolering",
    "betong": "betongvägg",
    "betong och cement": "betongvägg",
    "fönster, dörrar och glas": "fönster",
    "fönster och dörrar": "fönster",
    "glas": "fönster",
    "byggskivor": "innervägg",
    "skivor": "innervägg",
    "trävaror": "stomme",
    "trä": "stomme",
    "puts och bruk": "yttervägg",
    "tegel": "yttervägg",
    "takprodukter": "tak",
    "tak": "tak",
    "stål och metall": "ventilation",
    "stål": "ventilation",
    "golvmaterial": "golv",
    "golv": "golv",
    "ventilation": "ventilation",
    "belysning": "belysning",
    "hiss": "hiss",
}


# Sanity bound on a kg -> functional-unit conversion, as a multiple of the
# category's EPD typvärde for the target unit. Chosen from the catalog, not
# from taste: the widest spread any (category, unit) bucket shows between its
# typvärde and its own maximum is 15.5x (el per lm), then 13.0x (hiss per st),
# 12.9x (golv keramik per kg), 10.6x (fasadskikt per m2); every other bucket
# sits below 5x (measured 2026-09-14 over 1428 rows). Twenty therefore clears
# every genuine product the catalog holds, while the failure it exists to
# catch (a density read as an areal weight, a thickness assumed for a product
# sold by the piece, a wrong material density) is off by 50x to 1000x. A
# value above the bound is not a surprising product, it is a wrong unit.
_CONVERSION_SANITY_FACTOR = 20


def _conversion_is_implausible(
    comp_key: str, co2e_per_unit: float, unit: str, product_name: str,
) -> bool:
    """True when a converted per-unit value exceeds the category typvärde by
    more than _CONVERSION_SANITY_FACTOR. Logs why, so the fallback is visible.

    Returns False when no flat typvärde exists for (category, unit): the
    subcategorized categories (sanitet, belysning, vitvaror) have no
    meaningful category-wide figure, and a guard with no yardstick must stay
    open rather than guess.
    """
    from aida.data.epd_baseline_medians import get_baseline_typvärde

    typ = get_baseline_typvärde(comp_key, _normalize_unit(unit))
    if not typ:
        return False
    bound = typ["baseline_co2e_per_unit"] * _CONVERSION_SANITY_FACTOR
    if co2e_per_unit <= bound:
        return False
    logger.warning(
        "Unit conversion rejected for %r (%s): %.2f kg CO2e/%s is %.0fx the "
        "%s typvärde %.2f (bound %dx); keeping the kg figure instead",
        product_name, comp_key, co2e_per_unit, unit,
        co2e_per_unit / typ["baseline_co2e_per_unit"], comp_key,
        typ["baseline_co2e_per_unit"], _CONVERSION_SANITY_FACTOR,
    )
    return True


def _match_boverket_category(boverket_cat: str) -> str | None:
    """Match a Boverket category name to an Aida component key."""
    cat_lower = boverket_cat.lower().strip()
    if cat_lower in BOVERKET_TO_AIDA:
        return BOVERKET_TO_AIDA[cat_lower]
    # Substring match for variations
    for pattern, aida_key in BOVERKET_TO_AIDA.items():
        if pattern in cat_lower or cat_lower in pattern:
            return aida_key
    return None


@dataclass
class ClimateResult:
    """Result from a climate data lookup."""
    name: str
    co2e_per_unit: float
    cost_per_unit: float
    unit: str
    source: str
    source_layer: str = ""  # "boverket", "environdec", "llm"
    category: str = ""
    cache_key: str = ""  # product_name of the cache row this result came from


def _normalize_unit(unit: str) -> str:
    """Normalize a unit string for comparison (m² == m2, strip/case-fold)."""
    u = (unit or "").strip().lower()
    return {"m²": "m2", "m³": "m3", "pcs": "st", "styck": "st"}.get(u, u)


class ClimateProvider:
    """Layered climate data provider.

    Lookup order:
    1. Cache (SQLite) — returns immediately if fresh
    2. Boverket API — official Swedish building product climate data
    3. Environdec EPD database — product-specific EPDs (14,000+)
    4. (removed — old climate_data.py fallback deprecated)

    Source attribution: always "Boverket (typiskt värde)" for baseline data,
    regardless of whether fetched live or from cache.
    """

    def __init__(self, cache: ClimateCache | None = None):
        self._cache = cache or ClimateCache()
        self._boverket_synced = False
        self._environdec_client = None

    def lookup(
        self,
        product_name: str,
        component_hint: str = "",
    ) -> ClimateResult | None:
        """Look up climate data for a product. Returns None only for empty input.

        Args:
            product_name: Product or component name to look up.
            component_hint: Aida component category (e.g. "golv", "isolering")
                for unit conversion. If empty, tries to infer from product_name.
        """
        if not product_name or not product_name.strip():
            return None

        key = product_name.lower().strip()

        # Layer 1: Check cache (covers Boverket and Environdec entries)
        cached = self._cache.get(key)
        if cached:
            result = _entry_to_result(cached)
            result = self._maybe_convert_units(result, component_hint or key, cached.extra_json)
            return self._maybe_enrich_cost(result, key)

        # Layer 2: Try Boverket API (fuzzy search in cache after sync)
        result = self._try_boverket(key, component_hint)
        if result:
            return self._maybe_enrich_cost(result, key)

        # Layer 3: Environdec EPD database (product-specific EPDs)
        result = self._try_environdec(key, component_hint)
        if result:
            return self._maybe_enrich_cost(result, key)

        # Layer 4: Retry with normalized category key if full name didn't match
        # e.g. "Golvbeläggning, tambur (PVC från 2001)" → search for "golv" category
        result = self._try_normalized(key, component_hint)
        if result:
            return self._maybe_enrich_cost(result, key)

        return None

    def lookup_without_price(
        self,
        product_name: str,
        component_hint: str = "",
    ) -> ClimateResult | None:
        """Look up climate data without triggering price enrichment.

        Same fallback chain as lookup() but skips the per-component LLM
        web search for prices. Use this when price enrichment will be
        done separately in batch.
        """
        if not product_name or not product_name.strip():
            return None

        key = product_name.lower().strip()

        cached = self._cache.get(key)
        if cached:
            result = _entry_to_result(cached)
            return self._maybe_convert_units(result, component_hint or key, cached.extra_json)

        result = self._try_boverket(key, component_hint)
        if result:
            return result

        result = self._try_environdec(key, component_hint)
        if result:
            return result

        # Retry with normalized category key
        result = self._try_normalized(key, component_hint)
        if result:
            return result

        return None

    def ensure_synced(self) -> None:
        """Pre-load Boverket data if not already synced."""
        if not self._boverket_synced:
            if self._cache.count("boverket") > 0:
                self._boverket_synced = True
            else:
                try:
                    self.sync_boverket()
                except Exception as e:
                    logger.warning("Boverket pre-sync failed: %s", e)

    def _maybe_enrich_cost(self, result: ClimateResult, product_name: str) -> ClimateResult:
        """Try web search for current market price. Skips if already enriched."""
        # Enrich (and read back) against the actual cache row this result came
        # from. For fuzzy Boverket matches the matched product_name differs from
        # the lookup query; updating under the query key would hit 0 rows and the
        # expensive web search would re-run on every lookup of the same product.
        key = (result.cache_key or product_name).lower().strip()
        # Check if already price-enriched in cache
        cached = self._cache.get(key)
        if cached and cached.price_enriched:
            if cached.cost_per_unit > 0:
                # Use cached enriched price
                if result.cost_per_unit != cached.cost_per_unit:
                    return ClimateResult(
                        name=result.name, co2e_per_unit=result.co2e_per_unit,
                        cost_per_unit=cached.cost_per_unit, unit=result.unit,
                        source=result.source,
                        source_layer=result.source_layer,
                        category=getattr(result, 'category', ''),
                        cache_key=result.cache_key,
                    )
                return result
            return result  # Enrichment was attempted but found nothing
        try:
            from aida.data.pricing_provider import lookup_price
            pricing = lookup_price(product_name, result.unit)
            if pricing is None:
                # Mark as attempted so we don't retry every request
                self._cache.update_cost(key, result.cost_per_unit)
                return result
            price, price_unit, source = pricing
            # The web-searched price may be quoted in a different unit than the
            # climate result (e.g. SEK/m2 for a window priced per st). Storing it
            # as-is would later multiply by the wrong quantity. On mismatch, keep
            # the result unpriced and mark enrichment as attempted.
            if (price_unit and result.unit
                    and _normalize_unit(price_unit) != _normalize_unit(result.unit)):
                logger.debug(
                    "Price unit mismatch for '%s': got %s, expected %s — skipping",
                    product_name, price_unit, result.unit,
                )
                self._cache.update_cost(key, result.cost_per_unit)
                return result
            self._cache.update_cost(key, price)
            return ClimateResult(
                name=result.name, co2e_per_unit=result.co2e_per_unit,
                cost_per_unit=price, unit=result.unit,
                source=result.source,
                source_layer=result.source_layer,
                category=getattr(result, 'category', ''),
                cache_key=result.cache_key,
            )
        except Exception as e:
            logging.getLogger(__name__).debug("Price enrichment failed for '%s': %s", product_name, e)
            return result

    def _maybe_convert_units(
        self,
        result: ClimateResult,
        component_hint: str,
        extra_json: str = "",
    ) -> ClimateResult:
        """Convert kg-based values to functional units if possible."""
        if result.unit != "kg":
            return result  # already in functional unit

        from aida.data.climate_data import normalize_component_name
        from aida.data.unit_conversion import (
            convert_to_functional_unit,
            get_areal_density_from_extra,
            get_density_for_component,
        )

        comp_key = normalize_component_name(component_hint)

        # If hint didn't resolve, try inferring from Boverket category
        if not comp_key and extra_json:
            try:
                extra = json.loads(extra_json)
                bov_cat = extra.get("category", "")
                if bov_cat:
                    comp_key = self._cache.get_aida_component(bov_cat) or ""
            except (json.JSONDecodeError, TypeError):
                pass

        if not comp_key:
            return result

        density = get_density_for_component(comp_key, extra_json, product_name=result.name)
        areal_density = get_areal_density_from_extra(extra_json)
        co2e_converted, new_unit = convert_to_functional_unit(
            result.co2e_per_unit, comp_key, density, areal_density,
        )

        if new_unit != "kg":
            # A converted figure that dwarfs everything the catalog has seen
            # for this category is a unit error, not a product. Keep the kg
            # figure: downstream already handles a unit mismatch honestly,
            # whereas a silently accepted 3 000 kg CO2e/m2 floor becomes the
            # baseline and makes every alternative look miraculous.
            if _conversion_is_implausible(comp_key, co2e_converted, new_unit, result.name):
                return result
            return ClimateResult(
                name=result.name,
                co2e_per_unit=co2e_converted,
                cost_per_unit=result.cost_per_unit,
                unit=new_unit,
                source=result.source,
                source_layer=result.source_layer,
                category=result.category,
                cache_key=result.cache_key,
            )
        return result

    def sync_boverket(self) -> int:
        """Download all Boverket data and cache it. Returns count of entries cached."""
        from aida.data.boverket_client import BoverketClient

        client = BoverketClient()
        try:
            resources = client.get_all_resources()
        except Exception as e:
            logger.warning("Boverket API sync failed: %s", e)
            return 0

        entries = client.resources_to_cache_entries(resources)
        if entries:
            self._cache.put_many(entries)

        # Populate Boverket category → Aida component mappings
        self._sync_category_mappings(entries)
        self._boverket_synced = True

        return len(entries)

    def _sync_category_mappings(self, entries: list[CacheEntry]) -> None:
        """Extract Boverket categories from synced entries and store mappings."""
        categories: set[str] = set()
        for entry in entries:
            try:
                extra = json.loads(entry.extra_json or "{}")
                cat = extra.get("category", "")
                if cat:
                    categories.add(cat.lower())
            except (json.JSONDecodeError, TypeError):
                pass

        mappings = {}
        for cat in categories:
            aida_key = _match_boverket_category(cat)
            if aida_key:
                mappings[cat] = aida_key

        if mappings:
            self._cache.put_category_mappings(mappings)
            logger.info("Category mappings: %d/%d Boverket categories mapped",
                        len(mappings), len(categories))

    def _try_boverket(self, key: str, component_hint: str = "") -> ClimateResult | None:
        """Try Boverket API. Syncs on first call, then uses ranked search.

        On Vercel, the cache is pre-populated at build time so we check
        for existing data before attempting an API sync.
        """
        hint = component_hint or key
        if not self._boverket_synced:
            # Check if cache already has Boverket data (pre-populated)
            if self._cache.count("boverket") > 0:
                self._boverket_synced = True
            else:
                try:
                    count = self.sync_boverket()
                    if count == 0:
                        return None
                except Exception as e:
                    logger.warning("Boverket API unavailable: %s", e)
                    return None

        return self._search_boverket(key, hint)

    def _search_boverket(self, key: str, component_hint: str = "") -> ClimateResult | None:
        """Ranked search against Boverket cache entries.

        Ranking:
        1. Exact match on product_name
        2. Starts-with match (category-filtered if possible)
        3. Starts-with match (unfiltered) — skipped when category info exists
        4. Substring match (category-filtered)
        5. Substring match (unfiltered) — skipped when category info exists

        When category filtering is available, unfiltered fuzzy matches (3, 5)
        are skipped to avoid returning wrong-category materials.
        E.g. "golvmatta" with category "golv" must not match "golvskiva"
        (gipsskiva/Byggskivor).
        """
        if len(key) < 3:
            return None

        conn = self._cache._get_conn()
        hint = component_hint or key
        cat_likes = self._category_like_patterns(component_hint)
        cols = ("product_name, name, co2e_per_unit, cost_per_unit, unit, "
                "source, source_layer, fetched_at, expires_at, extra_json, price_enriched")

        def _result_from_row(row):
            entry = CacheEntry(**dict(row))
            result = _entry_to_result(entry)
            return self._maybe_convert_units(result, hint, entry.extra_json)

        # Rank 1: Exact match
        row = conn.execute(
            f"SELECT {cols} FROM climate_cache WHERE source_layer = 'boverket' "
            "AND product_name = ?",
            (key,),
        ).fetchone()
        if row:
            return _result_from_row(row)

        # Rank 2: Starts-with, category-filtered
        if cat_likes:
            for cat_like in cat_likes:
                row = conn.execute(
                    f"SELECT {cols} FROM climate_cache WHERE source_layer = 'boverket' "
                    "AND product_name LIKE ? AND LOWER(extra_json) LIKE ? "
                    "ORDER BY LENGTH(product_name) LIMIT 1",
                    (f"{key}%", cat_like),
                ).fetchone()
                if row:
                    return _result_from_row(row)

        # When a component_hint is provided, we know what type of product we
        # want.  Skip unfiltered fuzzy matches to avoid cross-category pollution
        # (e.g. "golvmatta" matching "golvskiva" which is a gypsum board).
        has_hint = bool(component_hint and component_hint.strip())

        # Rank 3: Starts-with, unfiltered (only when no hint at all)
        if not has_hint and not cat_likes:
            row = conn.execute(
                f"SELECT {cols} FROM climate_cache WHERE source_layer = 'boverket' "
                "AND product_name LIKE ? ORDER BY LENGTH(product_name) LIMIT 1",
                (f"{key}%",),
            ).fetchone()
            if row:
                return _result_from_row(row)

        # Rank 4: Substring, category-filtered
        if cat_likes:
            for cat_like in cat_likes:
                row = conn.execute(
                    f"SELECT {cols} FROM climate_cache WHERE source_layer = 'boverket' "
                    "AND product_name LIKE ? AND LOWER(extra_json) LIKE ? "
                    "ORDER BY LENGTH(product_name) LIMIT 1",
                    (f"%{key}%", cat_like),
                ).fetchone()
                if row:
                    return _result_from_row(row)

        # Rank 5: Substring, unfiltered (only when no hint at all)
        if not has_hint and not cat_likes:
            row = conn.execute(
                f"SELECT {cols} FROM climate_cache WHERE source_layer = 'boverket' "
                "AND product_name LIKE ? ORDER BY LENGTH(product_name) LIMIT 1",
                (f"%{key}%",),
            ).fetchone()
            if row:
                return _result_from_row(row)

        return None

    def _category_like_patterns(self, component_hint: str) -> list[str]:
        """Convert component_hint to LIKE patterns for extra_json category filtering."""
        if not component_hint:
            return []

        from aida.data.climate_data import normalize_component_name
        aida_key = normalize_component_name(component_hint)
        if not aida_key:
            return []

        categories = self._cache.get_categories_for_aida_key(aida_key)
        return [f'%"category": "{cat}"%' for cat in categories]

    def _get_environdec(self):
        """Lazy-init Environdec client."""
        if self._environdec_client is None:
            from aida.data.environdec_client import EnvirondecClient
            self._environdec_client = EnvirondecClient()
        return self._environdec_client

    def _try_environdec(self, key: str, component_hint: str = "") -> ClimateResult | None:
        """Search Environdec EPD database for product-specific EPDs."""
        client = self._get_environdec()
        try:
            matches = client.search_index(key, component_hint=component_hint, max_results=5)
        except Exception as e:
            logger.warning("Environdec search failed: %s", e)
            return None

        if not matches:
            return None

        # Fetch full EPD for best match
        best = matches[0]
        try:
            detail = client.fetch_epd_detail(best.uuid, best.version)
        except Exception as e:
            logger.warning("Environdec EPD fetch failed: %s", e)
            return None

        if detail is None:
            return None

        # Cache the result. epd_to_cache_entry returns None when GWP-fossil is
        # missing (e.g. legacy EPDs with only GWP-total). We drop those rather
        # than serve a misleading zero — they are not comparable to the
        # Boverket baseline.
        entry = client.epd_to_cache_entry(detail, key)
        if entry is None:
            return None
        self._cache.put(entry)

        result = _entry_to_result(entry)
        return self._maybe_convert_units(result, component_hint or key, entry.extra_json)

    def _try_normalized(self, key: str, component_hint: str = "") -> ClimateResult | None:
        """Retry lookup with normalized/simplified search terms.

        When intake produces verbose names like "Golvbeläggning, tambur (PVC från 2001)",
        extract material keywords and category to find matches in Boverket or Environdec.
        """
        from aida.data.climate_data import normalize_component_name

        # Try component_hint first (e.g. "golv" from intake category)
        comp_key = normalize_component_name(component_hint) if component_hint else ""
        if not comp_key:
            comp_key = normalize_component_name(key)
        if not comp_key:
            return None

        # Extract material keywords from the original name
        material_keywords = _extract_material_keywords(key)

        # Try each keyword against Boverket
        for kw in material_keywords:
            result = self._try_boverket(kw, comp_key)
            if result:
                logger.info("Normalized lookup hit: '%s' → '%s' (from '%s')", kw, result.name, key)
                return result

        # Try just the category key against Boverket
        if len(comp_key) >= 3:
            result = self._try_boverket(comp_key, comp_key)
            if result:
                logger.info("Category lookup hit: '%s' → '%s' (from '%s')", comp_key, result.name, key)
                return result

        # Try English translations against Environdec
        english_keywords = _get_english_search_terms(key, comp_key)
        for en_kw in english_keywords:
            result = self._try_environdec(en_kw, comp_key)
            if result:
                logger.info("Environdec normalized hit: '%s' → '%s' (from '%s')", en_kw, result.name, key)
                return result

        return None

    def sync_environdec_index(self) -> int:
        """Pre-fetch the Environdec EPD index. Returns count of EPDs indexed."""
        client = self._get_environdec()
        try:
            index = client.fetch_index(use_cached=False)
            return len(index)
        except Exception as e:
            logger.warning("Environdec index sync failed: %s", e)
            return 0

# Swedish product terms → English Environdec search terms.
# Used when Boverket has no match and the Swedish name didn't match Environdec directly.
_SWEDISH_TO_ENGLISH: dict[str, list[str]] = {
    # Floor coverings
    "golv": ["vinyl flooring", "floor covering"],
    "golvmatta": ["vinyl flooring", "floor covering", "carpet"],
    "plastgolv": ["vinyl flooring", "pvc floor covering"],
    "plastmatta": ["vinyl flooring", "pvc floor covering"],
    "textilmatta": ["carpet", "textile floor covering"],
    "heltäckningsmatta": ["carpet", "textile floor covering"],
    "vinylgolv": ["vinyl flooring"],
    "linoleum": ["linoleum flooring"],
    "parkett": ["parquet flooring", "wood floor"],
    "laminat": ["laminate flooring"],
    # klinker/kakel/keramik live in the renovation-materials block below.
    # Walls
    "gipsskiva": ["plasterboard", "gypsum board"],
    "innervägg": ["plasterboard", "gypsum board"],
    "yttervägg": ["facade", "exterior wall"],
    # Insulation
    "mineralull": ["mineral wool insulation", "stone wool"],
    "glasull": ["glass wool insulation"],
    "cellplast": ["eps insulation", "expanded polystyrene"],
    # Structural
    "betong": ["concrete", "precast concrete"],
    "stål": ["steel", "structural steel"],
    "trä": ["timber", "wood"],
    # Stomme (load-bearing frame) — specific structural elements.
    "stålbalk": ["structural steel beam", "steel section"],
    "stålpelare": ["structural steel column", "steel section"],
    "limträ": ["glulam", "glued laminated timber"],
    "kl-trä": ["cross-laminated timber", "clt"],
    "massivträ": ["solid timber", "clt"],
    "bjälklag": ["precast concrete slab", "hollow core slab"],
    "håldäck": ["hollow core slab"],
    # Appliances (no real white-goods EPDs in the source yet, but route the
    # search to the right English term so a future EPD is found, and so the
    # raw Swedish name is not matched against unrelated bilingual EPD names).
    "tvättmaskin": ["washing machine"],
    "värmepump": ["heat pump"],
    "diskmaskin": ["dishwasher"],
    "torktumlare": ["tumble dryer"],
    # Other
    "tak": ["roofing", "roof tile"],
    "fönster": ["window", "triple glazed window"],
    # Interior door first: an unqualified "dörr" in a renovation is almost
    # always interior, and bare "door" wrongly favours triple-glazed exterior
    # door EPDs that start with the word "Door".
    "dörr": ["interior door", "door"],
    # Renovation materials (added 2026-06-17).
    "kakel": ["ceramic tile", "wall tile"],
    "klinker": ["ceramic tile", "porcelain tile", "floor tile"],
    "keramik": ["ceramic tile"],
    "rör": ["pipe", "plumbing pipe"],
    "avloppsrör": ["drainage pipe", "sewer pipe"],
    "vattenrör": ["water pipe", "pex pipe"],
    "målning": ["paint", "interior paint"],
    "väggfärg": ["wall paint", "interior paint"],
    "färg": ["paint"],
    "elkabel": ["electrical cable", "power cable"],
    "kabel": ["cable", "installation cable"],
    "radiator": ["radiator", "panel radiator"],
    "värmeelement": ["radiator", "heating panel"],
}

# Broad category keys whose English terms are only a fallback. When a more
# specific material is named in the same string (linoleum, parkett, vinylgolv),
# the category default must NOT also be searched — otherwise e.g. a linoleum
# floor matches a generic vinyl-flooring EPD because "golv" is a substring of
# "golvbeläggning".
_CATEGORY_DEFAULT_KEYS = {
    "golv", "innervägg", "yttervägg", "betong", "stål", "trä",
    "tak", "fönster", "dörr",
}


def _get_english_search_terms(swedish_name: str, comp_key: str) -> list[str]:
    """Get English Environdec search terms for a Swedish product name.

    Specific material terms win over broad category defaults, ordered
    most-specific-first, so the search hits the right product before any
    generic fallback. Returns [] when nothing maps.
    """
    name_lower = swedish_name.lower().strip()

    specific: list[tuple[int, list[str]]] = []  # named materials
    generic: list[tuple[int, list[str]]] = []   # broad category defaults
    for sv_term, en_terms in _SWEDISH_TO_ENGLISH.items():
        if sv_term in name_lower:
            bucket = generic if sv_term in _CATEGORY_DEFAULT_KEYS else specific
            bucket.append((len(sv_term), en_terms))

    # If a specific material is named, ignore the broad category defaults.
    chosen = specific if specific else generic
    chosen.sort(key=lambda x: -x[0])  # longer Swedish key = more specific

    terms: list[str] = []
    for _, en_terms in chosen:
        for t in en_terms:
            if t not in terms:
                terms.append(t)

    # Fallback: nothing matched in the name; use the component category key.
    if not terms and comp_key in _SWEDISH_TO_ENGLISH:
        terms.extend(_SWEDISH_TO_ENGLISH[comp_key])

    return terms


def _extract_material_keywords(name: str) -> list[str]:
    """Extract likely material search terms from a verbose component name.

    "Golvbeläggning, tambur (PVC från 2001)" → ["pvc", "vinyl", "golvbeläggning"]
    "Toalettstol" → ["toalettstol"]
    """
    # Material synonyms that map to Boverket search terms
    material_map = {
        "pvc": ["pvc", "vinyl", "vinylgolv"],
        "vinyl": ["vinyl", "vinylgolv"],
        "linoleum": ["linoleum"],
        "klinker": ["klinker", "keramik"],
        "parkett": ["parkett", "trägolv"],
        "laminat": ["laminat", "laminatgolv"],
        "gips": ["gipsskiva", "gips"],
        "betong": ["betong"],
        "tegel": ["tegel"],
        "mineralull": ["mineralull"],
        "cellplast": ["cellplast", "eps"],
        "stål": ["stål"],
        "trä": ["trä", "träpanel"],
        "aluminium": ["aluminium"],
        "glas": ["glas"],
    }

    name_lower = name.lower()
    keywords = []

    for material, search_terms in material_map.items():
        if material in name_lower:
            keywords.extend(search_terms)

    # Also try the first word (often the main noun)
    first_word = name_lower.split(",")[0].split("(")[0].strip()
    if first_word and len(first_word) >= 3 and first_word not in keywords:
        keywords.append(first_word)

    return keywords


def _entry_to_result(entry: CacheEntry) -> ClimateResult:
    return ClimateResult(
        name=entry.name,
        co2e_per_unit=entry.co2e_per_unit,
        cost_per_unit=entry.cost_per_unit,
        unit=entry.unit,
        source=entry.source,
        source_layer=entry.source_layer,
        cache_key=entry.product_name,
    )


def main():
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Usage:", file=sys.stderr)
        print("  python -m aida.data.climate_provider --sync", file=sys.stderr)
        print("  python -m aida.data.climate_provider --sync-environdec", file=sys.stderr)
        print("  python -m aida.data.climate_provider --lookup <product>", file=sys.stderr)
        print("  python -m aida.data.climate_provider --epd-search <query>", file=sys.stderr)
        sys.exit(1)

    provider = ClimateProvider()

    if sys.argv[1] == "--sync":
        count = provider.sync_boverket()
        print(f"Synced {count} entries from Boverkets klimatdatabas")

    elif sys.argv[1] == "--sync-environdec":
        count = provider.sync_environdec_index()
        print(f"Indexed {count} EPDs from Environdec")

    elif sys.argv[1] == "--epd-search" and len(sys.argv) >= 3:
        query = " ".join(sys.argv[2:])
        client = provider._get_environdec()
        matches = client.search_index(query, max_results=10)
        if matches:
            for m in matches:
                print(f"  {m.name[:55]:55s} | {m.owner[:25]:25s} | {m.geo:5s} | {m.reg_no}")
        else:
            print(f"No EPDs found for: {query}")

    elif sys.argv[1] == "--lookup" and len(sys.argv) >= 3:
        product = " ".join(sys.argv[2:])
        result = provider.lookup(product)
        if result:
            print(f"name: {result.name}")
            print(f"co2e_per_unit: {result.co2e_per_unit}")
            print(f"unit: {result.unit}")
            print(f"source: {result.source}")
            print(f"cost_per_unit: {result.cost_per_unit}")
        else:
            print(f"No data found for: {product}")
            sys.exit(1)
    else:
        print(f"Unknown command: {sys.argv[1]}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
