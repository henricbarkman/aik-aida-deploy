"""Client for Environdec's EPD database (EPD International System).

Uses the soda4LCA data hub at data.environdec.com — no API key required.
Fetches EPD metadata index + individual EPD details with GWP values.

Data stock: Environdata (Digital EPD) — ~14,000 EPDs, mostly construction products.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from aida.data.climate_cache import TTL_BOVERKET, CacheEntry

logger = logging.getLogger(__name__)

DATA_HUB_URL = "https://data.environdec.com/resource"
ENVIRONDATA_STOCK = "a6c533b3-502e-47b9-885d-31304bf15c64"
REQUEST_TIMEOUT = 30
INDEX_PAGE_SIZE = 1000
INDEX_PATH = Path(__file__).parent / "environdec_index.json"

# GWP indicator UUIDs (EN 15804+A2)
GWP_FOSSIL_NAMES = {"gwp-fossil", "global warming potential - fossil fuels"}
GWP_TOTAL_NAMES = {"gwp-total", "global warming potential - total"}


@dataclass
class EPDSummary:
    """Lightweight EPD metadata from the index listing."""
    name: str
    uuid: str
    version: str
    geo: str
    owner: str
    reg_no: str
    classification: str
    valid_until: int


@dataclass
class EPDDetail:
    """Full EPD with GWP values extracted."""
    name: str
    uuid: str
    reg_no: str
    owner: str
    declared_unit: str
    gwp_fossil_a1a3: float | None
    gwp_total_a1a3: float | None
    gwp_biogenic_a1a3: float | None
    modules: dict[str, float]  # module code → GWP-fossil value
    geo: str


class EnvirondecClient:
    def __init__(self, base_url: str = DATA_HUB_URL):
        self.base_url = base_url
        self._index: list[EPDSummary] | None = None

    def fetch_index(self, use_cached: bool = True) -> list[EPDSummary]:
        """Fetch the full EPD index. Uses local JSON cache if available."""
        if self._index is not None:
            return self._index

        # Try local cache first
        if use_cached and INDEX_PATH.exists():
            age_days = (time.time() - INDEX_PATH.stat().st_mtime) / 86400
            if age_days < 30:
                self._index = self._load_index_file()
                if self._index:
                    logger.info("Loaded %d EPDs from cached index (%.0f days old)",
                                len(self._index), age_days)
                    return self._index

        # Fetch from API
        self._index = self._fetch_index_from_api()

        # Save to local cache
        self._save_index_file(self._index)

        return self._index

    def search_index(
        self,
        query: str,
        geo_filter: str = "",
        component_hint: str = "",
        max_results: int = 20,
    ) -> list[EPDSummary]:
        """Search the index with ranked scoring.

        Scoring factors:
        - Name match quality (exact > starts-with > word match > substring > owner-only)
        - Component hint alignment (product name contains hint keywords)
        - Geographic preference (SE > NORD > RER > GLO > other)
        - Name specificity (shorter = more specific = better)
        """
        index = self.fetch_index()
        query_lower = query.lower().strip()
        if not query_lower:
            return []

        query_tokens = set(query_lower.split())
        hint_keywords = _get_hint_keywords(component_hint) if component_hint else set()

        scored: list[tuple[float, EPDSummary]] = []
        for epd in index:
            if geo_filter and epd.geo != geo_filter:
                continue

            score = _score_match(epd, query_lower, query_tokens, hint_keywords, component_hint)
            if score > 0:
                scored.append((score, epd))

        scored.sort(key=lambda x: -x[0])
        return [epd for _, epd in scored[:max_results]]

    def fetch_epd_detail(self, uuid: str, version: str = "") -> EPDDetail | None:
        """Fetch full EPD data including GWP values."""
        url = f"{self.base_url}/processes/{uuid}"
        params = {"format": "json", "view": "extended"}
        if version:
            params["version"] = version

        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning("Failed to fetch EPD %s: %s", uuid, e)
            return None

        data = resp.json()
        return self._parse_epd_detail(data, uuid)

    def epd_to_cache_entry(self, detail: EPDDetail, product_name: str) -> CacheEntry | None:
        """Convert an EPDDetail to a CacheEntry for the climate cache.

        Returns None when GWP-fossil A1-A3 is missing. We do NOT fall back to
        GWP-total (which includes biogenic credit) since that is not
        comparable to the Boverket baseline. A None return signals callers to
        skip this EPD entirely rather than caching a misleading 0.0 value
        that downstream might treat as a real CO2e reading.
        """
        if detail.gwp_fossil_a1a3 is None:
            if detail.gwp_total_a1a3 is not None:
                logger.info(
                    "Skipping EPD %s (no GWP-fossil; only GWP-total available, "
                    "which is not comparable to Boverket baseline).",
                    detail.reg_no or detail.uuid[:8],
                )
            return None

        now = time.time()
        gwp = detail.gwp_fossil_a1a3

        extra = {
            "uuid": detail.uuid,
            "reg_no": detail.reg_no,
            "owner": detail.owner,
            "declared_unit": detail.declared_unit,
            "gwp_fossil_a1a3": detail.gwp_fossil_a1a3,
            "gwp_total_a1a3": detail.gwp_total_a1a3,
            "gwp_biogenic_a1a3": detail.gwp_biogenic_a1a3,
            "geo": detail.geo,
            "modules": detail.modules,
        }

        return CacheEntry(
            product_name=product_name.lower().strip(),
            name=detail.name,
            co2e_per_unit=gwp,
            cost_per_unit=0.0,
            unit=detail.declared_unit,
            source=f"Environdec EPD {detail.reg_no}" if detail.reg_no else f"Environdec EPD {detail.uuid[:8]}",
            source_layer="environdec",
            fetched_at=now,
            expires_at=now + TTL_BOVERKET,
            extra_json=json.dumps(extra, ensure_ascii=False),
        )

    # --- Internal ---

    def _fetch_index_from_api(self) -> list[EPDSummary]:
        """Fetch all EPDs from the soda4LCA data hub."""
        all_epds: list[EPDSummary] = []
        start = 0

        while True:
            url = (f"{self.base_url}/datastocks/{ENVIRONDATA_STOCK}"
                   f"/processes?format=json&pageSize={INDEX_PAGE_SIZE}&startIndex={start}")
            try:
                resp = requests.get(url, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
            except requests.RequestException as e:
                logger.warning("Environdec index fetch failed at offset %d: %s", start, e)
                break

            data = resp.json()
            total = data.get("totalCount", 0)
            batch = data.get("data", [])

            for item in batch:
                all_epds.append(EPDSummary(
                    name=item.get("name", "").strip(),
                    uuid=item.get("uuid", ""),
                    version=item.get("version", ""),
                    geo=item.get("geo", ""),
                    owner=item.get("owner", ""),
                    reg_no=item.get("regNo", ""),
                    classification=item.get("classific", ""),
                    valid_until=item.get("validUntil", 0),
                ))

            logger.info("Environdec index: %d/%d", len(all_epds), total)

            if len(all_epds) >= total or not batch:
                break
            start += INDEX_PAGE_SIZE
            time.sleep(0.5)

        return all_epds

    def _load_index_file(self) -> list[EPDSummary]:
        """Load index from local JSON cache."""
        try:
            with open(INDEX_PATH) as f:
                data = json.load(f)
            return [EPDSummary(
                name=item.get("name", ""),
                uuid=item.get("uuid", ""),
                version=item.get("version", ""),
                geo=item.get("geo", ""),
                owner=item.get("owner", ""),
                reg_no=item.get("regNo", item.get("reg_no", "")),
                classification=item.get("classific", item.get("classification", "")),
                valid_until=item.get("validUntil", item.get("valid_until", 0)),
            ) for item in data]
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load Environdec index: %s", e)
            return []

    def _save_index_file(self, index: list[EPDSummary]) -> None:
        """Save index to local JSON cache."""
        data = [
            {
                "name": e.name, "uuid": e.uuid, "version": e.version,
                "geo": e.geo, "owner": e.owner, "regNo": e.reg_no,
                "classific": e.classification, "validUntil": e.valid_until,
            }
            for e in index
        ]
        try:
            INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(INDEX_PATH, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info("Saved Environdec index: %d entries, %.1f MB",
                        len(data), INDEX_PATH.stat().st_size / 1024 / 1024)
        except OSError as e:
            logger.warning("Failed to save Environdec index: %s", e)

    def _parse_epd_detail(self, data: dict, uuid: str) -> EPDDetail:
        """Parse soda4LCA JSON response into EPDDetail."""
        pi = data.get("processInformation", {})
        desc = pi.get("dataSetInformation", {})

        # Name
        base_name = desc.get("name", {}).get("baseName", [{}])
        name = base_name[0].get("value", "unknown") if base_name else "unknown"
        name = name.strip()

        # Registration number
        reg_no = ""
        other_info = desc.get("other", {}).get("anies", [])
        for a in other_info:
            if isinstance(a, dict) and a.get("name") == "registrationNumber":
                reg_no = a.get("value", "")
                break

        # Owner
        owner = ""
        admin = data.get("administrativeInformation", {})
        pub_owner = admin.get("publicationAndOwnership", {})
        owner_ref = pub_owner.get("referenceToOwnershipOfDataSet", {})
        owner_desc = owner_ref.get("shortDescription", [{}])
        if owner_desc:
            owner = owner_desc[0].get("value", "")

        # Declared unit from reference flow
        declared_unit = self._extract_declared_unit(data)

        # GWP values
        gwp_fossil = None
        gwp_total = None
        gwp_biogenic = None
        modules: dict[str, float] = {}

        for result in data.get("LCIAResults", {}).get("LCIAResult", []):
            ref = result.get("referenceToLCIAMethodDataSet", {})
            indicator_desc = ref.get("shortDescription", [{}])
            indicator_name = indicator_desc[0].get("value", "").lower() if indicator_desc else ""

            is_fossil = any(n in indicator_name for n in GWP_FOSSIL_NAMES)
            is_total = any(n in indicator_name for n in GWP_TOTAL_NAMES)
            is_biogenic = "biogenic" in indicator_name

            anies = result.get("other", {}).get("anies", [])
            for a in anies:
                module = a.get("module", "")
                value_str = a.get("value", "")
                if value_str in ("ND", "MNA", "MND", ""):
                    continue
                try:
                    value = float(value_str)
                except (ValueError, TypeError):
                    continue

                if module == "A1-A3":
                    if is_fossil:
                        gwp_fossil = value
                    elif is_total:
                        gwp_total = value
                    elif is_biogenic:
                        gwp_biogenic = value

                if is_fossil and module:
                    modules[module] = value

        return EPDDetail(
            name=name,
            uuid=uuid,
            reg_no=reg_no,
            owner=owner,
            declared_unit=declared_unit,
            gwp_fossil_a1a3=gwp_fossil,
            gwp_total_a1a3=gwp_total,
            gwp_biogenic_a1a3=gwp_biogenic,
            modules=modules,
            geo=data.get("processInformation", {}).get("geography", {})
                .get("locationOfOperationSupplyOrProduction", {})
                .get("location", ""),
        )

    def _extract_declared_unit(self, data: dict) -> str:
        """Extract declared unit from the reference flow."""
        pi = data.get("processInformation", {})
        quant = pi.get("quantitativeReference", {})
        ref_indices = quant.get("referenceToReferenceFlow", [0])
        ref_idx = ref_indices[0] if ref_indices else 0

        exchanges = data.get("exchanges", {}).get("exchange", [])
        if ref_idx < len(exchanges):
            ref_ex = exchanges[ref_idx]
            flow_ref = ref_ex.get("referenceToFlowDataSet", {})
            flow_desc = flow_ref.get("shortDescription", [{}])
            if flow_desc:
                desc_text = flow_desc[0].get("value", "")
                return _parse_unit_from_description(desc_text)

        return "kg"


# --- Search scoring ---

# Swedish → English keyword mapping for component hints
_HINT_KEYWORDS: dict[str, set[str]] = {
    "golv": {"floor", "flooring", "vinyl", "linoleum", "parquet", "laminate", "tile",
             "carpet", "epoxy", "terrazzo", "rubber", "bamboo", "cork"},
    "innervägg": {"wall", "plasterboard", "gypsum", "drywall", "partition",
                  "board", "panel", "fibre board", "acoustic"},
    "yttervägg": {"facade", "brick", "render", "cladding", "exterior wall",
                  "curtain wall", "sandwich panel", "fibre cement"},
    "betongvägg": {"concrete", "betong", "precast", "reinforced"},
    "fönster": {"window", "glass", "glazing", "triple", "double"},
    "tak": {"roof", "tile", "roofing", "membrane", "bitumen", "shingle",
            "sedum", "green roof", "slate"},
    "isolering": {"insulation", "wool", "mineral", "cellulose", "eps", "xps",
                  "polyurethane", "pir", "glass wool", "stone wool", "hemp"},
    "dörr": {"door", "interior door", "wooden door", "fire door", "steel door"},
    "hiss": {"elevator", "lift", "escalator"},
    "belysning": {"luminaire", "lighting", "lamp", "led", "downlight", "spotlight"},
    "ventilation": {"ventilation", "duct", "air handling", "damper", "grille",
                    "diffuser", "fan", "ahu"},
    "storköksutrustning": {"dishwasher", "commercial kitchen", "storkök",
                           "industrial kitchen", "catering", "food service",
                           "warewash", "combi oven", "blast chiller"},
    "kylanläggning": {"refriger", "cooling", "chiller", "heat pump",
                      "air condition", "fan coil", "hvac", "compressor",
                      "condenser", "coolant"},
    "sanitet": {"toilet", "washbasin", "sanitary", "urinal", "faucet",
                "mixer", "sink", "shower", "bathtub", "cistern", "bidet",
                "wc", "lavatory", "tap"},
    "vitvaror": {"cooker hood", "washing machine", "tumble dryer",
                 "refrigerator", "fridge", "freezer", "oven", "stove",
                 "hob", "microwave", "hand dryer", "towel dryer",
                 "household appliance", "domestic appliance"},
}


def _get_hint_keywords(component_hint: str) -> set[str]:
    """Get English search keywords for a Swedish component hint."""
    hint_lower = component_hint.lower().strip()

    # Direct match
    if hint_lower in _HINT_KEYWORDS:
        return _HINT_KEYWORDS[hint_lower]

    # Try partial match (e.g. "golv" in "golvbeläggning")
    for key, keywords in _HINT_KEYWORDS.items():
        if key in hint_lower or hint_lower in key:
            return keywords

    return set()


# Terms that indicate an EPD is NOT a building material in the expected category.
# Used to reject e.g. furniture with linoleum surfaces when searching for floor coverings.
_NEGATIVE_TERMS: dict[str, set[str]] = {
    "golv": {"table", "desk", "desktop", "chair", "stool", "bench", "shelf",
             "cabinet", "wardrobe", "sofa", "bed", "furniture", "möbler",
             "powder coating", "coating", "covered board"},
    "innervägg": {"table", "desk", "furniture", "möbler"},
    "tak": {"table", "desk", "furniture", "möbler"},
}


def _score_match(
    epd: EPDSummary,
    query_lower: str,
    query_tokens: set[str],
    hint_keywords: set[str],
    component_hint: str = "",
) -> float:
    """Score an EPD match. Returns 0 for no match."""
    name_lower = epd.name.lower().strip()
    owner_lower = epd.owner.lower().strip()

    # Reject items containing negative terms for this category
    if component_hint:
        negatives = _NEGATIVE_TERMS.get(component_hint.lower(), set())
        if negatives and any(neg in name_lower for neg in negatives):
            return 0.0

    # --- Match detection ---
    name_match = False
    owner_match = False
    score = 0.0

    # Exact name match
    if query_lower == name_lower:
        score = 100.0
        name_match = True
    # Name starts with query
    elif name_lower.startswith(query_lower):
        score = 50.0
        name_match = True
    # All query tokens found in name
    elif query_tokens and all(t in name_lower for t in query_tokens):
        score = 45.0
        name_match = True
    # Query substring in name
    elif query_lower in name_lower:
        score = 40.0
        name_match = True
    # Match in owner name only
    elif query_lower in owner_lower or all(t in owner_lower for t in query_tokens):
        score = 20.0
        owner_match = True

    if not name_match and not owner_match:
        return 0.0

    # --- Component hint bonus ---
    if hint_keywords:
        name_words = set(name_lower.split())
        classification_lower = epd.classification.lower()

        # Product name contains hint-related keywords
        keyword_hits = hint_keywords & name_words
        if keyword_hits:
            score += 15.0 * len(keyword_hits)
        # Looser check: substring match for multi-word keywords
        elif any(kw in name_lower for kw in hint_keywords):
            score += 10.0
        # Classification match
        elif any(kw in classification_lower for kw in hint_keywords):
            score += 5.0
        # Owner-only match with no hint alignment: penalize
        elif owner_match and not name_match:
            score -= 10.0

    # --- Geographic preference (strong bias toward Swedish/Nordic) ---
    geo_scores = {"SE": 20, "NORD": 15, "DK": 12, "NO": 12, "FI": 12, "RER": 5, "GLO": 1}
    score += geo_scores.get(epd.geo, 0)

    # --- Specificity bonus (shorter names tend to be more relevant) ---
    if len(name_lower) < 40:
        score += 3.0
    elif len(name_lower) > 80:
        score -= 2.0

    return score


def _parse_unit_from_description(desc: str) -> str:
    """Parse unit from flow description text.

    Examples:
        "1 m2 of vinyl flooring" → "m2"
        "1 cubic meter(m³) of solid surface" → "m3"
        "1 kg of insulation" → "kg"
        "1 piece (pcs) of door" → "st"
    """
    desc_lower = desc.lower()

    if "m2" in desc_lower or "m²" in desc_lower or "square met" in desc_lower:
        return "m2"
    if "m3" in desc_lower or "m³" in desc_lower or "cubic met" in desc_lower:
        return "m3"
    if "pcs" in desc_lower or "piece" in desc_lower or "unit" in desc_lower:
        return "st"
    if "lm" in desc_lower or "linear met" in desc_lower or "running met" in desc_lower:
        return "lm"
    if "tonne" in desc_lower or "1000 kg" in desc_lower:
        return "ton"
    if "kg" in desc_lower:
        return "kg"
    if "kwh" in desc_lower:
        return "kWh"

    return "kg"  # default
