"""Client for Environdec's EPD database (EPD International System).

Uses the soda4LCA data hub at data.environdec.com — no API key required.
Fetches EPD metadata index + individual EPD details with GWP values.

Data stock: Environdata (Digital EPD) — ~14,000 EPDs, mostly construction products.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import requests

from aida.data.climate_cache import TTL_BOVERKET, CacheEntry

logger = logging.getLogger(__name__)

DATA_HUB_URL = "https://data.environdec.com/resource"
ENVIRONDATA_STOCK = "a6c533b3-502e-47b9-885d-31304bf15c64"
REQUEST_TIMEOUT = 30
INDEX_PAGE_SIZE = 1000
INDEX_PATH = Path(__file__).parent / "environdec_index.json"

# Physical plausibility ceiling for a per-kg cradle-to-gate GWP of a building
# material. Even primary aluminium sits near ~18 kg CO2e/kg; nothing legitimate
# in this domain approaches 40. A per-"kg" value above this means the declared
# unit was misdetected (e.g. a per-tonne or per-m2 figure read as per-kg), so we
# skip the EPD rather than cache a number that is wrong by orders of magnitude.
MAX_PLAUSIBLE_KG_CO2E = 40.0

# GWP indicator UUIDs (EN 15804+A2)
GWP_FOSSIL_NAMES = {"gwp-fossil", "global warming potential - fossil fuels"}
GWP_TOTAL_NAMES = {"gwp-total", "global warming potential - total"}
GWP_LULUC_NAMES = {"gwp-luluc", "land use and land use change"}
# Deliberately the parenthesised form: a bare "gwp-ghg" would also be a
# substring of nothing else, but the datahub writes "Global Warming Potential
# (GWP-GHG)" and matching the exact token keeps it away from the other four.
GWP_GHG_NAMES = {"(gwp-ghg)"}
# Pre-A2 declarations carry ONE indicator with no fossil/biogenic split. EPD
# Norge has 2 066 of them (measured 2026-09-14; the other 10 513 use the four
# names above). It is a total, not a fossil figure: Kebony's roofing declares
# -738 kg CO2e/m3 under it, which is biogenic uptake. Recorded as GWP-total so
# the builder can see the row and refuse it on the right grounds, rather than
# reading fossil=None and total=None and dropping it without a word. Matched
# on the full lowercased string so it cannot collide with the split names.
GWP_BARE_TOTAL_NAMES = {"global warming potential (gwp)"}

# EN 15804+A2: total = fossil + biogenic + luluc. Real declarations round, so
# accept the larger of 1 kg CO2e and 5% of the total before calling it broken.
GWP_SUM_ABS_TOLERANCE = 1.0
GWP_SUM_REL_TOLERANCE = 0.05


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


def gwp_components_consistent(
    fossil: float | None,
    biogenic: float | None,
    luluc: float | None,
    total: float | None,
) -> bool | None:
    """Does this declaration add up the way EN 15804+A2 says it must?

    Returns True/False, or None when the declaration does not carry enough
    indicators to judge (most EPDs omit luluc, and absence is not evidence of
    a defect).

    Why this exists. Sveden Trä's spruce cladding declares GWP-fossil -744 with
    GWP-total +74.9 and GWP-biogenic +3.97: the fossil row is carrying biogenic
    uptake. Taken at face value it would have put a facade with heavily negative
    emissions in front of a building manager writing a procurement document.
    Millworks' larch cladding, by contrast, declares fossil 203.0, biogenic
    -922.6, luluc 0.6, total -719.1, which sums exactly. Measured against the
    live datahub 2026-08-17.
    """
    if fossil is None or total is None:
        return None
    if biogenic is None and luluc is None:
        return None
    components = fossil + (biogenic or 0.0) + (luluc or 0.0)
    tolerance = max(GWP_SUM_ABS_TOLERANCE, abs(total) * GWP_SUM_REL_TOLERANCE)
    return abs(components - total) <= tolerance


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
    # Mass (kg) of one declared reference flow, set ONLY when the declared unit
    # is itself a mass (reference flow property == Mass). GWP is per reference
    # flow, so per-kg = gwp / reference_mass_kg. None for Area/Volume/piece-
    # declared EPDs, which keep their native functional unit instead.
    reference_mass_kg: float | None = None
    # Needed to tell a trustworthy declaration from a broken one. EN 15804+A2
    # has GWP-total = GWP-fossil + GWP-biogenic + GWP-luluc, so when those do
    # not add up the publisher has filed something in the wrong row and the
    # fossil value cannot be taken at face value. GWP-GHG is total excluding
    # biogenic, the nearest indicator to Boverket's basis, kept so a fallback
    # stays available.
    gwp_luluc_a1a3: float | None = None
    gwp_ghg_a1a3: float | None = None
    # Free text describing the reference flow ("250mm of EPS insulation board",
    # "1 m2 of EPS 80 with 31 mm thickness, R-value of 1 K*m2/W"). It is the
    # only place an area-declared insulation says WHAT AREA OF WHAT -- the
    # product name never carries a thickness (0 of 137 rows) and neither does
    # any structured field. Without it an m2 insulation figure cannot be
    # interpreted at all, let alone compared: 0.46 kg CO2e/m2 is meaningless
    # until you know whether that is 45 mm or 250 mm.
    flow_description: str = ""


class EnvirondecClient:
    # What a subclass changes to point the same protocol at another soda4LCA
    # hub (see epd_norge_client). `registry` is the label the catalog carries
    # in `source_registry`; the rest is where the index lives and how fast the
    # hub may be asked.
    registry = "environdec"
    datastock = ENVIRONDATA_STOCK
    index_path = INDEX_PATH
    index_page_sleep = 0.5

    def __init__(self, base_url: str = DATA_HUB_URL, request_interval: float = 0.0):
        self.base_url = base_url
        self.request_interval = request_interval
        # None, not 0.0: time.monotonic() has no defined zero, so a 0.0
        # sentinel would read as "just requested" on a fresh process.
        self._last_request_at: float | None = None
        self._index: list[EPDSummary] | None = None

    def _throttle(self) -> None:
        """Space requests `request_interval` seconds apart. No-op at 0."""
        if self.request_interval <= 0:
            return
        now = time.monotonic()
        if self._last_request_at is not None:
            wait = self.request_interval - (now - self._last_request_at)
            if wait > 0:
                time.sleep(wait)
        self._last_request_at = time.monotonic()

    def fetch_index(self, use_cached: bool = True) -> list[EPDSummary]:
        """Fetch the full EPD index. Uses local JSON cache if available."""
        if self._index is not None:
            return self._index

        # Try local cache first
        if use_cached and self.index_path.exists():
            age_days = (time.time() - self.index_path.stat().st_mtime) / 86400
            if age_days < 30:
                self._index = self._load_index_file()
                if self._index:
                    logger.info("Loaded %d EPDs from cached index (%.0f days old)",
                                len(self._index), age_days)
                    return self._index

        # Fetch from API
        fetched = self._fetch_index_from_api()
        self._index = self._reconcile_with_cached(fetched)

        # Only persist a complete index. A truncated fetch (mid-pagination
        # network error) would otherwise be cached for 30 days and silently
        # degrade every EPD search until expiry.
        if getattr(self, "_last_fetch_complete", False):
            self._save_index_file(self._index)
        else:
            logger.warning(
                "Environdec index incomplete (%d EPDs) — not caching", len(self._index)
            )

        return self._index

    def _reconcile_with_cached(self, fetched: list[EPDSummary]) -> list[EPDSummary]:
        """Merge a fresh fetch with the cached index instead of replacing it.

        A refresh must never be able to lose data on its own. Two ways it did
        (2026-08-14, measured against the committed index):

        1. The list endpoint stopped returning `owner` entirely, so
           `item.get("owner", "")` turned all 17567 rows into empty strings
           with no error anywhere. The value still exists upstream, but only in
           the per-EPD detail response, which is one request per EPD.
        2. Offset pagination is not stable while the collection changes under
           it. The refresh gained 2838 EPDs and dropped 1153, of which 1122
           were still valid and still resolve individually in the API. They
           were skipped pages, not withdrawals.

        So: carry an owner forward whenever the fetch has none, and keep any
        cached EPD the fetch did not return. A withdrawn EPD lingering in the
        index is a much smaller harm than a valid one silently disappearing,
        and `valid_until` already makes stale rows detectable.
        """
        if not fetched:
            return fetched

        cached = self._load_index_file()
        if not cached:
            return fetched

        by_uuid = {e.uuid: e for e in cached if e.uuid}
        restored_owners = 0
        merged = []
        for epd in fetched:
            prev = by_uuid.get(epd.uuid)
            if prev and not epd.owner.strip() and prev.owner.strip():
                epd = replace(epd, owner=prev.owner)
                restored_owners += 1
            merged.append(epd)

        seen = {e.uuid for e in fetched if e.uuid}
        unseen = [e for e in cached if e.uuid and e.uuid not in seen]
        merged.extend(unseen)

        if restored_owners:
            logger.warning(
                "Environdec list endpoint returned no owner for %d EPDs — "
                "carried the cached value forward. New EPDs keep an empty "
                "owner until someone backfills from the detail endpoint.",
                restored_owners,
            )
        if unseen:
            logger.warning(
                "Environdec refresh did not return %d cached EPDs (%d still "
                "valid) — kept them rather than deleting. Suspect unstable "
                "offset pagination, not withdrawal.",
                len(unseen),
                sum(1 for e in unseen if e.valid_until >= date.today().year),
            )

        return merged

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
            self._throttle()
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
        unit = detail.declared_unit

        # Normalize to per-kg when we know the reference flow's mass. The GWP is
        # reported per declared reference flow (1 section, 1 m2, 1 tonne...), and
        # reference_mass_kg is how many kg that flow weighs, so gwp / mass gives
        # the true per-kg figure that downstream unit conversion expects.
        if detail.reference_mass_kg and detail.reference_mass_kg > 0:
            gwp = gwp / detail.reference_mass_kg
            unit = "kg"

        # Plausibility backstop: a per-"kg" value above the physical ceiling means
        # the unit was still misdetected (no mass property to normalize against).
        # Skip rather than cache a value wrong by orders of magnitude — Boverket
        # and the baseline median cover the component far more reliably.
        if unit == "kg" and gwp > MAX_PLAUSIBLE_KG_CO2E:
            logger.warning(
                "Skipping EPD %s: implausible %.1f kg CO2e/kg (declared unit likely "
                "misdetected, no mass property to normalize).",
                detail.reg_no or detail.uuid[:8], gwp,
            )
            return None

        extra = {
            "uuid": detail.uuid,
            "reg_no": detail.reg_no,
            "owner": detail.owner,
            "declared_unit": unit,
            "reference_mass_kg": detail.reference_mass_kg,
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
            unit=unit,
            source=f"Environdec EPD {detail.reg_no}" if detail.reg_no else f"Environdec EPD {detail.uuid[:8]}",
            source_layer="environdec",
            fetched_at=now,
            expires_at=now + TTL_BOVERKET,
            extra_json=json.dumps(extra, ensure_ascii=False),
        )

    # --- Internal ---

    def _fetch_index_from_api(self) -> list[EPDSummary]:
        """Fetch all EPDs from the soda4LCA data hub.

        Sets self._last_fetch_complete = True only if pagination reached the
        reported total. On a mid-pagination network error it returns whatever
        was fetched so far with the flag left False, so the caller can avoid
        persisting a truncated index as a 30-day cache.
        """
        all_epds: list[EPDSummary] = []
        start = 0
        total = 0
        self._last_fetch_complete = False

        while True:
            url = (f"{self.base_url}/datastocks/{self.datastock}"
                   f"/processes?format=json&pageSize={INDEX_PAGE_SIZE}&startIndex={start}")
            try:
                self._throttle()
                resp = requests.get(url, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
            except requests.RequestException as e:
                logger.warning("Environdec index fetch failed at offset %d: %s", start, e)
                return all_epds

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
                self._last_fetch_complete = True
                break
            start += INDEX_PAGE_SIZE
            time.sleep(self.index_page_sleep)

        return all_epds

    def _load_index_file(self) -> list[EPDSummary]:
        """Load index from local JSON cache."""
        try:
            with open(self.index_path) as f:
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
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.index_path, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info("Saved %s index: %d entries, %.1f MB", self.registry,
                        len(data), self.index_path.stat().st_size / 1024 / 1024)
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

        # Declared unit + reference-flow mass from the reference exchange's
        # flow properties (authoritative). Falls back to text parsing when the
        # exchange carries no usable flow properties.
        declared_unit, reference_mass_kg = self._extract_reference_flow(data)

        # GWP values
        gwp_fossil = None
        gwp_total = None
        gwp_biogenic = None
        gwp_luluc = None
        gwp_ghg = None
        bare_total = None
        modules: dict[str, float] = {}

        for result in data.get("LCIAResults", {}).get("LCIAResult", []):
            ref = result.get("referenceToLCIAMethodDataSet", {})
            indicator_desc = ref.get("shortDescription", [{}])
            indicator_name = indicator_desc[0].get("value", "").lower() if indicator_desc else ""

            is_fossil = any(n in indicator_name for n in GWP_FOSSIL_NAMES)
            is_total = any(n in indicator_name for n in GWP_TOTAL_NAMES)
            is_biogenic = "biogenic" in indicator_name
            is_luluc = any(n in indicator_name for n in GWP_LULUC_NAMES)
            is_ghg = any(n in indicator_name for n in GWP_GHG_NAMES)
            is_bare = indicator_name.strip() in GWP_BARE_TOTAL_NAMES

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
                    elif is_luluc:
                        gwp_luluc = value
                    elif is_ghg:
                        gwp_ghg = value
                    elif is_bare:
                        bare_total = value

                if is_fossil and module:
                    modules[module] = value

        # A declaration that carries both a split GWP-total and the bare form
        # keeps the split one; the bare value only fills an otherwise empty
        # total, which is the only case it exists for.
        if gwp_total is None and bare_total is not None:
            gwp_total = bare_total

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
            reference_mass_kg=reference_mass_kg,
            gwp_luluc_a1a3=gwp_luluc,
            gwp_ghg_a1a3=gwp_ghg,
            flow_description=self._reference_flow_description(data),
        )

    def _extract_reference_flow(self, data: dict) -> tuple[str, float | None]:
        """Return (declared_unit, reference_mass_kg) for the reference flow.

        The declared unit and the reference flow's mass live in the reference
        exchange's ``flowProperties`` — the ``referenceFlowProperty`` entry is
        authoritative (Mass → kg, Area → m2, Volume → m3, pieces → st). The
        reference flow text description usually carries no unit, so the old text
        parse silently defaulted every such EPD to "kg", which made per-tonne
        figures (e.g. a 1490 kg CO2e / 1000 kg steel section) read as 1490 kg
        CO2e *per kg* — a 1000x error.

        ``reference_mass_kg`` is set ONLY when the declared unit is itself a
        mass — i.e. the reference flow property is Mass. Then ``gwp / mass`` is
        a pure per-kg figure and downstream applies no further conversion.
        For Area/Volume/piece-declared EPDs we deliberately keep the native
        functional unit (m2/m3/st) rather than divide by a secondary mass: the
        downstream unit conversion already turns per-kg into per-m2/m3 using a
        component density, so normalizing here via a different mass would make
        the value round-trip through two densities and drift.
        """
        exchanges = data.get("exchanges", {}).get("exchange", [])
        ref_ex = self._reference_exchange(data, exchanges)
        if ref_ex is None:
            return "kg", None

        declared_unit: str | None = None
        reference_mass_kg: float | None = None

        for prop in ref_ex.get("flowProperties", []):
            if not prop.get("referenceFlowProperty"):
                continue
            prop_name = " ".join(
                n.get("value", "") for n in prop.get("name", []) if isinstance(n, dict)
            )
            ref_unit = prop.get("referenceUnit") or ""
            declared_unit = _flow_property_unit(prop_name, ref_unit)
            mean = prop.get("meanValue")
            if "mass" in prop_name.lower() and not _is_per_product(prop_name) and mean:
                reference_mass_kg = _mass_to_kg(mean, ref_unit)
            break

        if declared_unit is None:
            # No usable flow property — fall back to parsing the text description.
            flow_ref = ref_ex.get("referenceToFlowDataSet", {})
            flow_desc = flow_ref.get("shortDescription", [{}])
            desc_text = flow_desc[0].get("value", "") if flow_desc else ""
            declared_unit = _parse_unit_from_description(desc_text)

        return declared_unit, reference_mass_kg

    def _reference_flow_description(self, data: dict) -> str:
        """Free text of the reference flow, or "" when absent.

        Kept separate from _extract_reference_flow rather than folded into its
        return tuple: that function is about the unit basis and is read by the
        per-kg normalisation, and widening its contract to carry an unrelated
        string would put a second concern inside a function whose correctness
        already had to be argued once (the 1000x per-tonne bug).
        """
        exchanges = data.get("exchanges", {}).get("exchange", [])
        ref_ex = self._reference_exchange(data, exchanges)
        if not ref_ex:
            return ""
        desc = ref_ex.get("referenceToFlowDataSet", {}).get("shortDescription", [])
        for entry in desc:
            if isinstance(entry, dict) and entry.get("value"):
                return str(entry["value"])
        return ""

    @staticmethod
    def _reference_exchange(data: dict, exchanges: list) -> dict | None:
        """Locate the reference exchange. ILCD marks it with ``referenceFlow:
        true``; we prefer that, then match ``dataSetInternalID`` against the
        ``referenceToReferenceFlow`` pointer, and only fall back to positional
        indexing (which assumes declaration order) as a last resort."""
        if not exchanges:
            return None
        for ex in exchanges:
            if ex.get("referenceFlow"):
                return ex
        quant = data.get("processInformation", {}).get("quantitativeReference", {})
        ref_indices = quant.get("referenceToReferenceFlow", [0])
        ref_idx = ref_indices[0] if ref_indices else 0
        for ex in exchanges:
            if ex.get("dataSetInternalID") == ref_idx:
                return ex
        return exchanges[ref_idx] if ref_idx < len(exchanges) else None


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
    "stomme": {"beam", "column", "girder", "structural steel", "steel section",
               "glulam", "laminated timber", "clt", "cross-laminated",
               "hollow core", "precast", "slab", "load-bearing", "structural"},
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
    "fast_inredning": {"kitchen cabinet", "kitchen front", "kitchen door",
                       "cabinet door", "bathroom cabinet", "mirror cabinet",
                       "vanity", "worktop", "countertop", "kitchen sink"},
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
    # Appliances/kitchen: reject component parts (a valve/hose/coupling is not a
    # machine). Guards e.g. "Dishwasher switch valve" or "connection hose for
    # dishwasher" from matching a search for tvättmaskin/diskmaskin.
    "vitvaror": {"valve", "ventil", "hose", "slang", "coupling", "koppling",
                 "connection", "fitting", "switch valve", "spare part"},
    "storköksutrustning": {"valve", "ventil", "hose", "slang", "coupling",
                           "koppling", "connection", "fitting", "spare part"},
    # Fixed interior is cabinets and worktops, which is exactly the vocabulary
    # of office furniture. Loose interior is outside the baseline for now. Only
    # words a fixed-interior name does not also carry ("shelf" and "storage"
    # do: a mirror cabinet with shelf is still a mirror cabinet).
    "fast_inredning": {"locker", "wardrobe", "desk", "chair", "office", "filing",
                       "möbler"},
}


# Query terms that must match as whole words. Everything else keeps the plain
# substring test, and the asymmetry is the whole point of this constant.
#
# On 2026-09-06 four contaminations were traced to bare substring matching in one
# evening -- "duct" inside "Product" put 23 rows of gravel, concrete piles and
# joinery into `ventilation`; "fan" inside "Fangyuan" and "ALFANAR" added more;
# "ahu" inside "Mahurangi" fetched eight New Zealand ready-mix concretes; "lack"
# inside "Black" offered roofing felt as paint. The obvious repair is to bound
# every term. It was written, and then measured against the real catalog before
# being kept, which is what stopped it: bounding every term makes 115 shipped
# rows unreachable by their own category's queries, and only about 40 of those
# are contamination.
#
# The reason no general rule works is visible once the losses are read one by
# one. Every collision above is a term sitting inside an unrelated longer word --
# but so is every legitimate compound the loose rule depends on:
#
#     wrong                        right
#     pro-DUCT     (not a duct)    Star-FLOOR, Mape-FLOOR  (are floors)
#     B-LACK       (not lack)      Polar-PIPE, Pipe-life   (are pipes)
#     as-PIR-e     (not PIR)       i-CABLE                 (is cable)
#     c-LAMP-s     (not a lamp)    sludge-PAINT            (is paint)
#     Ma-HAU-rangi (not an AHU)    Scot-LARCH              (is larch)
#     water-p-ROOFING              betong-produkter        (is concrete)
#
# Position does not separate them: both columns are suffix matches. Length does
# not either: "pipe" and "duct" are both four letters and land on opposite
# sides. The difference is semantic -- in the right-hand column the compound is a
# kind of the thing named, in the left-hand column the spelling is a coincidence
# -- and no rule available at this layer can see that. So the set is explicit,
# and each entry earns its place by an observed collision rather than by a
# property of the string.
#
# Read as a maintenance instruction: adding a term here is cheap and safe.
# Removing one, or bounding a term not listed, needs the same measurement --
# `scripts/audit_queue_fronts.py` will show what reached the front of a queue,
# but only a rebuild shows what stopped reaching the catalog at all.
# Measured on the shipped catalog with exactly this set: 1296 rows stay
# reachable, 37 stop being reachable, and reading all 37 by hand they are gravel,
# concrete piles, timber connectors, wall ties, spiral staircases, mooring wire,
# shower enclosures, two vinyl floors and a box of brass clamps. None is a
# building product for the category that was fetching it.
#
# The one near-miss is worth recording. "pir" also stops reaching "Glava Proff 34
# Plate m/papir", a real Norwegian mineral wool batt whose paper facing spells
# the term by accident. It survives anyway, because Glava AS is a recognised
# Nordic owner and the supplement fetches that route by supplier rather than by
# product word -- the two apertures cover each other, which is the argument for
# having both.
WORD_BOUNDED_TERMS: frozenset[str] = frozenset({
    "duct", "fan", "pir", "lamp", "laminate", "roofing", "ahu", "lack",
    # A negative term rather than a query: `golv` rejects on "bed", which sits
    # inside "bedroom" and "embedded". A negative that misfires is the invisible
    # half of this defect class -- a contaminated row announces itself at the
    # front of a queue, a wrongly excluded one announces nothing ever.
    "bed",
})


def _word_match(term: str, text: str) -> bool:
    """Does `term` occur in `text` as a whole word, allowing a plural or gerund?

    Both ends are bounded, unlike the sibling rule in
    `scripts/fetch_nordic_supplement.py`, which leaves the end open. There the
    terms are deliberate stems and "floor" must reach "flooring"; here an open
    end is exactly what lets "duct" reach "Product". The (s|es|ing) allowance
    covers the inflections that actually appear in EPD names -- "ducts",
    "ducting", "fans", "lamps".
    """
    return re.search(r"\b" + re.escape(term) + r"(s|es|ing)?\b", text) is not None


def _term_matches(term: str, text: str) -> bool:
    """Substring by default, whole word for the terms that have earned it."""
    if term in WORD_BOUNDED_TERMS:
        return _word_match(term, text)
    return term in text


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

    # Reject items containing negative terms for this category. Word-bounded for
    # the same reason as the match tests below, and here the cost of not doing it
    # falls the other way: a negative that fires on a substring silently DISCARDS
    # a legitimate product. `golv` rejects "bed", which sits inside "bedroom" and
    # "embedded", so a floor declared as "Bedroom vinyl" or a screed with
    # "embedded heating" would never have reached the catalog to be noticed
    # missing. That is the invisible half of this defect class: the contaminated
    # rows announce themselves at the front of a queue, the excluded ones never
    # announce anything at all.
    if component_hint:
        negatives = _NEGATIVE_TERMS.get(component_hint.lower(), set())
        if negatives and any(_term_matches(neg, name_lower) for neg in negatives):
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
    elif query_tokens and all(_term_matches(t, name_lower) for t in query_tokens):
        score = 45.0
        name_match = True
    # Query substring in name
    elif _term_matches(query_lower, name_lower):
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


def _is_per_product(prop_name: str) -> bool:
    """True for property names that report a per-product figure, not a mass.

    Guards against names like "Biogenic carbon content of the product
    (kg C/product)" being mistaken for the reference flow's mass.
    """
    low = prop_name.lower()
    return "content" in low or "/product" in low or "per product" in low


def _mass_to_kg(value: float, unit: str) -> float | None:
    """Convert a mass value to kg. EPD mass properties are almost always in kg
    already; tonne/gram are handled defensively. Returns None for non-positive."""
    if not value or value <= 0:
        return None
    u = (unit or "kg").strip().lower()
    if u in ("kg", "kilogram", "kilograms", ""):
        return value
    if u in ("t", "ton", "tonne", "tonnes", "metric ton"):
        return value * 1000.0
    if u in ("g", "gram", "grams"):
        return value / 1000.0
    if u in ("mg",):
        return value / 1_000_000.0
    # Unknown unit: assume kg but surface it — a silent wrong assumption here
    # (lb, short ton...) would re-introduce the very mis-scaling this fixes.
    logger.warning("Unrecognized mass unit %r on reference flow; assuming kg.", unit)
    return value


def _flow_property_unit(prop_name: str, ref_unit: str) -> str:
    """Map a reference flow property to Aida's declared-unit vocabulary."""
    low = prop_name.lower()
    if "mass" in low:
        return "kg"
    if "area" in low:
        return "m2"
    if "volume" in low:
        return "m3"
    if "piece" in low or "number of" in low:
        return "st"
    if "length" in low:
        return "lm"
    # Strip trailing punctuation/whitespace ("pcs." -> "pcs", "m² " -> "m²").
    u = (ref_unit or "").strip().strip(".").strip().lower()
    if u in ("m2", "m²"):
        return "m2"
    if u in ("m3", "m³"):
        return "m3"
    if u in ("m", "lm", "lfm"):
        return "lm"
    if u in ("t", "tonne", "ton", "g", "kg", "kilogram"):
        return "kg"
    if u in ("kwh",):
        return "kWh"
    if u in ("mj",):
        return "MJ"
    if u in ("piece", "pieces", "pcs", "item", "items", "st"):
        return "st"
    return "kg"


def _parse_unit_from_description(desc: str) -> str:
    """Parse unit from flow description text.

    Examples:
        "1 m2 of vinyl flooring" → "m2"
        "1 cubic meter(m³) of solid surface" → "m3"
        "1 kg of insulation" → "kg"
        "1 piece (pcs) of door" → "st"
    """
    import re
    desc_lower = desc.lower()
    # Strip the "declared/functional/reference unit" label so the literal word
    # "unit" in the label isn't mistaken for a pieces ("st") declared unit.
    # Without this, "Declared unit: 1 kg of adhesive" wrongly parses as "st".
    desc_lower = re.sub(
        r"\b(declared|functional|reference)\s+unit(\s+of\s+measurement)?\b[:\s]*",
        " ", desc_lower,
    )

    if "m2" in desc_lower or "m²" in desc_lower or "square met" in desc_lower:
        return "m2"
    if "m3" in desc_lower or "m³" in desc_lower or "cubic met" in desc_lower:
        return "m3"
    # "lm" must match as a word, not inside "film"/"laminate"; "unit" likewise.
    if "pcs" in desc_lower or "piece" in desc_lower or re.search(r"\bunits?\b", desc_lower):
        return "st"
    if re.search(r"\blm\b", desc_lower) or "linear met" in desc_lower or "running met" in desc_lower:
        return "lm"
    if "tonne" in desc_lower or "1000 kg" in desc_lower:
        return "ton"
    if "kg" in desc_lower:
        return "kg"
    if "kwh" in desc_lower:
        return "kWh"

    return "kg"  # default
