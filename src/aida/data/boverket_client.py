"""Client for Boverket's Klimatdatabas API.

API docs: https://api-portal.boverket.se (OpenAPI spec downloaded as klimatdatabas.json)
Base URL: https://api.boverket.se/klimatdatabas
No API key required — open data.
"""

from __future__ import annotations

import logging
import os
import time

import requests

from aida.data.climate_cache import TTL_BOVERKET, CacheEntry

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://api.boverket.se/klimatdatabas"
REQUEST_TIMEOUT = 15  # seconds


class BoverketClient:
    def __init__(self, base_url: str | None = None):
        self.base_url = (
            base_url
            or os.environ.get("BOVERKET_API_URL")
            or DEFAULT_API_URL
        )

    def get_all_resources(self, version: str = "senaste") -> list[dict]:
        url = f"{self.base_url}/api/Klimat/v2/GetAllResources/{version}/sv/json"
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return data.get("Resources", [])

    def resources_to_cache_entries(self, resources: list[dict]) -> list[CacheEntry]:
        now = time.time()
        entries: list[CacheEntry] = []
        seen_keys: set[str] = set()

        for r in resources:
            name = r.get("Name", "")
            if not name:
                continue

            co2e = _extract_co2e(r)
            if co2e is None:
                continue

            unit = r.get("InventoryUnit", "kg")
            category = _extract_category(r)
            bk04_code, bk04_text = _extract_bk04(r)
            extra = _build_extra(r, category, bk04_code, bk04_text)
            source = f"Boverkets klimatdatabas v{r.get('Version', '?')}"

            def _add(product_name: str, *, _n=name, _co2e=co2e, _u=unit,
                     _src=source, _extra=extra) -> None:
                pn = product_name.lower().strip()
                if not pn or len(pn) < 2 or pn in seen_keys:
                    return
                seen_keys.add(pn)
                entries.append(CacheEntry(
                    product_name=pn, name=_n, co2e_per_unit=_co2e,
                    cost_per_unit=0.0, unit=_u, source=_src,
                    source_layer="boverket",
                    fetched_at=now, expires_at=now + TTL_BOVERKET,
                    extra_json=_extra,
                ))

            # Primary: full product name
            _add(name)

            # Synonyms
            synonyms = r.get("Synonyms", "")
            if synonyms:
                for syn in synonyms.split(","):
                    _add(syn)

            # BK04 text (e.g., "Mineralull") as additional search term
            if bk04_text:
                _add(bk04_text)

            # Split compound name parts (e.g., "Glasull, fasadskivor" → each part)
            for part in _split_name_parts(name):
                _add(part)

        return entries


def _extract_co2e(resource: dict) -> float | None:
    """Extract A1-A3 Typical GWP value from a Boverket resource.

    Per NollCO2 Manual 1.2, sektion 5.2/tabell 3: Boverkets klimatdatabas
    ska använda 'typiskt värde' (A1-A3 Typical), inte Conservative (+25%).
    """
    for item in resource.get("DataItems", []):
        if "Global Warming" not in item.get("PropertyName", ""):
            continue
        for dv in item.get("DataValueItems", []):
            if dv.get("DataModuleCode") == "A1-A3 Typical":
                return dv.get("Value")
    return None


def _extract_category(resource: dict) -> str:
    """Extract Boverket category name."""
    for cat in resource.get("Categories", []):
        if cat.get("ClassificationType") == "Boverket":
            return cat.get("Text", "")
    return ""


def _extract_bk04(resource: dict) -> tuple[str, str]:
    """Extract BK04 classification code and text."""
    for cat in resource.get("Categories", []):
        if cat.get("ClassificationType") == "BK04":
            return cat.get("Code", ""), cat.get("Text", "")
    return "", ""


_STOP_WORDS = {"och", "med", "för", "typ", "av", "i", "på", "till", "utan", "eller"}


def _split_name_parts(name: str) -> list[str]:
    """Split compound product name into searchable parts.

    E.g. "Glasull, fasadskivor" → ["glasull", "fasadskivor"]
    """
    import re
    parts = []
    for chunk in re.split(r'[,/()]+', name):
        chunk = chunk.strip().lower()
        if len(chunk) >= 4 and chunk not in _STOP_WORDS:
            parts.append(chunk)
    return parts


def _build_extra(
    resource: dict, category: str,
    bk04_code: str = "", bk04_text: str = "",
) -> str:
    """Build extra JSON with useful metadata."""
    import json
    extra = {
        "resource_id": resource.get("ResourceId"),
        "category": category,
        "bk04_code": bk04_code,
        "bk04_text": bk04_text,
        "waste_factor": resource.get("WasteFactor"),
        "conservative_factor": resource.get("ConservativeDataConversionFactor"),
        "inventory_unit": resource.get("InventoryUnit"),
    }
    # Boverket reports conversions in different units depending on the product:
    # kg/m3 (true density) for volumetric materials, but kg/m2 (areal weight)
    # for sheet/membrane products (tatskikt, fonster, takpapp) and MJ/liter for
    # fuels. Storing whatever happens to be first as "density" silently corrupts
    # the area-based CO2e math (a 5.5 kg/m2 membrane treated as 5.5 kg/m3 then
    # multiplied by a thickness gives a result ~40x too low). Bucket each
    # conversion by its actual unit instead.
    for conv in resource.get("Conversions", []):
        unit = (conv.get("Unit") or "").replace(" ", "").lower()
        value = conv.get("Value")
        if value is None:
            continue
        if unit in ("kg/m³", "kg/m3") and "density_kg_m3" not in extra:
            extra["density_kg_m3"] = value
        elif unit in ("kg/m²", "kg/m2") and "areal_density_kg_m2" not in extra:
            extra["areal_density_kg_m2"] = value
    return json.dumps(extra, ensure_ascii=False)
