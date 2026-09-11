"""Pricing lookup via LLM web search.

Routes through OpenRouter (same key as rest of Aida).
Returns None silently if key is missing or any error occurs.
"""
from __future__ import annotations

import logging
import os
import re

import anthropic

from aida.api_client import LLM_CALL_TIMEOUT as PLATFORM_CALL_TIMEOUT
from aida.api_client import call_model
from aida.name_match import best_token_match, match_key

logger = logging.getLogger(__name__)

PRICING_MODEL = "anthropic/claude-sonnet-5"
# Pricing stays on Sonnet (cheap, web-search heavy, not the CO2 correctness
# path). Adaptive thinking + effort replaces the deprecated budget_tokens.
# Keep this id listed in api_client._ADAPTIVE_MODELS — otherwise PRICING_EFFORT
# is dropped without error and the call runs with no thinking at all.
PRICING_EFFORT = "medium"
PRICING_MAX_TOKENS = 8000  # room for adaptive thinking + a short price answer
MAX_SEARCH_USES = 3
OPENROUTER_BASE_URL = "https://openrouter.ai/api"


# Non-streaming; web search + adaptive thinking can run long, so allow headroom.
# Derived from the platform ceiling rather than hardcoded: this module used to
# carry a flat 300.0, which equalled Vercel's maxDuration exactly, so the SDK
# timeout could never fire first and a slow price search killed the whole
# function. Same defect api_client fixed for itself in M3 (2026-08-14).
LLM_CALL_TIMEOUT = PLATFORM_CALL_TIMEOUT

# Source labels carried on Alternative.price_basis, so the table and the report
# can say where a number came from instead of rendering three different kinds of
# figure identically.
BASIS_WEB_SEARCH = "market_estimate"   # web-searched typical installed price
BASIS_LLM_ESTIMATE = "llm_estimate"    # model's own estimate, no source found
BASIS_LISTING = "listing"              # a real Palats asking price


def _get_client() -> anthropic.Anthropic | None:
    """Return OpenRouter client for web search, or None if key not configured."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None
    return anthropic.Anthropic(
        api_key=api_key, base_url=OPENROUTER_BASE_URL, timeout=LLM_CALL_TIMEOUT,
    )


def _build_prompt(product_name: str, unit_hint: str) -> str:
    unit_phrase = f"per {unit_hint}" if unit_hint and unit_hint not in ("kg", "") else ""
    return (
        f"Vad kostar '{product_name}' installerat (material + arbete) "
        f"på den svenska byggmarknaden {unit_phrase}? "
        f"Sök efter aktuella priser hos svenska bygghandlare och entreprenörer. "
        f"Ange typiskt installerat pris i SEK exklusive moms. "
        f"Svara med exakt format: PRIS: [tal] SEK/[enhet]. "
        f"Om du hittar ett prisintervall, ange mittpunkten."
    )


def _extract_price(text: str, unit_hint: str) -> tuple[float, str] | None:
    """Extract price from LLM response text."""
    # Try structured format first: PRIS: 250 SEK/m2
    m = re.search(r'PRIS:\s*(\d[\d\s]*(?:[,.]\d+)?)\s*SEK\s*/\s*(\w+[²³]?)', text, re.IGNORECASE)
    if m:
        raw_num, raw_unit = m.group(1), m.group(2)
    else:
        # Fallback: any "N SEK/unit" or "N kr/unit" pattern
        pattern = r'(\d[\d\s]*(?:[,.]\d+)?)\s*(?:SEK|kr|kronor)\s*/\s*(\w+[²³]?)'
        matches = re.findall(pattern, text, re.IGNORECASE)
        if not matches:
            return None
        raw_num, raw_unit = matches[0]

    raw_num = raw_num.replace(" ", "").replace(",", ".")
    try:
        price = float(raw_num)
    except ValueError:
        return None

    unit_map = {"m²": "m2", "m2": "m2", "m³": "m3", "m3": "m3",
                "st": "st", "pcs": "st", "lm": "lm", "kg": "kg"}
    unit = unit_map.get(raw_unit.lower(), unit_hint or raw_unit.lower())

    if price <= 0 or price > 10_000_000:
        return None
    return price, unit


def _estimate_price_without_search(product_name: str, unit_hint: str) -> tuple[float, str, str] | None:
    """LLM estimate without web search — fallback when web search fails."""
    client = _get_client()
    if client is None:
        return None

    unit_phrase = f"per {unit_hint}" if unit_hint and unit_hint not in ("kg", "") else ""
    prompt = (
        f"Uppskatta vad '{product_name}' kostar installerat (material + arbete) "
        f"på den svenska byggmarknaden {unit_phrase}. "
        f"Basera din uppskattning på din kunskap om svenska byggpriser. "
        f"Svara med exakt format: PRIS: [tal] SEK/[enhet]"
    )

    try:
        response = call_model(
            client,
            model=PRICING_MODEL,
            max_tokens=PRICING_MAX_TOKENS,
            effort=PRICING_EFFORT,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        logger.warning("Price estimation failed for '%s': %s", product_name, e)
        return None

    text_parts = [b.text for b in (response.content or []) if hasattr(b, "type") and b.type == "text"]
    full_text = " ".join(text_parts)
    if not full_text:
        return None

    result = _extract_price(full_text, unit_hint)
    if result is None:
        return None

    price, unit = result
    logger.info("Price estimated for '%s': %.0f SEK/%s (no web search)", product_name, price, unit)
    return price, unit, "LLM-uppskattning"


def lookup_price(product_name: str, unit_hint: str = "") -> tuple[float, str, str] | None:
    """Search the web for current Swedish market price of a building material.

    Returns (price_sek, unit, source_description) or None on any failure.
    Falls back to LLM estimate without web search if web search fails.
    Never raises.
    """
    client = _get_client()
    if client is None:
        return None

    prompt = _build_prompt(product_name, unit_hint)

    try:
        response = call_model(
            client,
            model=PRICING_MODEL,
            max_tokens=PRICING_MAX_TOKENS,
            effort=PRICING_EFFORT,
            messages=[{"role": "user", "content": prompt}],
            tools=[{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": MAX_SEARCH_USES,
                "user_location": {
                    "type": "approximate",
                    "country": "SE",
                    "timezone": "Europe/Stockholm",
                },
            }],
        )
    except Exception as e:
        logger.warning("Pricing web search failed for '%s': %s", product_name, e)
        return _estimate_price_without_search(product_name, unit_hint)

    # Extract text and source URL from response
    text_parts = []
    source_url = ""
    for block in (response.content or []):
        if not hasattr(block, "type"):
            continue
        if block.type == "text":
            text_parts.append(block.text)
            if hasattr(block, "citations") and block.citations:
                for cit in block.citations:
                    if hasattr(cit, "url") and cit.url:
                        source_url = cit.url
                        break

    full_text = " ".join(text_parts)
    if not full_text:
        return _estimate_price_without_search(product_name, unit_hint)

    result = _extract_price(full_text, unit_hint)
    if result is None:
        logger.info("Could not extract price for '%s' from web search, trying estimate", product_name)
        return _estimate_price_without_search(product_name, unit_hint)

    price, unit = result
    source = f"Webbsökning ({source_url})" if source_url else "Webbsökning"
    logger.info("Price found for '%s': %.0f SEK/%s", product_name, price, unit)
    return price, unit, source


def lookup_prices_batch(
    products: list[tuple[str, str]],
) -> dict[str, tuple[float, str, str]]:
    """Look up prices for multiple products in a single LLM web search call.

    Args:
        products: list of (product_name, unit_hint) tuples

    Returns:
        dict mapping lowercase product_name -> (price_per_unit, unit, source)
    """
    if not products:
        return {}
    if len(products) == 1:
        name, unit = products[0]
        result = lookup_price(name, unit)
        return {name.lower(): result} if result else {}

    client = _get_client()
    if client is None:
        return {}

    # Format compliance is the weak link, not the searching. Live runs
    # 2026-08-20: the model reliably FINDS the prices and then decides how to
    # present them. Sometimes a fenced summary block at the end, sometimes
    # inline in prose with none of the requested lines at all, and that last
    # shape loses every price in the batch at once. It also spent 4654 output
    # tokens on five products, most of it prose, so at a realistic 12 to 15
    # alternatives the summary would be pushed past max_tokens and truncated
    # away even when it was going to be written.
    #
    # Hence: the lines are the whole answer, not a summary appended to one. And
    # asking it to echo the name unchanged attacks the matching problem at the
    # source rather than only in the parser.
    prompt = (
        f"Sök upp aktuella installerade priser (material + arbete) på den svenska "
        f"byggmarknaden för följande produkter:\n\n"
        f"{_batch_prompt_lines(products)}\n\n"
        f"Sök hos svenska bygghandlare och entreprenörer. "
        f"Ange typiska installerade priser i SEK exklusive moms. "
        f"Hittar du ett prisintervall, ange mittpunkten.\n\n"
        f"SVARSFORMAT. Svara med ENBART en rad per produkt, i den här formen:\n"
        f"PRODUKT: [produktnamn] | PRIS: [tal] SEK/[enhet]\n\n"
        f"Skriv ingen brödtext, inga rubriker, inga källförteckningar och inga "
        f"kommentarer före eller efter raderna. Skriv produktnamnet exakt som "
        f"det står i listan ovan, tecken för tecken, även om du hittade priset "
        f"under ett annat namn. En rad per produkt, alla {len(products)} "
        f"produkterna, i samma ordning som listan."
    )

    try:
        response = call_model(
            client,
            model=PRICING_MODEL,
            max_tokens=PRICING_MAX_TOKENS,
            effort=PRICING_EFFORT,
            messages=[{"role": "user", "content": prompt}],
            tools=[{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": min(MAX_SEARCH_USES + len(products), 8),
                "user_location": {
                    "type": "approximate",
                    "country": "SE",
                    "timezone": "Europe/Stockholm",
                },
            }],
        )
    except Exception as e:
        logger.warning("Batch pricing web search failed: %s", e)
        return {}

    text_parts = []
    source_urls: list[str] = []
    for block in (response.content or []):
        if not hasattr(block, "type"):
            continue
        if block.type == "text":
            text_parts.append(block.text)
            if hasattr(block, "citations") and block.citations:
                for cit in block.citations:
                    url = getattr(cit, "url", "")
                    if url and url not in source_urls:
                        source_urls.append(url)

    full_text = "\n".join(text_parts)
    if not full_text:
        return {}

    # One batch call searches for every product at once, and the citations come
    # back attached to the response as a whole, not to individual price lines.
    # Naming a single URL per product was therefore false provenance: a live
    # check on 2026-08-14 gave four different facade products the same
    # hantverkskollen article as their "source". In a document meant for
    # procurement that is worse than admitting the source is unresolved, so we
    # say how many sources the search used and let the reader go look.
    if len(source_urls) == 1:
        source_label = f"Webbsökning ({source_urls[0]})"
    elif source_urls:
        source_label = (
            f"Webbsökning ({len(source_urls)} källor, bl.a. {source_urls[0]})"
        )
    else:
        source_label = "Webbsökning"

    return _parse_batch_lines(full_text, products, source_label)


def _batch_prompt_lines(products: list[tuple[str, str]]) -> str:
    return "\n".join(
        f"- {name} (enhet: {unit})" if unit and unit not in ("kg", "")
        else f"- {name}"
        for name, unit in products
    )


def _parse_batch_lines(
    full_text: str,
    products: list[tuple[str, str]],
    source_label: str,
) -> dict[str, tuple[float, str, str]]:
    """Parse structured lines: PRODUKT: name | PRIS: 250 SEK/m2.

    Note: no unstructured single-price fallback. Applying one text-extracted
    price to several unmatched products would assign wrong prices; products the
    model did not return in the structured format are left for the caller to
    handle (the estimate pass, then the unpriced path) rather than guessed at
    here.
    """
    results: dict[str, tuple[float, str, str]] = {}
    # Normalised form -> the name the caller asked about, since the caller looks
    # results up by name.lower().
    normalised = {match_key(name): name for name, _unit in products}

    for line in full_text.split("\n"):
        pm = re.search(
            r'PRODUKT:\s*(.+?)\s*\|\s*PRIS:\s*(\d[\d\s]*(?:[,.]\d+)?)\s*SEK\s*/\s*(\w+[²³]?)',
            line, re.IGNORECASE,
        )
        if not pm:
            continue

        prod_name = pm.group(1)
        raw_num = pm.group(2).replace(" ", "").replace(" ", "").replace(",", ".")
        raw_unit = pm.group(3)

        try:
            price = float(raw_num)
        except ValueError:
            continue

        if price <= 0 or price > 10_000_000:
            continue

        unit_map = {"m²": "m2", "m2": "m2", "m³": "m3", "m3": "m3",
                    "st": "st", "pcs": "st", "lm": "lm", "kg": "kg"}
        unit = unit_map.get(raw_unit.lower(), raw_unit.lower())

        # Match the name the model wrote back to the name we asked about.
        #
        # This used to be a bare .lower() on both sides, which threw away
        # prices the search had genuinely found. Live check 2026-08-20: the
        # model returned all five prices and three were discarded, because it
        # wrote an en dash for a hyphen, a slash for a comma, and a Swedish
        # decimal comma for a point. Exactly the rewrites that broke EPD
        # matching in #557 and #558, which is why the normalisation is now
        # shared rather than reimplemented here.
        needle = match_key(prod_name)
        matched_key = None
        if needle in normalised:
            matched_key = normalised[needle]
        else:
            # Containment either way, longest wins: "Innerdörr vit standard"
            # beats the shorter "Dörr" that also substring-matches.
            #
            # But longest-wins is only right when the candidates are NESTED.
            # "profiled cladding larch" is contained in both "BCLarch profiled
            # cladding larch" and "Scotlarch profiled cladding larch", which are
            # different products, and there the longer name is not the better
            # answer, it is a coin toss. Require that the winner contains every
            # other candidate; otherwise treat it as ambiguous and fall through.
            candidates = [
                (norm, original) for norm, original in normalised.items()
                if norm in needle or needle in norm
            ]
            longest = max(candidates, key=lambda c: len(c[0]))[1] if candidates else None
            if longest is not None:
                winner = match_key(longest)
                nested = all(norm in winner or winner in norm for norm, _ in candidates)
                matched_key = longest if nested else None
            if matched_key is None:
                # Reordered or re-punctuated past what containment can follow.
                # Returns None unless one candidate is clearly ahead: pricing
                # the wrong product is worse than leaving it to the estimate.
                hit = best_token_match(prod_name, list(normalised.values()))
                matched_key = hit

        if matched_key:
            results[matched_key.lower()] = (price, unit, source_label)
            logger.info("Batch price for '%s': %.0f SEK/%s", matched_key, price, unit)
        else:
            # Silence here is how the discarded prices went unnoticed for a
            # week. A found-but-unmatched price is a matcher problem, not a
            # search problem, and the log has to be able to tell them apart.
            logger.warning(
                "Batch price for '%s' (%.0f SEK/%s) matched no requested product; discarded",
                prod_name.strip(), price, unit,
            )

    if not results and full_text.strip():
        # Zero rows out of a non-empty answer means the model ignored the
        # format and wrote prose instead, which loses every price in the batch
        # at once. It is invisible otherwise: no discard warnings fire, because
        # there was nothing to discard. Keep a sample so the next occurrence is
        # diagnosable without another live run.
        logger.warning(
            "No price lines parsed from a %d-char answer for %d products. "
            "First 400 chars: %s",
            len(full_text), len(products), full_text.strip()[:400],
        )

    return results


def estimate_prices_batch(
    products: list[tuple[str, str]],
    *,
    timeout: float | None = None,
) -> dict[str, tuple[float, str, str]]:
    """Model's own price estimate for products web search could not resolve.

    Henric, 2026-08-20: "Om det verkligen inte går att hitta ett pris med
    webbsök så ska ett LLM-genererat pris anges (och källangivelsen ska då säga
    det)." A blank in the cost column is useless to someone deciding between
    two materials; a labelled estimate is not.

    The single-product path already had this via _estimate_price_without_search,
    but lookup_prices_batch never called it, so in any analysis with more than
    one unpriced alternative the fallback was unreachable. That is the whole
    reason "Pris saknas" showed up as often as it did.

    One call for the whole remainder, never per product: sequential lookups are
    what caused the 5+ minute timeouts this module was restructured to avoid.
    """
    if not products:
        return {}

    client = _get_client()
    if client is None:
        return {}

    prompt = (
        f"Uppskatta typiska installerade priser (material + arbete) på den "
        f"svenska byggmarknaden för följande produkter:\n\n"
        f"{_batch_prompt_lines(products)}\n\n"
        f"Du har ingen webbsökning. Basera uppskattningen på din kunskap om "
        f"svenska byggpriser och ange en rimlig mittpunkt hellre än att "
        f"utelämna en produkt. Priser i SEK exklusive moms.\n\n"
        f"SVARSFORMAT. Svara med ENBART en rad per produkt, i den här formen:\n"
        f"PRODUKT: [produktnamn] | PRIS: [tal] SEK/[enhet]\n\n"
        f"Ingen brödtext, inga rubriker, inga kommentarer. Skriv produktnamnet "
        f"exakt som det står i listan ovan, tecken för tecken. En rad per "
        f"produkt, alla {len(products)} produkterna."
    )

    try:
        response = call_model(
            client,
            model=PRICING_MODEL,
            max_tokens=PRICING_MAX_TOKENS,
            effort=PRICING_EFFORT,
            messages=[{"role": "user", "content": prompt}],
            **({"timeout": timeout} if timeout else {}),
        )
    except Exception as e:
        logger.warning("Batch price estimation failed: %s", e)
        return {}

    text_parts = [
        b.text for b in (response.content or [])
        if hasattr(b, "type") and b.type == "text"
    ]
    full_text = "\n".join(text_parts)
    if not full_text:
        return {}

    return _parse_batch_lines(full_text, products, "LLM-uppskattning (ingen källa hittad)")
