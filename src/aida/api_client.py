"""Shared API client configuration: two routes to a model, tried in order.

1. Anthropic directly, with the keys in AIDA_ANTHROPIC_KEYS (comma-separated,
   in the order to try). Since 2026-10-21: the Max subscriptions' monthly API
   credits expire at the end of each month, so they go first.
2. OpenRouter, with OPENROUTER_API_KEY, as before. Without AIDA_ANTHROPIC_KEYS
   it is the only route and nothing behaves differently from before.

The model tables (DEFAULT_MODEL, PRICING_MODEL, ...) stay keyed by OpenRouter's
ids; DIRECT_MODEL_IDS translates on the wire for the direct route. The request
body is the same on both routes: everything Aida sends (adaptive thinking,
output_config.effort, tools, documents, forced tool_choice on the classifier)
is Anthropic's own Messages API, which OpenRouter proxies unchanged.

A direct route that answers with an error waiting will not fix (credits gone,
key refused, rate limit, overload, a server error, a rejected request, a
connection that never opened) hands the call to the next route at once, with a
WARNING naming the class. A key whose credits are gone or that is refused is
skipped for EXHAUSTED_FOR_S. A timeout while waiting for the answer is not
handed on: the request's time budget is spent.

One time budget per call, not per route: the caller's timeout (or
LLM_CALL_TIMEOUT) is the deadline, every route gets what is left of it, and a
route is not tried with less than MIN_HANDOFF_BUDGET_S. Otherwise a direct
failure plus a fresh OpenRouter attempt would outlive the Vercel function.

A key never appears in a log line or an error: the WARNINGs redact it, and
direct_keys() drops a value the SDK could not put in a header.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections.abc import Callable

import anthropic

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api"

# The direct keys live under Aida's own name on purpose. ANTHROPIC_API_KEY sits
# first in the SDK's credential order and would take over every Anthropic call
# on the machine (and on the machine running the tests).
DIRECT_KEYS_VAR = "AIDA_ANTHROPIC_KEYS"

ROUTE_OPENROUTER = "openrouter"

# Pinned on the direct clients: the SDK reads ANTHROPIC_BASE_URL from the
# environment even when api_key is explicit, and this route must not follow it.
ANTHROPIC_BASE_URL = "https://api.anthropic.com"

# A connect that stalls on the way to Anthropic must not eat the whole budget
# before OpenRouter gets a go. A connect that works takes well under a second.
DIRECT_CONNECT_TIMEOUT = 10.0

# A route is not tried with less of the budget left than this: the call could
# not finish, and the caller's own error is more useful than a second timeout.
MIN_HANDOFF_BUDGET_S = 10.0

# OpenRouter id -> Anthropic id, for the direct route. A model missing here is
# only ever sent to OpenRouter (with one warning per process), and
# scripts/test_llm_routes.py fails when a configured model is missing.
DIRECT_MODEL_IDS = {
    "anthropic/claude-opus-5.5": "claude-opus-5-5",
    "anthropic/claude-sonnet-5.5": "claude-sonnet-5-5",
    "anthropic/claude-haiku-5.5": "claude-haiku-5-5",
}

# USD per million tokens: input, output, cache read, cache write (5 min), from
# platform.claude.com/docs/en/about-claude/pricing on 2026-10-21. OpenRouter
# charges the same per token for Anthropic's models and does not send the cost
# in a Messages-API response, so one table prices both routes. Haiku 5.5 costs
# 5x above 100k prompt tokens; the classifier prompt is a few thousand.
PRICES = {
    "anthropic/claude-opus-5.5": (4.0, 20.0, 0.20, 5.0),
    "anthropic/claude-sonnet-5.5": (2.0, 10.0, 0.10, 2.50),
    "anthropic/claude-haiku-5.5": (0.10, 0.50, 0.01, 0.125),
}
# USD per web search (the pricing step's server tool), same source.
WEB_SEARCH_USD = 10.0 / 1000

# How long a direct key is skipped after "credits gone" or "key refused".
EXHAUSTED_FOR_S = 600.0

# Vercel kills the function at maxDuration (see vercel.json). AIDA_MAX_DURATION
# lets another host override it without a code change.
# An empty value (a copied .env.example line) means the default, not float("").
PLATFORM_MAX_DURATION = float(os.environ.get("AIDA_MAX_DURATION") or "300")

# Left for Flask to build and return the answer, including the 504 body when a
# step really does run long.
_RESPONSE_HEADROOM = 30.0

# Per-call client timeout (seconds). MUST stay below PLATFORM_MAX_DURATION.
# It was 600 against a 300s ceiling, so the SDK timeout could never fire first:
# Vercel killed the function and the browser got a gateway error page instead of
# the app's own "Analysen tog för lång tid". That is what Johanna reported as
# "för lång tid att analysera, fick timeout" in June 2026. Calls are
# non-streaming with max_tokens <= 16k, under the SDK's long-request guard.
LLM_CALL_TIMEOUT = max(30.0, PLATFORM_MAX_DURATION - _RESPONSE_HEADROOM)


def remaining_budget(started_at: float, now: float | None = None) -> float:
    """Seconds left of the request before the platform kills the function.

    A step that makes a second call (intake's repair round-trip) must not hand
    the SDK a timeout longer than the request has left, or the function dies
    mid-call and the user gets a gateway page instead of our own message.

    ``now`` is a time.monotonic() reading, for callers that take an injectable
    clock; omitted, the real clock is read.
    """
    spent = (time.monotonic() if now is None else now) - started_at
    return max(0.0, LLM_CALL_TIMEOUT - spent)


def get_client() -> anthropic.Anthropic:
    """Anthropic SDK client routed through OpenRouter (OPENROUTER_API_KEY).

    This is the last route and the only one without AIDA_ANTHROPIC_KEYS, so it
    is required. Every model id in Aida is in OpenRouter's format
    ("anthropic/claude-opus-5.5"); the direct route translates it on the wire
    (DIRECT_MODEL_IDS) and builds its own clients in create_message(). A
    missing OpenRouter key fails here, by name, instead of as a model-not-found
    error on the first call. ANTHROPIC_API_KEY is never read.

    max_retries=0: the SDK's default of two silent retries covers timeouts,
    429 and 5xx, so a call that timed out at LLM_CALL_TIMEOUT started again
    instead of raising, and Vercel killed the function mid-retry. The timeout
    budget above only holds if one call is one attempt. Callers that can
    afford a second try (intake's repair, the pricing estimate pass) make it
    themselves, with the remaining budget. The direct clients get the same.
    """
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Aida calls every model through "
            "OpenRouter (after the keys in AIDA_ANTHROPIC_KEYS, when set); "
            "ANTHROPIC_API_KEY is not used."
        )
    return anthropic.Anthropic(
        api_key=openrouter_key,
        base_url=OPENROUTER_BASE_URL,
        timeout=LLM_CALL_TIMEOUT,
        max_retries=0,
    )


# --- The direct route: keys, clients, and the memory of a key that is out ------

# What may stand in an HTTP header value: visible ASCII, no whitespace.
_KEY_FORM = re.compile(r"^[\x21-\x7e]+$")


def direct_keys() -> list[str]:
    """The keys in AIDA_ANTHROPIC_KEYS, in order; empty when unset or blank.

    A value that cannot go in a header (two keys with a newline between them
    instead of a comma, say) is dropped with one WARNING that does not print
    it. Left in, the SDK raises before anything is sent, the error carries
    the value, and OpenRouter is never tried."""
    # The literal name, not DIRECT_KEYS_VAR: scripts/test_env_example.py reads
    # the variables the code uses off the source text.
    raw = os.environ.get("AIDA_ANTHROPIC_KEYS", "")
    keys: list[str] = []
    for i, part in enumerate(raw.split(","), start=1):
        key = part.strip()
        if not key:
            continue
        if not _KEY_FORM.match(key):
            with _LOCK:
                first_time = key not in _BAD_KEY_WARNED
                _BAD_KEY_WARNED.add(key)
            if first_time:
                logger.warning("%s: key %d has characters that cannot go in an HTTP header; "
                               "skipping it", DIRECT_KEYS_VAR, i)
            continue
        keys.append(key)
    return keys


class _KeyState:
    """Per-process memory of one direct key.

    ``exhausted_at`` is a time.monotonic() reading, or None for never. Not 0:
    monotonic time can start near 0, and "never" must not look like "at
    boot" (code-craft.md, the sentinel section)."""

    __slots__ = ("exhausted_at",)

    def __init__(self) -> None:
        self.exhausted_at: float | None = None

    def skipped(self, now: float) -> bool:
        return self.exhausted_at is not None and now - self.exhausted_at < EXHAUSTED_FOR_S


_LOCK = threading.Lock()
_KEY_STATE: dict[str, _KeyState] = {}
_DIRECT_CLIENTS: dict[str, anthropic.Anthropic] = {}
_NO_DIRECT_ID_WARNED: set[str] = set()
_BAD_KEY_WARNED: set[str] = set()


def _key_state(key: str) -> _KeyState:
    with _LOCK:
        state = _KEY_STATE.get(key)
        if state is None:
            state = _KEY_STATE[key] = _KeyState()
        return state


def _direct_client(key: str) -> anthropic.Anthropic:
    """One client per key for the life of the process: Anthropic's own address
    (pinned, see ANTHROPIC_BASE_URL), the same read timeout and one-attempt
    rule as the OpenRouter client, and a short connect timeout. The route
    tests replace this; test_the_direct_client_is_pinned builds the real one."""
    with _LOCK:
        client = _DIRECT_CLIENTS.get(key)
        if client is None:
            client = anthropic.Anthropic(
                api_key=key,
                base_url=ANTHROPIC_BASE_URL,
                timeout=anthropic.Timeout(LLM_CALL_TIMEOUT, connect=DIRECT_CONNECT_TIMEOUT),
                max_retries=0,
            )
            _DIRECT_CLIENTS[key] = client
        return client


def _route_timeout(route: Route, left: float):
    """The per-request timeout for one route from the seconds left of the
    budget. The direct route keeps its short connect timeout; OpenRouter gets
    one number, as before."""
    if route.key is None:
        return left
    return anthropic.Timeout(left, connect=min(DIRECT_CONNECT_TIMEOUT, left))


def reset_route_state() -> None:
    """Forget exhausted keys, cached clients and warnings (tests)."""
    with _LOCK:
        _KEY_STATE.clear()
        _DIRECT_CLIENTS.clear()
        _NO_DIRECT_ID_WARNED.clear()
        _BAD_KEY_WARNED.clear()


class Route:
    """One way to send a request: a name for the logs, a client, and the model
    id that client understands. ``key`` is set on the direct routes, so a
    failure can mark the right key."""

    __slots__ = ("name", "client", "wire_model", "key")

    def __init__(self, name: str, client, wire_model: str, key: str | None = None) -> None:
        self.name = name
        self.client = client
        self.wire_model = wire_model
        self.key = key


def routes_for(client, model: str, now: float | None = None) -> list[Route]:
    """The routes for one call, in order: every direct key that is not skipped,
    then OpenRouter. Always ends with OpenRouter, so a request that every
    direct route hands on still reaches a model."""
    now = time.monotonic() if now is None else now
    routes: list[Route] = []
    keys = direct_keys()
    direct_id = DIRECT_MODEL_IDS.get(model)
    if keys and direct_id is None:
        if model not in _NO_DIRECT_ID_WARNED:
            _NO_DIRECT_ID_WARNED.add(model)
            logger.warning("No direct Anthropic id for %s (DIRECT_MODEL_IDS); "
                           "it goes to OpenRouter only", model)
    elif direct_id is not None:
        for i, key in enumerate(keys, start=1):
            if _key_state(key).skipped(now):
                continue
            try:
                direct = _direct_client(key)
            except Exception as e:  # noqa: BLE001 - one bad key must not take OpenRouter down with it
                logger.warning("Direct Anthropic key %d cannot be used (%s: %s); skipping it",
                               i, type(e).__name__, _short(e, key))
                continue
            routes.append(Route(f"anthropic:{i}", direct, direct_id, key))
    routes.append(Route(ROUTE_OPENROUTER, client, model))
    return routes


# --- Which errors hand the call on ----------------------------------------------

# Failure classes that also mark the key as exhausted for EXHAUSTED_FOR_S. The
# other classes (rate limit, overload, server, connection, a rejected request)
# hand the call on without marking: the next call may well get through.
MARKS_EXHAUSTED = frozenset({"credits", "auth", "permission"})

# Anthropic reports an empty balance as 400 invalid_request_error, not 402
# (402 billing_error exists too). A spend limit set in the console is a 400
# as well ("You have reached your specified API usage limits"). The words are
# the only tell.
_CREDIT_WORDS = re.compile(
    r"credit balance|spend limit|usage limits?|insufficient (?:credit|fund)|billing", re.I,
)


def failover_kind(e: BaseException) -> str | None:
    """The class of a direct-route error that the next route may get past, or
    None when handing on would not help: a timeout while waiting for the
    answer (the request's time is spent, and the SDK already waited it out),
    or anything that is not an API error (a bug in our code, not in a route).

    A connect timeout is the exception among timeouts: nothing was sent, only
    DIRECT_CONNECT_TIMEOUT seconds went by, and the next route has its own
    address. The SDK raises APITimeoutError from the httpx exception, so the
    kind of timeout is in __cause__."""
    if isinstance(e, anthropic.APITimeoutError):
        return "connection" if type(e.__cause__).__name__ == "ConnectTimeout" else None
    if isinstance(e, anthropic.APIConnectionError):
        return "connection"
    if isinstance(e, anthropic.AuthenticationError):
        return "auth"
    if isinstance(e, anthropic.PermissionDeniedError):
        return "permission"
    if isinstance(e, anthropic.RateLimitError):
        return "rate_limit"
    if isinstance(e, anthropic.APIStatusError):
        status = e.status_code
        if status == 402 or (status == 400 and _CREDIT_WORDS.search(str(e))):
            return "credits"
        if status == 529:
            return "overloaded"
        if status >= 500:
            return "server"
        if status == 400:
            return "bad_request"
        return "rejected"
    return None


# --- Usage: one row per call, with the route ------------------------------------

# Called with every usage row (a dict, see usage_record). The web app logs it;
# tests read the route from it.
USAGE_LISTENERS: list[Callable[[dict], None]] = []


def _tokens(usage, name: str) -> int:
    return int(getattr(usage, name, 0) or 0) if usage is not None else 0


def estimate_usd(model: str, usage) -> float | None:
    """List price of one call from its token counts, or None for a model not in
    PRICES. Anthropic's input_tokens exclude the cached tokens, which are
    counted in their own two fields."""
    price = PRICES.get(model)
    if price is None or usage is None:
        return None
    inp, out, read, write = price
    usd = (_tokens(usage, "input_tokens") * inp + _tokens(usage, "output_tokens") * out
           + _tokens(usage, "cache_read_input_tokens") * read
           + _tokens(usage, "cache_creation_input_tokens") * write) / 1e6
    server = getattr(usage, "server_tool_use", None)
    searches = int(getattr(server, "web_search_requests", 0) or 0) if server is not None else 0
    return usd + searches * WEB_SEARCH_USD


def usage_record(route: str, model: str, response, seconds: float) -> dict:
    """One flat row per call: the route that served it, the model as the tables
    name it, tokens, the estimated price and how long it took."""
    usage = getattr(response, "usage", None)
    return {
        "route": route,
        "model": model,
        "input": _tokens(usage, "input_tokens"),
        "output": _tokens(usage, "output_tokens"),
        "cache_read": _tokens(usage, "cache_read_input_tokens"),
        "cache_write": _tokens(usage, "cache_creation_input_tokens"),
        "seconds": round(seconds, 1),
        "stop_reason": getattr(response, "stop_reason", None),
        "usd": estimate_usd(model, usage),
    }


def _emit_usage(record: dict) -> None:
    usd = record["usd"]
    logger.info(
        "LLM usage route=%s model=%s input=%d output=%d cache_read=%d cache_write=%d "
        "usd=%s seconds=%.1f stop=%s",
        record["route"], record["model"], record["input"], record["output"],
        record["cache_read"], record["cache_write"],
        "n/a" if usd is None else f"{usd:.4f}", record["seconds"], record["stop_reason"],
    )
    for listener in list(USAGE_LISTENERS):
        try:
            listener(record)
        except Exception:  # noqa: BLE001 - a listener must not fail the call
            logger.exception("usage listener failed")


def _short(e: BaseException, key: str | None = None) -> str:
    """One line of the error for a log row, without the route's key: an SDK
    error about a header it could not build quotes the whole value."""
    text = " ".join(str(e).split())
    if key:
        text = text.replace(key, "[key]")
    return text if len(text) <= 160 else text[:157] + "..."


def _budget(req: dict) -> float | None:
    """The caller's time budget in seconds: its numeric ``timeout``, or
    LLM_CALL_TIMEOUT when it gave none. None for a timeout of another shape
    (an httpx.Timeout), which is passed through untouched and not budgeted."""
    given = req.get("timeout")
    if given is None:
        return LLM_CALL_TIMEOUT
    if isinstance(given, (int, float)) and not isinstance(given, bool):
        return float(given)
    return None


def create_message(client, req: dict):
    """Send one Messages request over the routes in order and return the
    Message. ``client`` is the OpenRouter client from get_client(); the direct
    clients are built here per key. A route's error that the next route may
    get past (failover_kind) hands the call on with a WARNING; the last
    route's error, and a read timeout on any route, reach the caller unchanged.

    The budget is one per call: the caller's ``timeout`` or LLM_CALL_TIMEOUT,
    counted from here. Every route gets what is left of it as its own timeout,
    and a route is skipped (the previous error raised instead) when less than
    MIN_HANDOFF_BUDGET_S is left, so the routes together never outlive the
    platform's clock.
    """
    model = req["model"]
    routes = routes_for(client, model)
    budget = _budget(req)
    deadline = None if budget is None else time.monotonic() + budget
    for i, route in enumerate(routes):
        last = i == len(routes) - 1
        started = time.monotonic()
        sent = {**req, "model": route.wire_model}
        if deadline is not None:
            sent["timeout"] = _route_timeout(route, max(0.0, deadline - started))
        try:
            response = route.client.messages.create(**sent)
        except Exception as e:
            kind = None if last else failover_kind(e)
            if kind is None:
                raise
            if kind in MARKS_EXHAUSTED and route.key is not None:
                _key_state(route.key).exhausted_at = time.monotonic()
            left = None if deadline is None else deadline - time.monotonic()
            if left is not None and left < MIN_HANDOFF_BUDGET_S:
                logger.warning("LLM route %s failed: %s (%s: %s); %.0fs of the budget left, not trying %s",
                               route.name, kind, type(e).__name__, _short(e, route.key), left,
                               routes[i + 1].name)
                raise
            logger.warning("LLM route %s failed: %s (%s: %s); trying %s",
                           route.name, kind, type(e).__name__, _short(e, route.key), routes[i + 1].name)
            continue
        _emit_usage(usage_record(route.name, model, response, time.monotonic() - started))
        return response
    raise RuntimeError("no LLM route")  # unreachable: OpenRouter is always last


def is_truncated(response) -> bool:
    """True when the model stopped at max_tokens, so its text is cut off.

    A cut-off answer can still contain a parseable fragment (one inner object
    of a list, the first lines of a price list), and a fragment read as the
    answer is worse than an error: it looks complete."""
    return getattr(response, "stop_reason", None) == "max_tokens"


# Default model for Aida's reasoning agents (OpenRouter format).
# Opus 5.5 (from Opus 5, 2026-09-23): adaptive thinking only. budget_tokens AND
# thinking={"type": "disabled"} both 400 here, forced tool_choice ("any"/"tool")
# 400s too, and effort defaults to "medium" where Opus 5 defaulted to "high".
# So every Opus call names its effort explicitly; one that omits it silently
# thinks less than it did on Opus 5.
DEFAULT_MODEL = "anthropic/claude-opus-5.5"

# Adaptive-thinking effort levels (Opus 5.5 / Sonnet 5.5). These replace the old
# budget_tokens scheme, which returns 400 on both. Prior budget -> effort:
#   LOW (1024) -> medium · STANDARD (5000) -> high · DEEP (10000) -> high.
# "high" is already strong; bump the correctness steps (routing, baseline) to
# "max" only if matching errors resurface — "max" can overthink.
# Thinking cannot be switched off on Opus 5.5 at all. Lower the effort to cut
# spend instead.
EFFORT_MEDIUM = "medium"
EFFORT_HIGH = "high"

# Output ceiling for reasoning steps. Adaptive thinking tokens count toward
# max_tokens, so this leaves room for thinking + the answer. Kept at 16k so
# calls stay non-streaming (the SDK refuses longer non-streaming requests) and
# well under Vercel's maxDuration. Adaptive thinking self-scales, so steps
# rarely approach this; a project huge enough to truncate would switch that one
# step to streaming.
REASONING_MAX_TOKENS = 16000

# Models that support adaptive thinking + effort. Anything else (the Haiku
# classifier) degrades to no thinking rather than erroring.
# Must stay in sync with PRICING_MODEL in data/pricing_provider.py: that module
# passes PRICING_EFFORT to call_model, and an id missing from this set silently
# drops both thinking and effort instead of erroring.
_ADAPTIVE_MODELS = {"anthropic/claude-opus-5.5", "anthropic/claude-sonnet-5.5"}


def _thinking_request(model: str, effort: str | None) -> dict:
    """Request kwargs for adaptive thinking at `effort`. Empty when thinking is
    off or the model can't do adaptive thinking. output_config goes via
    extra_body because that works whether or not the installed SDK types it
    (0.86 does not), so the request body is the same on every SDK version:
    local venvs and a fresh Vercel build have not always resolved the same
    one. OpenRouter forwards it to the model (verified against Opus 4.8), and
    on the direct route it is Anthropic's own field."""
    if not effort or model not in _ADAPTIVE_MODELS:
        return {}
    return {
        "thinking": {"type": "adaptive"},
        "extra_body": {"output_config": {"effort": effort}},
    }


def call_model(
    client: anthropic.Anthropic,
    *,
    model: str,
    max_tokens: int,
    effort: str | None = None,
    extra_body: dict | None = None,
    **kwargs,
):
    """One entry point for an LLM call: adaptive thinking + effort, and the
    route choice (create_message). Returns the Message, so callers keep using
    extract_text() / block iteration unchanged. Every call in Aida goes
    through here, the classifier included; scripts/test_llm_routes.py fails
    on a bare client.messages.create() elsewhere.

    Non-streaming: Aida's reasoning calls cap max_tokens at REASONING_MAX_TOKENS
    (16k), under the SDK's long-request guard and Vercel's maxDuration. Adaptive
    thinking self-scales, so this is ample headroom for thinking + the answer.
    """
    req: dict = dict(model=model, max_tokens=max_tokens, **kwargs)
    think = _thinking_request(model, effort)
    merged_extra = dict(extra_body or {})
    if think:
        req["thinking"] = think["thinking"]
        # Deep-merge output_config so a caller's other output_config keys survive
        # (a shallow update would drop them when we add effort).
        think_oc = think["extra_body"].get("output_config", {})
        merged_oc = {**merged_extra.get("output_config", {}), **think_oc}
        merged_extra = {**merged_extra, "output_config": merged_oc}
    if merged_extra:
        req["extra_body"] = merged_extra
    return create_message(client, req)


def extract_text(response) -> str:
    """Extract text content, works with both thinking and non-thinking responses."""
    for block in (response.content or []):
        if block.type == "text":
            return block.text
    return ""
