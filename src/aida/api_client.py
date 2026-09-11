"""Shared API client configuration. Routes through OpenRouter."""

from __future__ import annotations

import os

import anthropic

OPENROUTER_BASE_URL = "https://openrouter.ai/api"

# Vercel kills the function at maxDuration (see vercel.json). AIDA_MAX_DURATION
# lets another host override it without a code change.
PLATFORM_MAX_DURATION = float(os.environ.get("AIDA_MAX_DURATION", "300"))

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


def remaining_budget(started_at: float) -> float:
    """Seconds left of the request before the platform kills the function.

    A step that makes a second call (intake's repair round-trip) must not hand
    the SDK a timeout longer than the request has left, or the function dies
    mid-call and the user gets a gateway page instead of our own message.
    """
    import time

    spent = time.monotonic() - started_at
    return max(0.0, LLM_CALL_TIMEOUT - spent)


def get_client() -> anthropic.Anthropic:
    """Get Anthropic client routed through OpenRouter.

    Uses OPENROUTER_API_KEY (primary), falls back to direct Anthropic access.
    """
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        return anthropic.Anthropic(
            api_key=openrouter_key,
            base_url=OPENROUTER_BASE_URL,
            timeout=LLM_CALL_TIMEOUT,
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        return anthropic.Anthropic(api_key=api_key, timeout=LLM_CALL_TIMEOUT)

    raise RuntimeError(
        "No API key found. Set OPENROUTER_API_KEY or ANTHROPIC_API_KEY."
    )


# Default model for Aida's reasoning agents (OpenRouter format).
# Opus 5: adaptive thinking only — budget_tokens 400s here (see call_model).
DEFAULT_MODEL = "anthropic/claude-opus-5"

# Adaptive-thinking effort levels (Opus 5 / Sonnet 5). These replace the old
# budget_tokens scheme, which returns 400 on both. Prior budget -> effort:
#   LOW (1024) -> medium · STANDARD (5000) -> high · DEEP (10000) -> high.
# Opus 5 "high" is already strong; bump the correctness steps (routing,
# baseline) to "max" only if matching errors resurface — "max" can overthink.
# Do NOT reach for thinking={"type": "disabled"} to save cost on Opus 5: it
# returns 400 at effort xhigh/max, and below that the model sometimes writes a
# tool call into visible text instead of a tool_use block. Lower the effort
# instead — that cuts spend without either failure mode.
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
_ADAPTIVE_MODELS = {"anthropic/claude-opus-5", "anthropic/claude-sonnet-5"}


def _thinking_request(model: str, effort: str | None) -> dict:
    """Request kwargs for adaptive thinking at `effort`. Empty when thinking is
    off or the model can't do adaptive thinking. output_config goes via
    extra_body because the pinned SDK (0.86) doesn't type it; OpenRouter
    forwards it to the model (verified against Opus 4.8)."""
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
    """One entry point for an LLM call: adaptive thinking + effort. Returns the
    Message, so callers keep using extract_text() / block iteration unchanged.

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
    return client.messages.create(**req)


def extract_text(response) -> str:
    """Extract text content, works with both thinking and non-thinking responses."""
    for block in (response.content or []):
        if block.type == "text":
            return block.text
    return ""
