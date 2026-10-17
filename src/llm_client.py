"""
Unified LLM client supporting Anthropic and OpenAI.

Messages are provider-agnostic {"role": "user"|"assistant", "content": str}
dicts. Tools are only used as single forced calls (critic report, endorsement
vote), whose arguments come back as ToolCall(id, name, input).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    id: str
    name: str
    input: Dict[str, Any]


@dataclass
class LLMResponse:
    text: Optional[str]
    tool_calls: List[ToolCall]


def _unsupported_param_error(exc: Exception, param: str) -> bool:
    """True if `exc` looks like the server rejecting `param` as unknown/unsupported
    (rather than a genuine bad value). Matched on the message because the concrete
    exception type differs between the OpenAI SDK and compatible third-party servers."""
    msg = str(exc).lower()
    if param.lower() not in msg:
        return False
    return any(
        marker in msg
        for marker in ("unrecognized", "unsupported", "unknown", "not supported",
                       "extra_forbidden", "unexpected keyword")
    )


def _usage_detail(usage: Any, details_field: str, key: str) -> int:
    """Read `usage.<details_field>.<key>`, defaulting to 0. The details object is a
    pydantic model on the OpenAI SDK but often a plain dict (or absent entirely) on
    OpenAI-compatible servers, so handle both shapes."""
    details = getattr(usage, details_field, None)
    if details is None:
        return 0
    value = details.get(key) if isinstance(details, dict) else getattr(details, key, 0)
    return value or 0


def _item_to_dict(item: Any) -> dict:
    """Response output item -> plain dict, for replaying it in a later request."""
    return item.model_dump(exclude_none=True) if hasattr(item, "model_dump") else dict(item)


# Markers of an account-level failure: the request was well-formed but the key,
# the project or the balance will reject every later request the same way. A
# plain 429 is NOT one of these — a rate limit clears on its own — so it only
# counts when the body says the account is out of money, not out of tokens.
_UNRECOVERABLE_STATUS = (401, 403, 404)
_UNRECOVERABLE_MARKERS = (
    "insufficient_quota",           # OpenAI: exhausted quota / spend limit
    "spend_limit",                  # OpenAI: project_spend_limit_exceeded
    "billing_not_active",           # OpenAI: billing disabled
    "exceeded your current quota",  # OpenAI: legacy wording
    "credit balance is too low",    # Anthropic: out of credit (arrives as a 400)
)


def unrecoverable_api_error(exc: BaseException) -> Optional[str]:
    """A short reason string if `exc` will hit every subsequent call too — a bad
    key, a project without access to the model, or exhausted quota — so a caller
    running a batch can stop instead of retrying the same failure per item.
    None for transient failures (rate limits, 5xx, timeouts).

    Classified by status code and message rather than by SDK exception type, so
    the same check covers both providers and OpenAI-compatible proxies."""
    if getattr(exc, "status_code", None) in _UNRECOVERABLE_STATUS:
        return f"{type(exc).__name__}: {exc}"
    text = str(exc).lower()
    if any(marker in text for marker in _UNRECOVERABLE_MARKERS):
        return f"{type(exc).__name__}: {exc}"
    return None


def _is_openai_reasoning_model(model: str) -> bool:
    """Whether `model` accepts the Responses API `reasoning` parameter. The GPT-5+
    and o-series lines are reasoning models; the GPT-4.x/3.5 lines are not and
    reject the parameter outright."""
    m = model.lower()
    return not (m.startswith("gpt-4") or m.startswith("gpt-3") or m.startswith("chatgpt"))


class LLMClient:
    def __init__(self, provider: str, model: str, api_key: str, max_tokens: int = 2000,
                 base_url: Optional[str] = None):
        self.provider   = provider
        self.model      = model
        self.max_tokens = max_tokens

        # Cumulative API usage across every call made through this client.
        self.usage: Dict[str, int] = {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }

        # Per-call record: {ts, stage, model, input_tokens, output_tokens,
        # cache_creation_input_tokens, cache_read_input_tokens, reasoning_tokens,
        # cost_usd, latency_ms}. `stage` is the caller-supplied cost-tracking tag
        # (e.g. "s2_insight", "cg_filter", "stage2_clash.expert_speech").
        # `reasoning_tokens` is a subset of output_tokens (0 on non-reasoning
        # models), reported so a run can be diagnosed when speeches come out short.
        self.call_log: List[Dict[str, Any]] = []

        # Chat Completions token-cap parameter. `max_tokens` is deprecated and is
        # rejected outright by reasoning models (o-series, gpt-5.x), so we send
        # `max_completion_tokens` and downgrade once, permanently, if an endpoint
        # (typically an older OpenAI-compatible proxy) doesn't know it.
        self._openai_token_param = "max_completion_tokens"

        if base_url is None:  # explicit "" from a caller = force the SDK default
            from config import ANTHROPIC_BASE_URL, OPENAI_BASE_URL
            base_url = ANTHROPIC_BASE_URL if provider == "anthropic" else OPENAI_BASE_URL
        self.base_url = base_url or None

        # Added to every caller-supplied max_tokens when the model spends part of
        # its output budget on a hidden reasoning chain, so a call site's number
        # means "answer budget" on every model instead of "answer + chain" on
        # some. 0 whenever no chain is produced (Anthropic, gpt-4.x, effort
        # "none"/unset), leaving those callers' caps exactly as written.
        self.reasoning_reserve = 0

        if provider == "openai":
            from config import (
                OPENAI_API_STYLE,
                OPENAI_REASONING_EFFORT,
                REASONING_TOKEN_RESERVE,
            )
            self.openai_api_style = OPENAI_API_STYLE
            self.reasoning_effort = (
                OPENAI_REASONING_EFFORT if _is_openai_reasoning_model(model) else ""
            )
            if self.reasoning_effort and self.reasoning_effort != "none":
                self.reasoning_reserve = REASONING_TOKEN_RESERVE

        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
            logger.info("%s client using custom base_url: %s", provider, self.base_url)

        if provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(**client_kwargs)
        elif provider == "openai":
            import openai
            self._client = openai.OpenAI(**client_kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    # ── Usage / cost tracking ─────────────────────────────────────────────────

    def _track_usage(self, input_tokens: int, output_tokens: int,
                     cache_creation: int = 0, cache_read: int = 0, *,
                     stage: str, latency_ms: float, reasoning_tokens: int = 0) -> None:
        self.usage["calls"] += 1
        self.usage["input_tokens"] += input_tokens or 0
        self.usage["output_tokens"] += output_tokens or 0
        self.usage["cache_creation_input_tokens"] += cache_creation or 0
        self.usage["cache_read_input_tokens"] += cache_read or 0

        call_usage = {
            "input_tokens": input_tokens or 0,
            "output_tokens": output_tokens or 0,
            "cache_creation_input_tokens": cache_creation or 0,
            "cache_read_input_tokens": cache_read or 0,
        }
        self.call_log.append({
            "ts": time.time(),
            "stage": stage,
            "model": self.model,
            **call_usage,
            "reasoning_tokens": reasoning_tokens or 0,  # subset of output_tokens
            "cost_usd": self.estimated_cost_usd(call_usage),
            "latency_ms": round(latency_ms, 1),
        })
        logger.debug(
            "%s call #%d [%s]: %d in + %d out tokens (%.0fms)",
            self.model, self.usage["calls"], stage, input_tokens or 0, output_tokens or 0, latency_ms,
        )

    def usage_snapshot(self) -> Dict[str, int]:
        """Copy of the cumulative usage counters (for computing per-phase deltas)."""
        return dict(self.usage)

    def usage_since(self, snapshot: Dict[str, int]) -> Dict[str, int]:
        """Usage accumulated since a usage_snapshot()."""
        return {k: self.usage[k] - snapshot.get(k, 0) for k in self.usage}

    def call_log_snapshot(self) -> int:
        """Length of the call log right now (for computing per-phase call slices)."""
        return len(self.call_log)

    def calls_since(self, snapshot: int) -> List[Dict[str, Any]]:
        """Per-call records logged since a call_log_snapshot()."""
        return self.call_log[snapshot:]

    def model_prices(self) -> Optional[Dict[str, float]]:
        """Per-MTok price row for this client's model from config.API_PRICES_PER_MTOK:
        exact id first, then the longest key the model id extends with a `-` suffix,
        so dated snapshots ("gpt-5.4-mini-2026-03-17", "claude-haiku-4-5-20251001")
        resolve to their base model. Requiring the `-` keeps a genuinely new model
        from silently inheriting an older one's prices ("gpt-5.7" must not fall back
        to "gpt-5"); those return None, i.e. token counts logged without a cost."""
        from config import API_PRICES_PER_MTOK
        if self.model in API_PRICES_PER_MTOK:
            return API_PRICES_PER_MTOK[self.model]
        matches = [k for k in API_PRICES_PER_MTOK if self.model.startswith(k + "-")]
        if not matches:
            return None
        return API_PRICES_PER_MTOK[max(matches, key=len)]

    def estimated_cost_usd(self, usage: Optional[Dict[str, int]] = None) -> Optional[float]:
        """Estimated USD cost of `usage` (default: this client's cumulative usage),
        using config.API_PRICES_PER_MTOK. None if the model has no price entry."""
        prices = self.model_prices()
        if prices is None:
            return None
        u = usage if usage is not None else self.usage
        return (
            u.get("input_tokens", 0) * prices["input"]
            + u.get("output_tokens", 0) * prices["output"]
            + u.get("cache_creation_input_tokens", 0) * prices.get("cache_write", prices["input"] * 1.25)
            + u.get("cache_read_input_tokens", 0) * prices.get("cache_read", prices["input"] * 0.1)
        ) / 1_000_000

    @staticmethod
    def format_cost(cost: Optional[float]) -> str:
        return f"${cost:.2f}" if cost is not None else "n/a (no price entry for model)"

    # ── Public ────────────────────────────────────────────────────────────────

    def complete(
        self,
        system_prompt: str,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        max_tokens: Optional[int] = None,
        required_tool_name: Optional[str] = None,
        *,
        stage: str,
    ) -> LLMResponse:
        """`stage` is a required cost-tracking tag identifying which pipeline
        step this call belongs to (e.g. "s2_insight", "cg_filter",
        "stage2_clash.expert_speech"). Every call site must pass
        one so no API spend goes untracked."""
        # Caller caps state the visible-answer budget; the hidden reasoning chain
        # gets its own headroom on top (see config.REASONING_TOKEN_RESERVE).
        mt = (max_tokens or self.max_tokens) + self.reasoning_reserve
        if required_tool_name and not any(
            tool.get("name") == required_tool_name for tool in (tools or [])
        ):
            raise ValueError(
                f"Required tool {required_tool_name!r} is not present in the tool schema list."
            )
        if self.provider == "anthropic":
            return self._complete_anthropic(
                system_prompt, messages, tools, mt, required_tool_name, stage=stage,
            )
        else:
            return self._complete_openai(
                system_prompt, messages, tools, mt, required_tool_name, stage=stage,
            )

    # ── Anthropic ─────────────────────────────────────────────────────────────

    def _complete_anthropic(
        self, system_prompt, messages, tools, max_tokens, required_tool_name=None, *, stage: str,
    ):
        kwargs: dict = dict(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
        )
        if system_prompt:  # empty string = no system prompt (e.g. eval judge calls)
            kwargs["system"] = system_prompt
        if tools:
            # Anthropic tool schema: {name, description, input_schema}
            kwargs["tools"] = tools
        if required_tool_name:
            kwargs["tool_choice"] = {"type": "tool", "name": required_tool_name}

        t0 = time.time()
        resp = self._client.messages.create(**kwargs)
        latency_ms = (time.time() - t0) * 1000

        if getattr(resp, "usage", None) is not None:
            self._track_usage(
                getattr(resp.usage, "input_tokens", 0),
                getattr(resp.usage, "output_tokens", 0),
                getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
                getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
                stage=stage, latency_ms=latency_ms,
            )

        text = None
        tool_calls: List[ToolCall] = []

        for block in resp.content:
            if block.type == "text":
                text = block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=block.input))

        return LLMResponse(text=text, tool_calls=tool_calls)

    # ── OpenAI ────────────────────────────────────────────────────────────────

    def _complete_openai(
        self, system_prompt, messages, tools, max_tokens, required_tool_name=None, *, stage: str,
    ):
        if self.openai_api_style == "responses":
            return self._complete_openai_responses(
                system_prompt, messages, tools, max_tokens, required_tool_name, stage=stage,
            )
        return self._complete_openai_chat(
            system_prompt, messages, tools, max_tokens, required_tool_name, stage=stage,
        )

    # ── OpenAI: Responses API (/v1/responses) ─────────────────────────────────

    def _complete_openai_responses(
        self, system_prompt, messages, tools, max_tokens, required_tool_name=None, *, stage: str,
    ):
        """The endpoint reasoning models require: unlike Chat Completions it accepts
        function tools together with a non-'none' reasoning effort."""
        kwargs: dict = dict(
            model=self.model,
            input=[{"role": m["role"], "content": m["content"]} for m in messages],
            max_output_tokens=max_tokens,
        )
        if system_prompt:
            # `instructions` is the Responses equivalent of a system message. It is
            # NOT carried over between requests, so it must be resent every call —
            # which is what every call site here does anyway.
            kwargs["instructions"] = system_prompt
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        if tools:
            # Responses tool schema is flat: {type, name, description, parameters}
            kwargs["tools"] = [self._anthropic_tool_to_responses(t) for t in tools]
            if required_tool_name:
                kwargs["tool_choice"] = {"type": "function", "name": required_tool_name}
                kwargs["parallel_tool_calls"] = False

        t0 = time.time()
        resp = self._client.responses.create(**kwargs)
        latency_ms = (time.time() - t0) * 1000

        usage = getattr(resp, "usage", None)
        if usage is not None:
            # input_tokens INCLUDES cache hits here too — subtract for disjoint buckets.
            input_tokens = getattr(usage, "input_tokens", 0) or 0
            cached = _usage_detail(usage, "input_tokens_details", "cached_tokens")
            self._track_usage(
                max(input_tokens - cached, 0),
                getattr(usage, "output_tokens", 0) or 0,
                0,        # OpenAI caches automatically: no cache-write surcharge
                cached,
                stage=stage, latency_ms=latency_ms,
                reasoning_tokens=_usage_detail(usage, "output_tokens_details", "reasoning_tokens"),
            )

        if getattr(resp, "status", None) == "incomplete":
            reason = getattr(getattr(resp, "incomplete_details", None), "reason", "unknown")
            logger.warning(
                "[%s] response incomplete (%s) at max_output_tokens=%d (answer budget %d "
                "+ REASONING_TOKEN_RESERVE %d) — this budget is shared with the hidden "
                "reasoning chain, so raise REASONING_TOKEN_RESERVE, raise the stage's "
                "answer cap, or lower OPENAI_REASONING_EFFORT.",
                stage, reason, max_tokens, max_tokens - self.reasoning_reserve,
                self.reasoning_reserve,
            )

        tool_calls: List[ToolCall] = []
        for item in (_item_to_dict(item) for item in resp.output):
            if item.get("type") != "function_call":
                continue
            try:
                args = json.loads(item["arguments"])
            except json.JSONDecodeError:
                # Truncated mid-arguments (the reasoning chain can eat the whole
                # output budget). Drop the call so callers see "no tool call" and
                # take their existing retry path instead of dying on a parse error.
                logger.warning(
                    "[%s] dropping %s call with unparseable arguments (%d chars, "
                    "likely truncated).", stage, item.get("name"), len(item["arguments"]),
                )
                continue
            tool_calls.append(ToolCall(id=item["call_id"], name=item["name"], input=args))
        text = resp.output_text or None
        return LLMResponse(text=text, tool_calls=tool_calls)

    @staticmethod
    def _anthropic_tool_to_responses(tool: dict) -> dict:
        return {
            "type": "function",
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
        }

    # ── OpenAI: Chat Completions (/v1/chat/completions) ───────────────────────

    def _complete_openai_chat(
        self, system_prompt, messages, tools, max_tokens, required_tool_name=None, *, stage: str,
    ):
        openai_msgs = [{"role": "system", "content": system_prompt}] if system_prompt else []
        openai_msgs += [{"role": m["role"], "content": m["content"]} for m in messages]

        kwargs: dict = dict(
            model=self.model,
            messages=openai_msgs,
        )
        kwargs[self._openai_token_param] = max_tokens
        if tools:
            # Convert Anthropic-style tool schema to OpenAI function schema
            kwargs["tools"] = [self._anthropic_tool_to_openai(t) for t in tools]
            kwargs["tool_choice"] = (
                {
                    "type": "function",
                    "function": {"name": required_tool_name},
                }
                if required_tool_name else "auto"
            )
            if required_tool_name:
                kwargs["parallel_tool_calls"] = False

        t0 = time.time()
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            if not (self._openai_token_param == "max_completion_tokens"
                    and _unsupported_param_error(exc, "max_completion_tokens")):
                raise
            logger.warning(
                "Endpoint rejected max_completion_tokens (%s); falling back to the "
                "deprecated max_tokens for the rest of this client's calls.", exc,
            )
            self._openai_token_param = "max_tokens"
            kwargs.pop("max_completion_tokens", None)
            kwargs["max_tokens"] = max_tokens
            t0 = time.time()
            resp = self._client.chat.completions.create(**kwargs)
        latency_ms = (time.time() - t0) * 1000

        usage = getattr(resp, "usage", None)
        if usage is not None:
            # OpenAI's prompt_tokens INCLUDES cache hits, unlike Anthropic's
            # input_tokens — subtract them so the two providers' buckets mean the
            # same thing and cached tokens aren't also billed at the full input rate.
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            cached = _usage_detail(usage, "prompt_tokens_details", "cached_tokens")
            self._track_usage(
                max(prompt_tokens - cached, 0),
                getattr(usage, "completion_tokens", 0) or 0,
                0,        # OpenAI caches automatically: no cache-write surcharge
                cached,
                stage=stage, latency_ms=latency_ms,
                reasoning_tokens=_usage_detail(usage, "completion_tokens_details", "reasoning_tokens"),
            )

        choice = resp.choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            logger.warning(
                "[%s] response truncated at max_tokens=%d (answer budget %d + "
                "REASONING_TOKEN_RESERVE %d) before the model finished.",
                stage, max_tokens, max_tokens - self.reasoning_reserve,
                self.reasoning_reserve,
            )
        msg = choice.message

        text = msg.content
        tool_calls: List[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        input=json.loads(tc.function.arguments),
                    )
                )
        return LLMResponse(text=text, tool_calls=tool_calls)

    @staticmethod
    def _anthropic_tool_to_openai(tool: dict) -> dict:
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
