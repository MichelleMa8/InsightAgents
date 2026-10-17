"""
Free-form JSON completions for the steps that ask the LLM for a JSON object
without a tool schema. Parses the first JSON object in the response (repairing
trailing commas) and re-asks with the parse error on failure.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict

from llm_client import LLMClient

logger = logging.getLogger(__name__)


def _strip_trailing_commas(text: str) -> str:
    """Drop commas that directly precede a closing } or ] (whitespace between
    allowed), leaving string contents untouched. Models trained on JS/Python
    write `"finding": "...",\n}` — gpt-5.6-luna does so systematically on the
    whole-graph insight call (the parse fails with "Expecting property name"
    at the closing brace of a claim), and the re-ask does not cure it."""
    out = []
    in_str = escaped = False
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
            out.append(ch)
        elif ch == ",":
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j < n and text[j] in "}]":
                i += 1
                continue  # trailing comma — drop it
            out.append(ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def parse_json_response(text: str) -> Dict[str, Any]:
    """Extract and parse the first JSON object in an LLM response, ignoring any
    trailing text and markdown fences. Trailing commas are repaired rather than
    re-asked for: the meaning is unambiguous, and a re-ask repeats them."""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text.strip(), flags=re.MULTILINE)
    text = text.strip()
    try:
        obj, _ = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError as first_error:
        try:
            obj, _ = json.JSONDecoder().raw_decode(_strip_trailing_commas(text))
        except json.JSONDecodeError:
            raise first_error from None
        logger.info("Parsed JSON after removing trailing comma(s).")
    return obj


def complete_json(
    llm_client: LLMClient,
    system_prompt: str,
    user_msg: str,
    max_tokens: int,
    stage: str,
    max_retries: int = 1,
) -> Dict[str, Any]:
    """One JSON completion, retried up to `max_retries` times on a parse
    failure. Raises RuntimeError once the budget is spent."""
    last_error = "no response"
    for attempt in range(1, max_retries + 2):
        retry_note = ""
        if attempt > 1:
            retry_note = (
                f"\n\nThe previous response was invalid JSON ({last_error}). Return ONLY a "
                "single valid JSON object — no markdown fences, no surrounding text, and make "
                "sure every string value is properly escaped (no literal line breaks or "
                "unescaped quotes)."
            )
        response = llm_client.complete(
            system_prompt=system_prompt,
            messages=[{"role": "user", "content": user_msg + retry_note}],
            tools=None,
            max_tokens=max_tokens,
            stage=stage,
        )
        raw = response.text or ""
        try:
            return parse_json_response(raw)
        except json.JSONDecodeError as e:
            last_error = str(e)
            # The head of the response alone often hides the failure point,
            # which can sit well past it, so show the text around it too.
            logger.warning(
                "Failed to parse %s JSON on attempt %d/%d: %s\nRaw: %s\nAround the error: %r",
                stage, attempt, max_retries + 1, e, raw[:500],
                e.doc[max(0, e.pos - 200):e.pos + 100],
            )

    raise RuntimeError(
        f"{stage}: LLM did not return valid JSON after {max_retries + 1} attempt(s): {last_error}"
    )
