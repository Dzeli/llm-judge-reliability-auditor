import os
import re
import time
from typing import Optional

try:
    from openai import OpenAI
except Exception:  # optional until runtime
    OpenAI = None  # type: ignore

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    _HAS_TENACITY = True
except Exception:
    _HAS_TENACITY = False

from models.test_case import JudgeDecision


def _with_retry(fn):
    """Wrap fn with exponential-backoff retry if tenacity is available."""
    if not _HAS_TENACITY:
        return fn
    return retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )(fn)


class OpenRouterJudge:
    def __init__(self, model: str, temperature: float = 0.0, timeout: float = 60.0):
        self.model = model
        self.temperature = temperature
        if OpenAI is None:
            raise RuntimeError("The openai package is required to call OpenRouter. Install requirements.txt first.")
        self.client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            timeout=timeout,
        )
        self._call_with_retry = _with_retry(self._call_api)

    def _call_api(self, prompt: str):
        return self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )

    def call(self, prompt: str, variant_id: str, case_id: str = "user_case") -> JudgeDecision:
        start = time.perf_counter()
        response = self._call_with_retry(prompt)
        latency_ms = (time.perf_counter() - start) * 1000
        raw = response.choices[0].message.content or ""
        winner, parse_failed = _parse_winner(raw)
        return JudgeDecision(
            variant_id=variant_id,
            case_id=case_id,
            winner=winner,
            reasoning=raw.strip(),
            raw_response=raw,
            latency_ms=latency_ms,
            tokens_used=getattr(response.usage, "total_tokens", None) if response.usage else None,
            parse_failed=parse_failed,
        )


def _parse_winner(text: str) -> tuple[str, bool]:
    """Parse WINNER: A/B/TIE from the judge response.

    Returns (winner, parse_failed). parse_failed=True means the expected
    format was absent; the verdict defaulted to 'tie' and metrics may be
    unreliable for this call.
    """
    match = re.search(r"WINNER\s*:\s*(A|B|TIE)", text, flags=re.IGNORECASE)
    if not match:
        return "tie", True
    value = match.group(1).upper()
    return ("tie" if value == "TIE" else value), False
