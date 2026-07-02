import os
import re
import time
try:
    from openai import OpenAI
except Exception:  # optional until runtime
    OpenAI = None  # type: ignore
from models.test_case import JudgeDecision


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

    def call(self, prompt: str, variant_id: str, case_id: str = "user_case") -> JudgeDecision:
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        raw = response.choices[0].message.content or ""
        return JudgeDecision(
            variant_id=variant_id,
            case_id=case_id,
            winner=_parse_winner(raw),
            reasoning=raw.strip(),
            raw_response=raw,
            latency_ms=latency_ms,
            tokens_used=getattr(response.usage, "total_tokens", None) if response.usage else None,
        )


def _parse_winner(text: str) -> str:
    match = re.search(r"WINNER\s*:\s*(A|B|TIE)", text, flags=re.IGNORECASE)
    if not match:
        return "tie"
    value = match.group(1).upper()
    return "tie" if value == "TIE" else value
