from models.input import AuditInput, AuditMode, TestType
from agents import perturbation_generator as pg
from agents.perturbation_generator import generate_test_cases


def test_style_cases_validate_both_plain_and_polished_rewrites(monkeypatch):
    def fake_llm(prompt: str, fallback: str) -> str:
        # Keep tests offline and deterministic while still exercising validation.
        return fallback

    monkeypatch.setattr(pg, "_llm", fake_llm)
    inp = AuditInput(
        audit_mode=AuditMode.SINGLE_PAIR,
        question="Why is sleep important?",
        answer_a="Sleep supports memory, immune function, and recovery.",
        answer_b="Sleep is magical and fixes everything.",
        expected_winner="A",
        rubric="Choose the more accurate answer.",
        judge_model="dummy/model",
        tests=[TestType.STYLE],
    )

    style_cases = [c for c in generate_test_cases(inp) if c.test_type == TestType.STYLE]

    assert len(style_cases) == 2
    for case in style_cases:
        assert case.validation is not None
        assert case.validation.metrics["validated_rewrites"] == 2
        assert "a_expected_style" in case.validation.metrics
        assert "b_expected_style" in case.validation.metrics
        assert case.metadata["style_a_expected"] in {"plain", "polished"}
        assert case.metadata["style_b_expected"] in {"plain", "polished"}
