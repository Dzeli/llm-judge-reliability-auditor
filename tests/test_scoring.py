from models.input import AuditInput, AuditMode, TestType
from agents.perturbation_generator import generate_test_cases
from agents.diagnostic_suite import load_builtin_cases, DEFAULT_PATH


def test_diagnostic_cases_load():
    assert DEFAULT_PATH.exists(), "diagnostics/builtin_cases.jsonl must be shipped with the project"
    cases = load_builtin_cases(limit=3)
    assert len(cases) == 3
    assert cases[0].expected_winner in {"A", "B", "tie"}


def test_diagnostic_cases_cover_core_bias_dimensions():
    cases = load_builtin_cases()
    covered = {c.target_bias for c in cases}
    assert {TestType.POSITION, TestType.VERBOSITY, TestType.STYLE, TestType.CONSISTENCY, TestType.RUBRIC, TestType.REFERENCE} <= covered


def test_single_pair_generates_baseline_and_position():
    inp = AuditInput(
        audit_mode=AuditMode.SINGLE_PAIR,
        question="Q?",
        answer_a="Correct answer.",
        answer_b="Wrong answer.",
        expected_winner="A",
        rubric="Choose the correct answer.",
        judge_model="dummy/model",
        tests=[TestType.POSITION],
    )
    cases = generate_test_cases(inp)
    assert any(c.variant_id.endswith("::baseline") for c in cases)
    assert any(c.variant_id.endswith("position_swapped") for c in cases)
    swapped = next(c for c in cases if c.variant_id.endswith("position_swapped"))
    assert swapped.answer_order == ("B", "A")
