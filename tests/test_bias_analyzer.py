from agents.bias_analyzer import analyze
from models.input import TestType
from models.test_case import JudgeDecision, TestCase


def _case(test_type, variant_id, expected="A", case_id="case_1", role="variant"):
    return TestCase(
        test_type=test_type,
        variant_id=variant_id,
        role=role,
        case_id=case_id,
        prompt="prompt",
        expected_winner=expected,
        mutation_description="test",
    )


def _decision(variant_id, winner, case_id="case_1"):
    return JudgeDecision(
        variant_id=variant_id,
        case_id=case_id,
        winner=winner,
        reasoning="reason",
        raw_response=f"WINNER: {winner}",
        latency_ms=1.0,
    )


def test_position_reports_accuracy_and_invariance_separately():
    cases = [
        _case(TestType.POSITION, "case_1::baseline"),
        _case(TestType.POSITION, "case_1::position_swapped"),
    ]
    decisions = [
        _decision("case_1::baseline", "A"),
        _decision("case_1::position_swapped", "B"),
    ]

    result = analyze(decisions, cases)[TestType.POSITION]

    assert result.baseline_accuracy == 1.0
    assert result.robust_accuracy == 0.0
    assert result.invariance == 0.0
    assert result.bias_score == 1.0
    assert result.verdict == "HIGH"


def test_stably_wrong_judge_has_low_position_bias_but_low_quality():
    cases = [
        _case(TestType.POSITION, "case_1::baseline", expected="A"),
        _case(TestType.POSITION, "case_1::position_swapped", expected="A"),
    ]
    decisions = [
        _decision("case_1::baseline", "B"),
        _decision("case_1::position_swapped", "B"),
    ]

    result = analyze(decisions, cases)[TestType.POSITION]

    assert result.baseline_accuracy == 0.0
    assert result.robust_accuracy == 0.0
    assert result.invariance == 1.0
    assert result.bias_score == 0.0  # not position-biased; it is consistently wrong
    assert result.quality_score == 0.4


def test_consistency_includes_single_pair_baseline_as_first_run_and_uses_quality():
    cases = [
        _case(TestType.POSITION, "case_1::baseline"),
        _case(TestType.CONSISTENCY, "case_1::consistency_run_2"),
        _case(TestType.CONSISTENCY, "case_1::consistency_run_3"),
    ]
    decisions = [
        _decision("case_1::baseline", "A"),
        _decision("case_1::consistency_run_2", "A"),
        _decision("case_1::consistency_run_3", "B"),
    ]

    result = analyze(decisions, cases)[TestType.CONSISTENCY]

    assert len(result.decisions) == 3
    assert result.stability == 0.667
    assert result.consistency_accuracy == 0.667
    assert result.consistency_quality == 0.444
    assert result.bias_score == 0.556


def test_reference_result_measures_reference_guided_accuracy():
    cases = [
        _case(TestType.REFERENCE, "case_1::reference_with"),
        _case(TestType.REFERENCE, "case_1::reference_without"),
    ]
    decisions = [
        _decision("case_1::reference_with", "A"),
        _decision("case_1::reference_without", "B"),
    ]

    result = analyze(decisions, cases)[TestType.REFERENCE]

    assert result.reference_accuracy_with == 1.0
    assert result.reference_accuracy_without == 0.0
    assert result.reference_delta == 1.0
    assert result.bias_score == 0.0
    assert any("Reference accuracy delta" in e for e in result.evidence)


def test_consistency_only_does_not_create_spurious_position_result():
    cases = [
        _case(TestType.POSITION, "case_1::baseline", expected="A"),
        _case(TestType.CONSISTENCY, "case_1::consistency_run_2", expected="A"),
    ]
    decisions = [
        _decision("case_1::baseline", "A"),
        _decision("case_1::consistency_run_2", "A"),
    ]

    results = analyze(decisions, cases)

    assert TestType.CONSISTENCY in results
    assert TestType.POSITION not in results
    assert results[TestType.CONSISTENCY].consistency_profile == "stable_and_correct"


def test_consistency_profile_labels_stable_wrong_and_stable_correct_cases():
    stable_wrong_cases = [
        _case(TestType.POSITION, "case_1::baseline", expected="A"),
        _case(TestType.CONSISTENCY, "case_1::consistency_run_2", expected="A"),
        _case(TestType.CONSISTENCY, "case_1::consistency_run_3", expected="A"),
    ]
    stable_wrong_decisions = [
        _decision("case_1::baseline", "B"),
        _decision("case_1::consistency_run_2", "B"),
        _decision("case_1::consistency_run_3", "B"),
    ]

    stable_wrong = analyze(stable_wrong_decisions, stable_wrong_cases)[TestType.CONSISTENCY]
    assert stable_wrong.stability == 1.0
    assert stable_wrong.consistency_accuracy == 0.0
    assert stable_wrong.consistency_quality == 0.0
    assert stable_wrong.consistency_profile == "stable_but_inaccurate"

    stable_correct_cases = [
        _case(TestType.POSITION, "case_2::baseline", expected="A", case_id="case_2"),
        _case(TestType.CONSISTENCY, "case_2::consistency_run_2", expected="A", case_id="case_2"),
        _case(TestType.CONSISTENCY, "case_2::consistency_run_3", expected="A", case_id="case_2"),
        _case(TestType.CONSISTENCY, "case_2::consistency_run_4", expected="A", case_id="case_2"),
    ]
    stable_correct_decisions = [
        _decision("case_2::baseline", "A", case_id="case_2"),
        _decision("case_2::consistency_run_2", "A", case_id="case_2"),
        _decision("case_2::consistency_run_3", "A", case_id="case_2"),
        _decision("case_2::consistency_run_4", "B", case_id="case_2"),
    ]

    stable_correct = analyze(stable_correct_decisions, stable_correct_cases)[TestType.CONSISTENCY]
    assert stable_correct.stability == 0.75
    assert stable_correct.consistency_accuracy == 0.75
    assert stable_correct.consistency_quality == 0.562
    assert stable_correct.consistency_profile == "stable_and_correct"


def test_role_field_routes_baseline_without_naming_convention():
    """The role='baseline' field must work as the primary routing mechanism,
    independent of the ::baseline naming convention."""
    cases = [
        _case(TestType.POSITION, "case_1::orig", expected="A", role="baseline"),
        _case(TestType.POSITION, "case_1::swapped", expected="A", role="variant"),
    ]
    decisions = [
        _decision("case_1::orig", "A"),
        _decision("case_1::swapped", "B"),
    ]

    result = analyze(decisions, cases)[TestType.POSITION]

    assert result.baseline_accuracy == 1.0
    assert result.invariance == 0.0
    assert result.bias_score == 1.0
