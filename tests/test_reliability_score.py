from models.input import AuditMode, TestType
from models.report import TestResult
from models.test_case import JudgeDecision
from utils.scoring import compute_reliability_score, confidence_level, compute_metric_summary


def _decision():
    return JudgeDecision(
        variant_id="case::variant",
        case_id="case",
        winner="A",
        reasoning="reason",
        raw_response="WINNER: A",
        latency_ms=1.0,
    )


def _result(test_type, bias_score, **kwargs):
    return TestResult(
        test_type=test_type,
        decisions=[_decision()],
        bias_score=bias_score,
        verdict="LOW" if bias_score <= 0.25 else "MEDIUM" if bias_score <= 0.55 else "HIGH",
        evidence=["evidence"],
        n_cases=1,
        n_failures=0,
        **kwargs,
    )


def test_reliability_score_uses_quality_scores_when_available():
    results = {
        TestType.POSITION: _result(TestType.POSITION, 0.0, quality_score=1.0),
        TestType.STYLE: _result(TestType.STYLE, 0.5, quality_score=0.5),
    }
    # equal weights for position/style in v3 -> average quality=(1 + .5)/2=.75
    assert compute_reliability_score(results) == 75.0


def test_reliability_falls_back_to_bias_for_unlabelled_single_pair():
    results = {
        TestType.POSITION: _result(TestType.POSITION, 0.0),
        TestType.STYLE: _result(TestType.STYLE, 0.5),
    }
    assert compute_reliability_score(results) == 75.0


def test_metric_summary_separates_accuracy_invariance_stability_and_consistency_quality():
    results = {
        TestType.POSITION: _result(
            TestType.POSITION,
            0.2,
            baseline_accuracy=0.7,
            robust_accuracy=0.8,
            invariance=0.8,
            quality_score=0.77,
            metric_family="invariance",
        ),
        TestType.CONSISTENCY: _result(
            TestType.CONSISTENCY,
            0.4,
            stability=0.75,
            consistency_accuracy=0.5,
            consistency_quality=0.375,
            quality_score=0.375,
            metric_family="stability",
        ),
        TestType.REFERENCE: _result(
            TestType.REFERENCE,
            0.1,
            reference_accuracy_with=0.9,
            reference_accuracy_without=0.7,
            reference_delta=0.2,
            quality_score=0.9,
            metric_family="accuracy",
        ),
    }
    summary = compute_metric_summary(results)
    assert summary.baseline_accuracy == 0.7
    assert summary.robust_accuracy == 0.8
    assert summary.invariance == 0.8
    assert summary.stability == 0.75
    assert summary.consistency_accuracy == 0.5
    assert summary.consistency_quality == 0.375
    assert summary.consistency_profile is None
    assert summary.reference_accuracy_with == 0.9
    assert summary.reference_accuracy_without == 0.7
    assert summary.reference_helpfulness == 0.2
    assert summary.reference_helpfulness_label == "helpful"


def test_confidence_is_low_for_single_pair_even_with_many_mutations():
    assert confidence_level(AuditMode.SINGLE_PAIR, n_cases=100, n_valid_mutations=100) == "low"
    assert confidence_level(AuditMode.DIAGNOSTIC_SUITE, n_cases=24, n_valid_mutations=16) == "high"
    assert confidence_level(AuditMode.DIAGNOSTIC_SUITE, n_cases=18, n_valid_mutations=12) == "medium"
