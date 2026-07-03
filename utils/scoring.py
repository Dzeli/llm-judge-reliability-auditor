from models.input import AuditMode, TestType
from models.report import MetricSummary, TestResult

WEIGHTS: dict[TestType, float] = {
    TestType.POSITION: 0.18,
    TestType.VERBOSITY: 0.15,
    TestType.STYLE: 0.18,
    TestType.CONSISTENCY: 0.19,
    TestType.RUBRIC: 0.15,
    TestType.REFERENCE: 0.15,
}

VERDICT_THRESHOLDS = {"LOW": 0.25, "MEDIUM": 0.55}


def bias_score_to_verdict(score: float) -> str:
    if score <= VERDICT_THRESHOLDS["LOW"]:
        return "LOW"
    if score <= VERDICT_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "HIGH"


def compute_reliability_score(results: dict[TestType, TestResult]) -> float:
    """Compute a composite reliability score without hiding submetrics.

    Does not use pure invariance alone, because a consistently wrong judge can
    be invariant. Each TestResult exposes a quality_score that combines the
    relevant dimensions for that test type:
      - perturbation tests (position, verbosity, style, rubric):
            0.60 × robust_accuracy + 0.40 × invariance
            baseline_accuracy is kept as diagnostic context but excluded from
            the composite to avoid double-counting with robust_accuracy.
      - consistency: stability × accuracy when gold labels exist
      - reference: accuracy with reference when gold labels exist

    The UI/report displays all submetrics separately; this score is the summary.
    """
    scored = {t: r for t, r in results.items() if r.quality_score is not None}
    if not scored:
        # Backward-compatible fallback for single-pair probes without gold labels.
        if not results:
            return 0.0
        total = sum(WEIGHTS[t] for t in results)
        weighted_bias = sum(WEIGHTS[t] * r.bias_score for t, r in results.items())
        return round((1 - weighted_bias / total) * 100, 1)

    total = sum(WEIGHTS[t] for t in scored)
    weighted_quality = sum(WEIGHTS[t] * (r.quality_score or 0.0) for t, r in scored.items())
    return round((weighted_quality / total) * 100, 1)


def compute_metric_summary(results: dict[TestType, TestResult]) -> MetricSummary:
    baseline_accs = [r.baseline_accuracy for r in results.values() if r.baseline_accuracy is not None]
    robust_accs = [r.robust_accuracy for r in results.values() if r.robust_accuracy is not None]
    invariances = [r.invariance for r in results.values() if r.invariance is not None]
    stabilities = [r.stability for r in results.values() if r.stability is not None]
    consistency_accs = [r.consistency_accuracy for r in results.values() if r.consistency_accuracy is not None]
    consistency_qualities = [r.consistency_quality for r in results.values() if r.consistency_quality is not None]
    consistency_profile = None
    if TestType.CONSISTENCY in results:
        consistency_profile = results[TestType.CONSISTENCY].consistency_profile

    ref = results.get(TestType.REFERENCE)
    ref_with = ref.reference_accuracy_with if ref else None
    ref_without = ref.reference_accuracy_without if ref else None
    ref_delta = ref.reference_delta if ref else None

    # Bias susceptibility is deliberately about non-invariance only.
    non_invariance = [1 - inv for inv in invariances]

    return MetricSummary(
        baseline_accuracy=_mean(baseline_accs),
        robust_accuracy=_mean(robust_accs),
        invariance=_mean(invariances),
        stability=_mean(stabilities),
        consistency_accuracy=_mean(consistency_accs),
        consistency_quality=_mean(consistency_qualities),
        consistency_profile=consistency_profile,
        reference_accuracy_with=ref_with,
        reference_accuracy_without=ref_without,
        reference_helpfulness=ref_delta,
        bias_susceptibility=_mean(non_invariance),
        reference_helpfulness_label=_reference_label(ref_delta),
    )


def confidence_level(audit_mode: AuditMode, n_cases: int, n_valid_mutations: int) -> str:
    if audit_mode == AuditMode.SINGLE_PAIR:
        return "low"
    if n_cases >= 24 and n_valid_mutations >= 16:
        return "high"
    if n_cases >= 10:
        return "medium"
    return "low"


def score_interpretation(audit_mode: AuditMode, n_cases: int, confidence: str) -> str:
    if audit_mode == AuditMode.SINGLE_PAIR:
        return (
            "Single-pair probe: the score describes fragility on the supplied example only. "
            "V4 separates baseline accuracy, robust accuracy, and invariance when a gold winner is provided."
        )
    return (
        f"Diagnostic-suite estimate based on {n_cases} controlled cases. V4 separates accuracy from invariance; "
        f"perturbation quality is 0.6 × robust_accuracy + 0.4 × invariance, and "
        f"consistency quality is stability × accuracy. Confidence is {confidence}."
    )


def generate_warnings(results: dict[TestType, TestResult], confidence: str) -> list[str]:
    warnings: list[str] = []
    if confidence == "low":
        warnings.append("LOW CONFIDENCE: Treat this as a diagnostic probe, not a general model ranking.")

    summary = compute_metric_summary(results)
    if summary.baseline_accuracy is not None and summary.baseline_accuracy < 0.65:
        warnings.append("Low baseline accuracy: the judge often selects the wrong answer before any perturbation is applied.")
    if summary.invariance is not None and summary.invariance < 0.75:
        warnings.append("Low invariance: the judge changes verdicts under transformations that should preserve the winner.")
    if summary.consistency_quality is not None and summary.consistency_quality < 0.65:
        warnings.append("Low consistency quality: repeated judgments are not both stable and correct.")

    if TestType.STYLE in results and results[TestType.STYLE].verdict == "HIGH":
        warnings.append("CRITICAL: The judge appears susceptible to presentation/style changes.")
    if TestType.POSITION in results and results[TestType.POSITION].verdict != "LOW":
        warnings.append("The judge is sensitive to answer order. Run pairwise evaluations in both directions.")
    if TestType.CONSISTENCY in results and (results[TestType.CONSISTENCY].stability or 1.0) < 0.75:
        warnings.append("The judge is unstable across repeated calls. Use temperature=0 and aggregate multiple runs.")
    if TestType.VERBOSITY in results and results[TestType.VERBOSITY].verdict != "LOW":
        warnings.append("The judge appears length-sensitive. Control answer length or instruct it to ignore verbosity.")
    if TestType.RUBRIC in results and results[TestType.RUBRIC].verdict == "HIGH":
        warnings.append("The judge is highly sensitive to semantically similar rubric wording.")
    return warnings


def generate_recommendations(results: dict[TestType, TestResult]) -> list[str]:
    recommendations: list[str] = []
    ref = results.get(TestType.REFERENCE)
    if ref and ref.reference_delta is not None:
        if ref.reference_delta > 0.05:
            recommendations.append("Reference-guided judging improved accuracy on controlled reference cases.")
        elif ref.reference_delta < -0.05:
            recommendations.append("Reference-guided judging reduced accuracy in this run; inspect reference-case prompts and rationales.")
        else:
            recommendations.append("Reference answers did not materially change accuracy on these cases.")

    if TestType.CONSISTENCY in results:
        cons = results[TestType.CONSISTENCY]
        recommendations.append(
            "Interpret consistency using all four values: stability, consistency accuracy, consistency_quality = stability × accuracy, and the consistency profile."
        )

    recommendations.append("Inspect generated variants and validation warnings before trusting any bias finding.")
    recommendations.append("For stronger claims, add close-call cases from your real task distribution.")
    return recommendations


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def _reference_label(delta: float | None) -> str | None:
    if delta is None:
        return None
    if delta > 0.05:
        return "helpful"
    if delta < -0.05:
        return "harmful"
    return "neutral"
