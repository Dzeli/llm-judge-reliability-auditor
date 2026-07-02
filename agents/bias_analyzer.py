from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from models.input import TestType
from models.report import TestResult
from models.test_case import JudgeDecision, TestCase
from utils.scoring import bias_score_to_verdict


def analyze(decisions: list[JudgeDecision], test_cases: list[TestCase]) -> dict[TestType, TestResult]:
    """Analyze judge decisions using V3's separated metric families.

    V2 compared most perturbation decisions directly against the expected winner.
    That mixed two different properties:
      1. accuracy: did the judge pick the gold winner?
      2. invariance: did the judge's verdict stay stable under an irrelevant mutation?

    V3 computes both. A judge can therefore be:
      - correct and invariant,
      - correct but perturbation-sensitive,
      - consistently wrong,
      - wrong and unstable.
    """
    case_by_variant = {c.variant_id: c for c in test_cases}
    decisions_by_type: dict[TestType, list[JudgeDecision]] = defaultdict(list)
    baselines_by_case: dict[str, JudgeDecision] = {}

    for decision in decisions:
        case = case_by_variant.get(decision.variant_id)
        if not case:
            continue
        is_baseline = case.role == "baseline" or decision.variant_id.endswith("::baseline")
        if is_baseline:
            baselines_by_case[decision.case_id] = decision
        else:
            decisions_by_type[case.test_type].append(decision)

    results: dict[TestType, TestResult] = {}
    for test_type, variants in decisions_by_type.items():
        case_ids = {d.case_id for d in variants}
        paired_baselines = [b for cid, b in baselines_by_case.items() if cid in case_ids]
        ds = paired_baselines + variants
        if test_type == TestType.CONSISTENCY:
            results[test_type] = _consistency(ds, case_by_variant)
        elif test_type == TestType.REFERENCE:
            # Reference variants contain explicit with/without prompts; paired baselines
            # are kept as a fallback for older reports but no longer create spurious tests.
            results[test_type] = _reference(ds, case_by_variant)
        else:
            results[test_type] = _perturbation_invariance(test_type, ds, case_by_variant)
    return results


def _perturbation_invariance(
    test_type: TestType,
    decisions: list[JudgeDecision],
    case_by_variant: dict[str, TestCase],
) -> TestResult:
    baseline_decisions = [d for d in decisions if d.variant_id.endswith("::baseline")]
    variant_decisions = [d for d in decisions if not d.variant_id.endswith("::baseline")]
    eval_variants = variant_decisions or []

    baseline_by_case = {d.case_id: d for d in baseline_decisions}
    expected_by_case = {
        c.case_id: c.expected_winner
        for c in case_by_variant.values()
        if c.expected_winner is not None
    }

    baseline_known = [d for d in baseline_decisions if expected_by_case.get(d.case_id)]
    baseline_correct = [
        d.winner == expected_by_case[d.case_id]
        for d in baseline_known
    ]
    baseline_accuracy = _safe_mean_bool(baseline_correct)

    robust_known = [d for d in eval_variants if expected_by_case.get(d.case_id)]
    robust_correct = [
        d.winner == expected_by_case[d.case_id]
        for d in robust_known
    ]
    robust_accuracy = _safe_mean_bool(robust_correct)

    invariant_flags: list[bool] = []
    invariance_failures: list[JudgeDecision] = []
    robust_failures: list[JudgeDecision] = []
    evidence: list[str] = []

    for d in eval_variants:
        baseline = baseline_by_case.get(d.case_id)
        expected = expected_by_case.get(d.case_id)
        if baseline:
            invariant = d.winner == baseline.winner
            invariant_flags.append(invariant)
            if not invariant:
                invariance_failures.append(d)
                evidence.append(
                    f"{d.case_id}: invariance failure — baseline='{baseline.winner}', "
                    f"variant='{d.winner}' ({d.variant_id})."
                )
        if expected and d.winner != expected:
            robust_failures.append(d)

    invariance = _safe_mean_bool(invariant_flags)

    # For invariance-style tests, bias_score is pure non-invariance, not inaccuracy.
    # If no perturbation variants exist, fall back to robust accuracy failure only.
    if invariance is not None:
        bias_score = round(1 - invariance, 3)
        n_failures = len(invariance_failures)
    elif robust_accuracy is not None:
        bias_score = round(1 - robust_accuracy, 3)
        n_failures = len(robust_failures)
    else:
        bias_score = 0.0
        n_failures = 0

    quality_score = _quality_for_perturbation(baseline_accuracy, robust_accuracy, invariance)

    # Add accuracy evidence separately so stable wrongness is visible but not mislabeled as bias.
    if baseline_accuracy is not None and baseline_accuracy < 1:
        wrong_baselines = [
            d for d in baseline_known if d.winner != expected_by_case[d.case_id]
        ]
        for d in wrong_baselines[:4]:
            evidence.append(
                f"{d.case_id}: baseline accuracy failure — expected='{expected_by_case[d.case_id]}', "
                f"got='{d.winner}'. This is accuracy error, not necessarily {test_type.value} bias."
            )
    if robust_accuracy is not None and robust_accuracy < 1:
        for d in robust_failures[:4]:
            evidence.append(
                f"{d.case_id}: robust accuracy failure under perturbation — expected='{expected_by_case[d.case_id]}', "
                f"got='{d.winner}' ({d.variant_id})."
            )

    if not evidence:
        msg_parts = [f"{test_type.value}: no invariance failures detected"]
        if baseline_accuracy is not None:
            msg_parts.append(f"baseline accuracy={baseline_accuracy:.0%}")
        if robust_accuracy is not None:
            msg_parts.append(f"robust accuracy={robust_accuracy:.0%}")
        if invariance is not None:
            msg_parts.append(f"invariance={invariance:.0%}")
        evidence = ["; ".join(msg_parts) + "."]

    return TestResult(
        test_type=test_type,
        decisions=eval_variants or decisions,
        bias_score=bias_score,
        verdict=bias_score_to_verdict(bias_score),
        evidence=evidence[:16],
        n_cases=len({d.case_id for d in eval_variants or decisions}),
        n_failures=n_failures,
        baseline_accuracy=baseline_accuracy,
        robust_accuracy=robust_accuracy,
        invariance=invariance,
        accuracy=robust_accuracy,
        quality_score=quality_score,
        metric_family="invariance",
    )


def _consistency(decisions: list[JudgeDecision], case_by_variant: dict[str, TestCase]) -> TestResult:
    by_case: dict[str, list[JudgeDecision]] = defaultdict(list)
    for d in decisions:
        by_case[d.case_id].append(d)

    stability_values: list[float] = []
    accuracy_values: list[float] = []
    quality_values: list[float] = []
    evidence: list[str] = []
    unstable_cases = 0

    for case_id, ds in by_case.items():
        winners = [d.winner for d in ds]
        most_common, count = Counter(winners).most_common(1)[0]
        stability = count / len(winners)
        stability_values.append(stability)
        if stability < 1:
            unstable_cases += 1

        expected = next(
            (case_by_variant[d.variant_id].expected_winner for d in ds if case_by_variant[d.variant_id].expected_winner),
            None,
        )
        accuracy = None
        quality = None
        if expected:
            accuracy = sum(1 for d in ds if d.winner == expected) / len(ds)
            quality = stability * accuracy
            accuracy_values.append(accuracy)
            quality_values.append(quality)

        if accuracy is None:
            evidence.append(
                f"{case_id}: winners={winners}; stability={stability:.0%}; no gold winner available."
            )
        else:
            evidence.append(
                f"{case_id}: winners={winners}; stability={stability:.0%}; "
                f"accuracy={accuracy:.0%}; consistency_quality=stability×accuracy={quality:.0%}."
            )

    stability = round(mean(stability_values), 3) if stability_values else None
    consistency_accuracy = round(mean(accuracy_values), 3) if accuracy_values else None
    consistency_quality = round(mean(quality_values), 3) if quality_values else stability
    consistency_profile = _consistency_profile(stability, consistency_accuracy)

    # For consistency, bias_score reflects the quality loss when gold labels exist.
    # Without gold labels, it reflects pure instability.
    if consistency_quality is not None:
        bias_score = round(1 - consistency_quality, 3)
    elif stability is not None:
        bias_score = round(1 - stability, 3)
    else:
        bias_score = 0.0

    if consistency_profile:
        evidence.insert(
            0,
            f"Consistency profile: {consistency_profile.replace('_', ' ')} "
            f"(stability={stability:.0%}, accuracy={consistency_accuracy:.0%})."
            if stability is not None and consistency_accuracy is not None
            else f"Consistency profile: {consistency_profile.replace('_', ' ')}."
        )

    return TestResult(
        test_type=TestType.CONSISTENCY,
        decisions=decisions,
        bias_score=bias_score,
        verdict=bias_score_to_verdict(bias_score),
        evidence=evidence[:16],
        n_cases=len(by_case),
        n_failures=unstable_cases,
        stability=stability,
        consistency_accuracy=consistency_accuracy,
        consistency_quality=consistency_quality,
        consistency_profile=consistency_profile,
        accuracy=consistency_accuracy,
        quality_score=consistency_quality,
        metric_family="stability",
    )


def _reference(decisions: list[JudgeDecision], case_by_variant: dict[str, TestCase]) -> TestResult:
    by_case: dict[str, dict[str, JudgeDecision]] = defaultdict(dict)
    for d in decisions:
        if d.variant_id.endswith("reference_with"):
            by_case[d.case_id]["with"] = d
        elif d.variant_id.endswith("reference_without"):
            by_case[d.case_id]["without"] = d
        elif d.variant_id.endswith("baseline"):
            by_case[d.case_id].setdefault("with", d)

    with_correct: list[bool] = []
    without_correct: list[bool] = []
    influence_flags: list[bool] = []
    evidence: list[str] = []

    for case_id, modes in by_case.items():
        expected = next(
            (case_by_variant[d.variant_id].expected_winner for d in modes.values() if case_by_variant[d.variant_id].expected_winner),
            None,
        )
        with_dec = modes.get("with")
        without_dec = modes.get("without")

        if with_dec and without_dec:
            influence_flags.append(with_dec.winner != without_dec.winner)

        if expected and with_dec:
            with_correct.append(with_dec.winner == expected)
        if expected and without_dec:
            without_correct.append(without_dec.winner == expected)

        if with_dec and without_dec:
            evidence.append(
                f"{case_id}: with reference='{with_dec.winner}', without reference='{without_dec.winner}', expected='{expected}'."
            )

    with_acc = _safe_mean_bool(with_correct)
    without_acc = _safe_mean_bool(without_correct)
    influence_rate = _safe_mean_bool(influence_flags)
    delta = None
    if with_acc is not None and without_acc is not None:
        delta = round(with_acc - without_acc, 3)
        evidence.insert(
            0,
            f"Reference accuracy delta: {with_acc:.0%} with reference vs {without_acc:.0%} without reference ({delta:+.0%}).",
        )

    if with_acc is not None:
        bias_score = round(1 - with_acc, 3)
        quality_score = round(with_acc, 3)
    elif influence_rate is not None:
        # No gold label: report influence, not good/bad helpfulness.
        bias_score = round(influence_rate, 3)
        quality_score = None
    else:
        bias_score = 0.0
        quality_score = None

    if not evidence:
        evidence = ["Reference test had insufficient paired with/without decisions."]

    return TestResult(
        test_type=TestType.REFERENCE,
        decisions=decisions,
        bias_score=bias_score,
        verdict=bias_score_to_verdict(bias_score),
        evidence=evidence[:16],
        n_cases=len(by_case),
        n_failures=sum(1 for ok in with_correct if not ok) if with_correct else 0,
        reference_accuracy_with=with_acc,
        reference_accuracy_without=without_acc,
        reference_delta=delta,
        accuracy=with_acc,
        quality_score=quality_score,
        metric_family="accuracy",
    )


def _quality_for_perturbation(
    baseline_accuracy: float | None,
    robust_accuracy: float | None,
    invariance: float | None,
) -> float | None:
    """Composite score used only for the overall reliability index.

    The visible report keeps baseline accuracy, robust accuracy, and invariance separate.
    The composite intentionally uses only robust accuracy and invariance:
      quality = 0.60 * robust_accuracy + 0.40 * invariance

    Baseline accuracy remains visible as diagnostic context, but it is not added to
    this composite because robust accuracy is the more direct measure of whether the
    judge is correct under the perturbation being audited.
    """
    if robust_accuracy is not None and invariance is not None:
        return round(0.60 * robust_accuracy + 0.40 * invariance, 3)
    if invariance is not None:
        return round(invariance, 3)
    if robust_accuracy is not None:
        return round(robust_accuracy, 3)
    return None


def _consistency_profile(stability: float | None, accuracy: float | None) -> str:
    """Classify the failure mode behind consistency_quality.

    The product stability × accuracy is compact but symmetric; the profile makes
    the underlying asymmetry explicit in the report. Thresholds are intentionally
    simple and documented: >= 0.75 counts as acceptable for each component.
    """
    if stability is None:
        return "insufficient_data"
    if accuracy is None:
        return "stability_only"
    stable = stability >= 0.75
    accurate = accuracy >= 0.75
    if stable and accurate:
        return "stable_and_correct"
    if stable and not accurate:
        return "stable_but_inaccurate"
    if accurate and not stable:
        return "accurate_but_unstable"
    return "unstable_and_inaccurate"


def _safe_mean_bool(values: list[bool]) -> float | None:
    if not values:
        return None
    return round(sum(1 for v in values if v) / len(values), 3)
