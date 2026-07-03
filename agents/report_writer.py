from models.input import AuditInput
from models.report import AuditReport, TestResult
from models.test_case import TestCase
from utils.scoring import (
    compute_metric_summary,
    compute_reliability_score,
    confidence_level,
    generate_recommendations,
    generate_warnings,
    score_interpretation,
)


def build_report(audit_input: AuditInput, test_results: dict, test_cases: list[TestCase], total_api_calls: int) -> AuditReport:
    validations = [c.validation for c in test_cases if c.validation is not None]
    valid = sum(1 for v in validations if v.is_valid)
    invalid = sum(1 for v in validations if not v.is_valid)
    n_cases = len({c.case_id for c in test_cases})
    confidence = confidence_level(audit_input.audit_mode, n_cases, valid)
    reliability = compute_reliability_score(test_results)
    metric_summary = compute_metric_summary(test_results)

    # Collect all decisions across test results to check for parse failures.
    all_decisions = [d for result in test_results.values() for d in result.decisions]
    parse_failures = [d for d in all_decisions if d.parse_failed]
    if parse_failures:
        # Count as additional invalid mutations so confidence scoring reflects them.
        invalid += len(parse_failures)

    generated_variants = [
        {
            "case_id": c.case_id,
            "variant_id": c.variant_id,
            "test_type": c.test_type.value,
            "mutation_description": c.mutation_description,
            "expected_winner": c.expected_winner,
            "validation_valid": c.validation.is_valid if c.validation else None,
            "validation_warnings": "; ".join(c.validation.warnings) if c.validation and c.validation.warnings else "",
        }
        for c in test_cases
    ]

    warnings = generate_warnings(test_results, confidence)
    if parse_failures:
        warnings.insert(
            0,
            f"PARSE FAILURE: {len(parse_failures)} judge response(s) did not contain 'WINNER: A/B/TIE' and "
            f"defaulted to 'tie'. Affected variant IDs: "
            + ", ".join(d.variant_id for d in parse_failures[:5])
            + ("." if len(parse_failures) <= 5 else f" ... and {len(parse_failures) - 5} more."),
        )

    return AuditReport(
        judge_model=audit_input.judge_model,
        audit_mode=audit_input.audit_mode,
        reliability_score=reliability,
        confidence_level=confidence,
        score_interpretation=score_interpretation(audit_input.audit_mode, n_cases, confidence),
        metric_summary=metric_summary,
        test_results=test_results,
        warnings=warnings,
        recommendations=generate_recommendations(test_results),
        total_api_calls=total_api_calls,
        n_cases=n_cases,
        n_valid_mutations=valid,
        n_invalid_mutations=invalid,
        mutation_validations=validations,
        generated_variants=generated_variants,
        audit_input=audit_input,
    )
