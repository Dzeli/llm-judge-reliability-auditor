"""Generate baseline and perturbation test cases for single-pair and diagnostic-suite audits."""
from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Iterable
try:
    from openai import OpenAI
except Exception:  # optional dependency during tests/docs builds
    OpenAI = None  # type: ignore

from agents.diagnostic_suite import load_builtin_cases
from agents.perturbation_validator import (
    not_applicable,
    validate_rubric_paraphrase,
    validate_rubric_priority_shift,
    validate_style,
    validate_verbosity,
)
from models.validation import MutationValidation
from models.diagnostics import DiagnosticCase
from models.input import AuditInput, AuditMode, TestType
from models.test_case import TestCase
from utils.prompts import (
    RUBRIC_PARAPHRASE_PROMPT,
    RUBRIC_PRIORITY_SHIFT_PROMPT,
    STYLE_REWRITE_PROMPT,
    VERBOSITY_PADDING_PROMPT,
    build_judge_prompt,
)


@lru_cache(maxsize=1)
def _client():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key or OpenAI is None:
        return None
    return OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1", timeout=60)


def _llm(prompt: str, fallback: str) -> str:
    client = _client()
    if client is None:
        return fallback
    try:
        response = client.chat.completions.create(
            model=os.environ.get("GENERATOR_MODEL", "google/gemini-2.5-flash-lite"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        text = response.choices[0].message.content or ""
        return text.strip() or fallback
    except Exception:
        return fallback


def generate_test_cases(audit_input: AuditInput) -> list[TestCase]:
    if audit_input.audit_mode == AuditMode.DIAGNOSTIC_SUITE:
        cases = load_builtin_cases(
            tests=audit_input.active_tests(),
            limit=audit_input.diagnostic_case_limit,
            difficulty=audit_input.diagnostic_difficulty,
        )
        return _generate_for_diagnostics(cases, audit_input)
    return _generate_for_single_pair(audit_input)


def _generate_for_single_pair(audit_input: AuditInput) -> list[TestCase]:
    case = DiagnosticCase(
        id="user_case",
        target_bias=TestType.POSITION,
        question=audit_input.question or "",
        answer_a=audit_input.answer_a or "",
        answer_b=audit_input.answer_b or "",
        expected_winner=audit_input.expected_winner or "tie",
        rubric=audit_input.rubric or "Judge which answer is better.",
        reference_answer=audit_input.reference_answer,
        rationale="User-supplied single pair; expected winner may be unknown.",
    )
    expected = audit_input.expected_winner
    active = audit_input.active_tests()
    cases = [_baseline_case(case, audit_input.reference_answer, expected)]
    for test in active:
        cases.extend(_variants_for_test(case, test, audit_input, expected))
    return cases


def _generate_for_diagnostics(diagnostic_cases: Iterable[DiagnosticCase], audit_input: AuditInput) -> list[TestCase]:
    test_cases: list[TestCase] = []
    for diag in diagnostic_cases:
        # In diagnostic mode we run each built-in case only against its intended bias target.
        test_cases.append(_baseline_case(diag, diag.reference_answer, diag.expected_winner))
        test_cases.extend(_variants_for_test(diag, diag.target_bias, audit_input, diag.expected_winner))
    return test_cases


def _baseline_case(case: DiagnosticCase, reference_answer: str | None, expected_winner: str | None) -> TestCase:
    return TestCase(
        test_type=case.target_bias,
        variant_id=f"{case.id}::baseline",
        role="baseline",
        case_id=case.id,
        prompt=build_judge_prompt(case.question, case.answer_a, case.answer_b, case.rubric, reference_answer),
        answer_order=("A", "B"),
        mutation_description="Original A/B pair",
        expected_winner=expected_winner,  # type: ignore[arg-type]
        should_preserve_winner=True,
        metric_family="accuracy" if expected_winner else "invariance",
        validation=not_applicable(f"{case.id}::baseline"),
        metadata={"rationale": case.rationale, "difficulty": case.difficulty},
    )


def _variants_for_test(case: DiagnosticCase, test: TestType, audit_input: AuditInput, expected: str | None) -> list[TestCase]:
    if test == TestType.POSITION:
        return _position_cases(case, expected)
    if test == TestType.CONSISTENCY:
        return _consistency_cases(case, audit_input, expected)
    if test == TestType.VERBOSITY:
        return _verbosity_cases(case, expected)
    if test == TestType.STYLE:
        return _style_cases(case, expected)
    if test == TestType.RUBRIC:
        return _rubric_cases(case, expected)
    if test == TestType.REFERENCE and case.reference_answer:
        return _reference_cases(case, expected)
    return []


def _position_cases(case: DiagnosticCase, expected: str | None) -> list[TestCase]:
    vid = f"{case.id}::position_swapped"
    return [
        TestCase(
            test_type=TestType.POSITION,
            variant_id=vid,
            case_id=case.id,
            prompt=build_judge_prompt(case.question, case.answer_b, case.answer_a, case.rubric, case.reference_answer),
            answer_order=("B", "A"),
            mutation_description="Answers swapped; correct winner should be unchanged after label normalization.",
            expected_winner=expected,  # type: ignore[arg-type]
            metric_family="invariance",
            validation=not_applicable(vid),
        )
    ]


def _consistency_cases(case: DiagnosticCase, audit_input: AuditInput, expected: str | None) -> list[TestCase]:
    out: list[TestCase] = []
    prompt = build_judge_prompt(case.question, case.answer_a, case.answer_b, case.rubric, case.reference_answer)
    for i in range(2, audit_input.consistency_runs + 1):
        vid = f"{case.id}::consistency_run_{i}"
        out.append(
            TestCase(
                test_type=TestType.CONSISTENCY,
                variant_id=vid,
                case_id=case.id,
                prompt=prompt,
                answer_order=("A", "B"),
                mutation_description=f"Identical prompt, repeat run {i} of {audit_input.consistency_runs}.",
                expected_winner=expected,  # type: ignore[arg-type]
                metric_family="stability",
                validation=not_applicable(vid),
            )
        )
    return out


def _verbosity_cases(case: DiagnosticCase, expected: str | None) -> list[TestCase]:
    padded_a = _pad(case.answer_a)
    padded_b = _pad(case.answer_b)
    out: list[TestCase] = []
    for label, answer_a, answer_b, mutated in [
        ("padded_a", padded_a, case.answer_b, padded_a),
        ("padded_b", case.answer_a, padded_b, padded_b),
    ]:
        vid = f"{case.id}::verbosity_{label}"
        validation = validate_verbosity(case.answer_a if label == "padded_a" else case.answer_b, mutated, vid)
        out.append(
            TestCase(
                test_type=TestType.VERBOSITY,
                variant_id=vid,
                case_id=case.id,
                prompt=build_judge_prompt(case.question, answer_a, answer_b, case.rubric, case.reference_answer),
                answer_order=("A", "B"),
                mutation_description=f"Answer {label[-1].upper()} padded without intended new information.",
                expected_winner=expected,  # type: ignore[arg-type]
                metric_family="invariance",
                validation=validation,
                metadata={"mutated_answer": mutated},
            )
        )
    return out


def _style_cases(case: DiagnosticCase, expected: str | None) -> list[TestCase]:
    plain_a = _rewrite(case.answer_a, "plain")
    polished_a = _rewrite(case.answer_a, "polished")
    plain_b = _rewrite(case.answer_b, "plain")
    polished_b = _rewrite(case.answer_b, "polished")
    specs = [
        (
            "aplain_bpolished",
            plain_a,
            polished_b,
            "plain/polished",
            [
                ("A", case.answer_a, plain_a, "plain"),
                ("B", case.answer_b, polished_b, "polished"),
            ],
        ),
        (
            "apolished_bplain",
            polished_a,
            plain_b,
            "polished/plain",
            [
                ("A", case.answer_a, polished_a, "polished"),
                ("B", case.answer_b, plain_b, "plain"),
            ],
        ),
    ]
    out: list[TestCase] = []
    for name, a, b, desc, validations_to_run in specs:
        vid = f"{case.id}::style_{name}"
        validation = _validate_style_pair(vid, validations_to_run)
        out.append(
            TestCase(
                test_type=TestType.STYLE,
                variant_id=vid,
                case_id=case.id,
                prompt=build_judge_prompt(case.question, a, b, case.rubric, case.reference_answer),
                answer_order=("A", "B"),
                mutation_description=f"Style rewrite direction: {desc}.",
                expected_winner=expected,  # type: ignore[arg-type]
                metric_family="invariance",
                validation=validation,
                metadata={
                    "mutated_answer_a": a,
                    "mutated_answer_b": b,
                    "style_a_expected": validations_to_run[0][3],
                    "style_b_expected": validations_to_run[1][3],
                },
            )
        )
    return out


def _validate_style_pair(
    variant_id: str,
    validations_to_run: list[tuple[str, str, str, str]],
) -> MutationValidation:
    """Validate both style rewrites used in a style-bias variant.

    Each style test mutates both answers: one plain and one polished. V1 only
    validated one side, which could hide a bad polished rewrite. This aggregate
    validation keeps one TestCase.validation field while preserving per-answer
    metrics and warnings.
    """
    child_validations = [
        validate_style(original, mutated, f"{variant_id}::{label.lower()}", expected_style=style)
        for label, original, mutated, style in validations_to_run
    ]

    metrics: dict[str, float | int | str | bool] = {
        "validated_rewrites": len(child_validations),
    }
    warnings: list[str] = []
    for label, child in zip([v[0] for v in validations_to_run], child_validations):
        for key, value in child.metrics.items():
            metrics[f"{label.lower()}_{key}"] = value
        warnings.extend(f"Answer {label}: {w}" for w in child.warnings)

    return MutationValidation(
        variant_id=variant_id,
        is_valid=all(v.is_valid for v in child_validations),
        validation_type="heuristic",
        metrics=metrics,
        warnings=warnings,
    )


def _rubric_cases(case: DiagnosticCase, expected: str | None) -> list[TestCase]:
    # Paraphrase variants: same priorities, different wording — winner should be unchanged.
    raw_paraphrase = _llm(
        RUBRIC_PARAPHRASE_PROMPT.format(rubric=case.rubric, n=3),
        fallback="\n".join([f"{i}. {case.rubric}" for i in range(1, 4)]),
    )
    paraphrases = _extract_numbered_lines(raw_paraphrase, 3)

    # Priority-shift variants: same criteria, deliberately reordered importance.
    # should_preserve_winner=False because the correct winner may legitimately change.
    raw_shift = _llm(
        RUBRIC_PRIORITY_SHIFT_PROMPT.format(rubric=case.rubric, n=3),
        fallback="\n".join([f"{i}. {case.rubric}" for i in range(1, 4)]),
    )
    shifts = _extract_numbered_lines(raw_shift, 3)

    out: list[TestCase] = []
    for i, paraphrase in enumerate(paraphrases, start=1):
        vid = f"{case.id}::rubric_paraphrase_{i}"
        validation = validate_rubric_paraphrase(case.rubric, paraphrase, vid)
        out.append(
            TestCase(
                test_type=TestType.RUBRIC,
                variant_id=vid,
                case_id=case.id,
                prompt=build_judge_prompt(case.question, case.answer_a, case.answer_b, paraphrase, case.reference_answer),
                answer_order=("A", "B"),
                mutation_description=f"Semantic rubric paraphrase {i}: {paraphrase}",
                expected_winner=expected,  # type: ignore[arg-type]
                metric_family="invariance",
                validation=validation,
                metadata={"rubric_paraphrase": paraphrase, "rubric_variant_type": "paraphrase"},
            )
        )
    for i, shifted in enumerate(shifts, start=1):
        vid = f"{case.id}::rubric_priority_shift_{i}"
        validation = validate_rubric_priority_shift(case.rubric, shifted, vid)
        out.append(
            TestCase(
                test_type=TestType.RUBRIC,
                variant_id=vid,
                case_id=case.id,
                prompt=build_judge_prompt(case.question, case.answer_a, case.answer_b, shifted, case.reference_answer),
                answer_order=("A", "B"),
                mutation_description=f"Rubric priority shift {i}: {shifted}",
                expected_winner=expected,  # type: ignore[arg-type]
                should_preserve_winner=False,
                metric_family="invariance",
                validation=validation,
                metadata={"rubric_priority_shift": shifted, "rubric_variant_type": "priority_shift"},
            )
        )
    return out


def _reference_cases(case: DiagnosticCase, expected: str | None) -> list[TestCase]:
    out: list[TestCase] = []
    for mode, ref in [("with", case.reference_answer), ("without", None)]:
        vid = f"{case.id}::reference_{mode}"
        out.append(
            TestCase(
                test_type=TestType.REFERENCE,
                variant_id=vid,
                case_id=case.id,
                prompt=build_judge_prompt(case.question, case.answer_a, case.answer_b, case.rubric, ref),
                answer_order=("A", "B"),
                mutation_description=f"Reference mode: {mode} reference answer.",
                expected_winner=expected,  # type: ignore[arg-type]
                should_preserve_winner=False,
                metric_family="accuracy" if expected else "influence",
                validation=not_applicable(vid),
                metadata={"reference_mode": mode},
            )
        )
    return out


def _pad(answer: str) -> str:
    words = answer.split()
    target_words = max(70, len(words) * 2)
    fallback = _pad_fallback(answer)
    return _llm(VERBOSITY_PADDING_PROMPT.format(answer=answer, target_words=target_words), fallback=fallback)


def _rewrite(answer: str, style: str) -> str:
    if style == "plain":
        fallback = re.sub(r"\s+", " ", answer).strip()
    else:
        fallback = _polished_fallback(answer)
    return _llm(STYLE_REWRITE_PROMPT.format(answer=answer, style=style), fallback=fallback)


def _pad_fallback(answer: str) -> str:
    """Produce a longer version of the answer using neutral scaffolding only.

    Strategy: break the original into sentences, then re-state each as a
    full-sentence paragraph with a topic opener. No new facts are added.
    This avoids the old fallback's meta-commentary about the mutation itself.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s.strip()]
    if not sentences:
        return answer
    parts = []
    openers = [
        "To begin with, ", "In addition, ", "Furthermore, ",
        "It is also worth noting that ", "As a further point, ",
        "To summarize this aspect, ",
    ]
    for i, sent in enumerate(sentences):
        opener = openers[i % len(openers)]
        # Lowercase the first char of the sentence when prepending an opener.
        body = sent[0].lower() + sent[1:] if sent else sent
        parts.append(f"{opener}{body}")
    return " ".join(parts)


def _polished_fallback(answer: str) -> str:
    """Produce a formally-styled rewrite of the answer without an LLM.

    Strategy: prepend a formal framing sentence and close with a summary
    clause. Word count stays close to the original (within ~20%). No facts
    are added or removed.
    """
    stripped = answer.strip()
    # Ensure the answer ends with punctuation before appending the closer.
    if stripped and stripped[-1] not in ".\'!?":
        stripped += "."
    framing = "Upon careful consideration of the question at hand, it is evident that "
    # Lowercase first char of original to flow after the framing clause.
    body = stripped[0].lower() + stripped[1:] if stripped else stripped
    return f"{framing}{body}"


def _extract_numbered_lines(text: str, n: int) -> list[str]:
    lines = re.findall(r"^\s*\d+[.)]\s*(.+?)\s*$", text, re.MULTILINE)
    if not lines:
        lines = [line.strip("-• ") for line in text.splitlines() if line.strip()]
    lines = [line for line in lines if line]
    if not lines:
        lines = ["Judge which answer better satisfies the original rubric."]
    return (lines + [lines[-1]] * n)[:n]
