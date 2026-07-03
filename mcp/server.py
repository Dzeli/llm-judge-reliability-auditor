import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from fastmcp import FastMCP
from agents.orchestrator import run_audit_pipeline
from models.input import AuditInput, AuditMode, TestType

mcp = FastMCP("LLM Judge Reliability Auditor v3")


@mcp.tool()
def audit_judge_single_pair(
    question: str,
    answer_a: str,
    answer_b: str,
    judge_model: str,
    rubric: str,
    reference_answer: str = "",
    expected_winner: str = "unknown",
    tests: str = "all",
    consistency_runs: int = 5,
) -> str:
    """Probe a judge model for bias on a single custom question/answer pair.

    Use this when you have a specific case you want to test — the auditor mutates
    your pair (swaps positions, rewrites style, etc.) and checks if the judge changes
    its verdict. Confidence is always low (one pair = one data point).

    judge_model: OpenRouter model ID, e.g. 'google/gemini-2.5-flash'
    expected_winner: 'A', 'B', 'tie', or 'unknown'
    tests: comma-separated list of test types, or 'all'. Options: position, verbosity, style, consistency, rubric, reference
    consistency_runs: how many times to repeat the same judgment for consistency testing
    Returns: full AuditReport as JSON
    """
    selected_tests = _parse_tests(tests)
    report = run_audit_pipeline(AuditInput(
        audit_mode=AuditMode.SINGLE_PAIR,
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
        reference_answer=reference_answer or None,
        expected_winner=None if expected_winner == "unknown" else expected_winner,  # type: ignore[arg-type]
        judge_model=judge_model,
        rubric=rubric,
        tests=selected_tests,
        consistency_runs=consistency_runs,
    ))
    return report.model_dump_json(indent=2)


@mcp.tool()
def audit_judge_diagnostic_suite(
    judge_model: str,
    tests: str = "all",
    case_limit: int = 24,
    difficulty: str = "all",
    consistency_runs: int = 5,
) -> str:
    """Run a full bias audit on a judge model using built-in controlled cases.

    Use this to get a generalizable bias profile across all 6 bias dimensions.
    No custom input needed — the built-in case library handles everything.
    Confidence reaches medium or high depending on case count.

    judge_model: OpenRouter model ID, e.g. 'google/gemini-2.5-flash'
    tests: comma-separated list of test types, or 'all'. Options: position, verbosity, style, consistency, rubric, reference
    case_limit: maximum number of diagnostic cases to run (max 24)
    difficulty: filter cases by difficulty — 'all', 'easy', 'medium', or 'hard'
    consistency_runs: how many times to repeat the same judgment for consistency testing
    Returns: full AuditReport as JSON including reliability score, grade, and per-test breakdowns
    """
    report = run_audit_pipeline(AuditInput(
        audit_mode=AuditMode.DIAGNOSTIC_SUITE,
        judge_model=judge_model,
        tests=_parse_tests(tests),
        diagnostic_case_limit=case_limit,
        diagnostic_difficulty=difficulty,  # type: ignore[arg-type]
        consistency_runs=consistency_runs,
    ))
    return report.model_dump_json(indent=2)


@mcp.tool()
def list_builtin_diagnostic_tests() -> str:
    """List all available bias test types supported by the diagnostic suite.

    Returns: JSON object with a 'tests' array of valid test type names.
    Use the returned names as values for the 'tests' parameter in other tools.
    """
    return json.dumps({"tests": [t.value for t in TestType]}, indent=2)


def _parse_tests(tests_str: str) -> list[TestType]:
    if tests_str.strip().lower() == "all":
        return list(TestType)
    names = [t.strip().lower() for t in tests_str.split(",") if t.strip()]
    return [TestType(n) for n in names if n in TestType._value2member_map_]


if __name__ == "__main__":
    mcp.run()
