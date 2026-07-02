import json
try:
    from fastmcp import FastMCP
except Exception:
    from mcp.server.fastmcp import FastMCP
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
    return json.dumps({"tests": [t.value for t in TestType]}, indent=2)


def _parse_tests(tests_str: str) -> list[TestType]:
    if tests_str.strip().lower() == "all":
        return list(TestType)
    names = [t.strip().lower() for t in tests_str.split(",") if t.strip()]
    return [TestType(n) for n in names if n in TestType._value2member_map_]


if __name__ == "__main__":
    mcp.run()
