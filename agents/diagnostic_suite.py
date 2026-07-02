import json
from pathlib import Path
from models.diagnostics import DiagnosticCase
from models.input import TestType

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "diagnostics" / "builtin_cases.jsonl"


def load_builtin_cases(
    tests: list[TestType] | None = None,
    limit: int | None = None,
    difficulty: str = "all",
    path: Path = DEFAULT_PATH,
) -> list[DiagnosticCase]:
    cases: list[DiagnosticCase] = []
    wanted = set(tests or list(TestType))
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            case = DiagnosticCase.model_validate(json.loads(line))
            if case.target_bias not in wanted:
                continue
            if difficulty != "all" and case.difficulty != difficulty:
                continue
            cases.append(case)
    if limit:
        return cases[:limit]
    return cases
