import concurrent.futures

from judges import get_judge
from models.test_case import TestCase, JudgeDecision

_MAX_WORKERS = 8


def run_test_cases(test_cases: list[TestCase], judge_model: str) -> list[JudgeDecision]:
    """Call the judge model for each test case, concurrently.

    Uses a ThreadPoolExecutor so that the diagnostic-suite (which can
    generate 400+ calls for a full 18-case × 6-test × 5-run audit) does not
    bottleneck on sequential I/O. Results are returned in the same order as
    the input test_cases list.
    """
    judge = get_judge(judge_model)

    def _call_one(case: TestCase) -> JudgeDecision:
        decision = judge.call(prompt=case.prompt, variant_id=case.variant_id, case_id=case.case_id)
        if case.answer_order != ("A", "B"):
            decision = _translate_winner(decision, case.answer_order)
        return decision

    with concurrent.futures.ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futures = [pool.submit(_call_one, case) for case in test_cases]
        decisions = [f.result() for f in futures]

    return decisions


def _translate_winner(decision: JudgeDecision, answer_order: tuple[str, str]) -> JudgeDecision:
    if decision.winner == "tie":
        return decision
    return decision.model_copy(update={"winner": {"A": answer_order[0], "B": answer_order[1]}[decision.winner]})
