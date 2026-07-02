from judges import get_judge
from models.test_case import TestCase, JudgeDecision


def run_test_cases(test_cases: list[TestCase], judge_model: str) -> list[JudgeDecision]:
    judge = get_judge(judge_model)
    decisions: list[JudgeDecision] = []
    for case in test_cases:
        decision = judge.call(prompt=case.prompt, variant_id=case.variant_id, case_id=case.case_id)
        if case.answer_order != ("A", "B"):
            decision = _translate_winner(decision, case.answer_order)
        decisions.append(decision)
    return decisions


def _translate_winner(decision: JudgeDecision, answer_order: tuple[str, str]) -> JudgeDecision:
    if decision.winner == "tie":
        return decision
    return decision.model_copy(update={"winner": {"A": answer_order[0], "B": answer_order[1]}[decision.winner]})
