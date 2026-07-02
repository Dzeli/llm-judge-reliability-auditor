from judges.openrouter import OpenRouterJudge


def get_judge(model: str):
    return OpenRouterJudge(model=model)
