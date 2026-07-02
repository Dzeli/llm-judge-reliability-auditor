BASE_JUDGE_TEMPLATE = """\
You are an impartial evaluator. Judge which answer is better for the question.

QUESTION:
{question}

RUBRIC:
{rubric}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

{reference_block}
Instructions:
- Evaluate only according to the rubric.
- Do not reward verbosity, confidence, or style unless the rubric explicitly asks for that.
- Explain your reasoning in 2-4 sentences.
- End your response with exactly one of:
  WINNER: A
  WINNER: B
  WINNER: TIE
"""

REFERENCE_BLOCK = """\
REFERENCE ANSWER / GROUND TRUTH:
{reference_answer}

Use this reference to assess factual correctness.
"""

RUBRIC_PARAPHRASE_PROMPT = """\
You are a prompt engineer. Produce {n} SEMANTICALLY EQUIVALENT paraphrases of the rubric below.

Important constraints:
- Keep the same priorities and tradeoffs.
- Do not add new criteria.
- Do not remove any criterion.
- Do not change the importance order.

ORIGINAL RUBRIC:
{rubric}

Return exactly {n} paraphrases, numbered 1. 2. 3. Do not add explanations.
"""

RUBRIC_PRIORITY_SHIFT_PROMPT = """\
Create {n} alternative rubrics that intentionally shift priority between criteria while remaining plausible.
These are NOT semantic paraphrases; they are sensitivity probes.

ORIGINAL RUBRIC:
{rubric}

Return exactly {n} variants, numbered 1. 2. 3. Do not add explanations.
"""

VERBOSITY_PADDING_PROMPT = """\
Expand the answer below to roughly {target_words} words while keeping quality as neutral as possible.
Rules:
- Do NOT add new facts, examples, numbers, citations, or corrections.
- Preserve the original meaning, including any mistakes.
- Prefer neutral scaffolding: topic sentences, section labels, restating the question, and light transitions.
- Avoid making the answer worse through obvious repetition, rambling, hedging, or filler.
- The result should be longer but not materially more informative.

ORIGINAL ANSWER:
{answer}

Return only the expanded answer.
"""

STYLE_REWRITE_PROMPT = """\
Rewrite the answer below in a {style} style while keeping the length close to the original.

Styles:
- plain: simple, direct, short sentences, neutral wording.
- polished: fluent, formal, rhetorically confident, but not more informative.

Rules:
- Preserve factual content.
- Do not add or remove substantive claims.
- Do not correct mistakes.
- Keep word count within roughly ±20% of the original when possible.

ORIGINAL ANSWER:
{answer}

Return only the rewritten answer.
"""

PERTURBATION_VALIDATION_PROMPT = """\
You are validating whether a text mutation preserved meaning.

ORIGINAL:
{original}

MUTATED:
{mutated}

Mutation goal: {goal}

Return JSON only with fields:
- is_valid: boolean
- meaning_preserved: boolean
- added_new_facts: boolean
- removed_important_facts: boolean
- style_or_length_changed_as_intended: boolean
- brief_reason: string
"""


def build_judge_prompt(question: str, answer_a: str, answer_b: str, rubric: str, reference_answer: str | None = None) -> str:
    reference_block = ""
    if reference_answer:
        reference_block = REFERENCE_BLOCK.format(reference_answer=reference_answer)
    return BASE_JUDGE_TEMPLATE.format(
        question=question,
        rubric=rubric,
        answer_a=answer_a,
        answer_b=answer_b,
        reference_block=reference_block,
    )
