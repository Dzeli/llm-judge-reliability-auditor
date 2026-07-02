# LLM Judge Reliability Auditor v3.1

A diagnostic evaluation toolkit for testing whether an **LLM-as-a-judge** is accurate, stable, and robust before using it in an evaluation pipeline.

The project started as a single-pair perturbation probe and evolved into a more methodologically careful auditor. V3/V3.1 separates **accuracy** from **invariance** and treats consistency as useful only when it is both stable and correct.

Built for the Kaggle AI Agents Intensive Capstone using **Google ADK**, **Gradio**, **OpenRouter**, **Pydantic**, and **FastMCP**.

---

## Why this exists

LLM-as-a-judge is now common in RAG evaluation, leaderboard construction, preference modeling, automated review, and product analytics. But a judge model can be biased or fragile in ways that are hard to notice:

- It may prefer the first answer because of answer order.
- It may reward longer answers even when they add no useful content.
- It may overvalue polished, confident writing.
- It may change decisions when the rubric is paraphrased.
- It may be stable but consistently wrong.
- It may improve or degrade when given a reference answer.

This tool audits those risks empirically.

---

## What V3/V3.1 improves methodologically

Earlier versions treated many perturbation failures as generic “bias.” V3 is more careful.

For position, verbosity, style, and rubric tests, V3 reports three separate quantities:

| Metric | Meaning |
|---|---|
| `baseline_accuracy` | Did the judge pick the expected winner before perturbation? |
| `robust_accuracy` | Did the judge pick the expected winner after perturbation? |
| `invariance` | Did the judge keep the same verdict under a change that should not affect the winner? |

For the overall reliability index, V3.1 uses a deliberately simple perturbation-quality composite:

```text
perturbation_quality = 0.60 × robust_accuracy + 0.40 × invariance
```

`baseline_accuracy` is still reported, but it is not included in this inner composite because `robust_accuracy` is the direct measure of correctness under the perturbation being audited. The 60/40 split makes correctness under perturbation slightly more important than pure verdict stability, while still penalizing non-invariance.

This prevents a key mistake: a judge can be **consistently wrong** without being position-biased or style-biased.

For consistency, V3 reports:

| Metric | Meaning |
|---|---|
| `stability` | How often repeated calls agree with one another |
| `consistency_accuracy` | How often repeated calls match the expected winner |
| `consistency_quality` | `stability × consistency_accuracy` |

The report still shows stability and accuracy separately, but the overall reliability score uses the combined quality signal so stable wrongness is not rewarded. V3.1 also adds a `consistency_profile` label, such as `stable_and_correct`, `stable_but_inaccurate`, or `unstable_and_inaccurate`, so the same numeric quality score does not hide different failure modes.

---

## Audit modes

### 1. Single-pair probe

Use this mode when you want to debug one custom A/B evaluation example.

You provide:

- question or task
- Answer A
- Answer B
- rubric
- optional reference answer
- optional expected winner
- judge model

Interpretation: this is a **local perturbation probe**, not a global reliability certificate.

### 2. Diagnostic suite

Use this mode when you want to compare judge models across controlled built-in cases.

The built-in suite lives at:

```text
diagnostics/builtin_cases.jsonl
```

Each diagnostic case contains:

```json
{
  "id": "style_close_call_01",
  "target_bias": "style",
  "question": "...",
  "answer_a": "...",
  "answer_b": "...",
  "expected_winner": "A",
  "rubric": "...",
  "reference_answer": "...",
  "difficulty": "hard",
  "rationale": "Why the expected winner is known"
}
```

Interpretation: this gives a stronger empirical bias profile, but it is still limited by the size and design of the diagnostic case set.

---

## Bias dimensions tested

| Dimension | What it checks | Main V3 interpretation |
|---|---|---|
| Position | Does answer order affect verdicts? | Invariance after swapping A/B |
| Verbosity | Does length change the verdict? | Robustness to neutral expansion |
| Style | Does polish/confidence affect verdicts? | Robustness to register changes, with length validation |
| Consistency | Does the model repeat its verdict? | Stability, accuracy, `stability × accuracy`, and failure-mode profile |
| Rubric | Does paraphrasing the rubric change the verdict? | Invariance under equivalent rubric wording |
| Reference | Does a reference answer help or distort judging? | Accuracy with vs. without reference |

---

## Architecture

```text
User / Gradio UI / MCP client
        ↓
AuditInput
        ↓
Orchestrator
        ↓
Perturbation Generator
        ↓
Perturbation Validator
        ↓
Judge Runner
        ↓
Bias Analyzer
        ↓
Report Writer
        ↓
AuditReport
```

The direct Python pipeline is the main runtime path used by Gradio. The same stages are also exposed as Google ADK tools for agent-style orchestration and capstone compatibility.

---

## Project structure

```text
llm_judge_auditor_v3/
├── app.py                         # Gradio UI
├── agents/
│   ├── orchestrator.py            # Pipeline + ADK tool wrappers
│   ├── diagnostic_suite.py        # Loads built-in JSONL cases
│   ├── perturbation_generator.py  # Creates position/style/verbosity/rubric/reference variants
│   ├── perturbation_validator.py  # Checks mutation quality and confounds
│   ├── judge_runner.py            # Calls selected judge model
│   ├── bias_analyzer.py           # Computes accuracy, invariance, stability, quality
│   └── report_writer.py           # Builds final AuditReport
├── diagnostics/
│   └── builtin_cases.jsonl        # Built-in controlled diagnostic cases
├── judges/
│   └── openrouter.py              # OpenRouter judge adapter
├── models/
│   ├── input.py                   # AuditInput, AuditMode, TestType
│   ├── diagnostics.py             # DiagnosticCase schema
│   ├── test_case.py               # TestCase and JudgeDecision
│   ├── validation.py              # MutationValidation
│   └── report.py                  # TestResult, MetricSummary, AuditReport
├── mcp/
│   └── server.py                  # FastMCP tools
├── utils/
│   ├── prompts.py                 # Judge and perturbation prompts
│   └── scoring.py                 # Reliability score, warnings, recommendations
├── tests/                         # Unit tests
├── requirements.txt
├── .env.example
├── V3_CHANGES.md
└── REVIEW_AND_V2_CHANGES.md
```

---

## Quickstart

### 1. Create environment

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add API key

Create a `.env` file:

```bash
cp .env.example .env
```

Then add:

```env
OPENROUTER_API_KEY=sk-or-...
GENERATOR_MODEL=google/gemini-2.5-flash
```

`GENERATOR_MODEL` is optional. It controls the model used to generate perturbations, not the judge being audited.

### 3. Run the app

```bash
python app.py
```

Open the local Gradio URL, choose one or more judge models, then select either:

- **Single-pair probe**
- **Diagnostic suite**

---

## Python usage

### Diagnostic suite

```python
from agents.orchestrator import run_audit_pipeline
from models.input import AuditInput, AuditMode

report = run_audit_pipeline(AuditInput(
    audit_mode=AuditMode.DIAGNOSTIC_SUITE,
    judge_model="google/gemini-2.5-flash",
    diagnostic_case_limit=24,
    diagnostic_difficulty="all",
))

print(report.reliability_score)
print(report.confidence_level)
print(report.metric_summary)
```

### Single-pair probe

```python
from agents.orchestrator import run_audit_pipeline
from models.input import AuditInput, AuditMode

report = run_audit_pipeline(AuditInput(
    audit_mode=AuditMode.SINGLE_PAIR,
    judge_model="google/gemini-2.5-flash",
    question="Which answer better explains overfitting?",
    answer_a="Overfitting happens when a model memorizes training data and performs poorly on new data.",
    answer_b="Overfitting is when a model becomes more intelligent through repeated exposure to examples.",
    expected_winner="A",
    rubric="Choose the answer that is more accurate and useful.",
))

print(report.metric_summary.baseline_accuracy)
print(report.metric_summary.invariance)
```

---

## MCP usage

Run:

```bash
python mcp/server.py
```

Available MCP tools:

- `audit_judge_single_pair`
- `audit_judge_diagnostic_suite`
- `list_builtin_diagnostic_tests`

Example diagnostic-suite parameters:

```text
judge_model = "google/gemini-2.5-flash"
tests = "all"
case_limit = 24
difficulty = "all"
consistency_runs = 5
```

---

## Report fields to look at first

The most important fields in `AuditReport` are:

| Field | Meaning |
|---|---|
| `reliability_score` | Composite quality score, 0–100 |
| `confidence_level` | Low/medium/high based on audit mode and case coverage |
| `score_interpretation` | Plain-language warning about how to interpret the score |
| `metric_summary.baseline_accuracy` | Correctness before perturbation |
| `metric_summary.robust_accuracy` | Correctness after perturbation |
| `metric_summary.invariance` | Verdict stability under irrelevant transformations |
| `metric_summary.stability` | Repeated-call agreement |
| `metric_summary.consistency_accuracy` | Repeated-call correctness |
| `metric_summary.consistency_quality` | `stability × consistency_accuracy` |
| `metric_summary.consistency_profile` | Interpretable failure-mode label for consistency |
| `metric_summary.reference_helpfulness` | Accuracy delta with vs. without reference |
| `mutation_validations` | Warnings about confounded generated perturbations |
| `generated_variants` | Inspectable generated prompts/variants |

---

## Testing

Run:

```bash
python -m compileall .
pytest -q
```

Expected result:

```text
14 passed
```

The tests cover:

- diagnostic case loading
- coverage across bias dimensions
- separated accuracy and invariance scoring
- stable-wrongness behavior
- consistency stability, accuracy, quality, and profile labels
- reference-guided accuracy
- weighted reliability scoring with `0.60 × robust_accuracy + 0.40 × invariance` perturbation quality
- metric summary calculation
- style perturbation validation

---

## Important limitations

This project is a diagnostic toolkit, not a final scientific benchmark.

Current limitations:

- The built-in diagnostic suite is intentionally small.
- Results depend on the quality and representativeness of diagnostic cases.
- Generated perturbations can be imperfect and should be inspected.
- Mutation validation is heuristic, not a formal semantic guarantee.
- OpenRouter/provider model behavior can change over time.
- Strong claims require task-specific cases from the user’s real evaluation distribution.
- The current UI is optimized for clarity, not large-scale benchmarking.

The safest interpretation is:

```text
This tool estimates how a judge behaves on the supplied examples and controlled probes.
It should guide model selection and prompt debugging, not replace a full benchmark.
```

---

## Suggested next improvements

Good next steps for future versions:

- Add larger diagnostic suites by task type: factual QA, summarization, code, RAG, safety, medical, legal, creative writing.
- Add bootstrap confidence intervals over cases.
- Add dynamic OpenRouter model discovery instead of a hardcoded model list.
- Add HTML/PDF export for audit reports.
- Save audit runs and build a model comparison leaderboard.
- Add stronger semantic validation for generated perturbations.
- Add support for user-uploaded diagnostic JSONL files.
- Track judge settings explicitly: temperature, top-p, seed, provider, and model version.

---

## Summary

V3’s main methodological principle is simple:

```text
Reliability = correctness + robustness + stability.
```

A good judge should not only give the same answer repeatedly. It should give the right answer, remain stable under irrelevant transformations, and avoid being tipped by superficial presentation features.
