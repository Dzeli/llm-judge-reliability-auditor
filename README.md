---
title: LLM Judge Reliability Auditor
emoji: 🔬
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.19.0
python_version: '3.13'
app_file: app.py
pinned: false
license: mit
short_description: Multi-agent bias auditor for LLM-as-judge systems
---

# 🔬 LLM Judge Reliability Auditor

A multi-agent diagnostic toolkit that tests whether an **LLM-as-a-judge is trustworthy** — before you use it in an evaluation pipeline.

Built for the [Kaggle AI Agents Intensive Capstone](https://www.kaggle.com/competitions/google-ai-intensive-capstone) using **Google ADK**, **Gradio**, **OpenRouter**, **Pydantic v2**, and **FastMCP**.

---

## The problem this solves

LLM-as-judge has become a standard technique in RAG evaluation, leaderboard construction, preference modeling, and automated review. The idea: instead of human annotators, use a capable LLM to compare two answers and pick the better one.

But judge models can fail in subtle ways that are hard to notice from outputs alone:

| Failure mode | Example |
|---|---|
| **Position bias** | Always favors whichever answer appears first |
| **Verbosity bias** | Rewards longer answers even when length adds nothing |
| **Style bias** | Prefers polished, confident writing over accurate but plain writing |
| **Consistency failure** | Changes its verdict when asked the same question twice |
| **Rubric sensitivity** | Flips verdict when the rubric is paraphrased without changing meaning |
| **Reference distortion** | Performs worse — not better — when a reference answer is provided |

Any of these can quietly corrupt an evaluation pipeline. This tool makes them measurable.

---

## How it works

The auditor runs a **controlled perturbation experiment**:

1. Take a question + two candidate answers with a known correct winner
2. Generate a mutated variant of the pair that should *not* change the correct winner
3. Show both the original and the variant to the judge
4. Measure whether the judge's verdict changes

If a judge flips its verdict under a transformation that should be irrelevant (e.g. swapping answer positions), it has demonstrated that bias.

### The key methodological insight

Early approaches conflated *accuracy* and *stability*. This auditor keeps them separate:

- A judge that **always picks the wrong answer** is stable (invariant) but not reliable
- A judge that **sometimes picks the right answer** but flips under perturbation has a different problem
- A judge that **consistently picks the right answer** and stays stable under perturbation is trustworthy

This distinction drives all metric design in this tool.

---

## Metrics explained

### For perturbation tests (position, verbosity, style, rubric)

| Metric | What it measures |
|---|---|
| `baseline_accuracy` | Did the judge pick the correct winner **before** any perturbation? |
| `robust_accuracy` | Did the judge pick the correct winner **after** the perturbation? |
| `invariance` | Did the judge keep the **same verdict** regardless of which direction was correct? |
| `bias_score` | Rate of verdict flips due to this perturbation (pure instability signal) |
| `quality_score` | `0.60 × robust_accuracy + 0.40 × invariance` |

The quality formula weights correctness under perturbation more heavily than pure stability, because a stable but consistently wrong judge should not score well.

### For consistency tests

| Metric | What it measures |
|---|---|
| `stability` | How often repeated calls agree with each other |
| `consistency_accuracy` | How often repeated calls match the correct winner |
| `consistency_quality` | `stability × consistency_accuracy` |
| `consistency_profile` | One of: `stable_and_correct`, `stable_but_inaccurate`, `accurate_but_unstable`, `unstable_and_inaccurate` |

The profile label is important: two judges with identical `consistency_quality` scores can have very different failure modes.

### For reference tests

| Metric | What it measures |
|---|---|
| `reference_accuracy_with` | Accuracy when the judge is given a reference answer |
| `reference_accuracy_without` | Accuracy without a reference answer |
| `reference_delta` | The difference — positive means reference helps, negative means it hurts |

### Overall

| Metric | What it measures |
|---|---|
| `reliability_score` | Weighted composite of all quality scores, 0–100 |
| `confidence_level` | `low` / `medium` / `high` based on audit mode and case count |
| `grade` | Letter grade derived from reliability score |

---

## Audit modes

### Single-pair probe

**When to use:** You have a specific case you suspect a judge handles poorly.

You supply:
- A question or task
- Answer A and Answer B
- A rubric
- Optionally: a reference answer and the expected winner

The auditor mutates your pair across all selected test types and reports verdict stability and accuracy on that specific example.

**Confidence is always low** — one pair is one data point. Use this for spot-checking and debugging, not for general model comparison.

### Diagnostic suite

**When to use:** You want a generalizable bias profile across many models.

No custom input needed. The auditor runs its built-in library of 18 close-call cases — pairs where the quality gap between answers is small enough that presentation bias can actually tip the verdict.

Cases are tagged by:
- **Target bias type:** position, verbosity, style, consistency, rubric, reference
- **Difficulty:** easy / medium / hard

The auditor aggregates results across cases to produce a bias profile with `medium` or `high` confidence.

---

## Six bias dimensions

| Dimension | What it tests | How |
|---|---|---|
| **Position** | Does answer order affect verdicts? | Swap A↔B and check if the verdict flips |
| **Verbosity** | Does answer length affect verdicts? | Expand one answer with neutral filler content |
| **Style** | Does presentation style affect verdicts? | Rewrite one answer in a different register (plain↔polished) |
| **Consistency** | Does the judge repeat its verdict? | Ask the same question N times and measure agreement |
| **Rubric** | Does rubric wording affect verdicts? | Paraphrase the rubric without changing its meaning |
| **Reference** | Does a reference answer help or hurt? | Compare accuracy with vs. without a reference |

---

## Quickstart (local)

### 1. Set up environment

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add your API key

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENROUTER_API_KEY=sk-or-...
GENERATOR_MODEL=google/gemini-2.5-flash-lite
```

`OPENROUTER_API_KEY` is required. `GENERATOR_MODEL` controls which model generates perturbations (not the judge being audited) — defaults to `google/gemini-2.5-flash-lite` if omitted.

Get a free OpenRouter key at [openrouter.ai](https://openrouter.ai). It provides unified access to all supported judge models.

### 3. Run

```bash
python app.py
```

Open the local Gradio URL in your browser.

---

## Using the UI

1. **Choose audit mode** — Single-pair probe or Diagnostic suite
2. **Select judge model(s)** — run multiple models in parallel for comparison
3. **Choose test types** — all six are selected by default
4. **Run the audit**

For single-pair: fill in the question/answer fields or choose a preset example.  
For diagnostic suite: just click Run — the built-in cases load automatically.

Results appear as:
- A **comparison table** across all selected models
- **Per-model breakdowns** with all submetrics and evidence
- **Warnings** for specific failure modes detected
- **Raw JSON** for programmatic use

---

## Python API

```python
from agents.orchestrator import run_audit_pipeline
from models.input import AuditInput, AuditMode

# Diagnostic suite
report = run_audit_pipeline(AuditInput(
    audit_mode=AuditMode.DIAGNOSTIC_SUITE,
    judge_model="google/gemini-2.5-flash",
    diagnostic_case_limit=24,
    diagnostic_difficulty="all",
))

print(report.reliability_score)       # 0–100
print(report.confidence_level)        # "low" / "medium" / "high"
print(report.metric_summary)          # all submetrics
print(report.warnings)                # plain-language red flags
```

```python
# Single-pair probe
report = run_audit_pipeline(AuditInput(
    audit_mode=AuditMode.SINGLE_PAIR,
    judge_model="google/gemini-2.5-flash",
    question="Which answer better explains overfitting?",
    answer_a="Overfitting happens when a model memorizes training data and generalizes poorly.",
    answer_b="Overfitting is when a model becomes more intelligent through repeated exposure.",
    expected_winner="A",
    rubric="Choose the answer that is more accurate and useful.",
))

print(report.metric_summary.baseline_accuracy)
print(report.metric_summary.invariance)
print(report.metric_summary.consistency_profile)
```

---

## MCP server

```bash
python mcp/server.py
```

Exposes three tools for agent-style orchestration:

| Tool | Description |
|---|---|
| `audit_judge_single_pair` | Run a single-pair probe via MCP |
| `audit_judge_diagnostic_suite` | Run the diagnostic suite via MCP |
| `list_builtin_diagnostic_tests` | List available built-in test cases |

---

## Architecture

```
User / Gradio UI / MCP client
        │
        ▼
   AuditInput (Pydantic)
        │
        ▼
   Orchestrator  ──── Google ADK tools
        │
        ├─▶ Perturbation Generator  (creates position/style/verbosity/rubric/reference variants)
        │
        ├─▶ Perturbation Validator  (checks mutation quality; flags confounds)
        │
        ├─▶ Judge Runner  (calls judge model via OpenRouter)
        │
        ├─▶ Bias Analyzer  (computes accuracy, invariance, stability, quality scores)
        │
        └─▶ Report Writer  (builds AuditReport with warnings and recommendations)
```

The Gradio UI and the Python API both go through the same Orchestrator. The MCP server wraps the same pipeline for agent-style use.

---

## Project structure

```
├── app.py                          # Gradio UI
├── agents/
│   ├── orchestrator.py             # Pipeline + ADK tool wrappers
│   ├── diagnostic_suite.py         # Loads built-in JSONL cases
│   ├── perturbation_generator.py   # Generates test variants
│   ├── perturbation_validator.py   # Validates mutation quality
│   ├── judge_runner.py             # Calls judge model via OpenRouter
│   ├── bias_analyzer.py            # Computes all metrics
│   └── report_writer.py            # Builds AuditReport
├── diagnostics/
│   └── builtin_cases.jsonl         # 18 built-in close-call cases
├── judges/
│   └── openrouter.py               # OpenRouter adapter
├── mcp/
│   └── server.py                   # FastMCP tool definitions
├── models/
│   ├── input.py                    # AuditInput, AuditMode, TestType
│   ├── diagnostics.py              # DiagnosticCase schema
│   ├── test_case.py                # TestCase, JudgeDecision
│   ├── validation.py               # MutationValidation
│   └── report.py                   # TestResult, MetricSummary, AuditReport
├── utils/
│   ├── prompts.py                  # Judge and perturbation prompts
│   └── scoring.py                  # Reliability score, warnings, recommendations
├── tests/                          # Unit tests (pytest)
├── requirements.txt
└── .env.example
```

---

## Running tests

```bash
pytest -q
```

The test suite covers accuracy/invariance separation, stable-wrongness behavior, consistency profile labeling, reference accuracy, reliability score weighting, and role-based baseline routing.

---

## Limitations

This is a diagnostic toolkit, not a scientific benchmark.

- The built-in case library is intentionally small — 18 cases gives a signal, not a certificate
- Generated perturbations are LLM-produced and can be imperfect; inspect `mutation_validations` for warnings
- Strong claims require task-specific cases from your real evaluation distribution
- Model behavior on OpenRouter can vary across provider updates and versions
- Single-pair results describe fragility on the supplied example only

The safest interpretation:

> This tool estimates how a judge behaves on controlled probes. Use it to guide model selection and prompt debugging — not to replace a full human-curated benchmark.

---

## Key design principle

> **Reliability = correctness + robustness + stability.**
>
> A trustworthy judge picks the right answer, keeps picking it under irrelevant transformations, and gives the same answer when asked again.
> Stability alone is not enough — a consistently wrong judge is not reliable.
